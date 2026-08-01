import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader

from config.config import cfg
from loss.temporal_loss import TemporalYOPOLoss
from policy.state_transform import state_body2world
from policy.temporal_yopo_dataset import TemporalYOPODataset
from policy.temporal_yopo_network import TemporalYopoNetwork as TemporalYopoV1
from policy.temporal_yopo_network_v2 import TemporalYopoNetwork as TemporalYopoV2


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Measure whether Temporal YOPO uses frame history."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scenario", default="dynamic")
    parser.add_argument("--samples-per-episode", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model_class = (
        TemporalYopoV2
        if any(key.startswith("temporal_fusion.") for key in state_dict)
        else TemporalYopoV1
    )
    model = model_class().to(device).eval()
    model.load_state_dict(state_dict)
    return model


@torch.inference_mode()
def evaluate(model, loader, loss_function, device, mode: str):
    trajectory_count = int(cfg["traj_num"])
    totals = {
        "samples": 0,
        "collisions": 0,
        "margin_violations": 0,
        "backward": 0,
        "along_goal": 0.0,
        "clearance": 0.0,
    }
    action_histogram = torch.zeros(trajectory_count, dtype=torch.long)
    for batch in loader:
        depth, relative_pose, position, rotation, observation, boxes, _ = [
            value.to(device, non_blocking=True) for value in batch
        ]
        if mode in ("frozen", "single"):
            depth = depth[:, -1:].expand(
                -1, depth.shape[1], -1, -1
            ).contiguous()
        if mode in ("no_pose", "single"):
            relative_pose = torch.zeros_like(relative_pose)

        batch_size = depth.shape[0]
        goal_w, velocity_w, acceleration_w = state_body2world(
            position,
            rotation,
            observation[:, 6:9],
            observation[:, 0:3],
            observation[:, 3:6],
        )
        start_state = torch.stack(
            (position, velocity_w, acceleration_w), dim=1
        )
        endstate_b, score = model.inference(
            depth, observation, relative_pose
        )
        endstate_flat = endstate_b.permute(0, 2, 3, 1).reshape(
            batch_size * trajectory_count, 9
        )
        repeated_position = position.repeat_interleave(
            trajectory_count, dim=0
        )
        repeated_rotation = rotation.repeat_interleave(
            trajectory_count, dim=0
        )
        end_position, end_velocity, end_acceleration = state_body2world(
            repeated_position,
            repeated_rotation,
            endstate_flat[:, 0:3],
            endstate_flat[:, 3:6],
            endstate_flat[:, 6:9],
        )
        end_state = torch.stack(
            (end_position, end_velocity, end_acceleration), dim=1
        )
        repeated_start = start_state.repeat_interleave(
            trajectory_count, dim=0
        )
        repeated_goal = goal_w.repeat_interleave(trajectory_count, dim=0)
        _, _, _, _, clearance = loss_function(
            repeated_start, end_state, repeated_goal, boxes
        )
        clearance = clearance.reshape(batch_size, trajectory_count)
        score = score.reshape(batch_size, trajectory_count)
        selected_index = score.argmin(dim=1)
        selected_clearance = clearance.gather(
            1, selected_index[:, None]
        ).squeeze(1)

        end_positions_b = endstate_b.permute(0, 2, 3, 1).reshape(
            batch_size, trajectory_count, 9
        )[:, :, 0:3]
        selected_position = end_positions_b.gather(
            1, selected_index[:, None, None].expand(-1, 1, 3)
        ).squeeze(1)
        goal_b = observation[:, 6:9]
        goal_unit = goal_b / goal_b.norm(dim=1, keepdim=True).clamp_min(1e-6)
        along_goal = (selected_position * goal_unit).sum(dim=1)

        totals["samples"] += batch_size
        totals["collisions"] += int((selected_clearance <= 0.0).sum())
        totals["margin_violations"] += int(
            (selected_clearance < float(cfg["d0"])).sum()
        )
        totals["backward"] += int((along_goal <= 0.0).sum())
        totals["along_goal"] += float(along_goal.sum())
        totals["clearance"] += float(selected_clearance.sum())
        action_histogram += torch.bincount(
            selected_index.cpu(), minlength=trajectory_count
        )

    count = totals["samples"]
    return {
        "samples": count,
        "collision_rate": totals["collisions"] / count,
        "margin_violation_rate": totals["margin_violations"] / count,
        "backward_rate": totals["backward"] / count,
        "mean_along_goal_m": totals["along_goal"] / count,
        "mean_min_clearance_m": totals["clearance"] / count,
        "action_histogram": action_histogram.tolist(),
    }


def main():
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = (
        Path(__file__).resolve().parents[1]
        / cfg["temporal_dataset_paths"][args.scenario]
    ).resolve()
    dataset = TemporalYOPODataset(
        str(root),
        scenario=args.scenario,
        mode="valid",
        max_samples_per_episode=args.samples_per_episode,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    model = load_model(args.checkpoint, device)
    loss_function = TemporalYOPOLoss().to(device)
    for mode in ("normal", "frozen", "no_pose", "single"):
        print(mode, evaluate(model, loader, loss_function, device, mode))


if __name__ == "__main__":
    main()
