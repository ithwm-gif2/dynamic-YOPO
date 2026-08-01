import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from config.config import cfg
from loss.temporal_loss import TemporalYOPOLoss
from policy.state_transform import state_body2world
from policy.static_dynamic_planning_dataset import StaticDynamicPlanningDataset
from policy.static_dynamic_yopo_network import StaticDynamicYopoNetwork


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train geometry-aware static/dynamic YOPO."
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", default="static_dynamic_yopo")
    parser.add_argument("--motion-checkpoint", required=True)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    return parser.parse_args()


def configure_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def dataset_root(path):
    return (Path(__file__).resolve().parent / path).resolve()


def build_loaders(batch_size, workers, device):
    ratios = {name: float(value) for name, value in cfg["scenario_sampling"].items()}
    train_datasets = []
    validation_loaders = {}
    for scenario in ("static", "mixed", "dynamic"):
        root = dataset_root(cfg["sd_dataset_paths"][scenario])
        train_dataset = StaticDynamicPlanningDataset(
            str(root), scenario=scenario, mode="train"
        )
        validation_dataset = StaticDynamicPlanningDataset(
            str(root), scenario=scenario, mode="valid"
        )
        train_datasets.append(train_dataset)
        validation_loaders[scenario] = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=device.type == "cuda",
            persistent_workers=workers > 0,
        )
    concatenated = ConcatDataset(train_datasets)
    weights = []
    for dataset in train_datasets:
        weights.append(
            torch.full(
                (len(dataset),),
                ratios[dataset.scenario] / len(dataset),
                dtype=torch.double,
            )
        )
    train_loader = DataLoader(
        concatenated,
        batch_size=batch_size,
        sampler=WeightedRandomSampler(
            torch.cat(weights), len(concatenated), replacement=True
        ),
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        drop_last=True,
    )
    return train_loader, validation_loaders, ratios


def coarse_mask_target(mask):
    fraction = F.adaptive_avg_pool2d(mask[:, None], (3, 5))[:, 0]
    return (fraction >= 0.02).float()


class StaticDynamicYopoTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.traj_num = int(cfg["traj_num"])
        self.vertical = int(cfg["vertical_num"])
        self.horizontal = int(cfg["horizon_num"])
        self.future_bins = int(cfg["sd_future_bins"])
        self.mask_weight = float(cfg["sd_mask_loss_weight"])
        self.future_risk_weight = float(cfg["sd_future_risk_weight"])
        self.score_regression_weight = float(cfg["sd_score_regression_weight"])
        self.score_ranking_weight = float(cfg["sd_score_ranking_weight"])
        self.safety_ranking_weight = float(cfg["sd_safety_ranking_weight"])
        self.train_loader, self.validation_loaders, self.ratios = build_loaders(
            args.batch_size, args.workers, self.device
        )
        self.model = StaticDynamicYopoNetwork().to(self.device)
        self.model.load_motion_checkpoint(args.motion_checkpoint, self.device)
        motion_parameters = (
            list(self.model.motion_model.parameters())
            + list(self.model.velocity_head.parameters())
        )
        motion_ids = {id(parameter) for parameter in motion_parameters}
        new_parameters = [
            parameter
            for parameter in self.model.parameters()
            if id(parameter) not in motion_ids
        ]
        self.optimizer = torch.optim.AdamW(
            [
                {"params": motion_parameters, "lr": args.learning_rate * 0.2},
                {"params": new_parameters, "lr": args.learning_rate},
            ],
            weight_decay=1e-4,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, args.epochs),
            eta_min=args.learning_rate * 0.05,
        )
        self.loss_function = TemporalYOPOLoss().to(self.device)
        self.lattice_positions_b = (
            self.model.state_transform.lattice_primitive.lattice_pos_node
            .flip(0)
            .to(self.device)
        )
        self.run_dir = (
            Path(__file__).resolve().parent / "saved_static_dynamic" / args.run_name
        )
        self.run_dir.mkdir(parents=True, exist_ok=False)
        with open(self.run_dir / "run_config.json", "w", encoding="utf-8") as file:
            json.dump(vars(args), file, indent=2)
        self.best_selection = float("inf")
        self.epochs_without_improvement = 0

    def move_batch(self, batch):
        return [
            value.to(self.device, non_blocking=True)
            if torch.is_tensor(value)
            else value
            for value in batch
        ]

    @torch.no_grad()
    def future_dynamic_target(
        self,
        position,
        rotation_wb,
        start_state_w,
        dynamic_boxes,
    ):
        batch_size = position.shape[0]
        end_position_b = self.lattice_positions_b[None].expand(
            batch_size, -1, -1
        ).reshape(batch_size * self.traj_num, 3)
        zeros = torch.zeros_like(end_position_b)
        repeated_position = position.repeat_interleave(self.traj_num, dim=0)
        repeated_rotation = rotation_wb.repeat_interleave(self.traj_num, dim=0)
        end_position_w, end_velocity_w, end_acceleration_w = state_body2world(
            repeated_position,
            repeated_rotation,
            end_position_b,
            zeros,
            zeros,
        )
        fixed_end_state = torch.stack(
            (end_position_w, end_velocity_w, end_acceleration_w), dim=1
        )
        repeated_start = start_state_w.repeat_interleave(self.traj_num, dim=0)
        trajectory = self.loss_function.safety_loss.trajectory_positions(
            repeated_start.permute(0, 2, 1),
            fixed_end_state.permute(0, 2, 1),
        ).reshape(
            batch_size,
            self.traj_num,
            int(cfg["temporal_eval_points"]),
            3,
        )
        distance = self.loss_function.safety_loss.signed_distance_to_aabb(
            trajectory[:, :, :, None, :],
            dynamic_boxes[:, None, :, :, 0:3],
            dynamic_boxes[:, None, :, :, 3:6],
        ).amin(dim=-1) - float(cfg["drone_radius"])
        time_chunks = torch.tensor_split(distance, self.future_bins, dim=2)
        risks = [
            torch.sigmoid(
                (float(cfg["d0"]) - chunk.amin(dim=2)) / float(cfg["r"])
            )
            for chunk in time_chunks
        ]
        return torch.stack(risks, dim=1).reshape(
            batch_size,
            self.future_bins,
            self.vertical,
            self.horizontal,
        )

    @staticmethod
    def mask_loss(logits, target):
        positive = target.sum()
        negative = target.numel() - positive
        positive_weight = (negative / positive.clamp_min(1.0)).clamp(1.0, 10.0)
        return F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=positive_weight
        )

    def forward_batch(self, batch):
        (
            depth,
            relative_pose,
            position,
            rotation_wb,
            observation_b,
            future_boxes,
            dynamic_boxes,
            current_boxes,
            dynamic_mask,
            scenario_id,
        ) = self.move_batch(batch)
        del current_boxes
        batch_size = depth.shape[0]
        goal_w, velocity_w, acceleration_w = state_body2world(
            position,
            rotation_wb,
            observation_b[:, 6:9],
            observation_b[:, 0:3],
            observation_b[:, 3:6],
        )
        start_state_w = torch.stack(
            (position, velocity_w, acceleration_w), dim=1
        )
        (
            endstate_b,
            score,
            current_dynamic_logits,
            future_dynamic_logits,
            _,
        ) = self.model.inference(
            depth,
            observation_b,
            relative_pose,
            return_auxiliary=True,
        )
        endstate_flat = endstate_b.permute(0, 2, 3, 1).reshape(
            batch_size * self.traj_num, 9
        )
        repeated_position = position.repeat_interleave(self.traj_num, dim=0)
        repeated_rotation = rotation_wb.repeat_interleave(self.traj_num, dim=0)
        repeated_start = start_state_w.repeat_interleave(self.traj_num, dim=0)
        repeated_goal = goal_w.repeat_interleave(self.traj_num, dim=0)
        end_position_w, end_velocity_w, end_acceleration_w = state_body2world(
            repeated_position,
            repeated_rotation,
            endstate_flat[:, 0:3],
            endstate_flat[:, 3:6],
            endstate_flat[:, 6:9],
        )
        end_state_w = torch.stack(
            (end_position_w, end_velocity_w, end_acceleration_w), dim=1
        )
        smooth, safety, guidance, acceleration, clearance = self.loss_function(
            repeated_start, end_state_w, repeated_goal, future_boxes
        )
        total_cost = smooth + safety + guidance + acceleration
        trajectory_loss = total_cost.mean()
        score_matrix = score.reshape(batch_size, self.traj_num)
        clearance_matrix = clearance.reshape(batch_size, self.traj_num)
        selection_cost = (
            total_cost.detach().reshape(batch_size, self.traj_num)
            + float(cfg["sd_collision_rank_penalty"])
            * (clearance_matrix <= 0.0).float()
            + float(cfg["sd_margin_rank_penalty"])
            * (clearance_matrix < float(cfg["d0"])).float()
        )
        score_target = torch.log1p(selection_cost)
        score_regression = F.smooth_l1_loss(score_matrix, score_target)
        ranking_temperature = float(cfg["sd_score_ranking_temperature"])
        target_distribution = torch.softmax(
            -selection_cost / ranking_temperature, dim=1
        )
        score_ranking = -(
            target_distribution
            * F.log_softmax(-score_matrix / ranking_temperature, dim=1)
        ).sum(dim=1).mean()

        safe_mask = clearance_matrix > 0.0
        collision_mask = ~safe_mask
        best_safe_score = score_matrix.masked_fill(~safe_mask, float("inf")).amin(dim=1)
        best_collision_score = score_matrix.masked_fill(
            ~collision_mask, float("inf")
        ).amin(dim=1)
        valid_pair = safe_mask.any(dim=1) & collision_mask.any(dim=1)
        if valid_pair.any():
            safety_ranking = F.softplus(
                float(cfg["sd_collision_score_margin"])
                + best_safe_score[valid_pair]
                - best_collision_score[valid_pair]
            ).mean()
        else:
            safety_ranking = score_matrix.sum() * 0.0

        mask_target = coarse_mask_target(dynamic_mask)
        current_mask_loss = self.mask_loss(current_dynamic_logits, mask_target)
        future_target = self.future_dynamic_target(
            position, rotation_wb, start_state_w, dynamic_boxes
        )
        future_risk_loss = F.binary_cross_entropy_with_logits(
            future_dynamic_logits, future_target
        )
        future_probability = torch.sigmoid(future_dynamic_logits)
        total_loss = (
            trajectory_loss
            + self.score_regression_weight * score_regression
            + self.score_ranking_weight * score_ranking
            + self.safety_ranking_weight * safety_ranking
            + self.mask_weight * current_mask_loss
            + self.future_risk_weight * future_risk_loss
        )
        selected_index = score_matrix.argmin(dim=1, keepdim=True)
        selected_clearance = clearance_matrix.gather(1, selected_index).squeeze(1)
        oracle_clearance = clearance_matrix.amax(dim=1)
        with torch.no_grad():
            mask_prediction = torch.sigmoid(current_dynamic_logits) >= 0.5
            mask_binary = mask_target > 0.5
            mask_intersection = (mask_prediction & mask_binary).sum()
            mask_union = (mask_prediction | mask_binary).sum()
            future_prediction_binary = future_probability >= 0.5
            future_target_binary = future_target >= 0.5
            future_intersection = (
                future_prediction_binary & future_target_binary
            ).sum()
            future_union = (
                future_prediction_binary | future_target_binary
            ).sum()
            future_true_positive = future_intersection
            future_false_positive = (
                future_prediction_binary & ~future_target_binary
            ).sum()
            future_false_negative = (
                ~future_prediction_binary & future_target_binary
            ).sum()
        metrics = {
            "loss": total_loss,
            "trajectory": trajectory_loss,
            "score_regression": score_regression,
            "score_ranking": score_ranking,
            "safety_ranking": safety_ranking,
            "mask_loss": current_mask_loss,
            "future_risk_loss": future_risk_loss,
            "future_target_mean": future_target.mean(),
            "future_prediction_mean": future_probability.mean(),
            "future_brier": F.mse_loss(future_probability, future_target),
            "selected_collision_rate": (
                selected_clearance <= 0.0
            ).float().mean(),
            "selected_margin_violation_rate": (
                selected_clearance < float(cfg["d0"])
            ).float().mean(),
            "oracle_collision_rate": (
                oracle_clearance <= 0.0
            ).float().mean(),
            "mask_intersection": mask_intersection.float(),
            "mask_union": mask_union.float(),
            "future_intersection": future_intersection.float(),
            "future_union": future_union.float(),
            "future_true_positive": future_true_positive.float(),
            "future_false_positive": future_false_positive.float(),
            "future_false_negative": future_false_negative.float(),
        }
        return metrics, scenario_id

    def train_epoch(self, epoch):
        self.model.train()
        # Preserve the calibrated running statistics learned by the motion branch.
        # Its parameters still receive gradients through the low-LR optimizer group.
        self.model.motion_model.eval()
        totals = {}
        for batch in self.train_loader:
            self.optimizer.zero_grad(set_to_none=True)
            metrics, _ = self.forward_batch(batch)
            metrics["loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach())
        print(
            f"epoch {epoch:02d}: train_loss={totals['loss'] / len(self.train_loader):.4f}"
        )

    @torch.inference_mode()
    def validate(self):
        self.model.eval()
        results = {}
        for scenario, loader in self.validation_loaders.items():
            totals = {}
            for batch in loader:
                metrics, _ = self.forward_batch(batch)
                for name, value in metrics.items():
                    totals[name] = totals.get(name, 0.0) + float(value)
            averages = {
                name: value / max(1, len(loader))
                for name, value in totals.items()
            }
            averages["mask_iou"] = totals["mask_intersection"] / max(
                1.0, totals["mask_union"]
            )
            averages["future_iou"] = totals["future_intersection"] / max(
                1.0, totals["future_union"]
            )
            averages["future_precision"] = totals["future_true_positive"] / max(
                1.0, totals["future_true_positive"]
                + totals["future_false_positive"]
            )
            averages["future_recall"] = totals["future_true_positive"] / max(
                1.0, totals["future_true_positive"]
                + totals["future_false_negative"]
            )
            results[scenario] = averages
            print(
                f"validation {scenario:>7}: collision="
                f"{averages['selected_collision_rate']:.3f}, oracle="
                f"{averages['oracle_collision_rate']:.3f}, maskIoU="
                f"{averages['mask_iou']:.3f}, future="
                f"{averages['future_risk_loss']:.3f}, target/pred="
                f"{averages['future_target_mean']:.3f}/"
                f"{averages['future_prediction_mean']:.3f}, brier="
                f"{averages['future_brier']:.3f}, futureIoU/P/R="
                f"{averages['future_iou']:.3f}/"
                f"{averages['future_precision']:.3f}/"
                f"{averages['future_recall']:.3f}"
            )
        selection = sum(
            self.ratios[scenario]
            * (
                results[scenario]["selected_collision_rate"]
                + 0.25
                * results[scenario]["selected_margin_violation_rate"]
                + 0.1 * results[scenario]["oracle_collision_rate"]
            )
            for scenario in self.ratios
        )
        return results, selection

    def save(self, path, epoch):
        torch.save(
            {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "best_selection": self.best_selection,
                "architecture": "static_dynamic_geometry_yopo",
            },
            path,
        )

    def train(self):
        for epoch in range(self.args.epochs):
            self.train_epoch(epoch)
            results, selection = self.validate()
            self.scheduler.step()
            if selection < self.best_selection:
                self.best_selection = selection
                self.epochs_without_improvement = 0
                self.save(self.run_dir / "best.pth", epoch)
                with open(
                    self.run_dir / "best_metrics.json", "w", encoding="utf-8"
                ) as file:
                    json.dump(results, file, indent=2)
                print(f"saved new best SD-YOPO: {self.run_dir / 'best.pth'}")
            else:
                self.epochs_without_improvement += 1
            self.save(self.run_dir / "latest.pth", epoch)
            if self.epochs_without_improvement >= self.args.early_stopping_patience:
                print("early stopping SD-YOPO")
                break


def main():
    args = parse_arguments()
    configure_seed(args.seed)
    StaticDynamicYopoTrainer(args).train()


if __name__ == "__main__":
    main()
