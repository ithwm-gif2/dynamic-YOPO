import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config import cfg
from policy.geometry_temporal import EgoMotionDepthAligner
from policy.static_dynamic_dataset import StaticDynamicTemporalDataset


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Analyze depth residuals after ego-motion compensation."
    )
    parser.add_argument("--samples-per-episode", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--labeled", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def evaluate_scenario(
    scenario, samples_per_episode, batch_size, device, labeled=False
):
    yopo_root = Path(__file__).resolve().parents[1]
    dataset_paths = (
        cfg["sd_dataset_paths"] if labeled else cfg["temporal_dataset_paths"]
    )
    dataset_root = (yopo_root / dataset_paths[scenario]).resolve()
    dataset = StaticDynamicTemporalDataset(
        str(dataset_root),
        scenario=scenario,
        mode="valid",
        max_samples_per_episode=samples_per_episode,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    aligner = EgoMotionDepthAligner().to(device).eval()
    residual_samples = [[], []]
    valid_pixels = torch.zeros(2, dtype=torch.float64)
    absolute_sum = torch.zeros(2, dtype=torch.float64)
    threshold_counts = {
        threshold: torch.zeros(2, dtype=torch.float64)
        for threshold in (0.05, 0.10, 0.20, 0.40)
    }
    occlusion_count = torch.zeros(2, dtype=torch.float64)
    mask_positive_sum = torch.zeros(2, dtype=torch.float64)
    mask_positive_count = torch.zeros(2, dtype=torch.float64)

    for batch in loader:
        depth = batch[0].to(device, non_blocking=True)
        relative_pose = batch[1].to(device, non_blocking=True)
        dynamic_mask = batch[7].to(device, non_blocking=True)
        aligned = aligner(depth, relative_pose)
        absolute_m = (
            aligned["absolute_residual"]
            * float(cfg["sd_residual_scale"])
        )
        valid = aligned["valid"] > 0.5
        for interval in range(2):
            values = absolute_m[:, interval][valid[:, interval]]
            valid_pixels[interval] += values.numel()
            absolute_sum[interval] += values.double().sum().cpu()
            for threshold in threshold_counts:
                threshold_counts[threshold][interval] += (
                    values > threshold
                ).double().sum().cpu()
            occlusion_count[interval] += (
                aligned["occlusion"][:, interval] > 0.5
            ).double().sum().cpu()
            if values.numel() > 0:
                stride = max(1, values.numel() // 5000)
                residual_samples[interval].append(
                    values[::stride][:5000].float().cpu().numpy()
                )
            if (dynamic_mask >= 0.0).any():
                positive = (dynamic_mask > 0.5) & valid[:, interval]
                mask_positive_sum[interval] += absolute_m[:, interval][
                    positive
                ].double().sum().cpu()
                mask_positive_count[interval] += positive.double().sum().cpu()

    result = {"scenario": scenario, "samples": len(dataset), "intervals": []}
    for interval in range(2):
        samples = (
            np.concatenate(residual_samples[interval])
            if residual_samples[interval]
            else np.zeros(1, dtype=np.float32)
        )
        count = max(1.0, float(valid_pixels[interval]))
        interval_result = {
            "age_frames": int(cfg["sd_frame_offsets"][interval]),
            "valid_pixels": int(valid_pixels[interval]),
            "mean_abs_residual_m": float(absolute_sum[interval]) / count,
            "median_abs_residual_m": float(np.quantile(samples, 0.50)),
            "p90_abs_residual_m": float(np.quantile(samples, 0.90)),
            "p99_abs_residual_m": float(np.quantile(samples, 0.99)),
            "occlusion_fraction": float(occlusion_count[interval]) / count,
        }
        for threshold in threshold_counts:
            interval_result[f"fraction_gt_{threshold:.2f}m"] = (
                float(threshold_counts[threshold][interval]) / count
            )
        if mask_positive_count[interval] > 0:
            interval_result["dynamic_mask_mean_residual_m"] = (
                float(mask_positive_sum[interval])
                / float(mask_positive_count[interval])
            )
        result["intervals"].append(interval_result)
    return result


def synthetic_identity_check(device):
    aligner = EgoMotionDepthAligner().to(device).eval()
    depth = torch.full(
        (2, int(cfg["sd_history_length"]), int(cfg["image_height"]), int(cfg["image_width"])),
        0.25,
        device=device,
    )
    relative_pose = torch.zeros(
        2, int(cfg["sd_relative_pose_dim"]), device=device
    )
    aligned = aligner(depth, relative_pose)
    valid = aligned["valid"] > 0.5
    maximum = aligned["absolute_residual"][valid].max().item()
    return {"identity_max_abs_residual_normalized": maximum}


def main():
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(synthetic_identity_check(device))
    for scenario in ("static", "mixed", "dynamic"):
        print(
            evaluate_scenario(
                scenario,
                args.samples_per_episode,
                args.batch_size,
                device,
                labeled=args.labeled,
            )
        )


if __name__ == "__main__":
    main()
