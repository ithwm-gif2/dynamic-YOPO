import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config import cfg
from policy.geometry_temporal import EgoMotionDepthAligner
from policy.static_dynamic_dataset import StaticDynamicTemporalDataset


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate geometry residual as a dynamic-pixel baseline."
    )
    parser.add_argument("--samples-per-episode", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


@torch.inference_mode()
def evaluate(scenario, args, device):
    root = (
        Path(__file__).resolve().parents[1]
        / cfg["sd_dataset_paths"][scenario]
    ).resolve()
    dataset = StaticDynamicTemporalDataset(
        str(root),
        scenario=scenario,
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
    aligner = EgoMotionDepthAligner().to(device).eval()
    thresholds = (0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40)
    confusion = {
        threshold: torch.zeros(4, dtype=torch.float64)
        for threshold in thresholds
    }
    positive_pixels = 0
    valid_pixels = 0
    for batch in loader:
        depth = batch[0].to(device, non_blocking=True)
        relative_pose = batch[1].to(device, non_blocking=True)
        target = batch[7].to(device, non_blocking=True) > 0.5
        aligned = aligner(depth, relative_pose)
        residual_m = (
            aligned["absolute_residual"]
            * float(cfg["sd_residual_scale"])
        )
        validity = aligned["valid"] > 0.5
        residual_m = residual_m.masked_fill(~validity, 0.0)
        combined_residual = residual_m.amax(dim=1)
        combined_valid = validity.any(dim=1)
        target = target & combined_valid
        positive_pixels += int(target.sum())
        valid_pixels += int(combined_valid.sum())
        for threshold in thresholds:
            prediction = (combined_residual > threshold) & combined_valid
            true_positive = (prediction & target).sum()
            false_positive = (prediction & ~target & combined_valid).sum()
            false_negative = (~prediction & target).sum()
            true_negative = (~prediction & ~target & combined_valid).sum()
            confusion[threshold] += torch.stack(
                (true_positive, false_positive, false_negative, true_negative)
            ).double().cpu()

    rows = []
    for threshold, values in confusion.items():
        true_positive, false_positive, false_negative, true_negative = [
            float(value) for value in values
        ]
        precision = true_positive / max(1.0, true_positive + false_positive)
        recall = true_positive / max(1.0, true_positive + false_negative)
        f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
        iou = true_positive / max(
            1.0, true_positive + false_positive + false_negative
        )
        false_positive_rate = false_positive / max(
            1.0, false_positive + true_negative
        )
        rows.append(
            {
                "threshold_m": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "iou": iou,
                "static_false_positive_rate": false_positive_rate,
            }
        )
    return {
        "scenario": scenario,
        "samples": len(dataset),
        "dynamic_pixel_fraction": positive_pixels / max(1, valid_pixels),
        "best": max(rows, key=lambda row: row["f1"]),
        "thresholds": rows,
    }


def main():
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for scenario in ("static", "mixed", "dynamic"):
        print(evaluate(scenario, args, device))


if __name__ == "__main__":
    main()
