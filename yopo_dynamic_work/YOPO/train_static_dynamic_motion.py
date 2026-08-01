import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from config.config import cfg
from policy.static_dynamic_mask_dataset import StaticDynamicMaskDataset
from policy.static_dynamic_motion_network import StaticDynamicMotionNetwork


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Pretrain ego-compensated static/dynamic separation."
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", default="sd_motion")
    parser.add_argument("--early-stopping-patience", type=int, default=3)
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
        train_dataset = StaticDynamicMaskDataset(
            str(root), scenario=scenario, mode="train"
        )
        validation_dataset = StaticDynamicMaskDataset(
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
    sampler = WeightedRandomSampler(
        torch.cat(weights),
        num_samples=len(concatenated),
        replacement=True,
    )
    train_loader = DataLoader(
        concatenated,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        drop_last=True,
    )
    return train_loader, validation_loaders


def coarse_target(mask):
    occupancy_fraction = F.adaptive_avg_pool2d(mask[:, None], (3, 5))[:, 0]
    return (occupancy_fraction >= 0.02).float()


def mask_loss(logits, target):
    positive = target.sum()
    negative = target.numel() - positive
    positive_weight = (negative / positive.clamp_min(1.0)).clamp(1.0, 10.0)
    bce = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=positive_weight
    )
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(1, 2))
    denominator = probability.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return bce + 0.5 * dice, bce, dice


def velocity_loss(prediction, target, valid):
    valid_channels = valid[:, None]
    element_loss = F.smooth_l1_loss(prediction, target, reduction="none")
    denominator = (valid.sum() * prediction.shape[1]).clamp_min(1.0)
    return (element_loss * valid_channels).sum() / denominator


def velocity_error_sum(prediction, target, valid):
    scale = float(cfg["sd_velocity_scale"])
    valid_channels = valid[:, None]
    absolute_error = (prediction - target).abs() * scale * valid_channels
    count = valid.sum() * prediction.shape[1]
    return absolute_error.sum().double().cpu(), count.double().cpu()


def confusion(logits, target):
    prediction = torch.sigmoid(logits) >= 0.5
    target = target > 0.5
    return torch.stack(
        (
            (prediction & target).sum(),
            (prediction & ~target).sum(),
            (~prediction & target).sum(),
            (~prediction & ~target).sum(),
        )
    ).double().cpu()


def metrics_from_confusion(values):
    true_positive, false_positive, false_negative, true_negative = [
        float(value) for value in values
    ]
    precision = true_positive / max(1.0, true_positive + false_positive)
    recall = true_positive / max(1.0, true_positive + false_negative)
    iou = true_positive / max(
        1.0, true_positive + false_positive + false_negative
    )
    false_positive_rate = false_positive / max(
        1.0, false_positive + true_negative
    )
    return {
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "false_positive_rate": false_positive_rate,
    }


@torch.inference_mode()
def validate(model, velocity_head, loaders, device):
    model.eval()
    velocity_head.eval()
    results = {}
    velocity_weight = float(cfg["sd_velocity_loss_weight"])
    for scenario, loader in loaders.items():
        total_loss = 0.0
        total_confusion = torch.zeros(4, dtype=torch.float64)
        total_velocity_error = torch.tensor(0.0, dtype=torch.float64)
        total_velocity_count = torch.tensor(0.0, dtype=torch.float64)
        for (
            depth,
            relative_pose,
            mask,
            velocity_target,
            velocity_valid,
            _,
        ) in loader:
            depth = depth.to(device, non_blocking=True)
            relative_pose = relative_pose.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            velocity_target = velocity_target.to(device, non_blocking=True)
            velocity_valid = velocity_valid.to(device, non_blocking=True)
            target = coarse_target(mask)
            logits, motion_feature, _ = model(
                depth, relative_pose, return_alignment=True
            )
            velocity_prediction = torch.tanh(velocity_head(motion_feature))
            segmentation_loss, _, _ = mask_loss(logits, target)
            regression_loss = velocity_loss(
                velocity_prediction, velocity_target, velocity_valid
            )
            total_loss += float(
                segmentation_loss + velocity_weight * regression_loss
            )
            total_confusion += confusion(logits, target)
            error_sum, error_count = velocity_error_sum(
                velocity_prediction, velocity_target, velocity_valid
            )
            total_velocity_error += error_sum
            total_velocity_count += error_count
        result = metrics_from_confusion(total_confusion)
        result["loss"] = total_loss / max(1, len(loader))
        result["velocity_mae_mps"] = float(
            total_velocity_error / total_velocity_count.clamp_min(1.0)
        )
        result["velocity_values"] = int(total_velocity_count)
        results[scenario] = result
        print(
            f"validation {scenario:>7}: loss={result['loss']:.4f}, "
            f"IoU={result['iou']:.3f}, precision={result['precision']:.3f}, "
            f"recall={result['recall']:.3f}, FPR="
            f"{result['false_positive_rate']:.3f}, velocityMAE="
            f"{result['velocity_mae_mps']:.3f}m/s "
            f"(n={result['velocity_values']})"
        )
    velocity_scale = float(cfg["sd_velocity_scale"])
    results["selection"] = (
        0.6 * (1.0 - results["mixed"]["iou"])
        + 0.4 * (1.0 - results["dynamic"]["iou"])
        + 0.5 * results["static"]["false_positive_rate"]
        + 0.1
        * (
            results["mixed"]["velocity_mae_mps"]
            + results["dynamic"]["velocity_mae_mps"]
        )
        / (2.0 * velocity_scale)
    )
    return results


def save_checkpoint(
    path, epoch, model, velocity_head, optimizer, best_selection
):
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "velocity_head": velocity_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_selection": best_selection,
            "architecture": "ego_compensated_residual_convgru_velocity_aux",
        },
        path,
    )


def main():
    args = parse_arguments()
    configure_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(__file__).resolve().parent / "saved_static_dynamic" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    with open(run_dir / "run_config.json", "w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2)

    train_loader, validation_loaders = build_loaders(
        args.batch_size, args.workers, device
    )
    model = StaticDynamicMotionNetwork().to(device)
    velocity_head = torch.nn.Sequential(
        torch.nn.Conv2d(64, 64, kernel_size=1),
        torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(64, 3, kernel_size=1),
    ).to(device)
    velocity_weight = float(cfg["sd_velocity_loss_weight"])
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(velocity_head.parameters()),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs), eta_min=args.learning_rate * 0.05
    )
    best_selection = float("inf")
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        model.train()
        velocity_head.train()
        running = 0.0
        running_velocity = 0.0
        for step, (
            depth,
            relative_pose,
            mask,
            velocity_target,
            velocity_valid,
            _,
        ) in enumerate(train_loader):
            depth = depth.to(device, non_blocking=True)
            relative_pose = relative_pose.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            velocity_target = velocity_target.to(device, non_blocking=True)
            velocity_valid = velocity_valid.to(device, non_blocking=True)
            target = coarse_target(mask)
            optimizer.zero_grad(set_to_none=True)
            logits, motion_feature, _ = model(
                depth, relative_pose, return_alignment=True
            )
            velocity_prediction = torch.tanh(velocity_head(motion_feature))
            segmentation_loss, _, _ = mask_loss(logits, target)
            regression_loss = velocity_loss(
                velocity_prediction, velocity_target, velocity_valid
            )
            loss = segmentation_loss + velocity_weight * regression_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(velocity_head.parameters()), 1.0
            )
            optimizer.step()
            running += float(loss.detach())
            running_velocity += float(regression_loss.detach())
        scheduler.step()
        print(
            f"epoch {epoch:02d}: train_loss={running / len(train_loader):.4f}, "
            f"velocity={running_velocity / len(train_loader):.4f}"
        )
        validation = validate(model, velocity_head, validation_loaders, device)
        if validation["selection"] < best_selection:
            best_selection = validation["selection"]
            epochs_without_improvement = 0
            save_checkpoint(
                run_dir / "best.pth",
                epoch,
                model,
                velocity_head,
                optimizer,
                best_selection,
            )
            with open(
                run_dir / "best_metrics.json", "w", encoding="utf-8"
            ) as file:
                json.dump(validation, file, indent=2)
            print(f"saved new best motion model: {run_dir / 'best.pth'}")
        else:
            epochs_without_improvement += 1
        save_checkpoint(
            run_dir / "latest.pth",
            epoch,
            model,
            velocity_head,
            optimizer,
            best_selection,
        )
        if epochs_without_improvement >= args.early_stopping_patience:
            print("early stopping motion pretraining")
            break


if __name__ == "__main__":
    main()
