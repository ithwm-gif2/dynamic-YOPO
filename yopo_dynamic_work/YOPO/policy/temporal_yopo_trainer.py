import json
import os
import random
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    WeightedRandomSampler,
)
from torch.utils.tensorboard import SummaryWriter

from config.config import cfg
from loss.temporal_loss import TemporalYOPOLoss
from policy.state_transform import state_body2world
from policy.temporal_yopo_dataset_v2 import TemporalYOPODataset
from policy.temporal_yopo_network_v2 import TemporalYopoNetwork


class TemporalYopoTrainer:
    def __init__(
        self,
        learning_rate: float = 1.5e-4,
        batch_size: int = 16,
        workers: int = 4,
        epochs: int = 30,
        log_root: str = "saved_temporal",
        run_name: str = "",
        resume: str = "",
        max_samples_per_episode: int = 0,
        samples_per_epoch: int = 0,
        early_stopping_patience: int = 3,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.workers = workers
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.epochs_without_improvement = 0
        self.max_grad_norm = 1.0
        self.traj_num = int(cfg["traj_num"])
        self.scenario_ratios = {
            name: float(value) for name, value in cfg["scenario_sampling"].items()
        }
        if not np.isclose(sum(self.scenario_ratios.values()), 1.0):
            raise ValueError("Scenario sampling weights must sum to one")

        self.run_dir = self._make_run_dir(log_root, run_name)
        self.writer = SummaryWriter(str(self.run_dir))
        self._write_run_config(learning_rate)

        self.policy = TemporalYopoNetwork().to(self.device)
        self.loss_function = TemporalYOPOLoss().to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=learning_rate, weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, epochs), eta_min=learning_rate * 0.05
        )
        self.start_epoch = 0
        self.best_validation = float("inf")
        if resume:
            self._load_checkpoint(resume)
        else:
            print("Training temporal YOPO from random initialization")

        self.train_loader, self.validation_loaders = self._build_dataloaders(
            max_samples_per_episode=max_samples_per_episode,
            samples_per_epoch=samples_per_epoch,
        )

    @staticmethod
    def _dataset_root(configured_path: str) -> Path:
        yopo_root = Path(__file__).resolve().parents[1]
        return (yopo_root / configured_path).resolve()

    def _build_dataloaders(
        self, max_samples_per_episode: int, samples_per_epoch: int
    ):
        train_datasets = []
        validation_loaders = {}
        dataset_paths = cfg["temporal_dataset_paths"]
        for scenario in ("static", "mixed", "dynamic"):
            root = self._dataset_root(dataset_paths[scenario])
            train_dataset = TemporalYOPODataset(
                str(root),
                scenario=scenario,
                mode="train",
                max_samples_per_episode=max_samples_per_episode,
            )
            validation_dataset = TemporalYOPODataset(
                str(root),
                scenario=scenario,
                mode="valid",
                max_samples_per_episode=max_samples_per_episode,
            )
            train_datasets.append(train_dataset)
            validation_loaders[scenario] = DataLoader(
                validation_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.workers,
                pin_memory=self.device.type == "cuda",
                persistent_workers=self.workers > 0,
            )

        concatenated = ConcatDataset(train_datasets)
        weights = []
        for dataset in train_datasets:
            per_item_weight = self.scenario_ratios[dataset.scenario] / len(dataset)
            weights.append(
                torch.full((len(dataset),), per_item_weight, dtype=torch.double)
            )
        sampler_weights = torch.cat(weights)
        epoch_samples = samples_per_epoch or len(concatenated)
        sampler = WeightedRandomSampler(
            sampler_weights,
            num_samples=epoch_samples,
            replacement=True,
        )
        train_loader = DataLoader(
            concatenated,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=self.workers > 0,
            drop_last=True,
        )
        print(
            f"Training samples/epoch: {epoch_samples}; expected mix "
            f"{self.scenario_ratios}"
        )
        return train_loader, validation_loaders

    def _make_run_dir(self, log_root: str, run_name: str) -> Path:
        root = Path(__file__).resolve().parents[1] / log_root
        root.mkdir(parents=True, exist_ok=True)
        if run_name:
            run_dir = root / run_name
            run_dir.mkdir(parents=False, exist_ok=False)
            return run_dir

        indices = []
        for path in root.glob("TemporalYOPO_*"):
            try:
                indices.append(int(path.name.rsplit("_", 1)[1]))
            except ValueError:
                pass
        run_dir = root / f"TemporalYOPO_{max(indices, default=-1) + 1}"
        run_dir.mkdir(parents=False, exist_ok=False)
        return run_dir

    def _write_run_config(self, learning_rate: float) -> None:
        run_config = {
            "random_initialization": True,
            "learning_rate": learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "history_length": int(cfg["history_length"]),
            "relative_pose_dim": int(cfg["relative_pose_dim"]),
            "temporal_eval_points": int(cfg["temporal_eval_points"]),
            "temporal_peak_weight": float(cfg["temporal_peak_weight"]),
            "map_half_extent_xy": list(cfg["temporal_map_half_extent_xy"]),
            "map_ceiling": float(cfg["temporal_map_ceiling"]),
            "scenario_sampling": self.scenario_ratios,
            "dataset_paths": dict(cfg["temporal_dataset_paths"]),
        }
        with open(self.run_dir / "run_config.json", "w", encoding="utf-8") as file:
            json.dump(run_config, file, indent=2)

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        self.policy.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.best_validation = float(
            checkpoint.get("best_validation", float("inf"))
        )
        self.epochs_without_improvement = int(
            checkpoint.get("epochs_without_improvement", 0)
        )
        print(f"Resumed temporal YOPO from {checkpoint_path}")

    def save_checkpoint(self, epoch: int, name: str) -> Path:
        checkpoint_path = self.run_dir / name
        torch.save(
            {
                "epoch": epoch,
                "model": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "best_validation": self.best_validation,
                "epochs_without_improvement": self.epochs_without_improvement,
            },
            checkpoint_path,
        )
        return checkpoint_path

    def _move_batch(self, batch):
        return [
            value.to(self.device, non_blocking=True)
            if torch.is_tensor(value)
            else value
            for value in batch
        ]

    def forward_and_compute_loss(self, batch):
        (
            depth,
            relative_pose,
            position,
            rotation_wb,
            observation_b,
            future_boxes,
            scenario_id,
        ) = self._move_batch(batch)
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

        endstate_b, score = self.policy.inference(
            depth, observation_b, relative_pose
        )
        endstate_flat = (
            endstate_b.permute(0, 2, 3, 1)
            .reshape(batch_size * self.traj_num, 9)
        )
        score_flat = score.reshape(batch_size * self.traj_num)

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

        smooth, safety, guidance, acceleration, minimum_clearance = self.loss_function(
            repeated_start, end_state_w, repeated_goal, future_boxes
        )
        total_cost = smooth + safety + guidance + acceleration
        trajectory_loss = total_cost.mean()
        score_loss = F.smooth_l1_loss(score_flat, total_cost.detach())
        total_loss = trajectory_loss + score_loss
        score_matrix = score_flat.reshape(batch_size, self.traj_num)
        clearance_matrix = minimum_clearance.reshape(
            batch_size, self.traj_num
        )
        selected_index = score_matrix.argmin(dim=1, keepdim=True)
        selected_clearance = clearance_matrix.gather(
            1, selected_index
        ).squeeze(1)
        oracle_clearance = clearance_matrix.amax(dim=1)
        metrics = {
            "loss": total_loss,
            "trajectory": trajectory_loss,
            "score": score_loss,
            "smooth": smooth.mean(),
            "safety": safety.mean(),
            "guidance": guidance.mean(),
            "acceleration": acceleration.mean(),
            "selected_clearance": selected_clearance.mean(),
            "selected_collision_rate": (selected_clearance <= 0.0).float().mean(),
            "selected_margin_violation_rate": (
                selected_clearance < float(cfg["d0"])
            ).float().mean(),
            "oracle_collision_rate": (oracle_clearance <= 0.0).float().mean(),
        }
        return metrics, scenario_id

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.policy.train()
        totals: Dict[str, float] = {}
        scenario_counts = torch.zeros(3, dtype=torch.long)
        start_time = time.time()
        for step, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad(set_to_none=True)
            metrics, scenario_id = self.forward_and_compute_loss(batch)
            metrics["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.max_grad_norm
            )
            self.optimizer.step()

            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach())
            scenario_counts += torch.bincount(
                scenario_id.detach().cpu(), minlength=3
            )

            if (step + 1) % 200 == 0 or step + 1 == len(self.train_loader):
                elapsed = time.time() - start_time
                print(
                    f"epoch {epoch:02d} step {step + 1:05d}/"
                    f"{len(self.train_loader):05d} "
                    f"loss={totals['loss'] / (step + 1):.4g} "
                    f"{(step + 1) * self.batch_size / elapsed:.1f} samples/s"
                )

        steps = len(self.train_loader)
        averages = {name: value / steps for name, value in totals.items()}
        mix = scenario_counts.float() / scenario_counts.sum().clamp_min(1)
        averages.update(
            {
                "mix_static": float(mix[0]),
                "mix_mixed": float(mix[1]),
                "mix_dynamic": float(mix[2]),
            }
        )
        return averages

    @torch.inference_mode()
    def validate(self, epoch: int) -> Dict[str, float]:
        self.policy.eval()
        all_metrics: Dict[str, float] = {}
        for scenario, loader in self.validation_loaders.items():
            totals: Dict[str, float] = {}
            for batch in loader:
                metrics, _ = self.forward_and_compute_loss(batch)
                for name, value in metrics.items():
                    totals[name] = totals.get(name, 0.0) + float(value)
            scenario_metrics = {
                name: value / max(1, len(loader)) for name, value in totals.items()
            }
            for name, value in scenario_metrics.items():
                all_metrics[f"{scenario}/{name}"] = value
            print(
                f"validation {scenario:>7}: "
                f"trajectory={scenario_metrics['trajectory']:.4g}, "
                f"safety={scenario_metrics['safety']:.4g}, "
                f"collision={scenario_metrics['selected_collision_rate']:.3f}, "
                f"oracle_collision={scenario_metrics['oracle_collision_rate']:.3f}"
            )

        weighted_trajectory = sum(
            self.scenario_ratios[scenario]
            * all_metrics[f"{scenario}/trajectory"]
            for scenario in self.scenario_ratios
        )
        all_metrics["weighted_trajectory"] = weighted_trajectory
        for metric in (
            "selected_collision_rate",
            "selected_margin_violation_rate",
            "oracle_collision_rate",
        ):
            all_metrics[f"weighted_{metric}"] = sum(
                self.scenario_ratios[scenario]
                * all_metrics[f"{scenario}/{metric}"]
                for scenario in self.scenario_ratios
            )
        all_metrics["safety_selection"] = (
            all_metrics["weighted_selected_collision_rate"]
            + 0.25 * all_metrics["weighted_selected_margin_violation_rate"]
            + 0.10 * all_metrics["weighted_oracle_collision_rate"]
        )
        return all_metrics

    def _log_metrics(self, prefix: str, metrics: Dict[str, float], epoch: int):
        for name, value in metrics.items():
            self.writer.add_scalar(f"{prefix}/{name}", value, epoch)

    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            train_metrics = self.train_one_epoch(epoch)
            validation_metrics = self.validate(epoch)
            self.scheduler.step()
            self._log_metrics("train", train_metrics, epoch)
            self._log_metrics("validation", validation_metrics, epoch)
            self.writer.add_scalar(
                "train/learning_rate",
                self.optimizer.param_groups[0]["lr"],
                epoch,
            )

            if validation_metrics["safety_selection"] < self.best_validation:
                self.best_validation = validation_metrics["safety_selection"]
                self.epochs_without_improvement = 0
                best = self.save_checkpoint(epoch, "best.pth")
                print(f"saved new best checkpoint: {best}")
            else:
                self.epochs_without_improvement += 1
            latest = self.save_checkpoint(epoch, "latest.pth")
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch, f"epoch{epoch + 1}.pth")
            print(f"saved latest checkpoint: {latest}")
            self.writer.flush()
            if (
                self.early_stopping_patience > 0
                and self.epochs_without_improvement
                >= self.early_stopping_patience
            ):
                print(
                    "early stopping: safety selection did not improve for "
                    f"{self.early_stopping_patience} epochs"
                )
                break

        self.writer.close()
        print(f"Temporal YOPO training finished: {self.run_dir}")
