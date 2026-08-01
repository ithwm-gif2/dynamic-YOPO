import json

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from config.config import cfg
from policy.state_transform import state_body2world
from policy.temporal_yopo_dataset_v3 import TemporalYOPODataset
from policy.temporal_yopo_trainer import TemporalYopoTrainer as BaseTemporalTrainer


class TemporalYopoTrainer(BaseTemporalTrainer):
    """ConvGRU trainer with critical sampling, ranking, and motion supervision."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.score_regression_weight = float(
            cfg["temporal_score_regression_weight"]
        )
        self.score_ranking_weight = float(cfg["temporal_score_ranking_weight"])
        self.score_temperature = float(cfg["temporal_score_temperature"])
        self.collision_rank_penalty = float(
            cfg["temporal_collision_rank_penalty"]
        )
        self.margin_rank_penalty = float(cfg["temporal_margin_rank_penalty"])
        self.aux_risk_weight = float(cfg["temporal_aux_risk_weight"])
        self.aux_motion_weight = float(cfg["temporal_aux_motion_weight"])
        self.aux_presence_weight = float(cfg["temporal_aux_presence_weight"])
        self.aux_frozen_weight = float(cfg["temporal_aux_frozen_weight"])
        self.aux_positive_weight = float(cfg["temporal_aux_positive_weight"])
        self.lattice_positions_b = (
            self.policy.state_transform.lattice_primitive.lattice_pos_node
            .flip(0)
            .to(self.device)
        )

    def _write_run_config(self, learning_rate: float) -> None:
        super()._write_run_config(learning_rate)
        config_path = self.run_dir / "run_config.json"
        with open(config_path, "r", encoding="utf-8") as file:
            run_config = json.load(file)
        run_config.update(
            {
                "temporal_model": "shared_resnet18_convgru",
                "frame_feature_dim": int(cfg["temporal_frame_feature_dim"]),
                "gru_hidden_dim": int(cfg["temporal_gru_hidden_dim"]),
                "step_pose_dim": int(cfg["temporal_step_pose_dim"]),
                "critical_clearance": float(
                    cfg["temporal_critical_clearance"]
                ),
                "validation_samples_per_episode": int(
                    cfg["temporal_validation_samples_per_episode"]
                ),
                "critical_sampling_weight": float(
                    cfg["temporal_critical_sampling_weight"]
                ),
                "score_regression_weight": float(
                    cfg["temporal_score_regression_weight"]
                ),
                "score_ranking_weight": float(
                    cfg["temporal_score_ranking_weight"]
                ),
                "aux_risk_weight": float(cfg["temporal_aux_risk_weight"]),
                "aux_motion_weight": float(
                    cfg["temporal_aux_motion_weight"]
                ),
                "aux_presence_weight": float(
                    cfg["temporal_aux_presence_weight"]
                ),
                "aux_frozen_weight": float(
                    cfg["temporal_aux_frozen_weight"]
                ),
                "aux_positive_weight": float(
                    cfg["temporal_aux_positive_weight"]
                ),
                "aux_score_gain": float(cfg["temporal_aux_score_gain"]),
            }
        )
        with open(config_path, "w", encoding="utf-8") as file:
            json.dump(run_config, file, indent=2)

    def _build_dataloaders(
        self, max_samples_per_episode: int, samples_per_epoch: int
    ):
        train_datasets = []
        validation_loaders = {}
        validation_limit = (
            max_samples_per_episode
            or int(cfg["temporal_validation_samples_per_episode"])
        )
        for scenario in ("static", "mixed", "dynamic"):
            root = self._dataset_root(cfg["temporal_dataset_paths"][scenario])
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
                max_samples_per_episode=validation_limit,
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
            importance = torch.from_numpy(dataset.sample_importance).double()
            importance = importance / importance.sum().clamp_min(1e-12)
            weights.append(importance * self.scenario_ratios[dataset.scenario])
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
            f"{self.scenario_ratios}; critical-window sampling enabled"
        )
        return train_loader, validation_loaders

    @torch.no_grad()
    def _auxiliary_targets(
        self,
        position: torch.Tensor,
        rotation_wb: torch.Tensor,
        start_state_w: torch.Tensor,
        current_boxes: torch.Tensor,
        future_boxes: torch.Tensor,
    ):
        batch_size = position.shape[0]
        end_position_b = self.lattice_positions_b[None].expand(
            batch_size, -1, -1
        ).reshape(batch_size * self.traj_num, 3)
        zeros_b = torch.zeros_like(end_position_b)
        repeated_position = position.repeat_interleave(self.traj_num, dim=0)
        repeated_rotation = rotation_wb.repeat_interleave(
            self.traj_num, dim=0
        )
        end_position_w, end_velocity_w, end_acceleration_w = state_body2world(
            repeated_position,
            repeated_rotation,
            end_position_b,
            zeros_b,
            zeros_b,
        )
        fixed_end_state_w = torch.stack(
            (end_position_w, end_velocity_w, end_acceleration_w), dim=1
        )
        repeated_start = start_state_w.repeat_interleave(
            self.traj_num, dim=0
        )
        fixed_derivatives = repeated_start.permute(0, 2, 1)
        decision_derivatives = fixed_end_state_w.permute(0, 2, 1)

        _, future_clearance = self.loss_function.safety_loss(
            fixed_derivatives, decision_derivatives, future_boxes
        )
        static_boxes = current_boxes[:, None].expand(
            -1,
            int(cfg["temporal_eval_points"]),
            -1,
            -1,
        )
        _, static_clearance = self.loss_function.safety_loss(
            fixed_derivatives, decision_derivatives, static_boxes
        )
        risk_scale = float(cfg["r"])
        distance_margin = float(cfg["d0"])
        future_risk = torch.sigmoid(
            (distance_margin - future_clearance) / risk_scale
        )
        static_risk = torch.sigmoid(
            (distance_margin - static_clearance) / risk_scale
        )
        motion_delta = future_risk - static_risk
        motion_presence = (motion_delta.abs() / 0.15).clamp(0.0, 1.0)
        vertical = int(cfg["vertical_num"])
        horizontal = int(cfg["horizon_num"])
        return (
            future_risk.reshape(batch_size, vertical, horizontal),
            motion_delta.reshape(batch_size, vertical, horizontal),
            motion_presence.reshape(batch_size, vertical, horizontal),
        )

    def forward_and_compute_loss(self, batch):
        (
            depth,
            relative_pose,
            position,
            rotation_wb,
            observation_b,
            future_boxes,
            current_boxes,
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

        endstate_b, score, auxiliary = self.policy.inference(
            depth,
            observation_b,
            relative_pose,
            return_auxiliary=True,
        )
        frozen_auxiliary = None
        if self.policy.training:
            frozen_depth = depth[:, -1:].expand_as(depth).contiguous()
            frozen_relative_pose = relative_pose
            _, _, frozen_auxiliary = self.policy.inference(
                frozen_depth,
                observation_b,
                frozen_relative_pose,
                return_auxiliary=True,
            )
        endstate_flat = (
            endstate_b.permute(0, 2, 3, 1)
            .reshape(batch_size * self.traj_num, 9)
        )
        repeated_position = position.repeat_interleave(self.traj_num, dim=0)
        repeated_rotation = rotation_wb.repeat_interleave(
            self.traj_num, dim=0
        )
        repeated_start = start_state_w.repeat_interleave(
            self.traj_num, dim=0
        )
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

        smooth, safety, guidance, acceleration, minimum_clearance = (
            self.loss_function(
                repeated_start,
                end_state_w,
                repeated_goal,
                future_boxes,
            )
        )
        total_cost = smooth + safety + guidance + acceleration
        trajectory_loss = total_cost.mean()
        score_matrix = score.reshape(batch_size, self.traj_num)
        total_cost_matrix = total_cost.detach().reshape(
            batch_size, self.traj_num
        )
        clearance_matrix = minimum_clearance.reshape(
            batch_size, self.traj_num
        )

        score_target = torch.log1p(total_cost_matrix)
        score_regression_loss = F.smooth_l1_loss(
            score_matrix, score_target
        )
        selection_cost = (
            total_cost_matrix
            + self.collision_rank_penalty
            * (clearance_matrix <= 0.0).float()
            + self.margin_rank_penalty
            * (clearance_matrix < float(cfg["d0"])).float()
        )
        best_index = selection_cost.argmin(dim=1)
        score_ranking_loss = F.cross_entropy(
            -score_matrix / self.score_temperature,
            best_index,
        )

        zero = trajectory_loss.new_zeros(())
        motion_delta_target = zero
        auxiliary_risk_loss = zero
        auxiliary_motion_loss = zero
        auxiliary_presence_loss = zero
        frozen_motion_loss = zero
        if self.policy.training:
            (
                future_risk_target,
                motion_delta_target,
                motion_presence_target,
            ) = self._auxiliary_targets(
                position,
                rotation_wb,
                start_state_w,
                current_boxes,
                future_boxes,
            )
            auxiliary_risk_loss = F.binary_cross_entropy_with_logits(
                auxiliary[:, 0], future_risk_target
            )
            positive_weight = (
                1.0
                + (self.aux_positive_weight - 1.0)
                * motion_presence_target
            )
            auxiliary_motion_loss = (
                F.l1_loss(
                    torch.tanh(auxiliary[:, 1]),
                    motion_delta_target,
                    reduction="none",
                )
                * positive_weight
            ).mean()
            auxiliary_presence_loss = (
                F.binary_cross_entropy_with_logits(
                    auxiliary[:, 2],
                    motion_presence_target,
                    reduction="none",
                )
                * positive_weight
            ).mean()
            frozen_motion_loss = (
                torch.tanh(frozen_auxiliary[:, 1]).abs().mean()
                + F.binary_cross_entropy_with_logits(
                    frozen_auxiliary[:, 2],
                    torch.zeros_like(frozen_auxiliary[:, 2]),
                )
            )
        score_loss = (
            self.score_regression_weight * score_regression_loss
            + self.score_ranking_weight * score_ranking_loss
        )
        auxiliary_loss = (
            self.aux_risk_weight * auxiliary_risk_loss
            + self.aux_motion_weight * auxiliary_motion_loss
            + self.aux_presence_weight * auxiliary_presence_loss
            + self.aux_frozen_weight * frozen_motion_loss
        )
        total_loss = trajectory_loss + score_loss + auxiliary_loss

        selected_index = score_matrix.argmin(dim=1, keepdim=True)
        selected_clearance = clearance_matrix.gather(
            1, selected_index
        ).squeeze(1)
        oracle_clearance = clearance_matrix.amax(dim=1)
        selected_matches_oracle = (
            selected_index.squeeze(1) == best_index
        ).float().mean()
        metrics = {
            "loss": total_loss,
            "trajectory": trajectory_loss,
            "score": score_loss,
            "score_regression": score_regression_loss,
            "score_ranking": score_ranking_loss,
            "score_oracle_match": selected_matches_oracle,
            "auxiliary": auxiliary_loss,
            "auxiliary_risk": auxiliary_risk_loss,
            "auxiliary_motion": auxiliary_motion_loss,
            "auxiliary_presence": auxiliary_presence_loss,
            "auxiliary_frozen": frozen_motion_loss,
            "motion_target_abs": motion_delta_target.abs().mean(),
            "smooth": smooth.mean(),
            "safety": safety.mean(),
            "guidance": guidance.mean(),
            "acceleration": acceleration.mean(),
            "selected_clearance": selected_clearance.mean(),
            "selected_collision_rate": (
                selected_clearance <= 0.0
            ).float().mean(),
            "selected_margin_violation_rate": (
                selected_clearance < float(cfg["d0"])
            ).float().mean(),
            "oracle_collision_rate": (
                oracle_clearance <= 0.0
            ).float().mean(),
        }
        return metrics, scenario_id
