import torch
from torch import nn

from config.config import cfg
from policy.models.backbone import YopoBackbone
from policy.models.head import YopoHead
from policy.state_transform import StateTransform
from policy.static_dynamic_motion_network import StaticDynamicMotionNetwork


class StaticDynamicYopoNetwork(nn.Module):
    """YOPO with explicit static geometry and dynamic-motion branches."""

    def __init__(self, feature_dim: int = 64):
        super().__init__()
        self.state_transform = StateTransform()
        self.future_bins = int(cfg["sd_future_bins"])
        self.dynamic_score_gain = float(cfg["sd_dynamic_score_gain"])
        self.motion_model = StaticDynamicMotionNetwork(feature_dim=feature_dim)
        self.velocity_head = nn.Sequential(
            nn.Conv2d(feature_dim, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=1),
        )
        self.static_backbone = YopoBackbone(feature_dim, input_channels=1)
        self.pose_backbone = nn.Sequential(
            nn.Linear(int(cfg["sd_relative_pose_dim"]), 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
        )
        self.future_dynamic_head = nn.Sequential(
            nn.Conv2d(feature_dim + 9 + 3, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.future_bins, kernel_size=1),
        )
        head_channels = feature_dim * 2 + 9 + 32 + 1 + self.future_bins + 3
        self.yopo_head = YopoHead(head_channels, 10)

    def load_motion_checkpoint(self, checkpoint_path: str, device=None):
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device or "cpu",
            weights_only=False,
        )
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        self.motion_model.load_state_dict(state_dict)
        if "velocity_head" in checkpoint:
            self.velocity_head.load_state_dict(checkpoint["velocity_head"])

    def forward(
        self,
        depth_sequence: torch.Tensor,
        observation_grid: torch.Tensor,
        relative_pose: torch.Tensor,
        return_auxiliary: bool = False,
    ):
        current_depth = depth_sequence[:, -1:]
        static_feature = self.static_backbone(current_depth)
        current_dynamic_logits, motion_feature, alignment = self.motion_model(
            depth_sequence, relative_pose, return_alignment=True
        )
        predicted_velocity = torch.tanh(self.velocity_head(motion_feature))
        future_dynamic_logits = self.future_dynamic_head(
            torch.cat(
                (motion_feature, observation_grid, predicted_velocity), dim=1
            )
        )
        pose_feature = self.pose_backbone(relative_pose)
        pose_feature = pose_feature[:, :, None, None].expand(
            -1, -1, static_feature.shape[2], static_feature.shape[3]
        )
        current_dynamic_probability = torch.sigmoid(
            current_dynamic_logits
        )[:, None]
        future_dynamic_probability = torch.sigmoid(future_dynamic_logits)
        features = torch.cat(
            (
                static_feature,
                motion_feature,
                observation_grid,
                pose_feature,
                current_dynamic_probability,
                future_dynamic_probability,
                predicted_velocity,
            ),
            dim=1,
        )
        output = self.yopo_head(features)
        endstate = torch.tanh(output[:, :9])
        dynamic_risk = future_dynamic_probability.amax(dim=1)
        score = (
            torch.nn.functional.softplus(output[:, 9])
            + self.dynamic_score_gain * dynamic_risk
        )
        if return_auxiliary:
            return (
                endstate,
                score,
                current_dynamic_logits,
                future_dynamic_logits,
                alignment,
            )
        return endstate, score

    def inference(
        self,
        depth_sequence: torch.Tensor,
        observation: torch.Tensor,
        relative_pose: torch.Tensor,
        return_auxiliary: bool = False,
    ):
        normalized_observation = self.state_transform.normalize_obs(
            observation.clone()
        )
        observation_grid = self.state_transform.prepare_input(
            normalized_observation
        )
        output = self.forward(
            depth_sequence,
            observation_grid,
            relative_pose,
            return_auxiliary=return_auxiliary,
        )
        if return_auxiliary:
            endstate_prediction, score, current_logits, future_logits, alignment = output
        else:
            endstate_prediction, score = output
        endstate = self.state_transform.pred_to_endstate(endstate_prediction)
        if return_auxiliary:
            return endstate, score, current_logits, future_logits, alignment
        return endstate, score
