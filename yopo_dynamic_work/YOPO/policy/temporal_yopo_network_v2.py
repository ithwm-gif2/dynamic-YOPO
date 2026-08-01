import torch
from torch import nn

from config.config import cfg
from policy.models.backbone import YopoBackbone
from policy.models.conv_gru import ConvGRU
from policy.models.head import YopoHead
from policy.state_transform import StateTransform


class TemporalYopoNetwork(nn.Module):
    """YOPO with shared frame encoding and spatial ConvGRU fusion."""

    def __init__(self, observation_dim: int = 9, output_dim: int = 10):
        super().__init__()
        self.state_transform = StateTransform()
        self.history_length = int(cfg["history_length"])
        self.relative_pose_dim = int(cfg["relative_pose_dim"])
        self.pose_feature_dim = int(cfg["relative_pose_feature_dim"])
        self.translation_scale = float(cfg["relative_translation_scale"])
        self.rotation_scale = float(cfg["relative_rotation_scale"])
        self.frame_feature_dim = int(cfg["temporal_frame_feature_dim"])
        self.gru_hidden_dim = int(cfg["temporal_gru_hidden_dim"])
        self.step_pose_dim = int(cfg["temporal_step_pose_dim"])
        self.aux_score_gain = float(cfg["temporal_aux_score_gain"])

        self.image_backbone = YopoBackbone(
            self.frame_feature_dim, input_channels=1
        )
        self.step_pose_backbone = nn.Sequential(
            nn.Linear(6, self.step_pose_dim),
            nn.ReLU(inplace=True),
        )
        self.temporal_fusion = ConvGRU(
            input_channels=self.frame_feature_dim + self.step_pose_dim,
            hidden_channels=self.gru_hidden_dim,
        )
        self.pose_backbone = nn.Sequential(
            nn.Linear(self.relative_pose_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.pose_feature_dim),
            nn.ReLU(inplace=True),
        )
        self.yopo_head = YopoHead(
            self.gru_hidden_dim + observation_dim + self.pose_feature_dim + 3,
            output_dim,
        )
        self.future_risk_head = nn.Sequential(
            nn.Conv2d(
                self.gru_hidden_dim + self.pose_feature_dim,
                64,
                kernel_size=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=1),
        )

    def normalize_relative_pose(self, relative_pose: torch.Tensor) -> torch.Tensor:
        poses = relative_pose.reshape(relative_pose.shape[0], -1, 6).clone()
        poses[:, :, 0:3] /= self.translation_scale
        poses[:, :, 3:6] /= self.rotation_scale
        return poses.reshape(relative_pose.shape[0], -1)

    def encode_temporal_depth(
        self,
        depth: torch.Tensor,
        normalized_relative_pose: torch.Tensor,
    ) -> torch.Tensor:
        if depth.ndim != 4 or depth.shape[1] != self.history_length:
            raise ValueError(
                f"depth must be [B,{self.history_length},H,W], got {depth.shape}"
            )
        batch_size, time_steps, height, width = depth.shape
        frame_features = self.image_backbone(
            depth.reshape(batch_size * time_steps, 1, height, width)
        )
        feature_height, feature_width = frame_features.shape[-2:]
        frame_features = frame_features.reshape(
            batch_size,
            time_steps,
            self.frame_feature_dim,
            feature_height,
            feature_width,
        )

        pose_steps = normalized_relative_pose.reshape(
            batch_size, self.history_length - 1, 6
        )
        pose_steps = self.step_pose_backbone(pose_steps)
        initial_pose = pose_steps.new_zeros(batch_size, 1, self.step_pose_dim)
        pose_steps = torch.cat((initial_pose, pose_steps), dim=1)
        pose_steps = pose_steps[:, :, :, None, None].expand(
            -1, -1, -1, feature_height, feature_width
        )
        temporal_input = torch.cat((frame_features, pose_steps), dim=2)
        return self.temporal_fusion(temporal_input)

    def forward(
        self,
        depth: torch.Tensor,
        observation_grid: torch.Tensor,
        relative_pose: torch.Tensor,
        return_auxiliary: bool = False,
    ):
        normalized_relative_pose = self.normalize_relative_pose(relative_pose)
        depth_feature = self.encode_temporal_depth(
            depth, normalized_relative_pose
        )
        pose_feature = self.pose_backbone(normalized_relative_pose)
        pose_feature = pose_feature[:, :, None, None].expand(
            -1, -1, depth_feature.shape[2], depth_feature.shape[3]
        )
        temporal_context = torch.cat((depth_feature, pose_feature), dim=1)
        auxiliary = self.future_risk_head(temporal_context)
        auxiliary_features = torch.stack(
            (
                torch.sigmoid(auxiliary[:, 0]),
                torch.tanh(auxiliary[:, 1]),
                torch.sigmoid(auxiliary[:, 2]),
            ),
            dim=1,
        )
        features = torch.cat(
            (observation_grid, pose_feature, depth_feature, auxiliary_features),
            dim=1,
        )
        output = self.yopo_head(features)
        endstate = torch.tanh(output[:, :9])
        dynamic_risk = (
            torch.sigmoid(auxiliary[:, 0])
            + torch.relu(torch.tanh(auxiliary[:, 1]))
            * torch.sigmoid(auxiliary[:, 2])
        )
        score = (
            torch.nn.functional.softplus(output[:, 9])
            + self.aux_score_gain * dynamic_risk
        )
        if return_auxiliary:
            return endstate, score, auxiliary
        return endstate, score

    def inference(
        self,
        depth: torch.Tensor,
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
        network_output = self.forward(
            depth,
            observation_grid,
            relative_pose,
            return_auxiliary=return_auxiliary,
        )
        if return_auxiliary:
            endstate_prediction, score, auxiliary = network_output
        else:
            endstate_prediction, score = network_output
        endstate = self.state_transform.pred_to_endstate(endstate_prediction)
        if return_auxiliary:
            return endstate, score, auxiliary
        return endstate, score
