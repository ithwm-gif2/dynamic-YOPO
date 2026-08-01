import torch
from torch import nn

from config.config import cfg
from policy.models.backbone import YopoBackbone
from policy.models.head import YopoHead
from policy.state_transform import StateTransform


class TemporalYopoNetwork(nn.Module):
    """YOPO with four depth frames and ego-motion increments."""

    def __init__(
        self,
        observation_dim: int = 9,
        output_dim: int = 10,
        hidden_state: int = 64,
    ):
        super().__init__()
        self.state_transform = StateTransform()
        self.history_length = int(cfg["history_length"])
        self.relative_pose_dim = int(cfg["relative_pose_dim"])
        self.pose_feature_dim = int(cfg["relative_pose_feature_dim"])
        self.translation_scale = float(cfg["relative_translation_scale"])
        self.rotation_scale = float(cfg["relative_rotation_scale"])

        self.image_backbone = YopoBackbone(
            hidden_state, input_channels=self.history_length
        )
        self.pose_backbone = nn.Sequential(
            nn.Linear(self.relative_pose_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.pose_feature_dim),
            nn.ReLU(inplace=True),
        )
        self.yopo_head = YopoHead(
            hidden_state + observation_dim + self.pose_feature_dim, output_dim
        )

    def normalize_relative_pose(self, relative_pose: torch.Tensor) -> torch.Tensor:
        poses = relative_pose.reshape(relative_pose.shape[0], -1, 6).clone()
        poses[:, :, 0:3] /= self.translation_scale
        poses[:, :, 3:6] /= self.rotation_scale
        return poses.reshape(relative_pose.shape[0], -1)

    def forward(
        self,
        depth: torch.Tensor,
        observation_grid: torch.Tensor,
        relative_pose: torch.Tensor,
    ):
        depth_feature = self.image_backbone(depth)
        pose_feature = self.pose_backbone(
            self.normalize_relative_pose(relative_pose)
        )
        pose_feature = pose_feature[:, :, None, None].expand(
            -1, -1, depth_feature.shape[2], depth_feature.shape[3]
        )
        features = torch.cat(
            (observation_grid, pose_feature, depth_feature), dim=1
        )
        output = self.yopo_head(features)
        endstate = torch.tanh(output[:, :9])
        score = torch.nn.functional.softplus(output[:, 9])
        return endstate, score

    def inference(
        self,
        depth: torch.Tensor,
        observation: torch.Tensor,
        relative_pose: torch.Tensor,
    ):
        normalized_observation = self.state_transform.normalize_obs(
            observation.clone()
        )
        observation_grid = self.state_transform.prepare_input(
            normalized_observation
        )
        endstate_prediction, score = self.forward(
            depth, observation_grid, relative_pose
        )
        endstate = self.state_transform.pred_to_endstate(endstate_prediction)
        return endstate, score
