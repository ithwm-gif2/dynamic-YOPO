import torch
from torch import nn

from policy.geometry_temporal import EgoMotionDepthAligner
from policy.models.backbone import YopoBackbone
from policy.models.conv_gru import ConvGRU


class StaticDynamicMotionNetwork(nn.Module):
    """Predict coarse dynamic occupancy from ego-compensated depth residuals."""

    def __init__(self, feature_dim: int = 64):
        super().__init__()
        self.aligner = EgoMotionDepthAligner()
        self.motion_encoder = YopoBackbone(feature_dim, input_channels=5)
        self.temporal_fusion = ConvGRU(
            input_channels=feature_dim,
            hidden_channels=feature_dim,
        )
        self.dynamic_mask_head = nn.Sequential(
            nn.Conv2d(feature_dim, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    @staticmethod
    def residual_sequence(aligned):
        return torch.stack(
            (
                aligned["signed_residual"],
                aligned["absolute_residual"],
                aligned["valid"],
                aligned["occlusion"],
                aligned["disocclusion"],
            ),
            dim=2,
        )

    def forward(self, depth_sequence, relative_pose, return_alignment=False):
        aligned = self.aligner(depth_sequence, relative_pose)
        residual_sequence = self.residual_sequence(aligned)
        batch_size, time_steps, channels, height, width = (
            residual_sequence.shape
        )
        encoded = self.motion_encoder(
            residual_sequence.reshape(
                batch_size * time_steps, channels, height, width
            )
        )
        encoded = encoded.reshape(
            batch_size,
            time_steps,
            encoded.shape[1],
            encoded.shape[2],
            encoded.shape[3],
        )
        motion_feature = self.temporal_fusion(encoded)
        logits = self.dynamic_mask_head(motion_feature)[:, 0]
        if return_alignment:
            return logits, motion_feature, aligned
        return logits
