import torch
from torch import nn
from torch.nn import functional as F

from config.config import cfg


def rotvec_to_matrix(rotation_vector: torch.Tensor) -> torch.Tensor:
    """Convert [...,3] axis-angle vectors to rotation matrices."""
    x, y, z = rotation_vector.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    skew = torch.stack(
        (
            zeros, -z, y,
            z, zeros, -x,
            -y, x, zeros,
        ),
        dim=-1,
    ).reshape(*rotation_vector.shape[:-1], 3, 3)
    angle_squared = rotation_vector.square().sum(dim=-1, keepdim=True)
    angle = angle_squared.sqrt()
    small = angle_squared < 1e-8
    coefficient_a = torch.where(
        small,
        1.0 - angle_squared / 6.0,
        torch.sin(angle) / angle.clamp_min(1e-8),
    )
    coefficient_b = torch.where(
        small,
        0.5 - angle_squared / 24.0,
        (1.0 - torch.cos(angle)) / angle_squared.clamp_min(1e-8),
    )
    identity = torch.eye(
        3, device=rotation_vector.device, dtype=rotation_vector.dtype
    ).expand(*rotation_vector.shape[:-1], 3, 3)
    return (
        identity
        + coefficient_a[..., None] * skew
        + coefficient_b[..., None] * (skew @ skew)
    )


def compose_transform(
    rotation_ab: torch.Tensor,
    translation_ab: torch.Tensor,
    rotation_bc: torch.Tensor,
    translation_bc: torch.Tensor,
):
    """Compose transforms p_a=R_ab p_b+t_ab and p_b=R_bc p_c+t_bc."""
    rotation_ac = rotation_ab @ rotation_bc
    translation_ac = (
        rotation_ab @ translation_bc.unsqueeze(-1)
    ).squeeze(-1) + translation_ab
    return rotation_ac, translation_ac


class EgoMotionDepthAligner(nn.Module):
    """Remove camera motion before extracting depth-motion residuals."""

    def __init__(self):
        super().__init__()
        self.fx = float(cfg["sd_camera_fx"])
        self.fy = float(cfg["sd_camera_fy"])
        self.cx = float(cfg["sd_camera_cx"])
        self.cy = float(cfg["sd_camera_cy"])
        self.max_depth = float(cfg["sd_max_depth"])
        self.residual_scale = float(cfg["sd_residual_scale"])
        self.occlusion_threshold = float(cfg["sd_occlusion_threshold"])
        self.frame_count = int(cfg["sd_history_length"])
        self.relative_pose_dim = int(cfg["sd_relative_pose_dim"])
        self.register_buffer("pixel_grid", torch.empty(0), persistent=False)

    def _pixels(self, height: int, width: int, device, dtype):
        if (
            self.pixel_grid.numel() == 0
            or self.pixel_grid.shape[-2:] != (height, width)
            or self.pixel_grid.device != device
            or self.pixel_grid.dtype != dtype
        ):
            rows, columns = torch.meshgrid(
                torch.arange(height, device=device, dtype=dtype),
                torch.arange(width, device=device, dtype=dtype),
                indexing="ij",
            )
            self.pixel_grid = torch.stack((columns, rows), dim=0)
        return self.pixel_grid

    def _align_one(
        self,
        previous_depth: torch.Tensor,
        current_depth: torch.Tensor,
        rotation_previous_current: torch.Tensor,
        translation_previous_current: torch.Tensor,
    ):
        batch_size, _, height, width = current_depth.shape
        pixels = self._pixels(
            height, width, current_depth.device, current_depth.dtype
        )
        columns = pixels[0][None]
        rows = pixels[1][None]
        depth_current_m = current_depth[:, 0] * self.max_depth
        point_current = torch.stack(
            (
                depth_current_m,
                -(columns - self.cx) / self.fx * depth_current_m,
                -(rows - self.cy) / self.fy * depth_current_m,
            ),
            dim=1,
        ).reshape(batch_size, 3, -1)
        point_previous = (
            rotation_previous_current @ point_current
            + translation_previous_current[:, :, None]
        )
        previous_x = point_previous[:, 0].reshape(batch_size, height, width)
        previous_y = point_previous[:, 1].reshape(batch_size, height, width)
        previous_z = point_previous[:, 2].reshape(batch_size, height, width)

        safe_x = previous_x.clamp_min(1e-4)
        projected_u = self.cx - self.fx * previous_y / safe_x
        projected_v = self.cy - self.fy * previous_z / safe_x
        grid_x = 2.0 * projected_u / max(1, width - 1) - 1.0
        grid_y = 2.0 * projected_v / max(1, height - 1) - 1.0
        sampling_grid = torch.stack((grid_x, grid_y), dim=-1)
        sampled_previous_m = F.grid_sample(
            previous_depth * self.max_depth,
            sampling_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[:, 0]

        current_valid = (
            (depth_current_m > 0.04)
            & (depth_current_m < self.max_depth - 1e-3)
        )
        sampled_valid = (
            (sampled_previous_m > 0.04)
            & (sampled_previous_m < self.max_depth - 1e-3)
        )
        projection_valid = (
            (previous_x > 0.04)
            & (projected_u >= 0.0)
            & (projected_u <= width - 1)
            & (projected_v >= 0.0)
            & (projected_v <= height - 1)
        )
        valid = current_valid & sampled_valid & projection_valid
        residual_m = sampled_previous_m - previous_x
        occlusion = valid & (
            residual_m < -self.occlusion_threshold
        )
        disocclusion = valid & (
            residual_m > self.occlusion_threshold
        )
        signed_residual = torch.clamp(
            residual_m / self.residual_scale, -1.0, 1.0
        )
        signed_residual = signed_residual * valid
        return {
            "signed_residual": signed_residual,
            "absolute_residual": signed_residual.abs(),
            "valid": valid.float(),
            "occlusion": occlusion.float(),
            "disocclusion": disocclusion.float(),
            "sampled_previous_depth": sampled_previous_m / self.max_depth,
            "expected_previous_depth": previous_x / self.max_depth,
        }

    def forward(
        self,
        depth_sequence: torch.Tensor,
        relative_pose: torch.Tensor,
    ):
        if depth_sequence.ndim != 4:
            raise ValueError("depth_sequence must be [B,T,H,W]")
        if depth_sequence.shape[1] != self.frame_count:
            raise ValueError(
                f"expected {self.frame_count} frames, got {depth_sequence.shape[1]}"
            )
        if relative_pose.shape[1] != self.relative_pose_dim:
            raise ValueError(
                f"expected relative pose dim {self.relative_pose_dim}, "
                f"got {relative_pose.shape[1]}"
            )

        transforms = relative_pose.reshape(
            relative_pose.shape[0], self.frame_count - 1, 6
        )
        translations = transforms[:, :, 0:3]
        rotations = rotvec_to_matrix(transforms[:, :, 3:6])
        rotation_old_current, translation_old_current = compose_transform(
            rotations[:, 0],
            translations[:, 0],
            rotations[:, 1],
            translations[:, 1],
        )
        current_depth = depth_sequence[:, -1:]
        old_alignment = self._align_one(
            depth_sequence[:, 0:1],
            current_depth,
            rotation_old_current,
            translation_old_current,
        )
        recent_alignment = self._align_one(
            depth_sequence[:, 1:2],
            current_depth,
            rotations[:, 1],
            translations[:, 1],
        )
        return {
            key: torch.stack(
                (old_alignment[key], recent_alignment[key]), dim=1
            )
            for key in old_alignment
        }
