import numpy as np

from config.config import cfg
from policy.static_dynamic_dataset import StaticDynamicTemporalDataset


class StaticDynamicMaskDataset(StaticDynamicTemporalDataset):
    """Lightweight view used for dynamic-mask and velocity pretraining."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.velocity_scale = float(cfg["sd_velocity_scale"])
        self.fx = float(cfg["sd_camera_fx"])
        self.fy = float(cfg["sd_camera_fy"])
        self.cx = float(cfg["sd_camera_cx"])
        self.cy = float(cfg["sd_camera_cy"])
        self.grid_rows = int(cfg["vertical_num"])
        self.grid_columns = int(cfg["horizon_num"])
        self.corner_signs = np.asarray(
            [
                [-1, -1, -1], [-1, -1, 1],
                [-1, 1, -1], [-1, 1, 1],
                [1, -1, -1], [1, -1, 1],
                [1, 1, -1], [1, 1, 1],
            ],
            dtype=np.float32,
        )
        self.dynamic_flags = []
        for episode in self.episodes:
            flags = np.loadtxt(
                episode.path / "obstacles.csv",
                delimiter=",",
                skiprows=1,
                max_rows=episode.obstacle_count,
                usecols=3,
                dtype=np.int64,
            )
            self.dynamic_flags.append(np.atleast_1d(flags).astype(bool))

    def _coarse_velocity_target(
        self, episode_index: int, frame: int, dynamic_mask: np.ndarray
    ):
        episode = self.episodes[episode_index]
        obstacle_map = self._obstacle_map(episode_index)
        boxes = np.asarray(obstacle_map[frame], dtype=np.float32)
        next_boxes = np.asarray(obstacle_map[frame + 1], dtype=np.float32)
        centers_w = boxes[:, 0:3]
        half_sizes_w = boxes[:, 3:6]
        rotation_bw = episode.rotations_wb[frame].T
        position_w = episode.positions[frame]
        centers_b = np.einsum(
            "ij,nj->ni", rotation_bw, centers_w - position_w
        )
        velocity_w = (
            next_boxes[:, 0:3] - centers_w
        ) * float(episode.frame_rate)
        velocity_b = np.einsum("ij,nj->ni", rotation_bw, velocity_w)

        corners_w = (
            centers_w[:, None, :]
            + self.corner_signs[None, :, :] * half_sizes_w[:, None, :]
        )
        corners_b = np.einsum(
            "ij,nkj->nki", rotation_bw, corners_w - position_w
        )
        forward = corners_b[:, :, 0] > 0.05
        safe_x = np.maximum(corners_b[:, :, 0], 0.05)
        projected_u = self.cx - self.fx * corners_b[:, :, 1] / safe_x
        projected_v = self.cy - self.fy * corners_b[:, :, 2] / safe_x
        u_min = np.min(np.where(forward, projected_u, np.inf), axis=1)
        u_max = np.max(np.where(forward, projected_u, -np.inf), axis=1)
        v_min = np.min(np.where(forward, projected_v, np.inf), axis=1)
        v_max = np.max(np.where(forward, projected_v, -np.inf), axis=1)

        height, width = dynamic_mask.shape
        if height % self.grid_rows or width % self.grid_columns:
            raise ValueError("Image dimensions must divide the coarse grid")
        coarse_mask = dynamic_mask.reshape(
            self.grid_rows,
            height // self.grid_rows,
            self.grid_columns,
            width // self.grid_columns,
        ).mean(axis=(1, 3)) >= 0.02
        visible = (
            self.dynamic_flags[episode_index]
            & (centers_b[:, 0] > 0.05)
            & (u_max >= 0.0)
            & (u_min <= width)
            & (v_max >= 0.0)
            & (v_min <= height)
        )
        target = np.zeros(
            (3, self.grid_rows, self.grid_columns), dtype=np.float32
        )
        valid = np.zeros(
            (self.grid_rows, self.grid_columns), dtype=np.float32
        )
        cell_width = width / self.grid_columns
        cell_height = height / self.grid_rows
        for row in range(self.grid_rows):
            top = row * cell_height
            bottom = (row + 1) * cell_height
            for column in range(self.grid_columns):
                if not coarse_mask[row, column]:
                    continue
                left = column * cell_width
                right = (column + 1) * cell_width
                overlaps = (
                    visible
                    & (np.minimum(u_max, right) > np.maximum(u_min, left))
                    & (np.minimum(v_max, bottom) > np.maximum(v_min, top))
                )
                if not overlaps.any():
                    continue
                depth = np.where(overlaps, centers_b[:, 0], np.inf)
                obstacle_index = int(np.argmin(depth))
                target[:, row, column] = np.clip(
                    velocity_b[obstacle_index] / self.velocity_scale,
                    -1.0,
                    1.0,
                )
                valid[row, column] = 1.0
        return target, valid

    def __getitem__(self, item: int):
        episode_index, frame = self.samples[item]
        episode = self.episodes[episode_index]
        dynamic_mask = self._dynamic_mask(episode, frame)
        if (dynamic_mask < 0.0).any():
            raise FileNotFoundError(
                f"Dynamic mask label missing in {episode.path} at frame {frame}"
            )
        velocity_target, velocity_valid = self._coarse_velocity_target(
            episode_index, frame, dynamic_mask
        )
        return (
            self._read_sparse_depth(episode, frame),
            self._sparse_relative_pose(episode, frame),
            dynamic_mask,
            velocity_target,
            velocity_valid,
            np.int64(self.scenario_id),
        )
