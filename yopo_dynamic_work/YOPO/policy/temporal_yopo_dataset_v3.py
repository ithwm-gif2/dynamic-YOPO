import numpy as np

from config.config import cfg
from policy.temporal_yopo_dataset_v2 import TemporalYOPODataset as TemporalDatasetV2


def signed_distance_to_boxes(
    points: np.ndarray,
    boxes: np.ndarray,
    drone_radius: float,
) -> np.ndarray:
    q = np.abs(points[:, None, :] - boxes[:, :, 0:3]) - boxes[:, :, 3:6]
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(np.max(q, axis=-1), 0.0)
    return outside + inside - drone_radius


class TemporalYOPODataset(TemporalDatasetV2):
    """Focus dynamic sampling on hazards caused by obstacle displacement."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.scenario == "static":
            return

        clearance_threshold = float(cfg["temporal_critical_clearance"])
        sampling_weight = float(cfg["temporal_critical_sampling_weight"])
        displacement_advantage = float(
            cfg["temporal_motion_critical_advantage"]
        )
        drone_radius = float(cfg["drone_radius"])
        episode_flags = []
        for episode_index, episode in enumerate(self.episodes):
            obstacle_map = self._obstacle_map(episode_index)
            flags = np.zeros(episode.frames, dtype=bool)
            offsets = sorted(
                {
                    max(1, int(round(0.5 * episode.frame_rate))),
                    max(1, int(round(1.0 * episode.frame_rate))),
                    max(1, int(round(self.sgm_time * episode.frame_rate))),
                }
            )
            for offset in offsets:
                valid_count = episode.frames - offset
                if valid_count <= 0:
                    continue
                future_points = episode.positions[offset : offset + valid_count]
                actual_boxes = np.asarray(
                    obstacle_map[offset : offset + valid_count],
                    dtype=np.float32,
                )
                frozen_boxes = np.asarray(
                    obstacle_map[:valid_count], dtype=np.float32
                )
                actual_clearance = signed_distance_to_boxes(
                    future_points, actual_boxes, drone_radius
                ).min(axis=1)
                frozen_clearance = signed_distance_to_boxes(
                    future_points, frozen_boxes, drone_radius
                ).min(axis=1)
                flags[:valid_count] |= (
                    (actual_clearance < clearance_threshold)
                    & (
                        actual_clearance + displacement_advantage
                        < frozen_clearance
                    )
                )
            episode_flags.append(flags)

        self.sample_importance = np.asarray(
            [
                sampling_weight
                if episode_flags[episode_index][frame]
                else 1.0
                for episode_index, frame in self.samples
            ],
            dtype=np.float64,
        )
        critical_ratio = float(
            np.mean(self.sample_importance > 1.0)
        ) if len(self.sample_importance) else 0.0
        print(
            f"Motion-critical ratio {self.scenario:>7}: "
            f"{critical_ratio:.3f}, weight={sampling_weight:.1f}"
        )
