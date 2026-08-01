import math

import numpy as np

from config.config import cfg
from policy.temporal_yopo_dataset import TemporalYOPODataset as BaseTemporalDataset


class TemporalYOPODataset(BaseTemporalDataset):
    """Temporal dataset with current boxes and critical-window weights."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.max_samples_per_episode > 0:
            self.samples = []
            for episode_index, episode in enumerate(self.episodes):
                frames = episode.valid_frames
                if len(frames) > self.max_samples_per_episode:
                    indices = np.linspace(
                        0, len(frames) - 1, self.max_samples_per_episode
                    ).round().astype(np.int64)
                    frames = frames[indices]
                self.samples.extend(
                    (episode_index, int(frame)) for frame in frames
                )
        critical_clearance = float(cfg["temporal_critical_clearance"])
        critical_weight = float(cfg["temporal_critical_sampling_weight"])
        episode_critical = []
        for episode in self.episodes:
            clearances = np.loadtxt(
                episode.path / "drone_state.csv",
                delimiter=",",
                skiprows=1,
                usecols=24,
                dtype=np.float32,
            )
            horizon_frames = int(math.ceil(self.sgm_time * episode.frame_rate))
            windows = np.lib.stride_tricks.sliding_window_view(
                clearances, horizon_frames + 1
            )
            future_minimum = np.full_like(clearances, np.inf)
            future_minimum[: len(windows)] = windows.min(axis=1)
            episode_critical.append(future_minimum < critical_clearance)

        self.sample_importance = np.asarray(
            [
                critical_weight
                if episode_critical[episode_index][frame]
                else 1.0
                for episode_index, frame in self.samples
            ],
            dtype=np.float64,
        )
        critical_ratio = float(
            np.mean(self.sample_importance > 1.0)
        ) if len(self.sample_importance) else 0.0
        print(
            f"Critical-window ratio {self.scenario:>7}: "
            f"{critical_ratio:.3f}, weight={critical_weight:.1f}"
        )

    def __getitem__(self, item: int):
        base_sample = super().__getitem__(item)
        episode_index, frame = self.samples[item]
        current_boxes = np.array(
            self._obstacle_map(episode_index)[frame],
            dtype=np.float32,
            copy=True,
        )
        return (*base_sample[:-1], current_boxes, base_sample[-1])
