import numpy as np

from policy.static_dynamic_dataset import StaticDynamicTemporalDataset


class StaticDynamicPlanningDataset(StaticDynamicTemporalDataset):
    """Planning data with training-only dynamic-box separation labels."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
            flags = np.atleast_1d(flags).astype(bool)
            if flags.shape[0] != episode.obstacle_count:
                raise ValueError(
                    f"Dynamic flag count mismatch in {episode.path}"
                )
            self.dynamic_flags.append(flags)

    def __getitem__(self, item: int):
        base = super().__getitem__(item)
        episode_index, _ = self.samples[item]
        future_boxes = base[5]
        dynamic_boxes = np.array(future_boxes, copy=True)
        static_flags = ~self.dynamic_flags[episode_index]
        dynamic_boxes[:, static_flags, 0:3] = 1e4
        dynamic_boxes[:, static_flags, 3:6] = 0.0
        return (*base[:6], dynamic_boxes, *base[6:])
