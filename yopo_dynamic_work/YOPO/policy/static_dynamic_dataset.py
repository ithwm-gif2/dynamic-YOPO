import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from config.config import cfg
from policy.temporal_yopo_dataset import TemporalYOPODataset


class StaticDynamicTemporalDataset(TemporalYOPODataset):
    """Three sparsely sampled frames for geometry-aware motion separation."""

    def __init__(self, *args, **kwargs):
        requested_limit = int(kwargs.pop("max_samples_per_episode", 0))
        super().__init__(*args, max_samples_per_episode=0, **kwargs)
        self.frame_offsets = [int(value) for value in cfg["sd_frame_offsets"]]
        if self.frame_offsets[-1] != 0:
            raise ValueError("sd_frame_offsets must end with the current frame (0)")
        if sorted(self.frame_offsets, reverse=True) != self.frame_offsets:
            raise ValueError("sd_frame_offsets must be ordered oldest to newest")
        if len(self.frame_offsets) != int(cfg["sd_history_length"]):
            raise ValueError("sd_history_length and sd_frame_offsets disagree")

        samples = []
        oldest_offset = max(self.frame_offsets)
        for episode_index, episode in enumerate(self.episodes):
            frames = episode.valid_frames[episode.valid_frames >= oldest_offset]
            if requested_limit > 0 and len(frames) > requested_limit:
                indices = np.linspace(
                    0, len(frames) - 1, requested_limit
                ).round().astype(np.int64)
                frames = frames[indices]
            samples.extend((episode_index, int(frame)) for frame in frames)
        self.samples = samples
        print(
            f"StaticDynamic {self.mode:>5} {self.scenario:>7}: "
            f"{len(self.samples)} sparse samples, offsets={self.frame_offsets}"
        )

    def _read_sparse_depth(self, episode, frame: int) -> np.ndarray:
        images = []
        for offset in self.frame_offsets:
            image_frame = frame - offset
            image_path = episode.path / "depth" / f"img_{image_frame:06d}.png"
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise FileNotFoundError(image_path)
            image = cv2.resize(
                image,
                (self.width, self.height),
                interpolation=cv2.INTER_NEAREST,
            )
            images.append(image.astype(np.float32) / 65535.0)
        return np.stack(images, axis=0)

    @staticmethod
    def _relative_transform(episode, previous_frame: int, current_frame: int):
        rotation_previous = episode.rotations_wb[previous_frame]
        rotation_current = episode.rotations_wb[current_frame]
        translation = rotation_previous.T @ (
            episode.positions[current_frame] - episode.positions[previous_frame]
        )
        relative_rotation = rotation_previous.T @ rotation_current
        rotation_vector = Rotation.from_matrix(relative_rotation).as_rotvec()
        return np.concatenate((translation, rotation_vector)).astype(np.float32)

    def _sparse_relative_pose(self, episode, frame: int) -> np.ndarray:
        selected_frames = [frame - offset for offset in self.frame_offsets]
        transforms = [
            self._relative_transform(episode, previous, current)
            for previous, current in zip(
                selected_frames[:-1], selected_frames[1:]
            )
        ]
        relative_pose = np.concatenate(transforms).astype(np.float32)
        if relative_pose.shape[0] != int(cfg["sd_relative_pose_dim"]):
            raise RuntimeError("Static-dynamic relative pose size mismatch")
        return relative_pose

    def _dynamic_mask(self, episode, frame: int) -> np.ndarray:
        mask_path = episode.path / "dynamic_mask" / f"img_{frame:06d}.png"
        if not mask_path.exists():
            return np.full((self.height, self.width), -1.0, dtype=np.float32)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(mask_path)
        mask = cv2.resize(
            mask,
            (self.width, self.height),
            interpolation=cv2.INTER_NEAREST,
        )
        return (mask > 127).astype(np.float32)

    def __getitem__(self, item: int):
        episode_index, frame = self.samples[item]
        episode = self.episodes[episode_index]
        depth = self._read_sparse_depth(episode, frame)
        relative_pose = self._sparse_relative_pose(episode, frame)
        future_boxes = self._future_boxes(episode_index, frame)
        current_boxes = np.array(
            self._obstacle_map(episode_index)[frame],
            dtype=np.float32,
            copy=True,
        )
        return (
            depth,
            relative_pose,
            episode.positions[frame],
            episode.rotations_wb[frame],
            episode.observations_b[frame],
            future_boxes.astype(np.float32, copy=False),
            current_boxes,
            self._dynamic_mask(episode, frame),
            np.int64(self.scenario_id),
        )
