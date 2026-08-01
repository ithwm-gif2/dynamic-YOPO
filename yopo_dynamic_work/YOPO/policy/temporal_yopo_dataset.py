import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from ruamel.yaml import YAML
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

from config.config import cfg


cv2.setNumThreads(0)


@dataclass
class EpisodeData:
    path: Path
    scenario: str
    frames: int
    frame_rate: float
    obstacle_count: int
    valid_frames: np.ndarray
    positions: np.ndarray
    rotations_wb: np.ndarray
    observations_b: np.ndarray
    relative_pose_features: np.ndarray
    obstacle_tensor_path: Path


class TemporalYOPODataset(Dataset):
    """Continuous-frame YOPO data with training-only future obstacle boxes."""

    SCENARIO_TO_ID = {"static": 0, "mixed": 1, "dynamic": 2}

    def __init__(
        self,
        root: str,
        scenario: str,
        mode: str = "train",
        validation_episode_ratio: float = 0.2,
        max_samples_per_episode: int = 0,
    ):
        super().__init__()
        if scenario not in self.SCENARIO_TO_ID:
            raise ValueError(f"Unknown scenario: {scenario}")
        if mode not in ("train", "valid"):
            raise ValueError("mode must be train or valid")

        self.root = Path(root).expanduser().resolve()
        self.scenario = scenario
        self.scenario_id = self.SCENARIO_TO_ID[scenario]
        self.mode = mode
        self.height = int(cfg["image_height"])
        self.width = int(cfg["image_width"])
        self.history_length = int(cfg["history_length"])
        self.eval_points = int(cfg["temporal_eval_points"])
        self.sgm_time = float(cfg["sgm_time"])
        self.validation_episode_ratio = validation_episode_ratio
        self.max_samples_per_episode = max_samples_per_episode
        self._obstacle_maps: Dict[int, np.memmap] = {}

        episode_paths = sorted(
            path for path in self.root.glob("episode_*") if path.is_dir()
        )
        if not episode_paths:
            raise FileNotFoundError(f"No episode_* folders found in {self.root}")

        validation_count = max(
            1, int(round(len(episode_paths) * self.validation_episode_ratio))
        )
        selected_paths = (
            episode_paths[:-validation_count]
            if mode == "train"
            else episode_paths[-validation_count:]
        )
        if not selected_paths:
            raise RuntimeError(
                f"Not enough episodes in {self.root} for an episode-level split"
            )

        self.episodes: List[EpisodeData] = [
            self._load_episode(path) for path in selected_paths
        ]
        self.samples: List[Tuple[int, int]] = []
        for episode_index, episode in enumerate(self.episodes):
            frames = episode.valid_frames
            if self.max_samples_per_episode > 0:
                frames = frames[: self.max_samples_per_episode]
            self.samples.extend((episode_index, int(frame)) for frame in frames)

        print(
            f"Temporal {mode:>5} {scenario:>7}: "
            f"{len(self.episodes)} episodes, {len(self.samples)} samples"
        )

    def _load_episode(self, episode_path: Path) -> EpisodeData:
        yaml = YAML(typ="safe")
        with open(episode_path / "metadata.yaml", "r", encoding="utf-8") as file:
            metadata = yaml.load(file)

        metadata_scenario = metadata.get(
            "scenario", "dynamic" if metadata.get("all_obstacles_moving") else self.scenario
        )
        if metadata_scenario != self.scenario:
            raise ValueError(
                f"{episode_path} is {metadata_scenario}, expected {self.scenario}"
            )

        states = np.loadtxt(
            episode_path / "drone_state.csv",
            delimiter=",",
            skiprows=1,
            dtype=np.float32,
        )
        relative = np.loadtxt(
            episode_path / "relative_pose.csv",
            delimiter=",",
            skiprows=1,
            dtype=np.float32,
        )
        if states.ndim == 1:
            states = states[None, :]
        if relative.ndim == 1:
            relative = relative[None, :]

        frames = int(metadata["frames"])
        obstacle_count = int(metadata["obstacle_count"])
        if len(states) != frames or len(relative) != frames:
            raise ValueError(f"Frame metadata mismatch in {episode_path}")

        positions = states[:, 3:6].astype(np.float32, copy=False)
        quaternion_wxyz = states[:, 6:10]
        rotations = Rotation.from_quat(
            quaternion_wxyz[:, [1, 2, 3, 0]]
        ).as_matrix().astype(np.float32)

        velocity_w = states[:, 14:17]
        acceleration_w = states[:, 17:20]
        goal_vector_w = states[:, 20:23] - positions
        rotations_bw = np.transpose(rotations, (0, 2, 1))
        velocity_b = np.einsum("nij,nj->ni", rotations_bw, velocity_w)
        acceleration_b = np.einsum("nij,nj->ni", rotations_bw, acceleration_w)
        goal_b = np.einsum("nij,nj->ni", rotations_bw, goal_vector_w)
        observations = np.concatenate(
            (velocity_b, acceleration_b, goal_b), axis=1
        ).astype(np.float32)

        relative_quaternion_wxyz = relative[:, 6:10]
        relative_rotvec = Rotation.from_quat(
            relative_quaternion_wxyz[:, [1, 2, 3, 0]]
        ).as_rotvec().astype(np.float32)
        relative_features = np.concatenate(
            (relative[:, 3:6], relative_rotvec), axis=1
        ).astype(np.float32)

        valid_frames = np.flatnonzero(states[:, 2] > 0.5)
        valid_frames = valid_frames[valid_frames >= self.history_length - 1]
        collision = states[:, 23] > 0.5
        collision_prefix = np.concatenate(([0], np.cumsum(collision, dtype=np.int64)))
        history_start = valid_frames - self.history_length + 1
        history_collision_count = (
            collision_prefix[valid_frames + 1] - collision_prefix[history_start]
        )
        valid_frames = valid_frames[history_collision_count == 0]
        obstacle_tensor_path = episode_path / "obstacles.bin"
        expected_bytes = frames * obstacle_count * 6 * np.dtype(np.float32).itemsize
        if not obstacle_tensor_path.exists():
            raise FileNotFoundError(
                f"Missing {obstacle_tensor_path}; run tools/prepare_obstacle_tensor.py"
            )
        if obstacle_tensor_path.stat().st_size != expected_bytes:
            raise ValueError(
                f"Unexpected obstacle tensor size in {episode_path}: "
                f"{obstacle_tensor_path.stat().st_size} != {expected_bytes}"
            )

        return EpisodeData(
            path=episode_path,
            scenario=self.scenario,
            frames=frames,
            frame_rate=float(metadata["frame_rate"]),
            obstacle_count=obstacle_count,
            valid_frames=valid_frames,
            positions=positions,
            rotations_wb=rotations,
            observations_b=observations,
            relative_pose_features=relative_features,
            obstacle_tensor_path=obstacle_tensor_path,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_obstacle_maps"] = {}
        return state

    def _obstacle_map(self, episode_index: int) -> np.memmap:
        obstacle_map = self._obstacle_maps.get(episode_index)
        if obstacle_map is None:
            episode = self.episodes[episode_index]
            obstacle_map = np.memmap(
                episode.obstacle_tensor_path,
                dtype=np.float32,
                mode="r",
                shape=(episode.frames, episode.obstacle_count, 6),
            )
            self._obstacle_maps[episode_index] = obstacle_map
        return obstacle_map

    def _read_depth_history(self, episode: EpisodeData, frame: int) -> np.ndarray:
        images = []
        first = frame - self.history_length + 1
        for image_frame in range(first, frame + 1):
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

    def _future_boxes(self, episode_index: int, frame: int) -> np.ndarray:
        episode = self.episodes[episode_index]
        obstacle_map = self._obstacle_map(episode_index)
        frame_rate = episode.frame_rate
        times = np.linspace(
            self.sgm_time / self.eval_points,
            self.sgm_time,
            self.eval_points,
            dtype=np.float32,
        )
        offsets = times * frame_rate
        lower = np.floor(offsets).astype(np.int64)
        upper = np.ceil(offsets).astype(np.int64)
        alpha = (offsets - lower).astype(np.float32)[:, None, None]
        lower_boxes = np.asarray(obstacle_map[frame + lower], dtype=np.float32)
        upper_boxes = np.asarray(obstacle_map[frame + upper], dtype=np.float32)
        return lower_boxes * (1.0 - alpha) + upper_boxes * alpha

    def __getitem__(self, item: int):
        episode_index, frame = self.samples[item]
        episode = self.episodes[episode_index]
        depth = self._read_depth_history(episode, frame)
        pose_start = frame - self.history_length + 2
        relative_pose = episode.relative_pose_features[
            pose_start : frame + 1
        ].reshape(-1)
        if relative_pose.shape[0] != int(cfg["relative_pose_dim"]):
            raise RuntimeError("Relative-pose feature size mismatch")
        future_boxes = self._future_boxes(episode_index, frame)

        return (
            depth,
            relative_pose.astype(np.float32, copy=False),
            episode.positions[frame],
            episode.rotations_wb[frame],
            episode.observations_b[frame],
            future_boxes.astype(np.float32, copy=False),
            np.int64(self.scenario_id),
        )
