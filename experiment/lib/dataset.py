import json
from pathlib import Path

import h5py
import numpy as np
import torch

from experiment.lib.paths import DATASET_ROOT


Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


D4RL_DATASETS = {
    "hopper_medium": ("hopper_medium-v2.hdf5", "Hopper-v5"),
    "hopper_expert": ("hopper_expert-v2.hdf5", "Hopper-v5"),
    "hopper_hybrid": ("hopper_medium_expert-v2.hdf5", "Hopper-v5"),
    "hopper_medium_replay": ("hopper_medium_replay-v2.hdf5", "Hopper-v5"),
    "hopper_expert_replay": ("hopper_full_replay-v2.hdf5", "Hopper-v5"),
    "walker2d_medium": ("walker2d_medium-v2.hdf5", "Walker2d-v5"),
    "walker2d_expert": ("walker2d_expert-v2.hdf5", "Walker2d-v5"),
    "walker2d_hybrid": ("walker2d_medium_expert-v2.hdf5", "Walker2d-v5"),
    "walker2d_medium_replay": ("walker2d_medium_replay-v2.hdf5", "Walker2d-v5"),
    "walker2d_expert_replay": ("walker2d_full_replay-v2.hdf5", "Walker2d-v5"),
    "halfcheetah_medium": ("halfcheetah_medium-v2.hdf5", "HalfCheetah-v5"),
    "halfcheetah_expert": ("halfcheetah_expert-v2.hdf5", "HalfCheetah-v5"),
    "halfcheetah_hybrid": ("halfcheetah_medium_expert-v2.hdf5", "HalfCheetah-v5"),
    "halfcheetah_medium_replay": ("halfcheetah_medium_replay-v2.hdf5", "HalfCheetah-v5"),
    "halfcheetah_expert_replay": ("halfcheetah_full_replay-v2.hdf5", "HalfCheetah-v5"),
}

class Dataset:
    def __init__(self, dataset_id: str, path: Path, env_id: str, device: str) -> None:
        self.id = dataset_id
        self.path = path
        self.env_id = env_id
        self.device = device

    @property
    def obs_size(self) -> int:
        return int(np.prod(self.observations.shape[1:]))

    @property
    def act_size(self) -> int:
        return int(np.prod(self.actions.shape[1:]))

    @property
    def count(self) -> int:
        return int(self.observations.shape[0])

    def normalize_observation(self, observation) -> np.ndarray:
        observation_array = np.asarray(observation, dtype=np.float32)
        return (observation_array - self.obs_mean) / self.obs_std

    def sample_batch(self, batch_size: int) -> Batch:
        indexes = torch.randint(self.count, (batch_size,), device=self.device)
        return (
            self.observations[indexes],
            self.actions[indexes],
            self.rewards[indexes],
            self.next_observations[indexes],
            self.terminated[indexes],
        )

    def set_data(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        terminations: np.ndarray,
        truncations: np.ndarray,
    ) -> None:
        raw_observations = np.asarray(observations, dtype=np.float32)
        raw_next_observations = np.asarray(next_observations, dtype=np.float32)
        self.obs_mean = raw_observations.mean(axis=0, dtype=np.float64).astype(np.float32)
        self.obs_std = raw_observations.std(axis=0, dtype=np.float64).astype(np.float32) + 1e-6

        self.observations = torch.as_tensor(
            self.normalize_observation(raw_observations),
            dtype=torch.float32,
            device=self.device,
        )
        self.actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        self.rewards = torch.as_tensor(
            np.asarray(rewards).reshape(-1, 1),
            dtype=torch.float32,
            device=self.device,
        )
        self.next_observations = torch.as_tensor(
            self.normalize_observation(raw_next_observations),
            dtype=torch.float32,
            device=self.device,
        )
        self.terminated = torch.as_tensor(
            np.asarray(terminations).reshape(-1, 1),
            dtype=torch.float32,
            device=self.device,
        )
        self.truncated = torch.as_tensor(
            np.asarray(truncations).reshape(-1, 1),
            dtype=torch.float32,
            device=self.device,
        )


class D4RLDataset(Dataset):
    def __init__(self, dataset_id: str, device: str) -> None:
        filename, env_id = D4RL_DATASETS[dataset_id]
        super().__init__(dataset_id, DATASET_ROOT / filename, env_id, device)
        self.load()

    def load(self) -> None:
        with h5py.File(self.path, "r") as h5_file:
            self.set_data(
                observations=np.asarray(h5_file["observations"]),
                actions=np.asarray(h5_file["actions"]),
                rewards=np.asarray(h5_file["rewards"]),
                next_observations=np.asarray(h5_file["next_observations"]),
                terminations=np.asarray(h5_file["terminals"]),
                truncations=np.asarray(h5_file["timeouts"]),
            )


class MinariDataset(Dataset):
    def __init__(self, dataset_id: str, device: str) -> None:
        data_path = DATASET_ROOT / dataset_id / "data"
        with (data_path / "metadata.json").open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        super().__init__(dataset_id, data_path / "main_data.hdf5", metadata["env_id"], device)
        self.load()

    def load(self) -> None:
        observations = []
        actions = []
        rewards = []
        next_observations = []
        terminations = []
        truncations = []

        with h5py.File(self.path, "r") as h5_file:
            episode_ids = sorted(
                h5_file.keys(),
                key=lambda episode_id: int(episode_id.split("_")[-1]),
            )
            for episode_id in episode_ids:
                episode = h5_file[episode_id]
                episode_observations = np.asarray(episode["observations"])
                observations.append(episode_observations[:-1])
                next_observations.append(episode_observations[1:])
                actions.append(np.asarray(episode["actions"]))
                rewards.append(np.asarray(episode["rewards"]))
                terminations.append(np.asarray(episode["terminations"]))
                truncations.append(np.asarray(episode["truncations"]))

        self.set_data(
            observations=np.concatenate(observations),
            actions=np.concatenate(actions),
            rewards=np.concatenate(rewards),
            next_observations=np.concatenate(next_observations),
            terminations=np.concatenate(terminations),
            truncations=np.concatenate(truncations),
        )
