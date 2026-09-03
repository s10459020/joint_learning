import gymnasium as gym
import numpy as np

from experiment.agents.dynamics import Dynamic
from experiment.lib.paths import dynamic_path


def load_dynamic(dataset) -> Dynamic:
    dynamic = Dynamic(dataset.obs_size, dataset.act_size, device=dataset.device)
    path = dynamic_path(dataset)
    dynamic.load(path)
    return dynamic.eval().freeze()


def evaluate(agent, env: gym.Env, dataset, episodes: int, seed: int | None = None) -> list[float]:
    returns = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=None if seed is None else seed + episode)
        terminated = False
        truncated = False
        total_return = 0.0
        while not (terminated or truncated):
            action = agent.act(dataset.normalize_observation(observation))
            observation, reward, terminated, truncated, _ = env.step(action)
            total_return += float(reward)
        returns.append(total_return)
    return returns


def evaluate_action_noise(
    agent, env: gym.Env, dataset, noise_scale: float, episodes: int, seed: int
) -> list[float]:
    returns = []
    for episode in range(episodes):
        random_generator = np.random.default_rng(seed + episode)
        observation, _ = env.reset(seed=seed + episode)
        terminated = False
        truncated = False
        total_return = 0.0
        while not (terminated or truncated):
            action = agent.act(dataset.normalize_observation(observation))
            epsilon = random_generator.standard_normal(action.shape).astype(np.float32)
            action = action + noise_scale * epsilon
            observation, reward, terminated, truncated, _ = env.step(action)
            total_return += float(reward)
        returns.append(total_return)
    return returns


def evaluate_state_noise(
    agent, env: gym.Env, dataset, noise_scale: float, episodes: int, seed: int
) -> list[float]:
    state_std = np.concatenate(([0.0], dataset.obs_std))
    returns = []
    for episode in range(episodes):
        random_generator = np.random.default_rng(seed + episode)
        observation, _ = env.reset(seed=seed + episode)
        terminated = False
        truncated = False
        total_return = 0.0
        while not (terminated or truncated):
            action = agent.act(dataset.normalize_observation(observation))
            observation, reward, terminated, truncated, _ = env.step(action)
            total_return += float(reward)
            if terminated or truncated:
                continue

            base_env = env.unwrapped
            qpos = np.asarray(base_env.data.qpos, dtype=np.float64).copy()
            qvel = np.asarray(base_env.data.qvel, dtype=np.float64).copy()
            state = np.concatenate([qpos.reshape(-1), qvel.reshape(-1)])
            epsilon = random_generator.standard_normal(state.shape)
            noisy_state = state + noise_scale * state_std * epsilon
            qpos_end = qpos.size
            base_env.set_state(
                noisy_state[:qpos_end].reshape(qpos.shape),
                noisy_state[qpos_end:].reshape(qvel.shape),
            )
            observation = base_env._get_obs()
        returns.append(total_return)
    return returns
