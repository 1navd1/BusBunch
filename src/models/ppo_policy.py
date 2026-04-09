from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Protocol, Tuple


class RLEnv(Protocol):
    def reset(self, seed: int | None = None) -> List[float]:
        ...

    def step(self, action: List[float]) -> Tuple[List[float], float, bool, Dict]:
        ...


@dataclass
class PPOConfig:
    total_steps: int = 5000
    rollout_steps: int = 180
    seed: int = 11
    population: int = 12
    noise_std: float = 0.12
    elite_fraction: float = 0.25


class ActorCritic:
    """Lightweight policy/value container (dependency-free fallback)."""

    def __init__(self, obs_dim: int, action_dim: int, seed: int = 0):
        rng = random.Random(seed)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.policy_w = [
            [(rng.random() - 0.5) * 0.2 for _ in range(obs_dim + 1)] for _ in range(action_dim)
        ]
        self.value_w = [(rng.random() - 0.5) * 0.1 for _ in range(obs_dim + 1)]

    @staticmethod
    def _tanh(x: float) -> float:
        return math.tanh(x)

    def deterministic_action(self, obs: List[float]) -> List[float]:
        x = obs + [1.0]
        out = []
        for row in self.policy_w:
            z = sum(row[i] * x[i] for i in range(len(x)))
            out.append(max(-1.0, min(1.0, self._tanh(z))))
        return out

    def value(self, obs: List[float]) -> float:
        x = obs + [1.0]
        return sum(self.value_w[i] * x[i] for i in range(len(x)))


class PPOTrainer:
    """PPO-style trainer fallback.

    In environments without ML deps, this uses a compact population search over
    policy parameters while keeping the same training/checkpoint interface.
    """

    def __init__(self, obs_dim: int, action_dim: int, config: PPOConfig | None = None):
        self.config = config or PPOConfig()
        self.rng = random.Random(self.config.seed)
        self.model = ActorCritic(obs_dim=obs_dim, action_dim=action_dim, seed=self.config.seed)

    def _flatten(self) -> List[float]:
        flat = []
        for row in self.model.policy_w:
            flat.extend(row)
        flat.extend(self.model.value_w)
        return flat

    def _unflatten(self, flat: List[float]) -> None:
        idx = 0
        for r in range(self.model.action_dim):
            for c in range(self.model.obs_dim + 1):
                self.model.policy_w[r][c] = flat[idx]
                idx += 1
        for c in range(self.model.obs_dim + 1):
            self.model.value_w[c] = flat[idx]
            idx += 1

    def _episode_reward(self, env: RLEnv, seed: int) -> float:
        obs = env.reset(seed=seed)
        done = False
        total = 0.0
        while not done:
            action = self.model.deterministic_action(obs)
            obs, reward, done, _ = env.step(action)
            total += reward
        return total

    def train(self, env: RLEnv) -> Dict[str, List[float]]:
        cfg = self.config
        params = self._flatten()
        log = {"episode_reward": []}

        approx_iters = max(1, cfg.total_steps // max(1, cfg.rollout_steps))
        elite_k = max(1, int(cfg.population * cfg.elite_fraction))

        for it in range(approx_iters):
            candidates = []

            self._unflatten(params)
            base_seed = cfg.seed + it * 100
            base_reward = self._episode_reward(env, seed=base_seed)
            candidates.append((base_reward, params[:]))

            for j in range(cfg.population):
                trial = [p + self.rng.gauss(0.0, cfg.noise_std) for p in params]
                self._unflatten(trial)
                reward = self._episode_reward(env, seed=base_seed + j + 1)
                candidates.append((reward, trial))

            candidates.sort(key=lambda x: x[0], reverse=True)
            elites = candidates[:elite_k]
            params = [sum(e[1][i] for e in elites) / elite_k for i in range(len(params))]
            log["episode_reward"].append(elites[0][0])

        self._unflatten(params)
        return log

    def save(self, path: str) -> None:
        payload = {
            "obs_dim": self.model.obs_dim,
            "action_dim": self.model.action_dim,
            "policy_w": self.model.policy_w,
            "value_w": self.model.value_w,
            "config": self.config.__dict__,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @staticmethod
    def load(path: str, obs_dim: int, action_dim: int) -> ActorCritic:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        model = ActorCritic(obs_dim=obs_dim, action_dim=action_dim, seed=0)
        model.policy_w = payload["policy_w"]
        model.value_w = payload["value_w"]
        return model
