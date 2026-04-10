from __future__ import annotations

from pathlib import Path

import numpy as np

from src.models.contracts import ControlAction, ControlState, PredictionBundle

try:
    from stable_baselines3 import PPO
except Exception as exc:  # pragma: no cover
    raise RuntimeError("stable-baselines3 is required for RLPolicy inference.") from exc


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class RLPolicy:
    name = "ppo"

    def __init__(self, checkpoint_path: str, obs_dim: int, action_dim: int = 3):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Missing PPO checkpoint at {self.checkpoint_path}. Run `python3 -m src.train.train_ppo` first."
            )

        self.model = PPO.load(str(self.checkpoint_path), device="cpu")

    @staticmethod
    def _obs_from_control(control_state: ControlState, prediction_bundle: PredictionBundle) -> list[float]:
        return [
            control_state.current_bus_delay_sec / 600.0,
            control_state.forward_headway_sec / 600.0,
            control_state.backward_headway_sec / 600.0,
            control_state.occupancy_ratio,
            control_state.stop_demand_estimate / 80.0,
            prediction_bundle.congestion_score,
            prediction_bundle.bunching_risk_score,
            1.0 if control_state.is_terminal else 0.0,
        ]

    @staticmethod
    def _action_to_control(action: np.ndarray) -> ControlAction:
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.shape[0] < 3:
            raise ValueError(f"Expected 3 action dimensions, got {a.shape[0]}")
        return ControlAction(
            hold_sec=max(0.0, float(a[0])) * 60.0,
            speed_delta_pct=_clamp(float(a[1]) * 0.2, -0.2, 0.2),
            dispatch_offset_sec=max(0.0, float(a[2])) * 90.0,
        )

    def act(self, control_state: ControlState, prediction_bundle: PredictionBundle) -> ControlAction:
        obs = np.asarray(self._obs_from_control(control_state, prediction_bundle), dtype=np.float32)
        action, _ = self.model.predict(obs, deterministic=True)
        return self._action_to_control(action)

    def action_vector(self, observation: list[float]) -> list[float]:
        obs = np.asarray(observation, dtype=np.float32)
        action, _ = self.model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32).reshape(-1).tolist()
