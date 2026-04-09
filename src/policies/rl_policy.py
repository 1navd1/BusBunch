from __future__ import annotations

from src.models.contracts import ControlAction, ControlState, PredictionBundle
from src.models.ppo_policy import PPOTrainer


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class RLPolicy:
    name = "ppo"

    def __init__(self, checkpoint_path: str, obs_dim: int, action_dim: int = 3):
        self.model = PPOTrainer.load(checkpoint_path, obs_dim=obs_dim, action_dim=action_dim)

    @staticmethod
    def _obs_from_control(control_state: ControlState, prediction_bundle: PredictionBundle) -> list[float]:
        return [
            control_state.current_bus_delay_sec / 600.0,
            control_state.forward_headway_sec / 600.0,
            control_state.backward_headway_sec / 600.0,
            control_state.occupancy_ratio,
            control_state.stop_demand_estimate / 12.0,
            prediction_bundle.congestion_score,
            prediction_bundle.bunching_risk_score,
            1.0 if control_state.is_terminal else 0.0,
        ]

    def act(self, control_state: ControlState, prediction_bundle: PredictionBundle) -> ControlAction:
        obs = self._obs_from_control(control_state, prediction_bundle)
        a = self.model.deterministic_action(obs)
        return ControlAction(
            hold_sec=max(0.0, a[0]) * 60.0,
            speed_delta_pct=_clamp(a[1] * 0.2, -0.2, 0.2),
            dispatch_offset_sec=max(0.0, a[2]) * 90.0,
        )

    def action_vector(self, observation: list[float]) -> list[float]:
        return self.model.deterministic_action(observation)
