from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.models.contracts import ControlAction, ControlState
from src.models.stgnn_infer import STGNNPredictor
from src.sim.entities import EpisodeScenario
from src.sim.simulator import Simulator


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class EnvScenario:
    day_type: str = "weekday"
    peak_profile: str = "peak"
    max_steps: int = 180


class BusBunchEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: EnvScenario | None = None,
        predictor: STGNNPredictor | None = None,
        seed: int = 7,
    ):
        super().__init__()
        self.sim = Simulator()
        self.scenario = scenario or EnvScenario(max_steps=int(self.sim.corridor_cfg["sim"]["max_steps_peak"]))
        self.predictor = predictor or STGNNPredictor()
        self.default_seed = int(seed)

        self.max_hold = float(self.sim.corridor_cfg["controls"]["max_hold_sec"])
        self.max_speed = float(self.sim.corridor_cfg["controls"]["max_speed_delta_pct"])
        self.max_dispatch = float(self.sim.corridor_cfg["controls"]["max_dispatch_offset_sec"])

        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        self._last_control: ControlState | None = None

    @staticmethod
    def _focused_bus_idx(system_state) -> int:
        fwd = [b.headway_forward_sec for b in system_state.buses]
        return int(min(range(len(fwd)), key=lambda i: fwd[i]))

    def _control_state(self, system_state, prediction_bundle) -> ControlState:
        i = self._focused_bus_idx(system_state)
        bus = system_state.buses[i]
        demand_est = prediction_bundle.stop_demand_forecast[i % len(prediction_bundle.stop_demand_forecast)]
        return ControlState(
            focus_bus_id=bus.bus_id,
            focus_stop_id=bus.current_stop_id,
            focus_stop_name=bus.current_stop_name,
            current_bus_delay_sec=bus.delay_sec,
            forward_headway_sec=bus.headway_forward_sec,
            backward_headway_sec=bus.headway_backward_sec,
            occupancy_ratio=bus.occupancy,
            stop_demand_estimate=demand_est,
            predicted_corridor_congestion_score=prediction_bundle.congestion_score,
            predicted_bunching_risk=prediction_bundle.bunching_risk_score,
            is_terminal=(bus.status == "terminal"),
        )

    @staticmethod
    def obs_from_control(control_state: ControlState) -> np.ndarray:
        return np.array(
            [
                control_state.current_bus_delay_sec / 600.0,
                control_state.forward_headway_sec / 600.0,
                control_state.backward_headway_sec / 600.0,
                control_state.occupancy_ratio,
                control_state.stop_demand_estimate / 80.0,
                control_state.predicted_corridor_congestion_score,
                control_state.predicted_bunching_risk,
                1.0 if control_state.is_terminal else 0.0,
            ],
            dtype=np.float32,
        )

    def action_to_control(self, action: np.ndarray) -> ControlAction:
        a = np.asarray(action, dtype=np.float32).reshape(3)
        return ControlAction(
            hold_sec=max(0.0, float(a[0])) * self.max_hold,
            speed_delta_pct=_clip(float(a[1]) * self.max_speed, -self.max_speed, self.max_speed),
            dispatch_offset_sec=max(0.0, float(a[2])) * self.max_dispatch,
        )

    def _reward(self, info: Dict[str, float], action: ControlAction) -> float:
        bunching_penalty = 2.8 * float(info.get("bunching", 0.0))
        headway_penalty = 0.016 * float(info.get("headway_std", 0.0))
        wait_penalty = 0.004 * float(info.get("wait_proxy", 0.0))
        occupancy_penalty = 0.75 * float(info.get("occupancy_std", 0.0))

        intervention = (
            action.hold_sec / max(1.0, self.max_hold)
            + abs(action.speed_delta_pct) / max(1e-6, self.max_speed)
            + action.dispatch_offset_sec / max(1.0, self.max_dispatch)
        )
        intervention_penalty = 0.06 * intervention

        return -(
            bunching_penalty
            + headway_penalty
            + wait_penalty
            + occupancy_penalty
            + intervention_penalty
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> Tuple[np.ndarray, Dict]:
        del options
        seed = self.default_seed if seed is None else int(seed)
        super().reset(seed=seed)

        self.sim.reset(
            EpisodeScenario(
                day_type=self.scenario.day_type,
                peak_profile=self.scenario.peak_profile,
                max_steps=self.scenario.max_steps,
                seed=seed,
            )
        )

        state = self.sim.current_system_state()
        pred = self.predictor.predict(state)
        control = self._control_state(state, pred)
        self._last_control = control
        return self.obs_from_control(control), {"seed": seed}

    def step(self, action: np.ndarray):
        ctrl = self.action_to_control(action)
        next_state, _, done, info = self.sim.step(ctrl)

        pred = self.predictor.predict(next_state)
        control = self._control_state(next_state, pred)
        obs = self.obs_from_control(control)

        reward = self._reward(info, ctrl)
        self._last_control = control
        terminated = bool(done)
        truncated = False
        return obs, float(reward), terminated, truncated, info
