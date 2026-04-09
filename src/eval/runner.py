from __future__ import annotations

import math
import random
from dataclasses import asdict
from typing import Dict, List, Tuple

from src.models.contracts import (
    BusState,
    ControlAction,
    ControlState,
    EpisodeMetrics,
    EpisodeReport,
    SegmentState,
    StopState,
    SystemState,
)
from src.models.predictor import GraphAwarePredictor


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _std(values: List[float]) -> float:
    mean = sum(values) / len(values)
    return (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5


class ScenarioConfig:
    def __init__(
        self,
        n_buses: int = 3,
        max_steps: int = 180,
        cycle_time_sec: float = 900.0,
        bunching_threshold_sec: float = 180.0,
        day_type: str = "weekday",
        peak_profile: str = "peak",
    ):
        self.n_buses = n_buses
        self.max_steps = max_steps
        self.cycle_time_sec = cycle_time_sec
        self.bunching_threshold_sec = bunching_threshold_sec
        self.day_type = day_type
        self.peak_profile = peak_profile


class ScenarioGenerator:
    @staticmethod
    def create(day_type: str, peak_profile: str) -> ScenarioConfig:
        cfg = ScenarioConfig(day_type=day_type, peak_profile=peak_profile)
        if peak_profile == "off_peak":
            cfg.max_steps = 140
        return cfg


class MiniBusEnv:
    """Simplified corridor simulator for policy development and demo artifacts."""

    def __init__(self, config: ScenarioConfig | None = None):
        self.cfg = config or ScenarioConfig()
        self.predictor = GraphAwarePredictor(self.cfg.n_buses)
        self.rng = random.Random(7)

        self.step_idx = 0
        self.headways = [self.cfg.cycle_time_sec / self.cfg.n_buses for _ in range(self.cfg.n_buses)]
        self.occupancies = [0.5 for _ in range(self.cfg.n_buses)]
        self.total_hold = 0.0
        self.total_delay = 0.0
        self.total_fuel = 0.0
        self.bunching_count = 0
        self.history: List[Dict] = []

    @property
    def obs_dim(self) -> int:
        # compact ControlState-like features for one controlled bus
        return 8

    @property
    def action_dim(self) -> int:
        return 3

    def _context(self) -> Tuple[float, float, float, float]:
        phase = 2 * math.pi * (self.step_idx % 60) / 60.0
        time_sin = math.sin(phase)
        time_cos = math.cos(phase)
        traffic = 0.45 + 0.35 * max(0.0, time_sin)
        demand = 0.4 + 0.4 * max(0.0, time_sin)
        return time_sin, time_cos, demand, traffic

    def _focused_bus_idx(self) -> int:
        return min(range(self.cfg.n_buses), key=lambda i: self.headways[i])

    def _system_state(self) -> SystemState:
        time_sin, _, demand, traffic = self._context()
        mean_h = sum(self.headways) / len(self.headways)
        buses = []
        for i in range(self.cfg.n_buses):
            buses.append(
                BusState(
                    bus_id=f"bus_{i+1}",
                    stop_index=i,
                    position_progress=float((self.step_idx % 10) / 10.0),
                    occupancy=self.occupancies[i],
                    delay_sec=max(0.0, self.headways[i] - mean_h),
                    headway_forward_sec=self.headways[i],
                    headway_backward_sec=self.headways[(i - 1) % self.cfg.n_buses],
                    status="in_service" if i != 0 else "terminal" if self.step_idx % 15 == 0 else "in_service",
                )
            )

        stops = [StopState(stop_id=f"stop_{i+1}", queue_len=int(10 + 12 * demand)) for i in range(self.cfg.n_buses)]
        segments = [
            SegmentState(
                segment_id=f"seg_{i+1}",
                from_stop_id=f"stop_{i+1}",
                to_stop_id=f"stop_{(i + 1) % self.cfg.n_buses + 1}",
                base_travel_time_sec=90.0,
                traffic_multiplier=traffic,
            )
            for i in range(self.cfg.n_buses)
        ]

        return SystemState(
            timestamp=self.step_idx,
            buses=buses,
            stops=stops,
            segments=segments,
            terminal_queue={"main_terminal": 1 if self.step_idx % 15 == 0 else 0},
            global_traffic_context={"traffic_level": traffic, "demand_level": demand, "time_sin": time_sin},
        )

    def _control_state_and_prediction(self) -> Tuple[ControlState, object]:
        s = self._system_state()
        pred = self.predictor.predict(s)
        i = self._focused_bus_idx()
        mean_h = sum(self.headways) / len(self.headways)
        control = ControlState(
            current_bus_delay_sec=max(0.0, self.headways[i] - mean_h),
            forward_headway_sec=self.headways[i],
            backward_headway_sec=self.headways[(i - 1) % self.cfg.n_buses],
            occupancy_ratio=self.occupancies[i],
            stop_demand_estimate=pred.stop_demand_forecast[i],
            predicted_corridor_congestion_score=pred.congestion_score,
            predicted_bunching_risk=pred.bunching_risk_score,
            is_terminal=(i == 0 and self.step_idx % 15 == 0),
        )
        return control, pred

    def _obs(self) -> List[float]:
        control, pred = self._control_state_and_prediction()
        return [
            control.current_bus_delay_sec / 600.0,
            control.forward_headway_sec / 600.0,
            control.backward_headway_sec / 600.0,
            control.occupancy_ratio,
            control.stop_demand_estimate / 12.0,
            pred.congestion_score,
            pred.bunching_risk_score,
            1.0 if control.is_terminal else 0.0,
        ]

    def _vector_to_action(self, a: List[float]) -> ControlAction:
        vec = [_clip(x, -1.0, 1.0) for x in a]
        return ControlAction(
            hold_sec=max(0.0, vec[0]) * 60.0,
            speed_delta_pct=vec[1] * 0.2,
            dispatch_offset_sec=max(0.0, vec[2]) * 90.0,
        )

    def reset(self, seed: int | None = None) -> List[float]:
        if seed is not None:
            self.rng = random.Random(seed)

        self.step_idx = 0
        self.headways = [210.0, 320.0, 370.0]
        self.occupancies = [0.75, 0.45, 0.38]
        self.total_hold = 0.0
        self.total_delay = 0.0
        self.total_fuel = 0.0
        self.bunching_count = 0
        self.history = []
        return self._obs()

    def step_action(self, action: ControlAction):
        hold = action.hold_sec / 60.0
        speed_red = max(0.0, -action.speed_delta_pct / 0.2)
        dispatch_off = action.dispatch_offset_sec / 90.0

        self.total_hold += action.hold_sec
        self.total_delay += hold * 25.0 + speed_red * 10.0 + dispatch_off * 15.0

        time_sin, _, demand, traffic = self._context()
        mean_hw = sum(self.headways) / len(self.headways)
        imbalance = [h - mean_hw for h in self.headways]

        stochastic = [self.rng.gauss(0.0, 8.0) for _ in self.headways]
        traffic_push = [(traffic - 0.45) * 16.0, (traffic - 0.45) * -8.0, (traffic - 0.45) * -8.0]

        natural = [
            self.headways[i] + 0.16 * imbalance[i] + stochastic[i] + traffic_push[i]
            for i in range(self.cfg.n_buses)
        ]

        control_strength = 0.38 * hold + 0.25 * speed_red + 0.28 * dispatch_off
        controlled = [natural[i] - control_strength * imbalance[i] for i in range(self.cfg.n_buses)]
        self.headways = [_clip(x, 55.0, 560.0) for x in controlled]

        scale = self.cfg.cycle_time_sec / sum(self.headways)
        self.headways = [h * scale for h in self.headways]

        occ_noise = [self.rng.gauss(0.0, 0.03) for _ in self.occupancies]
        mean_h = sum(self.headways) / len(self.headways)
        self.occupancies = [
            _clip(
                self.occupancies[i] + 0.08 * demand - 0.06 * (self.headways[i] / mean_h) + occ_noise[i],
                0.08,
                1.0,
            )
            for i in range(self.cfg.n_buses)
        ]

        bunching = sum(1 for h in self.headways if h < self.cfg.bunching_threshold_sec)
        self.bunching_count += bunching

        hw_std = _std(self.headways)
        wait_proxy = sum(self.headways) / len(self.headways) / 2.0 + 0.35 * hw_std
        occ_std = _std(self.occupancies)

        fuel_step = 1.0 + 0.15 * hold + 0.1 * speed_red + 0.06 * dispatch_off + 0.2 * traffic
        self.total_fuel += fuel_step

        peak_weight = 1.2 if time_sin > 0.4 else 1.0
        reward = -(
            peak_weight * (2.8 * bunching + 0.015 * hw_std + 0.004 * wait_proxy + 0.8 * occ_std)
            + 0.18 * fuel_step
            + 0.05 * (hold + speed_red + dispatch_off)
        )

        self.step_idx += 1
        done = self.step_idx >= self.cfg.max_steps

        control_state, pred = self._control_state_and_prediction()
        self.history.append(
            {
                "step": self.step_idx,
                "system_state": asdict(self._system_state()),
                "control_state": asdict(control_state),
                "prediction_bundle": asdict(pred),
                "action": asdict(action),
                "bunching": int(bunching),
                "reward": round(float(reward), 4),
            }
        )

        info = {
            "bunching": bunching,
            "headway_std": hw_std,
            "wait_proxy": wait_proxy,
            "occupancy_std": occ_std,
            "fuel_proxy_step": fuel_step,
        }
        return self._obs(), float(reward), done, info

    def step(self, action_vector: List[float]):
        action = self._vector_to_action(action_vector)
        return self.step_action(action)


class ScenarioRunner:
    @staticmethod
    def run(policy, scenario: ScenarioConfig, seed: int = 7) -> EpisodeReport:
        env = MiniBusEnv(scenario)
        _ = env.reset(seed=seed)
        done = False
        total_reward = 0.0
        hw_stds: List[float] = []
        waits: List[float] = []
        occ_stds: List[float] = []
        fuels: List[float] = []

        while not done:
            control_state, prediction_bundle = env._control_state_and_prediction()
            action = policy.act(control_state, prediction_bundle)
            _, reward, done, info = env.step_action(action)
            total_reward += reward
            hw_stds.append(info["headway_std"])
            waits.append(info["wait_proxy"])
            occ_stds.append(info["occupancy_std"])
            fuels.append(info["fuel_proxy_step"])

        metrics = EpisodeMetrics(
            bunching_count=int(env.bunching_count),
            headway_std=float(sum(hw_stds) / len(hw_stds)),
            avg_wait_time=float(sum(waits) / len(waits)),
            occupancy_std=float(sum(occ_stds) / len(occ_stds)),
            fuel_proxy=float(sum(fuels)),
            total_delay=float(env.total_delay),
        )
        return EpisodeReport(
            policy_name=getattr(policy, "name", policy.__class__.__name__),
            seed=seed,
            total_reward=float(total_reward),
            metrics=metrics,
            trace=env.history,
        )


def run_episode(env: MiniBusEnv, policy, seed: int = 7) -> Tuple[EpisodeMetrics, List[Dict], float]:
    report = ScenarioRunner.run(policy, env.cfg, seed)
    return report.metrics, report.trace, report.total_reward


def metrics_dict(metrics: EpisodeMetrics) -> Dict:
    return asdict(metrics)
