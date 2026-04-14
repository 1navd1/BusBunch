from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

from src.models.contracts import ControlAction, ControlState, EpisodeMetrics, EpisodeReport
from src.models.predictor import GraphAwarePredictor, Predictor
from src.sim.entities import EpisodeScenario
from src.sim.simulator import Simulator


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
        sim = Simulator()
        base_steps = int(sim.corridor_cfg["sim"]["max_steps_peak"])
        if peak_profile == "off_peak":
            base_steps = int(sim.corridor_cfg["sim"]["max_steps_off_peak"])

        return ScenarioConfig(
            n_buses=int(sim.corridor_cfg["n_buses"]),
            max_steps=base_steps,
            cycle_time_sec=float(sum(sim.corridor_cfg["segment_base_travel_time_sec"])),
            bunching_threshold_sec=float(sim.corridor_cfg["sim"]["bunching_threshold_sec"]),
            day_type=day_type,
            peak_profile=peak_profile,
        )


class ControlEnv:
    """Replay/eval wrapper over simulator with a best-effort predictor backend."""

    def __init__(
        self,
        scenario: ScenarioConfig | None = None,
        stgnn_checkpoint: str = "artifacts/models/stgnn_best.pt",
        stgnn_norm: str = "artifacts/models/stgnn_norm.json",
        predictor: Predictor | None = None,
    ):
        self.scenario = scenario or ScenarioGenerator.create("weekday", "peak")
        self.sim = Simulator()
        self.predictor = predictor or self._build_predictor(
            stgnn_checkpoint=stgnn_checkpoint,
            stgnn_norm=stgnn_norm,
        )

    def _build_predictor(self, stgnn_checkpoint: str, stgnn_norm: str) -> Predictor:
        checkpoint = Path(stgnn_checkpoint)
        norm = Path(stgnn_norm)
        if checkpoint.exists() and norm.exists():
            try:
                from src.models.stgnn_infer import STGNNPredictor

                return STGNNPredictor(checkpoint_path=stgnn_checkpoint, norm_path=stgnn_norm)
            except Exception:
                pass

        return GraphAwarePredictor(num_buses=int(self.sim.corridor_cfg["n_buses"]))

    @property
    def obs_dim(self) -> int:
        return 8

    @property
    def action_dim(self) -> int:
        return 3

    def _focused_bus_idx(self, system_state) -> int:
        fwd = [b.headway_forward_sec for b in system_state.buses]
        return min(range(len(fwd)), key=lambda i: fwd[i])

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
    def _obs_from_control(control_state: ControlState) -> List[float]:
        return [
            control_state.current_bus_delay_sec / 600.0,
            control_state.forward_headway_sec / 600.0,
            control_state.backward_headway_sec / 600.0,
            control_state.occupancy_ratio,
            control_state.stop_demand_estimate / 80.0,
            control_state.predicted_corridor_congestion_score,
            control_state.predicted_bunching_risk,
            1.0 if control_state.is_terminal else 0.0,
        ]

    @staticmethod
    def vector_to_action(vec: List[float]) -> ControlAction:
        v0 = max(-1.0, min(1.0, vec[0]))
        v1 = max(-1.0, min(1.0, vec[1]))
        v2 = max(-1.0, min(1.0, vec[2]))
        return ControlAction(
            hold_sec=max(0.0, v0) * 60.0,
            speed_delta_pct=v1 * 0.2,
            dispatch_offset_sec=max(0.0, v2) * 90.0,
        )

    def reset(self, seed: int | None = None) -> List[float]:
        scenario = EpisodeScenario(
            day_type=self.scenario.day_type,
            peak_profile=self.scenario.peak_profile,
            max_steps=self.scenario.max_steps,
            seed=7 if seed is None else int(seed),
        )
        self.sim.reset(scenario)

        state = self.sim.current_system_state()
        pred = self.predictor.predict(state)
        control = self._control_state(state, pred)
        return self._obs_from_control(control)

    def step(self, action_vec: List[float]) -> Tuple[List[float], float, bool, Dict]:
        action = self.vector_to_action(action_vec)
        next_state, reward, done, info = self.sim.step(action)

        pred = self.predictor.predict(next_state)
        control = self._control_state(next_state, pred)
        obs = self._obs_from_control(control)
        return obs, reward, done, info


MiniBusEnv = ControlEnv


class ScenarioRunner:
    @staticmethod
    def run(policy, scenario: ScenarioConfig, seed: int = 7) -> EpisodeReport:
        env = ControlEnv(scenario)
        _ = env.reset(seed=seed)
        done = False
        total_reward = 0.0

        trace: List[Dict] = []

        while not done:
            system_state = env.sim.current_system_state()
            prediction_bundle = env.predictor.predict(system_state)
            control_state = env._control_state(system_state, prediction_bundle)

            action = policy.act(control_state, prediction_bundle)
            next_state, reward, done, info = env.sim.step(action)
            total_reward += reward

            trace.append(
                env.sim.snapshot_json(
                    system_state=next_state,
                    action=action,
                    reward=reward,
                    bunching=int(info["bunching"]),
                    prediction_bundle=asdict(prediction_bundle),
                    control_state=asdict(control_state),
                )
            )

        metrics = env.sim.episode_metrics()
        return EpisodeReport(
            policy_name=getattr(policy, "name", policy.__class__.__name__),
            seed=seed,
            total_reward=float(total_reward),
            metrics=metrics,
            trace=trace,
        )


def run_episode(env: ControlEnv, policy, seed: int = 7) -> Tuple[EpisodeMetrics, List[Dict], float]:
    report = ScenarioRunner.run(policy, env.scenario, seed)
    return report.metrics, report.trace, report.total_reward


def metrics_dict(metrics: EpisodeMetrics) -> Dict:
    return asdict(metrics)
