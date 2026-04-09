from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

from src.models.contracts import BusState, ControlAction, SegmentState, StopState, SystemState
from src.sim.dispatch import compute_dispatch_delay
from src.sim.entities import Bus, EpisodeScenario, RouteGraph, RouteSegment, Stop
from src.sim.metrics import MetricsTracker
from src.sim.passengers import board_alight, demand_context, traffic_multiplier, update_stop_queues


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class Simulator:
    def __init__(
        self,
        corridor_path: str = "src/data/corridor_config.json",
        stops_path: str = "src/data/stops.csv",
        demand_path: str = "src/data/demand_profile.json",
        traffic_path: str = "src/data/traffic_profile.json",
    ):
        self.corridor_cfg = json.loads(Path(corridor_path).read_text(encoding="utf-8"))
        self.demand_cfg = json.loads(Path(demand_path).read_text(encoding="utf-8"))
        self.traffic_cfg = json.loads(Path(traffic_path).read_text(encoding="utf-8"))

        self.stops = self._load_stops(stops_path)
        self.route_graph = self._build_route_graph(self.stops, self.corridor_cfg["segment_base_travel_time_sec"])

        self.n_buses = int(self.corridor_cfg["n_buses"])
        self.capacity = int(self.corridor_cfg["bus_capacity"])
        self.tick_sec = float(self.corridor_cfg["sim"]["tick_sec"])
        self.bunching_threshold_sec = float(self.corridor_cfg["sim"]["bunching_threshold_sec"])

        self.rng = random.Random(7)
        self.metrics = MetricsTracker()

        self.step_idx = 0
        self.max_steps = int(self.corridor_cfg["sim"]["max_steps_peak"])
        self.day_type = "weekday"
        self.peak_profile = "peak"

        self.headways: List[float] = []
        self.occupancies: List[int] = []
        self.stop_queues: List[int] = []
        self.delay_proxy: List[float] = []

    @staticmethod
    def _load_stops(path: str) -> List[Stop]:
        out: List[Stop] = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                out.append(
                    Stop(
                        stop_id=r["stop_id"],
                        stop_name=r["stop_name"],
                        lat=float(r["lat"]),
                        lon=float(r["lon"]),
                        order=int(r["order"]),
                    )
                )
        out.sort(key=lambda s: s.order)
        return out

    @staticmethod
    def _build_route_graph(stops: List[Stop], segment_times: List[float]) -> RouteGraph:
        segments: List[RouteSegment] = []
        n = len(stops)
        for i in range(n):
            segments.append(
                RouteSegment(
                    segment_id=f"seg_{i+1}",
                    from_stop_id=stops[i].stop_id,
                    to_stop_id=stops[(i + 1) % n].stop_id,
                    base_travel_time_sec=float(segment_times[i % len(segment_times)]),
                )
            )
        return RouteGraph.load(stops=stops, segments=segments)

    def _context_profiles(self) -> Tuple[Dict, Dict]:
        demand_profile = self.demand_cfg[self.day_type][self.peak_profile]
        traffic_profile = self.traffic_cfg[self.day_type][self.peak_profile]
        return demand_profile, traffic_profile

    def reset(self, scenario: EpisodeScenario) -> None:
        self.day_type = scenario.day_type
        self.peak_profile = scenario.peak_profile
        self.max_steps = scenario.max_steps
        self.step_idx = 0

        self.rng = random.Random(scenario.seed)
        self.metrics = MetricsTracker()

        self.headways = [float(x) for x in self.corridor_cfg["initial_headways_sec"]]
        self.occupancies = [52, 34, 29][: self.n_buses]
        if len(self.occupancies) < self.n_buses:
            self.occupancies.extend([30] * (self.n_buses - len(self.occupancies)))

        self.stop_queues = [self.rng.randint(8, 20) for _ in self.route_graph.stops]
        self.delay_proxy = [0.0 for _ in range(self.n_buses)]

    def _focused_bus_idx(self) -> int:
        return min(range(self.n_buses), key=lambda i: self.headways[i])

    def _build_system_state(self, time_sin: float, demand_level: float, traffic_mult: float) -> SystemState:
        mean_h = sum(self.headways) / len(self.headways)

        buses: List[BusState] = []
        for i in range(self.n_buses):
            occ_ratio = _clip(self.occupancies[i] / float(self.capacity), 0.0, 1.0)
            buses.append(
                BusState(
                    bus_id=f"bus_{i+1}",
                    stop_index=i % len(self.route_graph.stops),
                    position_progress=float((self.step_idx % 10) / 10.0),
                    occupancy=occ_ratio,
                    delay_sec=max(0.0, self.headways[i] - mean_h),
                    headway_forward_sec=float(self.headways[i]),
                    headway_backward_sec=float(self.headways[(i - 1) % self.n_buses]),
                    status="terminal" if i == 0 and self.step_idx % 15 == 0 else "in_service",
                )
            )

        stops = [
            StopState(stop_id=self.route_graph.stops[i].stop_id, queue_len=int(self.stop_queues[i]))
            for i in range(len(self.route_graph.stops))
        ]

        segments = [
            SegmentState(
                segment_id=s.segment_id,
                from_stop_id=s.from_stop_id,
                to_stop_id=s.to_stop_id,
                base_travel_time_sec=s.base_travel_time_sec,
                traffic_multiplier=float(traffic_mult),
            )
            for s in self.route_graph.segments
        ]

        return SystemState(
            timestamp=self.step_idx,
            buses=buses,
            stops=stops,
            segments=segments,
            terminal_queue={"main_terminal": 1 if self.step_idx % 15 == 0 else 0},
            global_traffic_context={
                "time_sin": float(time_sin),
                "demand_level": float(demand_level),
                "traffic_level": float(_clip((traffic_mult - 0.7) / 0.8, 0.0, 1.0)),
            },
        )

    def step(self, action: ControlAction) -> tuple[SystemState, float, bool, Dict[str, float]]:
        demand_profile, traffic_profile = self._context_profiles()

        self.stop_queues, time_sin, demand_level = update_stop_queues(
            queues=self.stop_queues,
            tick_sec=self.tick_sec,
            demand_profile=demand_profile,
            step_idx=self.step_idx,
            rng=self.rng,
        )
        traffic_mult = traffic_multiplier(self.step_idx, traffic_profile)

        focused = self._focused_bus_idx()
        hold_sec = _clip(action.hold_sec, 0.0, float(self.corridor_cfg["controls"]["max_hold_sec"]))
        speed_delta = _clip(
            action.speed_delta_pct,
            -float(self.corridor_cfg["controls"]["max_speed_delta_pct"]),
            float(self.corridor_cfg["controls"]["max_speed_delta_pct"]),
        )
        dispatch_offset = _clip(
            action.dispatch_offset_sec,
            0.0,
            float(self.corridor_cfg["controls"]["max_dispatch_offset_sec"]),
        )

        mean_hw = sum(self.headways) / len(self.headways)
        imbalance = [h - mean_hw for h in self.headways]

        stochastic = [self.rng.gauss(0.0, 8.0) for _ in self.headways]
        traffic_push = [(traffic_mult - 1.0) * 18.0, (traffic_mult - 1.0) * -9.0, (traffic_mult - 1.0) * -9.0]
        if len(traffic_push) < self.n_buses:
            traffic_push.extend([0.0] * (self.n_buses - len(traffic_push)))

        natural = [
            self.headways[i] + 0.17 * imbalance[i] + stochastic[i] + traffic_push[i]
            for i in range(self.n_buses)
        ]

        # Passenger interaction and dwell impact per focused bus stop.
        stop_idx = focused % len(self.stop_queues)
        board, alight, new_occ = board_alight(self.stop_queues[stop_idx], self.occupancies[focused], self.capacity, self.rng)
        self.stop_queues[stop_idx] = max(0, self.stop_queues[stop_idx] - board)
        self.occupancies[focused] = new_occ

        dwell_cfg = self.corridor_cfg["dwell"]
        dwell_time = min(
            float(dwell_cfg["max_sec"]),
            float(dwell_cfg["base_sec"]) + board * float(dwell_cfg["board_sec_per_passenger"]) + alight * float(dwell_cfg["alight_sec_per_passenger"]),
        )

        is_terminal = focused == 0 and self.step_idx % 15 == 0
        dispatch_delay = compute_dispatch_delay(
            is_terminal=is_terminal,
            forward_headway_sec=self.headways[focused],
            target_headway_sec=float(self.corridor_cfg["dispatch"]["target_headway_sec"]),
            dispatch_offset_sec=dispatch_offset,
        )

        control_strength = 0.34 * (hold_sec / 60.0) + 0.22 * max(0.0, -speed_delta / 0.2) + 0.26 * (dispatch_delay / 120.0)

        controlled = [natural[i] - control_strength * imbalance[i] for i in range(self.n_buses)]
        controlled[focused] += 0.22 * dwell_time + 0.18 * hold_sec + 0.12 * dispatch_delay
        controlled[focused] -= 35.0 * max(0.0, speed_delta)

        self.headways = [_clip(x, 55.0, 560.0) for x in controlled]
        cycle_time = sum(float(s.base_travel_time_sec) for s in self.route_graph.segments) * traffic_mult
        cycle_time = max(780.0, cycle_time)
        scale = cycle_time / sum(self.headways)
        self.headways = [h * scale for h in self.headways]

        # Occupancy drift for all buses.
        mean_headway = max(1e-6, sum(self.headways) / self.n_buses)
        for i in range(self.n_buses):
            # Longer headway buses pick up more passengers and become more crowded,
            # which increases occupancy imbalance under bunching.
            headway_load_effect = 0.20 * ((self.headways[i] / mean_headway) - 1.0)
            drift = 0.03 * demand_level + headway_load_effect
            self.occupancies[i] = int(
                _clip(self.occupancies[i] + drift * 8 + self.rng.gauss(0.0, 1.5), 6, self.capacity)
            )

        bunching_step = sum(1 for h in self.headways if h < self.bunching_threshold_sec)
        occupancies_ratio = [o / float(self.capacity) for o in self.occupancies]

        fuel_step = 1.0 + 0.01 * hold_sec + 1.5 * abs(speed_delta) + 0.002 * dispatch_delay + 0.45 * max(0.0, traffic_mult - 1.0)
        delay_step = 0.45 * hold_sec + 0.25 * dispatch_delay + 0.2 * dwell_time

        self.metrics.update(
            headways=self.headways,
            occupancies=occupancies_ratio,
            bunching_step=bunching_step,
            hold_sec=hold_sec,
            delay_sec=delay_step,
            fuel_step=fuel_step,
        )

        hw_std = self.metrics.headway_stds[-1]
        wait_proxy = self.metrics.wait_times[-1]
        occ_std = self.metrics.occupancy_stds[-1]
        peak_weight = 1.25 if time_sin > 0.4 else 1.0

        reward = -(
            peak_weight * (2.8 * bunching_step + 0.016 * hw_std + 0.004 * wait_proxy + 0.75 * occ_std)
            + 0.10 * fuel_step
            + 0.06 * (hold_sec / 60.0 + abs(speed_delta) / 0.2 + dispatch_delay / 90.0)
        )

        self.step_idx += 1
        done = self.step_idx >= self.max_steps

        state = self._build_system_state(time_sin=time_sin, demand_level=demand_level, traffic_mult=traffic_mult)
        info = {
            "bunching": float(bunching_step),
            "headway_std": float(hw_std),
            "wait_proxy": float(wait_proxy),
            "occupancy_std": float(occ_std),
            "fuel_proxy_step": float(fuel_step),
            "hold_sec": float(hold_sec),
            "dispatch_delay": float(dispatch_delay),
        }
        return state, float(reward), bool(done), info

    def current_system_state(self) -> SystemState:
        demand_profile, traffic_profile = self._context_profiles()
        time_sin, demand_level = demand_context(self.step_idx, demand_profile)
        traffic_mult = traffic_multiplier(self.step_idx, traffic_profile)
        return self._build_system_state(time_sin=time_sin, demand_level=demand_level, traffic_mult=traffic_mult)

    def episode_metrics(self):
        return self.metrics.to_episode_metrics()

    def snapshot_json(self, system_state: SystemState, action: ControlAction, reward: float, bunching: int, prediction_bundle: Dict | None = None, control_state: Dict | None = None) -> Dict:
        frame = {
            "step": self.step_idx,
            "system_state": asdict(system_state),
            "action": asdict(action),
            "bunching": int(bunching),
            "reward": round(float(reward), 4),
        }
        if prediction_bundle is not None:
            frame["prediction_bundle"] = prediction_bundle
        if control_state is not None:
            frame["control_state"] = control_state
        return frame
