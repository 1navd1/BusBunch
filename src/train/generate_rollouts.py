from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

from src.data.dataset import build_windows, compute_norm, normalize_x, save_dataset, save_norm_json, split_indices
from src.data.graph_builder import build_corridor_graph
from src.models.contracts import ControlAction, ControlState
from src.models.predictor import GraphAwarePredictor
from src.sim.entities import EpisodeScenario
from src.sim.simulator import Simulator
from src.policies.headway_policy import HeadwayPolicy
from src.policies.static_policy import StaticPolicy


class NoisyPolicy:
    name = "noisy"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def act(self, control_state, prediction_bundle):
        del control_state, prediction_bundle
        return ControlAction(
            hold_sec=max(0.0, self.rng.uniform(-0.1, 1.0)) * 60.0,
            speed_delta_pct=self.rng.uniform(-1.0, 1.0) * 0.2,
            dispatch_offset_sec=max(0.0, self.rng.uniform(-0.2, 1.0)) * 90.0,
        )


def collect_rollouts(
    seeds: List[int],
    peak_profile: str,
    include_noisy: bool = True,
) -> Dict[str, List[List[Dict]]]:
    policies = [StaticPolicy(), HeadwayPolicy()]
    if include_noisy:
        policies.append(NoisyPolicy(seed=17))

    data: Dict[str, List[List[Dict]]] = {p.name: [] for p in policies}
    sim = Simulator()
    max_steps = int(
        sim.corridor_cfg["sim"]["max_steps_peak"]
        if peak_profile == "peak"
        else sim.corridor_cfg["sim"]["max_steps_off_peak"]
    )
    predictor = GraphAwarePredictor(num_buses=int(sim.corridor_cfg["n_buses"]))

    for policy in policies:
        for seed in seeds:
            sim.reset(
                EpisodeScenario(
                    day_type="weekday",
                    peak_profile=peak_profile,
                    max_steps=max_steps,
                    seed=seed,
                )
            )

            trace: List[Dict] = []
            done = False
            while not done:
                sys = sim.current_system_state()
                pred = predictor.predict(sys)
                focus_idx = min(range(len(sys.buses)), key=lambda i: sys.buses[i].headway_forward_sec)
                bus = sys.buses[focus_idx]

                demand_est = pred.stop_demand_forecast[focus_idx % len(pred.stop_demand_forecast)]
                control_state = ControlState(
                    focus_bus_id=bus.bus_id,
                    focus_stop_id=bus.current_stop_id,
                    focus_stop_name=bus.current_stop_name,
                    current_bus_delay_sec=bus.delay_sec,
                    forward_headway_sec=bus.headway_forward_sec,
                    backward_headway_sec=bus.headway_backward_sec,
                    occupancy_ratio=bus.occupancy,
                    stop_demand_estimate=demand_est,
                    predicted_corridor_congestion_score=pred.congestion_score,
                    predicted_bunching_risk=pred.bunching_risk_score,
                    is_terminal=bus.status == "terminal",
                )

                action = policy.act(control_state=control_state, prediction_bundle=pred)
                next_state, reward, done, info = sim.step(action)
                trace.append(
                    sim.snapshot_json(
                        system_state=next_state,
                        action=action,
                        reward=reward,
                        bunching=int(info["bunching"]),
                        prediction_bundle={
                            "downstream_travel_time": pred.downstream_travel_time,
                            "stop_demand_forecast": pred.stop_demand_forecast,
                            "bunching_risk_score": pred.bunching_risk_score,
                            "congestion_score": pred.congestion_score,
                        },
                        control_state={
                            "focus_bus_id": control_state.focus_bus_id,
                            "focus_stop_id": control_state.focus_stop_id,
                            "focus_stop_name": control_state.focus_stop_name,
                            "current_bus_delay_sec": control_state.current_bus_delay_sec,
                            "forward_headway_sec": control_state.forward_headway_sec,
                            "backward_headway_sec": control_state.backward_headway_sec,
                            "occupancy_ratio": control_state.occupancy_ratio,
                            "stop_demand_estimate": control_state.stop_demand_estimate,
                            "predicted_corridor_congestion_score": control_state.predicted_corridor_congestion_score,
                            "predicted_bunching_risk": control_state.predicted_bunching_risk,
                            "is_terminal": control_state.is_terminal,
                        },
                    )
                )
            data[policy.name].append(trace)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic rollouts and STGNN dataset windows")
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--peak-profile", type=str, default="peak", choices=["peak", "off_peak"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[3, 7, 11, 17, 23, 29, 31, 37])
    parser.add_argument("--out-rollouts", type=str, default="artifacts/eval/rollouts.json")
    parser.add_argument("--out-dataset", type=str, default="artifacts/eval/stgnn_dataset.npz")
    parser.add_argument("--out-norm", type=str, default="artifacts/models/stgnn_norm.json")
    args = parser.parse_args()

    out_rollouts = Path(args.out_rollouts)
    out_rollouts.parent.mkdir(parents=True, exist_ok=True)

    rollouts_by_policy = collect_rollouts(seeds=args.seeds, peak_profile=args.peak_profile, include_noisy=True)
    out_rollouts.write_text(json.dumps(rollouts_by_policy), encoding="utf-8")

    flat_rollouts: List[List[Dict]] = []
    for traces in rollouts_by_policy.values():
        flat_rollouts.extend(traces)

    sim = Simulator()
    max_steps = int(
        sim.corridor_cfg["sim"]["max_steps_peak"]
        if args.peak_profile == "peak"
        else sim.corridor_cfg["sim"]["max_steps_off_peak"]
    )
    arrays = build_windows(flat_rollouts, history=args.history, horizon=args.horizon, max_steps=max_steps)
    norm = compute_norm(arrays["x"])
    arrays["x"] = normalize_x(arrays["x"], norm["mean"], norm["std"])

    split = split_indices(arrays["x"].shape[0], seed=7)
    graph = build_corridor_graph()

    metadata = {
        "history": args.history,
        "horizon": args.horizon,
        "n_samples": int(arrays["x"].shape[0]),
        "n_nodes": int(arrays["x"].shape[2]),
        "n_features": int(arrays["x"].shape[3]),
        "split": {k: int(v.shape[0]) for k, v in split.items()},
        "seeds": args.seeds,
        "policies": list(rollouts_by_policy.keys()),
        "peak_profile": args.peak_profile,
    }

    save_dataset(
        out_path=args.out_dataset,
        arrays=arrays,
        split=split,
        graph=graph,
        norm=norm,
        metadata=metadata,
    )
    save_norm_json(args.out_norm, norm=norm, graph=graph, history=args.history)

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
