from __future__ import annotations

import copy
import json
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.runner import ScenarioGenerator, ScenarioRunner
from src.policies.headway_policy import HeadwayPolicy
from src.policies.rl_policy import RLPolicy
from src.policies.static_policy import StaticPolicy

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CHECKPOINT_PATH = ARTIFACTS_DIR / "ppo_checkpoint.json"


def _std(values: List[float]) -> float:
    if not values:
        return 0.0
    m = sum(values) / len(values)
    return (sum((x - m) ** 2 for x in values) / len(values)) ** 0.5


@st.cache_data
def load_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data
def load_demo_seed() -> Dict[str, Any]:
    return load_json(str(ARTIFACTS_DIR / "demo_seed.json"))


@st.cache_data
def load_kpi_summary() -> Dict[str, Any]:
    return load_json(str(ARTIFACTS_DIR / "kpi_summary.json"))


@st.cache_data
def run_policy_report(policy_name: str, seed: int, peak_profile: str) -> Dict[str, Any]:
    scenario = ScenarioGenerator.create(day_type="weekday", peak_profile=peak_profile)

    if policy_name == "static":
        policy = StaticPolicy()
    elif policy_name == "headway":
        policy = HeadwayPolicy()
    elif policy_name == "ppo":
        policy = RLPolicy(str(CHECKPOINT_PATH), obs_dim=8)
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    report = ScenarioRunner.run(policy, scenario, seed=seed)
    return {
        "policy": policy_name,
        "seed": seed,
        "scenario": {"day_type": scenario.day_type, "peak_profile": scenario.peak_profile},
        "metrics": asdict(report.metrics),
        "total_reward": report.total_reward,
        "trace": report.trace,
    }


def apply_modifiers(trace: List[Dict[str, Any]], traffic_spike: bool, passenger_surge: bool) -> List[Dict[str, Any]]:
    if not traffic_spike and not passenger_surge:
        return trace

    out = copy.deepcopy(trace)
    n = len(out)

    for i, frame in enumerate(out):
        if traffic_spike and int(0.22 * n) <= i <= int(0.62 * n):
            for s in frame["system_state"]["segments"]:
                s["traffic_multiplier"] = min(1.8, s["traffic_multiplier"] * 1.2)
            frame["prediction_bundle"]["congestion_score"] = min(
                1.0, frame["prediction_bundle"]["congestion_score"] + 0.18
            )
            frame["prediction_bundle"]["bunching_risk_score"] = min(
                1.0, frame["prediction_bundle"]["bunching_risk_score"] + 0.08
            )
            for b in frame["system_state"]["buses"]:
                b["headway_forward_sec"] *= 1.1
                b["delay_sec"] *= 1.15

        if passenger_surge and int(0.33 * n) <= i <= int(0.78 * n):
            for stop in frame["system_state"]["stops"]:
                stop["queue_len"] = int(stop["queue_len"] * 1.4)
            for b in frame["system_state"]["buses"]:
                b["occupancy"] = min(1.0, b["occupancy"] * 1.18)
                b["delay_sec"] *= 1.08
            frame["action"]["hold_sec"] *= 1.15
            frame["prediction_bundle"]["bunching_risk_score"] = min(
                1.0, frame["prediction_bundle"]["bunching_risk_score"] + 0.07
            )

        bunch = sum(1 for b in frame["system_state"]["buses"] if b["headway_forward_sec"] < 180)
        frame["bunching"] = bunch

    return out


def summarize_trace(trace: List[Dict[str, Any]]) -> Dict[str, float]:
    if not trace:
        return {
            "bunching_count": 0.0,
            "avg_wait_time": 0.0,
            "occupancy_std": 0.0,
            "headway_std": 0.0,
            "fuel_proxy": 0.0,
            "total_delay": 0.0,
        }

    bunch = 0
    waits, occ_stds, hw_stds = [], [], []
    fuel_proxy, total_delay = 0.0, 0.0

    for frame in trace:
        buses = frame["system_state"]["buses"]
        hws = [float(b["headway_forward_sec"]) for b in buses]
        occ = [float(b["occupancy"]) for b in buses]

        bunch += int(frame["bunching"])
        waits.append(sum(hws) / len(hws) / 2.0)
        occ_stds.append(_std(occ))
        hw_stds.append(_std(hws))

        a = frame["action"]
        fuel_proxy += 1.0 + 0.01 * a["hold_sec"] + 1.5 * abs(a["speed_delta_pct"]) + 0.002 * a["dispatch_offset_sec"]
        total_delay += 0.6 * a["hold_sec"] + 0.3 * a["dispatch_offset_sec"]

    return {
        "bunching_count": round(float(bunch), 4),
        "avg_wait_time": round(float(mean(waits)), 4),
        "occupancy_std": round(float(mean(occ_stds)), 4),
        "headway_std": round(float(mean(hw_stds)), 4),
        "fuel_proxy": round(float(fuel_proxy), 4),
        "total_delay": round(float(total_delay), 4),
    }


@st.cache_data
def compare_reports(seed: int, peak_profile: str) -> Dict[str, Any]:
    static_r = run_policy_report("static", seed, peak_profile)
    headway_r = run_policy_report("headway", seed, peak_profile)
    ppo_r = run_policy_report("ppo", seed, peak_profile)

    def pct_improve(metric: str) -> float:
        base = static_r["metrics"][metric]
        new = ppo_r["metrics"][metric]
        if base == 0:
            return 0.0
        return round(100.0 * (base - new) / base, 2)

    return {
        "static": static_r,
        "headway": headway_r,
        "ppo": ppo_r,
        "improvement_vs_static": {
            "bunching_count_pct": pct_improve("bunching_count"),
            "avg_wait_time_pct": pct_improve("avg_wait_time"),
            "occupancy_std_pct": pct_improve("occupancy_std"),
        },
    }


def step_view(trace: List[Dict[str, Any]], step: int) -> Dict[str, Any]:
    idx = max(0, min(step, len(trace) - 1))
    return trace[idx]


def trace_series(trace: List[Dict[str, Any]], key: str) -> List[float]:
    out: List[float] = []
    for frame in trace:
        if key == "risk":
            out.append(float(frame["prediction_bundle"]["bunching_risk_score"]))
        elif key == "congestion":
            out.append(float(frame["prediction_bundle"]["congestion_score"]))
        elif key == "reward":
            out.append(float(frame["reward"]))
        elif key == "min_headway":
            buses = frame["system_state"]["buses"]
            out.append(min(float(b["headway_forward_sec"]) for b in buses))
        elif key == "avg_occupancy":
            buses = frame["system_state"]["buses"]
            out.append(mean(float(b["occupancy"]) for b in buses))
    return out


def explain_action(frame: Dict[str, Any]) -> str:
    action = frame["action"]
    ctrl = frame["control_state"]
    pred = frame["prediction_bundle"]

    reasons: List[str] = []
    if pred["bunching_risk_score"] > 0.6:
        reasons.append("high predicted bunching risk")
    if pred["congestion_score"] > 0.6:
        reasons.append("congestion forecast elevated")
    if ctrl["occupancy_ratio"] > 0.8:
        reasons.append("bus occupancy is high")
    if ctrl["is_terminal"]:
        reasons.append("bus is at terminal and can be dispatch-adjusted")

    if not reasons:
        reasons.append("service appears stable, so control remains light")

    return (
        f"Action: hold={action['hold_sec']:.1f}s, "
        f"speed_delta={action['speed_delta_pct']*100:.1f}%, "
        f"dispatch_offset={action['dispatch_offset_sec']:.1f}s. "
        f"Reasoning: {', '.join(reasons)}."
    )
