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
STATIC_ROUTE_COLOR = [185, 28, 28, 220]
AI_ROUTE_COLOR = [15, 118, 110, 230]


def _bus_icon_data(fill_hex: str) -> Dict[str, Any]:
    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" width="96" height="96" viewBox="0 0 96 96">
      <rect x="16" y="18" width="64" height="44" rx="12" fill="{fill_hex}" stroke="#10221a" stroke-width="4"/>
      <rect x="24" y="26" width="48" height="18" rx="4" fill="#f7f7f4" opacity="0.95"/>
      <rect x="24" y="48" width="14" height="10" rx="2" fill="#f7f7f4" opacity="0.9"/>
      <rect x="58" y="48" width="14" height="10" rx="2" fill="#f7f7f4" opacity="0.9"/>
      <circle cx="30" cy="70" r="8" fill="#10221a"/>
      <circle cx="66" cy="70" r="8" fill="#10221a"/>
    </svg>
    """.strip()
    encoded = svg.replace("#", "%23").replace("\n", "").replace('"', "'").replace("  ", "")
    return {
        "url": f"data:image/svg+xml;charset=utf-8,{encoded}",
        "width": 96,
        "height": 96,
        "anchorY": 72,
    }


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
                b["headway_forward_sec"] *= 0.92
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

        _refresh_bus_geometry(frame)
        bunch = sum(1 for b in frame["system_state"]["buses"] if b["headway_forward_sec"] < 180)
        frame["bunching"] = bunch
        frame["visual_event"] = "bunching_alert" if bunch else "monitoring"

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


def _route_stops(frame: Dict[str, Any]) -> List[Dict[str, Any]]:
    return sorted(frame["system_state"]["stops"], key=lambda stop: stop.get("order", 0))


def _route_segments(frame: Dict[str, Any]) -> List[Dict[str, Any]]:
    return sorted(frame["system_state"]["segments"], key=lambda seg: int(str(seg["segment_id"]).split("_")[-1]))


def _route_cycle_time(frame: Dict[str, Any]) -> float:
    segments = _route_segments(frame)
    return max(1.0, sum(float(seg["base_travel_time_sec"]) * float(seg["traffic_multiplier"]) for seg in segments))


def _position_to_geo(frame: Dict[str, Any], cycle_position_sec: float) -> Dict[str, Any]:
    stops = _route_stops(frame)
    segments = _route_segments(frame)
    route_total = _route_cycle_time(frame)
    route_progress = cycle_position_sec % route_total
    elapsed = 0.0

    for idx, seg in enumerate(segments):
        seg_time = float(seg["base_travel_time_sec"]) * float(seg["traffic_multiplier"])
        next_elapsed = elapsed + seg_time
        if route_progress <= next_elapsed or idx == len(segments) - 1:
            progress = 0.0 if seg_time <= 0 else (route_progress - elapsed) / seg_time
            progress = max(0.0, min(1.0, progress))
            current_stop = stops[idx]
            next_stop = stops[(idx + 1) % len(stops)]
            lat = float(current_stop["lat"]) + (float(next_stop["lat"]) - float(current_stop["lat"])) * progress
            lon = float(current_stop["lon"]) + (float(next_stop["lon"]) - float(current_stop["lon"])) * progress
            return {
                "stop_index": idx,
                "position_progress": progress,
                "lat": lat,
                "lon": lon,
                "current_stop_id": current_stop["stop_id"],
                "current_stop_name": current_stop["stop_name"],
                "next_stop_id": next_stop["stop_id"],
                "next_stop_name": next_stop["stop_name"],
            }
        elapsed = next_elapsed

    last_stop = stops[-1]
    return {
        "stop_index": len(stops) - 1,
        "position_progress": 1.0,
        "lat": float(last_stop["lat"]),
        "lon": float(last_stop["lon"]),
        "current_stop_id": last_stop["stop_id"],
        "current_stop_name": last_stop["stop_name"],
        "next_stop_id": stops[0]["stop_id"],
        "next_stop_name": stops[0]["stop_name"],
    }


def _refresh_bus_geometry(frame: Dict[str, Any]) -> None:
    buses = frame["system_state"]["buses"]
    if not buses:
        return

    cycle_time = _route_cycle_time(frame)
    anchor = float(buses[0].get("cycle_position_sec", 0.0)) % cycle_time
    positions = [anchor]
    for idx in range(1, len(buses)):
        prev = buses[idx - 1]
        positions.append((positions[idx - 1] - float(prev["headway_forward_sec"])) % cycle_time)

    for bus, position in zip(buses, positions):
        geo = _position_to_geo(frame, position)
        bus["cycle_position_sec"] = position
        bus["stop_index"] = geo["stop_index"]
        bus["position_progress"] = geo["position_progress"]
        bus["lat"] = geo["lat"]
        bus["lon"] = geo["lon"]
        bus["current_stop_id"] = geo["current_stop_id"]
        bus["current_stop_name"] = geo["current_stop_name"]
        bus["next_stop_id"] = geo["next_stop_id"]
        bus["next_stop_name"] = geo["next_stop_name"]

    for idx, bus in enumerate(buses):
        bus["headway_backward_sec"] = float(buses[idx - 1]["headway_forward_sec"])


def route_path(frame: Dict[str, Any]) -> List[List[float]]:
    stops = _route_stops(frame)
    points = [[float(stop["lon"]), float(stop["lat"])] for stop in stops]
    if points:
        points.append(points[0])
    return points


def primary_bunching_bus(frame: Dict[str, Any]) -> Dict[str, Any]:
    buses = frame["system_state"]["buses"]
    return min(buses, key=lambda bus: float(bus["headway_forward_sec"]))


def map_payload(frame: Dict[str, Any], policy_name: str) -> Dict[str, Any]:
    focus_bus_id = frame.get("control_state", {}).get("focus_bus_id")
    route_color = AI_ROUTE_COLOR if policy_name == "ppo" else STATIC_ROUTE_COLOR if policy_name == "static" else [217, 119, 6, 220]
    stops = _route_stops(frame)

    stop_points = []
    for stop in stops:
        queue_len = int(stop["queue_len"])
        stop_points.append(
            {
                "coordinates": [float(stop["lon"]), float(stop["lat"])],
                "stop_name": stop["stop_name"].replace("_", " "),
                "queue_len": queue_len,
                "fill_color": [240, 244, 242, 220] if not stop.get("is_terminal") else [251, 191, 36, 220],
                "label_color": [16, 34, 26, 240],
                "tooltip": f"{stop['stop_name'].replace('_', ' ')}<br/>Waiting passengers: {queue_len}",
            }
        )

    bus_points = []
    for bus in frame["system_state"]["buses"]:
        headway = float(bus["headway_forward_sec"])
        occupancy = float(bus["occupancy"])
        fill_color = [220, 38, 38, 230] if headway < 180 else [217, 119, 6, 230] if occupancy > 0.82 else [15, 118, 110, 230]
        if policy_name == "static" and headway < 180:
            fill_color = [190, 24, 93, 235]
        fill_hex = "#dc2626" if headway < 180 else "#d97706" if occupancy > 0.82 else "#0f766e"
        if policy_name == "static" and headway < 180:
            fill_hex = "#be185d"
        bus_points.append(
            {
                "coordinates": [float(bus["lon"]), float(bus["lat"])],
                "label": bus["bus_id"].upper(),
                "subtitle": f"Now near {bus['current_stop_name'].replace('_', ' ')} -> {bus['next_stop_name'].replace('_', ' ')}",
                "tooltip": (
                    f"{bus['bus_id'].upper()}<br/>"
                    f"Current stop: {bus['current_stop_name'].replace('_', ' ')}<br/>"
                    f"Next stop: {bus['next_stop_name'].replace('_', ' ')}<br/>"
                    f"Headway: {headway:.0f}s<br/>Occupancy: {occupancy * 100:.0f}%"
                ),
                "icon_data": _bus_icon_data(fill_hex),
                "icon_size": 1.0,
                "label_color": [16, 34, 26, 255],
                "is_focus": bus["bus_id"] == focus_bus_id,
            }
        )

    all_lats = [float(stop["lat"]) for stop in stops]
    all_lons = [float(stop["lon"]) for stop in stops]

    return {
        "route": [{"path": route_path(frame), "color": route_color}],
        "stops": stop_points,
        "buses": bus_points,
        "view": {
            "latitude": sum(all_lats) / max(1, len(all_lats)),
            "longitude": sum(all_lons) / max(1, len(all_lons)),
            "zoom": 12.1,
            "pitch": 35,
        },
    }


def frame_story(frame: Dict[str, Any], policy_name: str) -> Dict[str, str]:
    bunching_bus = primary_bunching_bus(frame)
    stop_name = bunching_bus["next_stop_name"].replace("_", " ")
    focus_stop = frame.get("control_state", {}).get("focus_stop_name", stop_name).replace("_", " ")
    bunching = int(frame["bunching"])
    action = frame["action"]

    if policy_name == "static":
        headline = f"Buses are compressing near {stop_name}" if bunching else f"Static service is drifting near {stop_name}"
        detail = "The delayed bus attracts more passengers, while the following bus catches up because it sees lighter demand."
    else:
        headline = f"AI is stabilizing spacing near {focus_stop}" if bunching == 0 else f"AI is intervening before bunching worsens near {focus_stop}"
        detail = explain_action(frame)

    return {"headline": headline, "detail": detail}


def comparison_story(static_frame: Dict[str, Any], ai_frame: Dict[str, Any]) -> List[str]:
    static_bus = primary_bunching_bus(static_frame)
    ai_bus = primary_bunching_bus(ai_frame)
    notes = [
        f"Static mode compresses buses near {static_bus['next_stop_name'].replace('_', ' ')}.",
        f"AI keeps the fleet more evenly spaced approaching {ai_bus['next_stop_name'].replace('_', ' ')}.",
    ]
    if ai_frame["action"]["hold_sec"] > 0:
        notes.append(f"The controller briefly holds {ai_frame['control_state']['focus_bus_id'].upper()} to rebuild headway.")
    if ai_frame["action"]["dispatch_offset_sec"] > 0:
        notes.append("Terminal dispatch is nudged to prevent the next bunching wave.")
    return notes


def action_chips(frame: Dict[str, Any]) -> List[str]:
    action = frame["action"]
    chips = []
    if action["hold_sec"] > 0:
        chips.append(f"Hold {action['hold_sec']:.0f}s")
    if abs(action["speed_delta_pct"]) > 0.01:
        chips.append(f"Speed {action['speed_delta_pct'] * 100:.0f}%")
    if action["dispatch_offset_sec"] > 0:
        chips.append(f"Dispatch +{action['dispatch_offset_sec']:.0f}s")
    if not chips:
        chips.append("No intervention")
    return chips


def _seconds_text(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(seconds, 60)
    if minutes and secs:
        return f"{minutes} min {secs} sec"
    if minutes:
        return f"{minutes} min"
    return f"{secs} sec"


def _risk_level(score: float) -> str:
    if score >= 0.7:
        return "High"
    if score >= 0.4:
        return "Moderate"
    return "Low"


def _crowd_level(occupancy_ratio: float, demand_estimate: float) -> str:
    combined = 0.65 * occupancy_ratio + 0.35 * min(1.0, demand_estimate / 12.0)
    if combined >= 0.78:
        return "Heavy"
    if combined >= 0.48:
        return "Moderate"
    return "Light"


def focus_bus(frame: Dict[str, Any]) -> Dict[str, Any]:
    focus_bus_id = frame.get("control_state", {}).get("focus_bus_id")
    buses = frame["system_state"]["buses"]
    if focus_bus_id:
        for bus in buses:
            if bus["bus_id"] == focus_bus_id:
                return bus
    return primary_bunching_bus(frame)


def driver_assist(frame: Dict[str, Any]) -> Dict[str, str]:
    ctrl = frame["control_state"]
    pred = frame["prediction_bundle"]
    action = frame["action"]
    bus = focus_bus(frame)

    risk = _risk_level(float(pred["bunching_risk_score"]))
    crowd = _crowd_level(float(bus["occupancy"]), float(ctrl["stop_demand_estimate"]))
    current_stop = bus["current_stop_name"].replace("_", " ")
    next_stop = bus["next_stop_name"].replace("_", " ")
    headway_ahead = _seconds_text(float(bus["headway_forward_sec"]))
    headway_behind = _seconds_text(float(bus["headway_backward_sec"]))

    if action["hold_sec"] > 0:
        instruction = f"Hold for {_seconds_text(action['hold_sec'])} at {next_stop}"
        expected = f"This should reopen the gap ahead before reaching {next_stop}."
    elif action["dispatch_offset_sec"] > 0:
        instruction = f"Leave terminal after {_seconds_text(action['dispatch_offset_sec'])}"
        expected = "This should smooth the next dispatch wave and reduce a new bunch forming."
    elif action["speed_delta_pct"] < -0.01:
        instruction = f"Reduce speed slightly for the next segment ({abs(action['speed_delta_pct']) * 100:.0f}% slower)"
        expected = f"This should stop the bus from catching the vehicle ahead before {next_stop}."
    elif action["speed_delta_pct"] > 0.01:
        instruction = f"Move slightly faster for the next segment ({action['speed_delta_pct'] * 100:.0f}% faster)"
        expected = f"This should recover spacing before reaching {next_stop}."
    else:
        instruction = "Maintain current speed and normal dwell"
        expected = "Spacing is acceptable; no intervention is needed right now."

    reasons = []
    if float(pred["bunching_risk_score"]) >= 0.55:
        reasons.append("bus spacing is getting tight")
    if float(pred["congestion_score"]) >= 0.6:
        reasons.append("traffic ahead is slowing the corridor")
    if crowd == "Heavy":
        reasons.append(f"{next_stop} is expected to be crowded")
    if float(bus["headway_backward_sec"]) < 180:
        reasons.append("the bus behind is too close")
    if not reasons:
        reasons.append("service spacing is currently stable")

    spoken = (
        f"{bus['bus_id'].upper()}: {instruction}. "
        f"Bus ahead gap is {headway_ahead}. Bus behind gap is {headway_behind}. "
        f"Risk is {risk.lower()}."
    )

    return {
        "bus_id": bus["bus_id"].upper(),
        "current_stop": current_stop,
        "next_stop": next_stop,
        "headway_ahead": headway_ahead,
        "headway_behind": headway_behind,
        "risk_level": risk,
        "crowd_level": crowd,
        "occupancy_pct": f"{float(bus['occupancy']) * 100:.0f}%",
        "instruction": instruction,
        "reason": "; ".join(reasons).capitalize() + ".",
        "expected_benefit": expected,
        "spoken_message": spoken,
    }
