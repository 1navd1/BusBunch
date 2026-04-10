from __future__ import annotations

import copy
import sys
from dataclasses import asdict
from pathlib import Path
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
CHECKPOINT_PATH = ARTIFACTS_DIR / "models" / "ppo_best.zip"
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


def map_payload(frame: Dict[str, Any], policy_name: str, selected_bus_id: str | None = None) -> Dict[str, Any]:
    focus_bus_id = selected_bus_id or frame.get("control_state", {}).get("focus_bus_id")
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
                "label": f"{bus['bus_id'].upper()}*" if bus["bus_id"] == focus_bus_id else bus["bus_id"].upper(),
                "subtitle": f"Now near {bus['current_stop_name'].replace('_', ' ')} -> {bus['next_stop_name'].replace('_', ' ')}",
                "tooltip": (
                    f"{bus['bus_id'].upper()}<br/>"
                    f"Current stop: {bus['current_stop_name'].replace('_', ' ')}<br/>"
                    f"Next stop: {bus['next_stop_name'].replace('_', ' ')}<br/>"
                    f"Headway: {headway:.0f}s<br/>Occupancy: {occupancy * 100:.0f}%"
                ),
                "icon_data": _bus_icon_data(fill_hex),
                "icon_size": 1.18 if bus["bus_id"] == focus_bus_id else 1.0,
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


def focus_bus(frame: Dict[str, Any], selected_bus_id: str | None = None) -> Dict[str, Any]:
    focus_bus_id = selected_bus_id or frame.get("control_state", {}).get("focus_bus_id")
    buses = frame["system_state"]["buses"]
    if focus_bus_id:
        for bus in buses:
            if bus["bus_id"] == focus_bus_id:
                return bus
    return primary_bunching_bus(frame)


def driver_assist(frame: Dict[str, Any], selected_bus_id: str | None = None) -> Dict[str, str]:
    ctrl = frame["control_state"]
    pred = frame["prediction_bundle"]
    action = frame["action"]
    bus = focus_bus(frame, selected_bus_id)

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
        "focus_bus_id": bus["bus_id"],
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


def _other_bus_updates(frame: Dict[str, Any], selected_bus_id: str, response: str) -> List[Dict[str, str]]:
    updates: List[Dict[str, str]] = []
    for bus in frame["system_state"]["buses"]:
        if bus["bus_id"] == selected_bus_id:
            continue

        instruction = "No change"
        reason = "Continue current plan."
        next_stop = bus["next_stop_name"].replace("_", " ")
        headway_forward = float(bus["headway_forward_sec"])
        headway_backward = float(bus["headway_backward_sec"])

        if response == "traffic_blocked":
            if float(bus["headway_forward_sec"]) < 220:
                speed_cut = max(8, min(18, int(round((220 - headway_forward) / 12.0))))
                instruction = f"Reduce speed by {speed_cut}% before {next_stop}"
                reason = "The bus ahead reported traffic blockage, so spacing must be widened before the blocked segment."
            else:
                dispatch_delay = max(30, min(90, int(round((headway_forward - 220) / 3.0))))
                instruction = f"Delay terminal release by {_seconds_text(dispatch_delay)}"
                reason = "Control room is holding this bus back so fewer vehicles enter the blocked segment together."
        elif response == "stop_overcrowded":
            if float(bus["headway_backward_sec"]) < 220:
                speed_up = max(6, min(15, int(round((220 - headway_backward) / 14.0))))
                instruction = f"Increase speed by {speed_up}% to support boarding at {next_stop}"
                reason = "The selected bus is clearing an overcrowded stop and this bus should help absorb spillover demand."
            else:
                dispatch_advance = max(20, min(60, int(round((headway_forward - 240) / 5.0))))
                instruction = f"Advance departure by {_seconds_text(dispatch_advance)} if ready"
                reason = "Passenger spillover may reach the next stop, so extra capacity should arrive earlier."
        elif response == "cannot_comply":
            correction = max(20, min(50, int(round(abs(headway_forward - headway_backward) / 6.0))))
            instruction = f"Adjust spacing target by {_seconds_text(correction)}"
            reason = "Selected driver could not apply the planned intervention, so corridor timing is being corrected around this bus."
        elif response == "acknowledged":
            monitor_window = max(30, min(90, int(round(headway_forward / 5.0))))
            instruction = f"Maintain current plan and review again in {_seconds_text(monitor_window)}"
            reason = "Selected bus is following the active plan and this bus should keep its current corridor spacing."

        updates.append(
            {
                "bus_id": bus["bus_id"].upper(),
                "current_stop": bus["current_stop_name"].replace("_", " "),
                "next_stop": bus["next_stop_name"].replace("_", " "),
                "instruction": instruction,
                "reason": reason,
            }
        )
    return updates


def reevaluate_after_driver_response(frame: Dict[str, Any], response: str, selected_bus_id: str | None = None) -> Dict[str, str | List[str] | bool]:
    assist = driver_assist(frame, selected_bus_id)
    pred = frame["prediction_bundle"]
    response = response.lower().strip()

    current_stop = assist["current_stop"]
    next_stop = assist["next_stop"]
    focus_bus = assist["bus_id"]
    risk_score = float(pred["bunching_risk_score"])
    congestion = float(pred["congestion_score"])

    route_status = "Stay on the same corridor. No route diversion is required."
    revised_risk = assist["risk_level"]
    response_label = response.replace("_", " ").title()
    corridor_wide = response in {"traffic_blocked", "stop_overcrowded"}

    if response == "acknowledged":
        revised_instruction = assist["instruction"]
        revised_reason = "Driver confirmed the instruction, so the controller keeps the original plan active."
        timing_adjustment = (
            f"Monitor spacing through {next_stop}. If the gap recovers, no extra dispatch correction is needed."
        )
        control_room_action = (
            f"Track {focus_bus} through the next two stops and release the terminal schedule only if spacing stays stable."
        )
        expected_outcome = "Headway should recover gradually without shifting work to other buses."
        status_tone = "green" if revised_risk == "Low" else "yellow"
        action_chips = ["Driver acknowledged", "Original plan stays active", f"Monitor through {next_stop}"]
    elif response == "cannot_comply":
        revised_instruction = f"Maintain service through {next_stop}; no manual hold is required from {focus_bus}."
        revised_reason = (
            "The driver cannot perform the requested action, so the AI pushes correction downstream to dispatch timing and the following bus."
        )
        timing_adjustment = (
            f"Delay the next terminal dispatch by 45 sec and ask the trailing bus to ease speed slightly for one segment."
        )
        control_room_action = (
            f"Control room should absorb the correction centrally and rebalance the corridor around {current_stop} and {next_stop}."
        )
        expected_outcome = "Bunching risk remains controlled, but recovery will be slower than the original plan."
        revised_risk = "Moderate" if risk_score < 0.75 else "High"
        status_tone = "yellow" if revised_risk == "Moderate" else "red"
        action_chips = ["Driver cannot comply", "Dispatch +45 sec", "Trailing bus slows slightly"]
    elif response == "traffic_blocked":
        revised_instruction = f"Hold position safely at {current_stop} until traffic clears; do not force the segment to {next_stop}."
        revised_reason = (
            "The segment ahead is blocked, so the AI switches from spacing recovery to congestion containment."
        )
        timing_adjustment = (
            f"Mark the {current_stop} -> {next_stop} segment as delayed, delay the following bus by 60 sec, and warn terminal dispatch."
        )
        control_room_action = (
            f"Push a corridor-wide timing revision around the blocked segment and prevent more buses from entering the queue."
        )
        expected_outcome = "The corridor avoids stacking more buses into the blocked segment while preserving safer recovery afterward."
        revised_risk = "High" if congestion >= 0.45 else "Moderate"
        status_tone = "red" if revised_risk == "High" else "yellow"
        route_status = f"Segment restriction active on {current_stop} -> {next_stop}. Corridor remains the same, but timing is being re-sequenced around the blockage."
        action_chips = ["Traffic blocked", "Segment delay +60 sec", "Gate following buses"]
    elif response == "stop_overcrowded":
        revised_instruction = f"Serve full boarding at {next_stop}; skip any extra holding and clear the stop safely."
        revised_reason = (
            "Passenger buildup is the main constraint, so the AI removes hold pressure from this bus and shifts recovery to supporting vehicles."
        )
        timing_adjustment = (
            f"Allow extended dwell at {next_stop}, dispatch the next bus 30 sec earlier if available, and tighten spacing behind this bus."
        )
        control_room_action = (
            f"Treat {next_stop} as a crowd hot spot and send supporting capacity behind {focus_bus}."
        )
        expected_outcome = "Passenger queues fall faster, and the following bus helps absorb crowding while headway is repaired later."
        revised_risk = "Moderate" if assist["crowd_level"] == "Heavy" else assist["risk_level"]
        status_tone = "yellow" if revised_risk != "Low" else "green"
        route_status = f"Crowd-management priority at {next_stop}. The corridor stays unchanged, but timing favors passenger clearance over holding."
        action_chips = ["Stop overcrowded", "Extended dwell allowed", "Support bus closes gap"]
    else:
        revised_instruction = assist["instruction"]
        revised_reason = "No special driver response was provided."
        timing_adjustment = "No timing changes applied."
        control_room_action = "No extra control-room action is required."
        expected_outcome = assist["expected_benefit"]
        status_tone = "green"
        action_chips = ["Normal operation"]

    voice_prompt = (
        f"{focus_bus}: updated plan. {revised_instruction}. "
        f"Reason: {revised_reason} "
        f"Control room update: {timing_adjustment}"
    )

    other_updates = _other_bus_updates(frame, assist["focus_bus_id"], response)

    return {
        "response_label": response_label,
        "revised_risk": revised_risk,
        "status_tone": status_tone,
        "revised_instruction": revised_instruction,
        "revised_reason": revised_reason,
        "timing_adjustment": timing_adjustment,
        "route_status": route_status,
        "control_room_action": control_room_action,
        "expected_outcome": expected_outcome,
        "voice_prompt": voice_prompt,
        "action_chips": action_chips,
        "corridor_wide": corridor_wide,
        "other_bus_updates": other_updates,
    }
