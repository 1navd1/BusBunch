from __future__ import annotations

import time

import streamlit as st

from lib.data import (
    action_chips,
    apply_modifiers,
    compare_reports,
    explain_action,
    map_payload,
    step_view,
)
from lib.ui import apply_theme, chip_row, divider, legend, render_corridor_map, story_card


def _event_label(frame: dict, mode: str) -> str:
    if frame["bunching"] > 0:
        return f"{mode}: Bunching detected"
    if frame["action"]["hold_sec"] > 0 or abs(frame["action"]["speed_delta_pct"]) > 0.01 or frame["action"]["dispatch_offset_sec"] > 0:
        return f"{mode}: Active intervention"
    return f"{mode}: Stable flow"


def _what_changed(static_frame: dict, ai_frame: dict) -> str:
    static_bunch = int(static_frame["bunching"])
    ai_bunch = int(ai_frame["bunching"])
    if static_bunch > ai_bunch:
        return "AI keeps more space between buses at this replay step."
    if ai_frame["action"]["hold_sec"] > 0:
        return f"AI is using a short hold to prevent the next bunch near {ai_frame['control_state']['focus_stop_name'].replace('_', ' ')}."
    if ai_frame["action"]["dispatch_offset_sec"] > 0:
        return "AI is re-sequencing dispatch timing to stop the next wave from compressing."
    return "Both views are stable here, but the AI view is still monitoring corridor risk."


def _render_step(static_frame: dict, ai_frame: dict, step: int) -> None:
    maps = st.columns(2)
    with maps[0]:
        st.subheader("Static Baseline")
        render_corridor_map(map_payload(static_frame, "static"), key=f"sim-static-{step}")
        chip_row(
            [
                _event_label(static_frame, "Static"),
                f"Bunching {static_frame['bunching']}",
                f"Risk {static_frame['prediction_bundle']['bunching_risk_score']:.2f}",
            ]
        )
        story_card("Current State", "No intervention is applied in the baseline view.")

    with maps[1]:
        st.subheader("AI Controlled")
        render_corridor_map(map_payload(ai_frame, "ppo"), key=f"sim-ai-{step}")
        chip_row(
            [
                _event_label(ai_frame, "AI"),
                f"Bunching {ai_frame['bunching']}",
                f"Risk {ai_frame['prediction_bundle']['bunching_risk_score']:.2f}",
            ]
            + action_chips(ai_frame)
        )
        story_card("Current Action", explain_action(ai_frame))

    legend()
    divider()
    story_card("What Changed", _what_changed(static_frame, ai_frame))


apply_theme()
st.title("Simulation")
st.caption("Split-screen replay of the full corridor process: static schedule vs AI control")

seed = st.sidebar.number_input("Scenario seed", min_value=1, max_value=999, value=11, step=1)
profile = st.sidebar.selectbox("Demand profile", ["peak", "off_peak"], index=0)
traffic_spike = st.sidebar.toggle("Traffic spike", value=False)
passenger_surge = st.sidebar.toggle("Passenger surge", value=False)

cmp = compare_reports(int(seed), profile)
if cmp["ppo"].get("warning"):
    st.warning(cmp["ppo"]["warning"])

static_trace = apply_modifiers(cmp["static"]["trace"], traffic_spike, passenger_surge)
ai_trace = apply_modifiers(cmp["ppo"]["trace"], traffic_spike, passenger_surge)

max_step = min(len(static_trace), len(ai_trace)) - 1
step = st.slider("Replay step", min_value=0, max_value=max_step, value=min(52, max_step))

play_pressed = st.button("Autoplay", use_container_width=False)

frame_placeholder = st.empty()

def render_at(index: int) -> None:
    with frame_placeholder.container():
        _render_step(step_view(static_trace, index), step_view(ai_trace, index), index)


render_at(step)

if play_pressed:
    for idx in range(step, max_step + 1):
        render_at(idx)
        time.sleep(0.18)
