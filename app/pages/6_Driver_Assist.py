from __future__ import annotations

import time

import streamlit as st

from lib.data import (
    apply_modifiers,
    driver_assist,
    map_payload,
    reevaluate_after_driver_response,
    run_policy_report,
    step_view,
)
from lib.ui import (
    action_box,
    apply_theme,
    chip_row,
    divider,
    driver_panel,
    info_box,
    legend,
    render_corridor_map,
    status_card,
    tablet_shell,
)


def _risk_tone(level: str) -> str:
    return {"Low": "green", "Moderate": "yellow", "High": "red"}[level]


def _render_driver_view(frame: dict, selected_bus_id: str, replan: dict | None, step: int) -> None:
    assist = driver_assist(frame, selected_bus_id)
    risk_tone = _risk_tone(replan["revised_risk"] if replan else assist["risk_level"])
    crowd_tone = {"Light": "green", "Moderate": "yellow", "Heavy": "red"}[assist["crowd_level"]]

    map_col, tablet_col = st.columns([1.25, 1.05])
    with map_col:
        render_corridor_map(map_payload(frame, "ppo", selected_bus_id=selected_bus_id), key=f"driver-map-{selected_bus_id}-{step}")
        legend()
        chip_row(
            [
                f"Selected {assist['bus_id']}",
                f"{assist['current_stop']} -> {assist['next_stop']}",
                f"Gap ahead {assist['headway_ahead']}",
                f"Gap behind {assist['headway_behind']}",
            ]
        )

    with tablet_col:
        tablet_shell(
            f"{assist['bus_id']} Driver Tablet",
            f"Current stop {assist['current_stop']} -> Next stop {assist['next_stop']}",
        )
        st.markdown("")

        status_cols = st.columns(2)
        with status_cols[0]:
            status_card(
                "Risk Level",
                replan["revised_risk"] if replan else assist["risk_level"],
                f"Gap ahead {assist['headway_ahead']}",
                risk_tone,
            )
        with status_cols[1]:
            status_card("Next Stop Crowd", assist["crowd_level"], f"Bus load {assist['occupancy_pct']}", crowd_tone)

        st.markdown("")
        action_box("Recommended Action", replan["revised_instruction"] if replan else assist["instruction"])
        st.markdown("")
        info_box("Why This Is Needed", replan["revised_reason"] if replan else assist["reason"])
        st.markdown("")
        info_box("Expected Result", replan["expected_outcome"] if replan else assist["expected_benefit"])

    divider()

    st.subheader("Driver Response")
    if replan:
        st.success(f"Latest driver response: {replan['response_label']}")
        repl_cols = st.columns(2)
        with repl_cols[0]:
            status_card("Re-Evaluated Risk", replan["revised_risk"], replan["timing_adjustment"], replan["status_tone"])
        with repl_cols[1]:
            status_card(
                "Corridor Plan",
                "Corridor-Wide" if replan["corridor_wide"] else "Local Update",
                replan["route_status"],
                replan["status_tone"],
            )
        st.markdown("")
        driver_panel("Route and Timing Revision", replan["route_status"] + " " + replan["timing_adjustment"])
        st.markdown("")
        driver_panel("Control Room Follow-Up", replan["control_room_action"])
        chip_row(replan["action_chips"])
    else:
        st.info("No driver response recorded yet. Choose a response on the right-side controls.")

    divider()
    st.subheader("Other Buses")
    updates = replan["other_bus_updates"] if replan else []
    if updates:
        cols = st.columns(len(updates))
        for col, update in zip(cols, updates):
            with col:
                story = f"{update['current_stop']} -> {update['next_stop']}. {update['instruction']}. {update['reason']}"
                driver_panel(update["bus_id"], story)
    else:
        st.info("Other buses keep their current instructions until the selected driver sends a disruptive response.")

    divider()
    st.subheader("Voice Prompt")
    driver_panel("What the driver hears", replan["voice_prompt"] if replan else assist["spoken_message"])


apply_theme()
st.title("Driver Assist")
st.caption("Choose a bus, watch it move stop-to-stop, and re-plan the corridor when the driver reports a problem")

seed = st.sidebar.number_input("Scenario seed", min_value=1, max_value=999, value=11, step=1)
profile = st.sidebar.selectbox("Demand profile", ["peak", "off_peak"], index=0)
traffic_spike = st.sidebar.toggle("Traffic spike", value=False)
passenger_surge = st.sidebar.toggle("Passenger surge", value=False)

report = run_policy_report("ppo", int(seed), profile)
trace = apply_modifiers(report["trace"], traffic_spike, passenger_surge)
step = st.slider("Replay step", min_value=0, max_value=len(trace) - 1, value=min(80, len(trace) - 1))
frame = step_view(trace, step)

bus_ids = [bus["bus_id"] for bus in frame["system_state"]["buses"]]
default_bus = frame["control_state"]["focus_bus_id"] if frame["control_state"]["focus_bus_id"] in bus_ids else bus_ids[0]
selected_bus_id = st.selectbox("Choose bus", bus_ids, index=bus_ids.index(default_bus), format_func=lambda bus_id: bus_id.upper())

response_state_key = f"driver_response::{seed}::{profile}::{selected_bus_id}::{step}"
if response_state_key not in st.session_state:
    st.session_state[response_state_key] = None

selected_response = st.session_state[response_state_key]
replan = reevaluate_after_driver_response(frame, selected_response, selected_bus_id) if selected_response else None

control_cols = st.columns([1, 1, 1, 1, 1, 1, 1])
if control_cols[0].button("Autoplay", key=f"driver_autoplay_{selected_bus_id}_{step}", use_container_width=True):
    autoplay_placeholder = st.empty()
    for idx in range(step, len(trace)):
        auto_frame = step_view(trace, idx)
        auto_response_key = f"driver_response::{seed}::{profile}::{selected_bus_id}::{idx}"
        auto_response = st.session_state.get(auto_response_key)
        auto_replan = (
            reevaluate_after_driver_response(auto_frame, auto_response, selected_bus_id) if auto_response else None
        )
        with autoplay_placeholder.container():
            _render_driver_view(auto_frame, selected_bus_id, auto_replan, idx)
        time.sleep(0.18)

if control_cols[1].button("Acknowledged", key=f"driver_ack_{selected_bus_id}_{step}", use_container_width=True):
    st.session_state[response_state_key] = "acknowledged"
    st.rerun()
if control_cols[2].button("Cannot Comply", key=f"driver_cannot_{selected_bus_id}_{step}", use_container_width=True):
    st.session_state[response_state_key] = "cannot_comply"
    st.rerun()
if control_cols[3].button("Traffic Blocked", key=f"driver_traffic_{selected_bus_id}_{step}", use_container_width=True):
    st.session_state[response_state_key] = "traffic_blocked"
    st.rerun()
if control_cols[4].button("Stop Overcrowded", key=f"driver_crowd_{selected_bus_id}_{step}", use_container_width=True):
    st.session_state[response_state_key] = "stop_overcrowded"
    st.rerun()
if control_cols[5].button("Play Audio", key=f"driver_audio_{selected_bus_id}_{step}", use_container_width=True):
    current_assist = driver_assist(frame, selected_bus_id)
    st.info(replan["voice_prompt"] if replan else current_assist["spoken_message"])
if control_cols[6].button("Clear", key=f"driver_clear_{selected_bus_id}_{step}", use_container_width=True):
    st.session_state[response_state_key] = None
    st.rerun()

_render_driver_view(frame, selected_bus_id, replan, step)
