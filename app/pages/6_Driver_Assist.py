from __future__ import annotations

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
    apply_theme,
    chip_row,
    divider,
    driver_panel,
    legend,
    render_corridor_map,
    status_card,
    tablet_shell,
)

apply_theme()
st.title("Driver Assist")
st.caption("An in-bus tablet view: one clear instruction, one reason, and one tap to acknowledge")

seed = st.sidebar.number_input("Scenario seed", min_value=1, max_value=999, value=11, step=1)
profile = st.sidebar.selectbox("Demand profile", ["peak", "off_peak"], index=0)
traffic_spike = st.sidebar.toggle("Traffic spike", value=False)
passenger_surge = st.sidebar.toggle("Passenger surge", value=False)

report = run_policy_report("ppo", int(seed), profile)
trace = apply_modifiers(report["trace"], traffic_spike, passenger_surge)
step = st.slider("Replay step", min_value=0, max_value=len(trace) - 1, value=min(80, len(trace) - 1))
frame = step_view(trace, step)
assist = driver_assist(frame)
response_state_key = f"driver_response::{seed}::{profile}::{step}"
if response_state_key not in st.session_state:
    st.session_state[response_state_key] = None

selected_response = st.session_state[response_state_key]
replan = reevaluate_after_driver_response(frame, selected_response) if selected_response else None

risk_tone = {"Low": "green", "Moderate": "yellow", "High": "red"}[assist["risk_level"]]
crowd_tone = {"Light": "green", "Moderate": "yellow", "Heavy": "red"}[assist["crowd_level"]]

map_col, tablet_col = st.columns([1.25, 1.05])
with map_col:
    render_corridor_map(map_payload(frame, "ppo"), key=f"driver-map-{step}")
    legend()

with tablet_col:
    tablet_shell(
        f"{assist['bus_id']} Driver Tablet",
        f"Current stop {assist['current_stop']} -> Next stop {assist['next_stop']}",
    )
    st.markdown("")

    status_cols = st.columns(2)
    with status_cols[0]:
        status_card("Risk Level", assist["risk_level"], f"Gap ahead {assist['headway_ahead']}", risk_tone)
    with status_cols[1]:
        status_card("Next Stop Crowd", assist["crowd_level"], f"Bus load {assist['occupancy_pct']}", crowd_tone)

    st.markdown("")
    driver_panel("Recommended Action", assist["instruction"])
    st.markdown("")
    driver_panel("Why This Is Needed", assist["reason"])
    st.markdown("")
    driver_panel("Expected Result", assist["expected_benefit"])

    st.markdown("")
    response_cols = st.columns(2)
    with response_cols[0]:
        if st.button("Acknowledged", key=f"driver_ack_{step}", use_container_width=True):
            st.session_state[response_state_key] = "acknowledged"
            st.rerun()
    with response_cols[1]:
        if st.button("Cannot Comply", key=f"driver_cannot_{step}", use_container_width=True):
            st.session_state[response_state_key] = "cannot_comply"
            st.rerun()

    response_cols_2 = st.columns(2)
    with response_cols_2[0]:
        if st.button("Traffic Blocked", key=f"driver_traffic_{step}", use_container_width=True):
            st.session_state[response_state_key] = "traffic_blocked"
            st.rerun()
    with response_cols_2[1]:
        if st.button("Stop Overcrowded", key=f"driver_crowd_{step}", use_container_width=True):
            st.session_state[response_state_key] = "stop_overcrowded"
            st.rerun()

    utility_cols = st.columns(2)
    with utility_cols[0]:
        if st.button("Play Audio Prompt", key=f"driver_audio_{step}", use_container_width=True):
            st.info(replan["voice_prompt"] if replan else assist["spoken_message"])
    with utility_cols[1]:
        if st.button("Clear Response", key=f"driver_clear_{step}", use_container_width=True):
            st.session_state[response_state_key] = None
            st.rerun()

divider()

st.subheader("Driver Response")
if selected_response:
    st.success(f"Latest driver response: {replan['response_label']}")
else:
    st.info("No driver response recorded yet. Tap one of the response buttons on the tablet.")

if replan:
    repl_cols = st.columns(2)
    with repl_cols[0]:
        status_card("Re-Evaluated Risk", replan["revised_risk"], replan["timing_adjustment"], replan["status_tone"])
    with repl_cols[1]:
        status_card("Corridor Plan", replan["response_label"], replan["expected_outcome"], replan["status_tone"])

    st.markdown("")
    driver_panel("Updated Driver Instruction", replan["revised_instruction"])
    st.markdown("")
    driver_panel("AI Re-Evaluation", replan["revised_reason"])
    st.markdown("")
    driver_panel("Route and Timing Revision", replan["route_status"] + " " + replan["timing_adjustment"])
    st.markdown("")
    driver_panel("Control Room Follow-Up", replan["control_room_action"])
    chip_row(replan["action_chips"])
    divider()

st.subheader("Quick Driver Readout")
readout = st.columns(4)
readout[0].markdown(f"**Current Stop**  \n`{assist['current_stop']}`")
readout[1].markdown(f"**Next Stop**  \n`{assist['next_stop']}`")
readout[2].markdown(f"**Gap Ahead**  \n`{assist['headway_ahead']}`")
readout[3].markdown(f"**Gap Behind**  \n`{assist['headway_behind']}`")

chip_row(
    [
        f"Bus {assist['bus_id']}",
        f"Risk {assist['risk_level']}",
        f"Load {assist['occupancy_pct']}",
        f"Next stop crowd {assist['crowd_level']}",
    ]
)

divider()
st.subheader("Voice Prompt")
driver_panel("What the driver hears", replan["voice_prompt"] if replan else assist["spoken_message"])
