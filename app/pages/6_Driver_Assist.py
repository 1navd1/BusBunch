from __future__ import annotations

import streamlit as st

from lib.data import apply_modifiers, driver_assist, map_payload, run_policy_report, step_view
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

    button_cols = st.columns(2)
    with button_cols[0]:
        if st.button("Acknowledge", key="driver_ack", use_container_width=True):
            st.success(f"{assist['bus_id']} acknowledged the instruction.")
    with button_cols[1]:
        if st.button("Play Audio Prompt", key="driver_audio", use_container_width=True):
            st.info(assist["spoken_message"])

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
driver_panel("What the driver hears", assist["spoken_message"])
