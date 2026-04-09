from __future__ import annotations

import time

import streamlit as st

from lib.data import apply_modifiers, run_policy_report, step_view
from lib.ui import apply_theme, divider

apply_theme()
st.title("Why Bunching Happens")
st.caption("Static schedules fail when delays accumulate unevenly")

seed = st.sidebar.number_input("Scenario seed", min_value=1, max_value=999, value=11, step=1)
profile = st.sidebar.selectbox("Demand profile", ["peak", "off_peak"], index=0)
traffic_spike = st.sidebar.toggle("Traffic spike", value=False)
passenger_surge = st.sidebar.toggle("Passenger surge", value=False)

report = run_policy_report("static", int(seed), profile)
trace = apply_modifiers(report["trace"], traffic_spike=traffic_spike, passenger_surge=passenger_surge)
max_step = len(trace) - 1

step = st.slider("Replay step", min_value=0, max_value=max_step, value=min(40, max_step))
frame = step_view(trace, step)

c1, c2 = st.columns([2, 1])
with c1:
    st.subheader("Corridor Snapshot")
    for bus in frame["system_state"]["buses"]:
        label = f"{bus['bus_id']} | headway={bus['headway_forward_sec']:.1f}s | occ={bus['occupancy']:.2f}"
        st.progress(min(1.0, float(bus["occupancy"])), text=label)

with c2:
    st.subheader("At This Step")
    st.metric("Bunching Events", frame["bunching"])
    st.metric("Predicted Risk", f"{frame['prediction_bundle']['bunching_risk_score']:.2f}")
    st.metric("Congestion Score", f"{frame['prediction_bundle']['congestion_score']:.2f}")

divider()
st.subheader("Headway Compression Replay")
play = st.button("Play 30-step animation")
placeholder = st.empty()
if play:
    for i in range(step, min(step + 30, max_step + 1)):
        f = step_view(trace, i)
        hws = [b["headway_forward_sec"] for b in f["system_state"]["buses"]]
        placeholder.line_chart({"min_headway": [min(hws)], "avg_headway": [sum(hws) / len(hws)]})
        time.sleep(0.05)

st.warning("Bunching cascade: delay increases dwell, following buses catch up, and headways collapse.")
