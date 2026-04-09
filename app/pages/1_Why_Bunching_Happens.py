from __future__ import annotations

import time

import streamlit as st

from app.lib.data import apply_modifiers, run_policy_report, step_view

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

col_a, col_b = st.columns([2, 1])

with col_a:
    st.subheader("Corridor Snapshot")
    buses = frame["system_state"]["buses"]
    for bus in buses:
        label = f"{bus['bus_id']} | headway={bus['headway_forward_sec']:.1f}s | occ={bus['occupancy']:.2f}"
        st.progress(min(1.0, float(bus["occupancy"])), text=label)

with col_b:
    st.subheader("At This Step")
    st.metric("Bunching Events (step)", frame["bunching"])
    st.metric("Predicted Risk", f"{frame['prediction_bundle']['bunching_risk_score']:.2f}")
    st.metric("Congestion Score", f"{frame['prediction_bundle']['congestion_score']:.2f}")

st.markdown("### Headway Compression Replay")
play = st.button("Play 30-step animation")
placeholder = st.empty()

if play:
    for i in range(step, min(step + 30, max_step + 1)):
        f = step_view(trace, i)
        hws = [b["headway_forward_sec"] for b in f["system_state"]["buses"]]
        placeholder.line_chart({"min_headway": [min(hws)], "avg_headway": [sum(hws) / len(hws)]})
        time.sleep(0.05)

st.warning(
    "Bunching cascade: one delayed bus collects more passengers, increases dwell, gets slower; "
    "the following bus catches up and headways collapse."
)
