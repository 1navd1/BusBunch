from __future__ import annotations

import streamlit as st

from lib.data import apply_modifiers, explain_action, run_policy_report, step_view, trace_series
from lib.ui import apply_theme, divider

apply_theme()
st.title("AI Brain")
st.caption("Prediction + control reasoning behind each intervention")

seed = st.sidebar.number_input("Scenario seed", min_value=1, max_value=999, value=11, step=1)
profile = st.sidebar.selectbox("Demand profile", ["peak", "off_peak"], index=0)
traffic_spike = st.sidebar.toggle("Traffic spike", value=False)
passenger_surge = st.sidebar.toggle("Passenger surge", value=False)

report = run_policy_report("ppo", int(seed), profile)
trace = apply_modifiers(report["trace"], traffic_spike, passenger_surge)

step = st.slider("Replay step", min_value=0, max_value=len(trace) - 1, value=min(90, len(trace) - 1))
frame = step_view(trace, step)

st.subheader("Hybrid Loop")
st.code(
    "SystemState -> Predictor.predict() -> PredictionBundle -> Policy.act() -> ControlAction -> Simulator.step()",
    language="text",
)

left, right = st.columns(2)
with left:
    st.subheader("Prediction Bundle")
    st.json(frame["prediction_bundle"])
with right:
    st.subheader("Control State")
    st.json(frame["control_state"])

divider()
st.subheader("Route Graph Snapshot")
segments = frame["system_state"]["segments"]
st.dataframe(
    [
        {
            "segment": s["segment_id"],
            "from": s["from_stop_id"],
            "to": s["to_stop_id"],
            "base_travel_time_sec": s["base_travel_time_sec"],
            "traffic_multiplier": round(s["traffic_multiplier"], 3),
        }
        for s in segments
    ],
    use_container_width=True,
)

st.subheader("Temporal Signals")
st.line_chart(
    {
        "bunching_risk": trace_series(trace, "risk"),
        "congestion": trace_series(trace, "congestion"),
        "avg_occupancy": trace_series(trace, "avg_occupancy"),
    }
)

st.subheader("Decision Explanation")
st.info(explain_action(frame))
