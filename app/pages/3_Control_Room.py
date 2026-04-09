from __future__ import annotations

import streamlit as st

from lib.data import apply_modifiers, explain_action, run_policy_report, step_view, trace_series
from lib.ui import apply_theme, divider

apply_theme()
st.title("Control Room")
st.caption("Operational view: alerts, action log, and per-bus monitoring")

seed = st.sidebar.number_input("Scenario seed", min_value=1, max_value=999, value=11, step=1)
profile = st.sidebar.selectbox("Demand profile", ["peak", "off_peak"], index=0)
traffic_spike = st.sidebar.toggle("Traffic spike", value=False)
passenger_surge = st.sidebar.toggle("Passenger surge", value=False)

report = run_policy_report("ppo", int(seed), profile)
trace = apply_modifiers(report["trace"], traffic_spike, passenger_surge)

step = st.slider("Replay step", min_value=0, max_value=len(trace) - 1, value=min(80, len(trace) - 1))
frame = step_view(trace, step)

alerts = []
if frame["prediction_bundle"]["bunching_risk_score"] > 0.65:
    alerts.append("High bunching risk in downstream stops")
if frame["prediction_bundle"]["congestion_score"] > 0.65:
    alerts.append("Traffic congestion rising")
for b in frame["system_state"]["buses"]:
    if b["occupancy"] > 0.85:
        alerts.append(f"{b['bus_id']} approaching overload")

st.subheader("Live Alerts")
if alerts:
    for a in alerts:
        st.error(a)
else:
    st.success("No critical alerts at this step")

st.subheader("Latest Action")
st.info(explain_action(frame))

st.subheader("Per-Bus Status")
for b in frame["system_state"]["buses"]:
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{b['bus_id']} Headway", f"{b['headway_forward_sec']:.1f}s")
    c2.metric(f"{b['bus_id']} Occupancy", f"{100*b['occupancy']:.1f}%")
    c3.metric(f"{b['bus_id']} Delay", f"{b['delay_sec']:.1f}s")

divider()
st.subheader("Trend Panels")
st.line_chart(
    {
        "risk": trace_series(trace, "risk"),
        "congestion": trace_series(trace, "congestion"),
        "min_headway": trace_series(trace, "min_headway"),
    }
)

st.subheader("Recent Action Log")
start = max(0, step - 8)
log_rows = []
for f in trace[start : step + 1]:
    a = f["action"]
    log_rows.append(
        {
            "step": f["step"],
            "hold_sec": round(a["hold_sec"], 1),
            "speed_delta_pct": round(a["speed_delta_pct"] * 100, 2),
            "dispatch_offset_sec": round(a["dispatch_offset_sec"], 1),
            "risk": f["prediction_bundle"]["bunching_risk_score"],
        }
    )
st.dataframe(log_rows, use_container_width=True)
