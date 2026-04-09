from __future__ import annotations

import streamlit as st

from lib.data import apply_modifiers, compare_reports, step_view, summarize_trace
from lib.ui import apply_theme, divider

apply_theme()
st.title("Baseline vs AI")
st.caption("Same seed, same corridor, different control policy")

seed = st.sidebar.number_input("Scenario seed", min_value=1, max_value=999, value=11, step=1)
profile = st.sidebar.selectbox("Demand profile", ["peak", "off_peak"], index=0)
traffic_spike = st.sidebar.toggle("Traffic spike", value=False)
passenger_surge = st.sidebar.toggle("Passenger surge", value=False)

cmp = compare_reports(int(seed), profile)
static_trace = apply_modifiers(cmp["static"]["trace"], traffic_spike, passenger_surge)
ppo_trace = apply_modifiers(cmp["ppo"]["trace"], traffic_spike, passenger_surge)

max_step = min(len(static_trace), len(ppo_trace)) - 1
step = st.slider("Replay step", min_value=0, max_value=max_step, value=min(40, max_step))

left, right = st.columns(2)
with left:
    st.subheader("Static Baseline")
    s_frame = step_view(static_trace, step)
    st.metric("Step Bunching", s_frame["bunching"])
    for b in s_frame["system_state"]["buses"]:
        st.write(f"{b['bus_id']}: headway={b['headway_forward_sec']:.1f}s, occ={b['occupancy']:.2f}")

with right:
    st.subheader("AI Controller (PPO)")
    a_frame = step_view(ppo_trace, step)
    st.metric("Step Bunching", a_frame["bunching"])
    for b in a_frame["system_state"]["buses"]:
        st.write(f"{b['bus_id']}: headway={b['headway_forward_sec']:.1f}s, occ={b['occupancy']:.2f}")

s_metrics = summarize_trace(static_trace)
a_metrics = summarize_trace(ppo_trace)

divider()
st.subheader("KPI Comparison")
rows = []
for m in ["bunching_count", "avg_wait_time", "occupancy_std", "headway_std", "fuel_proxy"]:
    rows.append({"metric": m, "static": round(s_metrics[m], 4), "ai": round(a_metrics[m], 4)})
st.dataframe(rows, use_container_width=True)


def pct(base: float, new: float) -> float:
    if base == 0:
        return 0.0
    return round(100.0 * (base - new) / base, 2)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Bunching Reduction", f"{pct(s_metrics['bunching_count'], a_metrics['bunching_count'])}%")
k2.metric("Wait-Time Reduction", f"{pct(s_metrics['avg_wait_time'], a_metrics['avg_wait_time'])}%")
k3.metric("Occupancy Balance", f"{pct(s_metrics['occupancy_std'], a_metrics['occupancy_std'])}%")
k4.metric("Headway Stability", f"{pct(s_metrics['headway_std'], a_metrics['headway_std'])}%")
