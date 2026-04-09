from __future__ import annotations

import json

import streamlit as st

from app.lib.data import ARTIFACTS_DIR, apply_modifiers, compare_reports, load_kpi_summary, summarize_trace

st.title("Outcome Summary")
st.caption("Judge-facing KPI outcomes and scalability narrative")

seed = st.sidebar.number_input("Scenario seed", min_value=1, max_value=999, value=11, step=1)
profile = st.sidebar.selectbox("Demand profile", ["peak", "off_peak"], index=0)
traffic_spike = st.sidebar.toggle("Traffic spike", value=False)
passenger_surge = st.sidebar.toggle("Passenger surge", value=False)

cmp = compare_reports(int(seed), profile)
kpi = load_kpi_summary()

s_metrics = summarize_trace(apply_modifiers(cmp["static"]["trace"], traffic_spike, passenger_surge))
h_metrics = summarize_trace(apply_modifiers(cmp["headway"]["trace"], traffic_spike, passenger_surge))
p_metrics = summarize_trace(apply_modifiers(cmp["ppo"]["trace"], traffic_spike, passenger_surge))

st.subheader("Headline Result")

def better(a: float, b: float) -> bool:
    return b < a

wins = sum(
    [
        better(s_metrics["bunching_count"], p_metrics["bunching_count"]),
        better(s_metrics["avg_wait_time"], p_metrics["avg_wait_time"]),
        better(s_metrics["occupancy_std"], p_metrics["occupancy_std"]),
    ]
)

st.success(f"KPI wins in current scenario settings: {wins}/3")

rows = [
    {"policy": "static", **{k: round(v, 3) for k, v in s_metrics.items() if k != "total_delay"}},
    {"policy": "headway", **{k: round(v, 3) for k, v in h_metrics.items() if k != "total_delay"}},
    {"policy": "ppo", **{k: round(v, 3) for k, v in p_metrics.items() if k != "total_delay"}},
]
st.dataframe(rows, use_container_width=True)


def pct(base: float, new: float) -> float:
    if base == 0:
        return 0.0
    return round(100.0 * (base - new) / base, 2)

c1, c2, c3 = st.columns(3)
c1.metric("Bunching Reduction", f"{pct(s_metrics['bunching_count'], p_metrics['bunching_count'])}%")
c2.metric("Wait-Time Reduction", f"{pct(s_metrics['avg_wait_time'], p_metrics['avg_wait_time'])}%")
c3.metric("Occupancy Variance Reduction", f"{pct(s_metrics['occupancy_std'], p_metrics['occupancy_std'])}%")

st.subheader("Scale Story: Corridor -> City")
st.markdown(
    """
- Corridor MVP proves intervention logic with deterministic replay.
- Each corridor is represented as a route graph with local demand/traffic priors.
- Same `Predictor` and `Policy` interfaces can be scaled route-by-route.
- City control room can aggregate corridor risk and intervention streams.
"""
)

st.subheader("Download Demo Assets")
for filename in [
    "kpi_summary.json",
    "demo_seed.json",
    "comparison_full.json",
    "pitch_script.md",
    "backup_assets.json",
]:
    path = ARTIFACTS_DIR / filename
    if path.exists():
        st.download_button(
            label=f"Download {filename}",
            data=path.read_bytes(),
            file_name=filename,
            mime="application/octet-stream",
        )

with st.expander("Raw KPI Summary"):
    st.code(json.dumps(kpi, indent=2), language="json")
