from __future__ import annotations

import json

import streamlit as st

from lib.data import ARTIFACTS_DIR, apply_modifiers, compare_reports, load_kpi_summary, summarize_trace
from lib.ui import apply_theme, divider

apply_theme()
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


def better(a: float, b: float) -> bool:
    return b < a

wins = sum(
    [
        better(s_metrics["bunching_count"], p_metrics["bunching_count"]),
        better(s_metrics["avg_wait_time"], p_metrics["avg_wait_time"]),
        better(s_metrics["occupancy_std"], p_metrics["occupancy_std"]),
    ]
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("KPI Wins", f"{wins}/3")
c2.metric("Bunching", f"{s_metrics['bunching_count']:.0f} -> {p_metrics['bunching_count']:.0f}")
c3.metric("Wait Time", f"{s_metrics['avg_wait_time']:.1f}s -> {p_metrics['avg_wait_time']:.1f}s")
c4.metric("Occupancy Std", f"{s_metrics['occupancy_std']:.3f} -> {p_metrics['occupancy_std']:.3f}")

divider()
st.dataframe(
    [
        {"policy": "static", **{k: round(v, 3) for k, v in s_metrics.items() if k != "total_delay"}},
        {"policy": "headway", **{k: round(v, 3) for k, v in h_metrics.items() if k != "total_delay"}},
        {"policy": "ppo", **{k: round(v, 3) for k, v in p_metrics.items() if k != "total_delay"}},
    ],
    use_container_width=True,
)

st.subheader("Scale Story: Corridor -> City")
st.markdown("- Corridor MVP validates interventions with deterministic replay")
st.markdown("- Each corridor can reuse the same Predictor/Policy interfaces")
st.markdown("- City control room can aggregate corridor risk/action streams")

st.subheader("Download Demo Assets")
for filename in ["kpi_summary.json", "demo_seed.json", "comparison_full.json", "pitch_script.md", "backup_assets.json"]:
    path = ARTIFACTS_DIR / filename
    if path.exists():
        st.download_button(label=f"Download {filename}", data=path.read_bytes(), file_name=filename)

with st.expander("Raw KPI Summary"):
    st.code(json.dumps(kpi, indent=2), language="json")
