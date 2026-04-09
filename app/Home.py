from __future__ import annotations

import streamlit as st

from lib.data import load_demo_seed, load_kpi_summary
from lib.ui import apply_theme, divider

st.set_page_config(page_title="BusBunch BMTC AI Demo", page_icon=":bus:", layout="wide")
apply_theme()

st.title("BusBunch: Hybrid DRL-GNN Bus Anti-Bunching for BMTC")
st.caption("One-corridor MVP with city-scale architecture story")

kpi = load_kpi_summary()
demo = load_demo_seed()
imp = kpi["improvement_vs_static"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Bunching Reduction", f"{imp['bunching_count_pct']}%")
c2.metric("Wait-Time Reduction", f"{imp['avg_wait_time_pct']}%")
c3.metric("Occupancy Balance", f"{imp['occupancy_std_pct']}%")
c4.metric("Headline KPI Wins", f"{kpi['headline_kpi_win']['wins_out_of_3']} / 3")

divider()

left, right = st.columns(2)
with left:
    st.subheader("What This Demo Shows")
    st.markdown("- Why static schedules create bunching under delays and demand shocks")
    st.markdown("- Baseline vs AI control on the same seeded scenario")
    st.markdown("- Control-room operations with alerts and interventions")
    st.markdown("- AI-brain view with risk predictions and action rationale")

with right:
    st.subheader("Navigation")
    st.markdown("1. `Why Bunching Happens`")
    st.markdown("2. `Baseline vs AI`")
    st.markdown("3. `Control Room`")
    st.markdown("4. `AI Brain`")
    st.markdown("5. `Outcome Summary`")

st.info(
    f"Frozen demo seed: seed={demo['seed']} | policy={demo['policy']} | "
    f"profile={demo.get('scenario', {}).get('peak_profile', 'peak')}"
)
