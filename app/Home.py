from __future__ import annotations

import streamlit as st

from app.lib.data import load_demo_seed, load_kpi_summary

st.set_page_config(page_title="BusBunch BMTC AI Demo", page_icon=":bus:", layout="wide")

st.title("BusBunch: Hybrid DRL-GNN Bus Anti-Bunching for BMTC")
st.caption("One-corridor MVP with city-scale architecture story")

kpi = load_kpi_summary()
demo = load_demo_seed()

c1, c2, c3, c4 = st.columns(4)
imp = kpi["improvement_vs_static"]

c1.metric("Bunching Reduction", f"{imp['bunching_count_pct']}%")
c2.metric("Wait Time Reduction", f"{imp['avg_wait_time_pct']}%")
c3.metric("Occupancy Balance Gain", f"{imp['occupancy_std_pct']}%")
c4.metric("Headline KPI Wins", f"{kpi['headline_kpi_win']['wins_out_of_3']} / 3")

st.markdown("""
### What This Demo Shows
- Why static schedules create bus bunching under delay and demand shocks.
- Baseline vs AI control on the same seeded scenario.
- A control-room style view with intervention logs.
- AI brain outputs: predicted risk, congestion, and action rationale.
- Outcome summary for judge-friendly KPI comparison.
""")

st.markdown("""
### Scenario Controls (available on each page)
- `Peak/Off-peak` demand profile
- `Traffic spike` stress toggle
- `Passenger surge` stress toggle
""")

st.markdown("""
### Navigation
Use the left sidebar to open:
1. `Why Bunching Happens`
2. `Baseline vs AI`
3. `Control Room`
4. `AI Brain`
5. `Outcome Summary`
""")

st.info(
    f"Frozen demo seed loaded from artifacts: seed={demo['seed']}, "
    f"policy={demo['policy']}, profile={demo.get('scenario', {}).get('peak_profile', 'peak')}"
)
