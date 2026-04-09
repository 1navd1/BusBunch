from __future__ import annotations

import json

import streamlit as st

from lib.data import apply_modifiers, compare_reports, frame_story, load_kpi_summary, map_payload, step_view
from lib.ui import apply_theme, divider, render_corridor_map, story_card

apply_theme()
st.title("Outcome Summary")
st.caption("A final visual wrap-up for judges: what changed on the corridor and why it matters")

seed = st.sidebar.number_input("Scenario seed", min_value=1, max_value=999, value=11, step=1)
profile = st.sidebar.selectbox("Demand profile", ["peak", "off_peak"], index=0)
traffic_spike = st.sidebar.toggle("Traffic spike", value=False)
passenger_surge = st.sidebar.toggle("Passenger surge", value=False)

cmp = compare_reports(int(seed), profile)
kpi = load_kpi_summary()

static_trace = apply_modifiers(cmp["static"]["trace"], traffic_spike, passenger_surge)
ppo_trace = apply_modifiers(cmp["ppo"]["trace"], traffic_spike, passenger_surge)
step = st.slider("Replay step", min_value=0, max_value=min(len(static_trace), len(ppo_trace)) - 1, value=60)

static_frame = step_view(static_trace, step)
ppo_frame = step_view(ppo_trace, step)

left, right = st.columns(2)
with left:
    st.subheader("Before: Static Schedule")
    render_corridor_map(map_payload(static_frame, "static"), key=f"summary-static-{step}")
    story = frame_story(static_frame, "static")
    story_card(story["headline"], story["detail"])

with right:
    st.subheader("After: AI Control")
    render_corridor_map(map_payload(ppo_frame, "ppo"), key=f"summary-ai-{step}")
    story = frame_story(ppo_frame, "ppo")
    story_card(story["headline"], story["detail"])

divider()
st.subheader("Judge Takeaways")
takeaways = st.columns(3)
takeaways[0].markdown(
    "**Spacing becomes readable**  \nInstead of buses stacking up in one location, they spread back out along the route."
)
takeaways[1].markdown(
    "**Stops feel less chaotic**  \nPassenger buildup is easier to manage when the corridor does not send two buses together."
)
takeaways[2].markdown(
    "**This scales corridor-by-corridor**  \nThe same predictor-policy loop can be reused across BMTC corridors."
)

divider()
with st.expander("Technical Summary"):
    st.code(json.dumps(kpi, indent=2), language="json")
