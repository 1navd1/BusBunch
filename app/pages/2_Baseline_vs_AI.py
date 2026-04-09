from __future__ import annotations

import time

import streamlit as st

from lib.data import apply_modifiers, comparison_story, compare_reports, frame_story, map_payload, step_view
from lib.ui import apply_theme, chip_row, divider, legend, render_corridor_map, story_card

apply_theme()
st.title("Baseline vs AI")
st.caption("Same corridor. Same replay seed. One side bunches; the other side stays spaced out.")

seed = st.sidebar.number_input("Scenario seed", min_value=1, max_value=999, value=11, step=1)
profile = st.sidebar.selectbox("Demand profile", ["peak", "off_peak"], index=0)
traffic_spike = st.sidebar.toggle("Traffic spike", value=False)
passenger_surge = st.sidebar.toggle("Passenger surge", value=False)

cmp = compare_reports(int(seed), profile)
static_trace = apply_modifiers(cmp["static"]["trace"], traffic_spike, passenger_surge)
ppo_trace = apply_modifiers(cmp["ppo"]["trace"], traffic_spike, passenger_surge)

max_step = min(len(static_trace), len(ppo_trace)) - 1
step = st.slider("Replay step", min_value=0, max_value=max_step, value=min(52, max_step))

static_frame = step_view(static_trace, step)
ai_frame = step_view(ppo_trace, step)
static_story = frame_story(static_frame, "static")
ai_story = frame_story(ai_frame, "ppo")

left, right = st.columns(2)
with left:
    st.subheader("Static Schedule")
    render_corridor_map(map_payload(static_frame, "static"), key=f"cmp-static-{step}")
    story_card(static_story["headline"], static_story["detail"])

with right:
    st.subheader("AI Controller")
    render_corridor_map(map_payload(ai_frame, "ppo"), key=f"cmp-ai-{step}")
    story_card(ai_story["headline"], ai_story["detail"])

legend()
divider()

notes = comparison_story(static_frame, ai_frame)
story_cols = st.columns(len(notes))
for col, note in zip(story_cols, notes):
    with col:
        story_card("Judge Cue", note)

chip_row(
    [
        f"Static bunching: {static_frame['bunching']}",
        f"AI bunching: {ai_frame['bunching']}",
        f"Focus stop: {ai_frame['control_state']['focus_stop_name'].replace('_', ' ')}",
    ]
)

divider()
if st.button("Play side-by-side replay"):
    static_placeholder = left.empty()
    ai_placeholder = right.empty()
    for i in range(step, min(step + 18, max_step + 1)):
        s_frame = step_view(static_trace, i)
        a_frame = step_view(ppo_trace, i)
        with static_placeholder.container():
            st.subheader("Static Schedule")
            render_corridor_map(map_payload(s_frame, "static"), key=f"cmp-static-play-{i}")
        with ai_placeholder.container():
            st.subheader("AI Controller")
            render_corridor_map(map_payload(a_frame, "ppo"), key=f"cmp-ai-play-{i}")
        time.sleep(0.18)
