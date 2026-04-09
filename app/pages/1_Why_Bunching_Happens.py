from __future__ import annotations

import time

import streamlit as st

from lib.data import apply_modifiers, frame_story, map_payload, run_policy_report, step_view
from lib.ui import apply_theme, chip_row, divider, legend, render_corridor_map, story_card

apply_theme()
st.title("Why Bunching Happens")
st.caption("Watch how small delays turn into buses catching each other on the real BMTC corridor map")

seed = st.sidebar.number_input("Scenario seed", min_value=1, max_value=999, value=11, step=1)
profile = st.sidebar.selectbox("Demand profile", ["peak", "off_peak"], index=0)
traffic_spike = st.sidebar.toggle("Traffic spike", value=False)
passenger_surge = st.sidebar.toggle("Passenger surge", value=False)

report = run_policy_report("static", int(seed), profile)
trace = apply_modifiers(report["trace"], traffic_spike=traffic_spike, passenger_surge=passenger_surge)
max_step = len(trace) - 1
step = st.slider("Replay step", min_value=0, max_value=max_step, value=min(48, max_step))

frame = step_view(trace, step)
story = frame_story(frame, "static")

top, side = st.columns([1.6, 1.0])
with top:
    render_corridor_map(map_payload(frame, "static"), key=f"why-map-{step}")
    legend()

with side:
    story_card(story["headline"], story["detail"])
    st.markdown("")
    if frame["bunching"] > 0:
        story_card(
            "What Judges Should Notice",
            "Two buses are visually collapsing into the same part of the corridor. That is the bunching cascade in action.",
        )
    else:
        story_card(
            "What Judges Should Notice",
            "Even before buses fully bunch, spacing becomes visibly uneven and the next collapse is already forming.",
        )
    st.markdown("")
    chip_row(
        [
            f"Bunching alerts: {frame['bunching']}",
            f"Risk {frame['prediction_bundle']['bunching_risk_score']:.2f}",
            f"Congestion {frame['prediction_bundle']['congestion_score']:.2f}",
        ]
    )

divider()
st.subheader("Autoplay")
play_cols = st.columns([1, 5])
if play_cols[0].button("Play 20 steps"):
    placeholder = play_cols[1].empty()
    for i in range(step, min(step + 20, max_step + 1)):
        replay_frame = step_view(trace, i)
        with placeholder.container():
            render_corridor_map(map_payload(replay_frame, "static"), key=f"why-map-play-{i}")
            chip_row(
                [
                    replay_frame["visual_event"].replace("_", " ").title(),
                    f"Near {replay_frame['system_state']['buses'][0]['next_stop_name'].replace('_', ' ')}",
                ]
            )
        time.sleep(0.18)

st.warning(
    "Narration tip: point at the map and explain that a delayed bus boards more passengers, slows further, and the next bus catches up from behind."
)
