from __future__ import annotations

import streamlit as st

from lib.data import (
    action_chips,
    apply_modifiers,
    driver_assist,
    explain_action,
    frame_story,
    map_payload,
    run_policy_report,
    step_view,
)
from lib.ui import apply_theme, chip_row, divider, legend, render_corridor_map, story_card

apply_theme()
st.title("Control Room")
st.caption("A visual operations desk: where the controller sees risk and decides what to do next")

seed = st.sidebar.number_input("Scenario seed", min_value=1, max_value=999, value=11, step=1)
profile = st.sidebar.selectbox("Demand profile", ["peak", "off_peak"], index=0)
traffic_spike = st.sidebar.toggle("Traffic spike", value=False)
passenger_surge = st.sidebar.toggle("Passenger surge", value=False)

report = run_policy_report("ppo", int(seed), profile)
trace = apply_modifiers(report["trace"], traffic_spike, passenger_surge)
step = st.slider("Replay step", min_value=0, max_value=len(trace) - 1, value=min(80, len(trace) - 1))
frame = step_view(trace, step)
story = frame_story(frame, "ppo")
assist = driver_assist(frame)

map_col, side = st.columns([1.65, 1.0])
with map_col:
    render_corridor_map(map_payload(frame, "ppo"), key=f"control-map-{step}")
    legend()

with side:
    story_card(story["headline"], story["detail"])
    st.markdown("")
    story_card("Latest Intervention", explain_action(frame))
    st.markdown("")
    story_card("Driver Instruction", assist["instruction"])
    st.markdown("")
    chip_row(action_chips(frame))

divider()

st.subheader("Driver Assist")
assist_cols = st.columns(4)
assist_cols[0].markdown(f"**Bus**  \n`{assist['bus_id']}`")
assist_cols[1].markdown(f"**Gap Ahead**  \n`{assist['headway_ahead']}`")
assist_cols[2].markdown(f"**Gap Behind**  \n`{assist['headway_behind']}`")
assist_cols[3].markdown(f"**Risk**  \n`{assist['risk_level']}`")

driver_cols = st.columns(2)
driver_cols[0].markdown(
    f"**Reason**  \n{assist['reason']}  \n\n**Expected Crowd At Next Stop**  \n`{assist['crowd_level']}`"
)
driver_cols[1].markdown(
    f"**Driver Message**  \n{assist['spoken_message']}  \n\n**Expected Benefit**  \n{assist['expected_benefit']}"
)

divider()

st.subheader("What The Controller Is Watching")
watch_cols = st.columns(3)
watch_cols[0].markdown(
    f"**Focus Bus**  \n`{frame['control_state']['focus_bus_id'].upper()}` near `{frame['control_state']['focus_stop_name'].replace('_', ' ')}`"
)
watch_cols[1].markdown(
    f"**Predicted Risk**  \nBunching risk is `{frame['prediction_bundle']['bunching_risk_score']:.2f}` on the downstream corridor."
)
watch_cols[2].markdown(
    f"**Congestion Outlook**  \nTraffic score is `{frame['prediction_bundle']['congestion_score']:.2f}` for this replay moment."
)

divider()
st.subheader("Bus-by-Bus Field View")
bus_cols = st.columns(len(frame["system_state"]["buses"]))
for col, bus in zip(bus_cols, frame["system_state"]["buses"]):
    with col:
        story_card(
            bus["bus_id"].upper(),
            (
                f"Near {bus['current_stop_name'].replace('_', ' ')} -> {bus['next_stop_name'].replace('_', ' ')}. "
                f"Occupancy {bus['occupancy'] * 100:.0f}%. "
                f"Forward headway {bus['headway_forward_sec']:.0f}s."
            ),
        )
