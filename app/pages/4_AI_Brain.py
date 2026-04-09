from __future__ import annotations

import streamlit as st

from lib.data import action_chips, apply_modifiers, explain_action, map_payload, run_policy_report, step_view
from lib.ui import apply_theme, chip_row, divider, legend, render_corridor_map, story_card

apply_theme()
st.title("AI Brain")
st.caption("A simple view of what the predictor sees and how the controller turns that into action")

seed = st.sidebar.number_input("Scenario seed", min_value=1, max_value=999, value=11, step=1)
profile = st.sidebar.selectbox("Demand profile", ["peak", "off_peak"], index=0)
traffic_spike = st.sidebar.toggle("Traffic spike", value=False)
passenger_surge = st.sidebar.toggle("Passenger surge", value=False)

report = run_policy_report("ppo", int(seed), profile)
trace = apply_modifiers(report["trace"], traffic_spike, passenger_surge)
step = st.slider("Replay step", min_value=0, max_value=len(trace) - 1, value=min(90, len(trace) - 1))
frame = step_view(trace, step)

left, right = st.columns([1.55, 1.0])
with left:
    render_corridor_map(map_payload(frame, "ppo"), key=f"brain-map-{step}")
    legend()

with right:
    story_card(
        "Predictor Sees",
        (
            f"Congestion score {frame['prediction_bundle']['congestion_score']:.2f} and bunching risk "
            f"{frame['prediction_bundle']['bunching_risk_score']:.2f} around "
            f"{frame['control_state']['focus_stop_name'].replace('_', ' ')}."
        ),
    )
    st.markdown("")
    story_card("Controller Does", explain_action(frame))
    st.markdown("")
    chip_row(action_chips(frame))

divider()
st.subheader("Decision Loop")
loop_cols = st.columns(4)
loop_cols[0].markdown("**1. Observe**  \nRead bus spacing, stop queues, and congestion.")
loop_cols[1].markdown("**2. Predict**  \nEstimate where bunching will happen next.")
loop_cols[2].markdown("**3. Decide**  \nChoose hold, speed, or dispatch intervention.")
loop_cols[3].markdown("**4. Replay**  \nShow the effect directly on the geographic corridor map.")

divider()
st.subheader("Current Focus")
focus = frame["control_state"]
focus_bus = next(bus for bus in frame["system_state"]["buses"] if bus["bus_id"] == focus["focus_bus_id"])
story_card(
    f"{focus['focus_bus_id'].upper()} is the decision target",
    (
        f"It is currently between {focus_bus['current_stop_name'].replace('_', ' ')} and "
        f"{focus_bus['next_stop_name'].replace('_', ' ')}. "
        f"Occupancy is {focus_bus['occupancy'] * 100:.0f}% and forward headway is "
        f"{focus_bus['headway_forward_sec']:.0f}s."
    ),
)
