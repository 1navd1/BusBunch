from __future__ import annotations

import streamlit as st

from lib.data import action_chips, frame_story, load_demo_seed, load_kpi_summary, map_payload, step_view
from lib.ui import apply_theme, chip_row, divider, legend, render_corridor_map, story_card

st.set_page_config(page_title="BusBunch BMTC AI Demo", page_icon=":bus:", layout="wide")
apply_theme()

demo = load_demo_seed()
kpi = load_kpi_summary()
frame = step_view(demo["trace"], min(24, len(demo["trace"]) - 1))
story = frame_story(frame, demo["policy"])

st.title("BusBunch: AI Control Room for BMTC")
st.caption("A map-first demo of how AI prevents bus bunching on the KBS - MG Road - Domlur corridor")

lead, side = st.columns([1.55, 1.0])
with lead:
    render_corridor_map(map_payload(frame, demo["policy"]), key="home-map")
    legend()

with side:
    story_card("What Judges Will See", story["headline"])
    st.markdown("")
    story_card(
        "Why This Works",
        "Instead of reacting after buses bunch, the system predicts risk ahead of time and gently changes holding, speed, or dispatch timing.",
    )
    st.markdown("")
    chip_row(action_chips(frame))
    st.info(
        f"Frozen replay: seed {demo['seed']} | corridor profile {demo.get('scenario', {}).get('peak_profile', 'peak')}"
    )

divider()

left, right = st.columns(2)
with left:
    st.subheader("Judge-Friendly Flow")
    st.markdown("- `Why Bunching Happens`: watch buses visually collapse into a bunch on the real corridor map")
    st.markdown("- `Baseline vs AI`: same route, same seed, but one side bunches while the other stays evenly spaced")
    st.markdown("- `Control Room`: see the intervention being triggered on the map")
    st.markdown("- `AI Brain`: understand why the controller acted at that stop")
    st.markdown("- `Driver Assist`: see the plain-language instruction a driver would actually follow")

with right:
    st.subheader("Headline Story")
    st.markdown("- Fewer bunching events")
    st.markdown("- Smoother spacing between buses")
    st.markdown("- Less crowd shock at key stops")
    st.markdown(
        f"- Current demo artifacts show `{kpi['headline_kpi_win']['wins_out_of_3']}/3` headline outcomes moving in the right direction"
    )
