from __future__ import annotations

from pathlib import Path

import streamlit as st

st.set_page_config(page_title="BusBunch Simulation", page_icon=":bus:", layout="wide")

app_dir = Path(__file__).resolve().parent
navigation = st.navigation(
    [
        st.Page(app_dir / "simulation_page.py", title="Simulation", icon=":material/play_circle:"),
        st.Page(app_dir / "pages" / "6_Driver_Assist.py", title="Driver Assist", icon=":material/directions_bus:"),
    ]
)
navigation.run()
