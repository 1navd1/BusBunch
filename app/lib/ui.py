from __future__ import annotations

from typing import Any, Dict, List

import pydeck as pdk
import streamlit as st


def apply_theme() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Source+Sans+3:wght@400;600;700&display=swap');

:root {
  --bg-a: #f4efe5;
  --bg-b: #dfe8df;
  --ink: #132033;
  --muted: #556173;
  --line: rgba(19, 32, 51, 0.14);
  --accent: #0f6d7a;
  --accent-2: #c56a1a;
  --card: rgba(255,255,255,0.74);
  --sidebar-bg: linear-gradient(180deg, #152235, #101926);
}

html[data-theme="light"] {
  --bg-a: #f4efe5;
  --bg-b: #dfe8df;
  --ink: #132033;
  --muted: #556173;
  --line: rgba(19, 32, 51, 0.14);
  --accent: #0f6d7a;
  --accent-2: #c56a1a;
  --card: rgba(255,255,255,0.74);
  --sidebar-bg: linear-gradient(180deg, #152235, #101926);
}

html[data-theme="dark"] {
  --bg-a: #0f1624;
  --bg-b: #152133;
  --ink: #eef3f9;
  --muted: #b7c3d1;
  --line: rgba(228, 236, 245, 0.16);
  --accent: #52c7d8;
  --accent-2: #ef9b45;
  --card: rgba(19, 31, 48, 0.74);
  --sidebar-bg: linear-gradient(180deg, #09111b, #0d1623);
}

.stApp {
  background:
    radial-gradient(circle at 12% 10%, rgba(15,109,122,0.16), transparent 28%),
    radial-gradient(circle at 84% 16%, rgba(197,106,26,0.12), transparent 26%),
    linear-gradient(145deg, var(--bg-a), var(--bg-b));
  color: var(--ink);
  font-family: "Source Sans 3", sans-serif;
}

h1, h2, h3 {
  font-family: "Space Grotesk", sans-serif !important;
  letter-spacing: -0.02em;
  color: var(--ink);
}

h1 {
  font-size: clamp(2.2rem, 3.6vw, 3.3rem) !important;
}

p, li, .stCaption {
  color: var(--muted);
}

[data-testid="stHeader"] {
  background: transparent;
}

[data-testid="stMetric"] {
  border: 1px solid var(--line);
  border-radius: 18px;
  background: var(--card);
  backdrop-filter: blur(12px);
  box-shadow: 0 14px 30px rgba(19, 32, 51, 0.08);
  padding: 0.45rem 0.65rem;
}

[data-testid="stSidebar"] {
  background: var(--sidebar-bg);
}

[data-testid="stSidebar"] * {
  color: #edf3f8 !important;
}

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stCaption {
  color: rgba(237,243,248,0.76) !important;
}

.stButton > button {
  border-radius: 16px;
  background: linear-gradient(135deg, var(--accent), #1f8fa0);
  border: 1px solid rgba(15,109,122,0.24);
  color: white;
  font-weight: 700;
  box-shadow: 0 10px 24px rgba(15,109,122,0.24);
}

[data-testid="stDataFrame"] {
  border-radius: 18px;
  overflow: hidden;
}

.block-divider {
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--line), transparent);
  margin: 0.75rem 0 1rem;
}

.story-card {
  border: 1px solid var(--line);
  border-radius: 22px;
  background: var(--card);
  padding: 1.05rem 1.05rem 0.95rem;
  backdrop-filter: blur(12px);
  box-shadow: 0 16px 34px rgba(19, 32, 51, 0.08);
}

.story-card h4 {
  margin: 0 0 0.35rem;
  font-family: "Space Grotesk", sans-serif;
  color: var(--ink);
}

.story-card p {
  margin: 0;
  color: var(--muted);
}

.chip-row {
  display: flex;
  gap: 0.45rem;
  flex-wrap: wrap;
}

.chip {
  display: inline-flex;
  align-items: center;
  border-radius: 999px;
  padding: 0.3rem 0.65rem;
  background: rgba(15,109,122,0.10);
  border: 1px solid rgba(15,109,122,0.20);
  color: var(--ink);
  font-size: 0.92rem;
  font-weight: 600;
}

.legend {
  display: flex;
  gap: 0.75rem;
  flex-wrap: wrap;
  margin-top: 0.4rem;
}

.legend-item {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  color: var(--muted);
  font-size: 0.9rem;
}

.legend-dot {
  width: 11px;
  height: 11px;
  border-radius: 999px;
  display: inline-block;
}

.tablet-shell {
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 28px;
  background:
    radial-gradient(circle at 85% 18%, rgba(239,155,69,0.16), transparent 28%),
    linear-gradient(180deg, #16253b, #0e1827);
  padding: 1rem;
  box-shadow: 0 22px 50px rgba(8, 14, 22, 0.34);
}

.tablet-shell h4,
.tablet-shell p,
.tablet-shell span,
.tablet-shell strong {
  color: #f1f6f3 !important;
}

.tablet-title {
  font-family: "Space Grotesk", sans-serif;
  font-size: 1.1rem;
  margin: 0 0 0.2rem;
}

.tablet-subtitle {
  color: rgba(241,246,243,0.72) !important;
  font-size: 0.93rem;
  margin: 0;
}

.status-card {
  border-radius: 22px;
  padding: 1rem 1rem 0.9rem;
  min-height: 134px;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.18);
}

.status-card h5 {
  margin: 0 0 0.35rem;
  font-size: 0.9rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.status-card strong {
  display: block;
  font-size: 2rem;
  line-height: 1.05;
  margin-bottom: 0.25rem;
  font-family: "Space Grotesk", sans-serif;
}

.status-card p {
  margin: 0;
  font-size: 0.92rem;
}

.status-green {
  background: linear-gradient(180deg, #17896f, #0f5c4c);
}

.status-yellow {
  background: linear-gradient(180deg, #d88a2d, #9b5514);
}

.status-red {
  background: linear-gradient(180deg, #de5a54, #a62c2c);
}

.driver-panel {
  border-radius: 22px;
  padding: 1rem;
}

.driver-panel-light {
  background: var(--card);
  border: 1px solid var(--line);
}

.driver-panel-dark {
  background: rgba(255,255,255,0.08);
  border: 1px solid rgba(255,255,255,0.10);
}

.driver-panel h5 {
  margin: 0 0 0.35rem;
  font-size: 0.9rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.driver-panel-light h5 {
  color: var(--ink);
}

.driver-panel-light p {
  color: var(--muted);
}

.driver-panel-dark h5 {
  color: #f1f6f3;
}

.driver-panel-dark p {
  margin: 0;
  color: rgba(241,246,243,0.88);
}

.action-box {
  border-radius: 24px;
  padding: 1rem 1rem 0.95rem;
  background: linear-gradient(135deg, #fff2d8, #f6d7a9);
  border: 1px solid rgba(197,106,26,0.22);
  box-shadow: 0 14px 34px rgba(197,106,26,0.18);
}

.action-box h5 {
  margin: 0 0 0.4rem;
  color: #7a4311;
  font-size: 0.9rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.action-box p {
  margin: 0;
  color: #1d2430;
  font-size: 1.1rem;
  font-weight: 700;
  line-height: 1.35;
}

.info-box {
  border-radius: 24px;
  padding: 1rem 1rem 0.95rem;
  background: linear-gradient(135deg, #eef4fb, #dce8f5);
  border: 1px solid rgba(15,109,122,0.16);
  box-shadow: 0 14px 30px rgba(15,109,122,0.10);
}

.info-box h5 {
  margin: 0 0 0.4rem;
  color: #23415a;
  font-size: 0.9rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.info-box p {
  margin: 0;
  color: #1e2c3b;
  font-size: 1rem;
  font-weight: 600;
  line-height: 1.4;
}

@media (max-width: 768px) {
  [data-testid="stMetric"] { padding: 0.25rem 0.4rem; }
}
</style>
        """,
        unsafe_allow_html=True,
    )


def divider() -> None:
    st.markdown('<div class="block-divider"></div>', unsafe_allow_html=True)


def story_card(title: str, body: str) -> None:
    st.markdown(
        f"""
<div class="story-card">
  <h4>{title}</h4>
  <p>{body}</p>
</div>
        """,
        unsafe_allow_html=True,
    )


def chip_row(labels: List[str]) -> None:
    chips = "".join(f'<span class="chip">{label}</span>' for label in labels)
    st.markdown(f'<div class="chip-row">{chips}</div>', unsafe_allow_html=True)


def legend() -> None:
    st.markdown(
        """
<div class="legend">
  <span class="legend-item"><span class="legend-dot" style="background:#0f766e"></span>Stable flow</span>
  <span class="legend-item"><span class="legend-dot" style="background:#d97706"></span>Crowded bus</span>
  <span class="legend-item"><span class="legend-dot" style="background:#dc2626"></span>Bunching risk</span>
  <span class="legend-item"><span class="legend-dot" style="background:#fbbf24"></span>Terminal / key stop</span>
</div>
        """,
        unsafe_allow_html=True,
    )


def tablet_shell(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
<div class="tablet-shell">
  <div class="tablet-title">{title}</div>
  <p class="tablet-subtitle">{subtitle}</p>
</div>
        """,
        unsafe_allow_html=True,
    )


def status_card(title: str, value: str, detail: str, tone: str) -> None:
    tone_class = {
        "green": "status-green",
        "yellow": "status-yellow",
        "red": "status-red",
    }.get(tone, "status-green")
    st.markdown(
        f"""
<div class="status-card {tone_class}">
  <h5>{title}</h5>
  <strong>{value}</strong>
  <p>{detail}</p>
</div>
        """,
        unsafe_allow_html=True,
    )


def driver_panel(title: str, body: str, tone: str = "light") -> None:
    panel_class = "driver-panel-dark" if tone == "dark" else "driver-panel-light"
    st.markdown(
        f"""
<div class="driver-panel {panel_class}">
  <h5>{title}</h5>
  <p>{body}</p>
</div>
        """,
        unsafe_allow_html=True,
    )


def action_box(title: str, body: str) -> None:
    st.markdown(
        f"""
<div class="action-box">
  <h5>{title}</h5>
  <p>{body}</p>
</div>
        """,
        unsafe_allow_html=True,
    )


def info_box(title: str, body: str) -> None:
    st.markdown(
        f"""
<div class="info-box">
  <h5>{title}</h5>
  <p>{body}</p>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_corridor_map(payload: Dict[str, Any], key: str, height: int = 500) -> None:
    layers = [
        pdk.Layer(
            "PathLayer",
            payload["route"],
            get_path="path",
            get_width=8,
            width_min_pixels=4,
            rounded=True,
            pickable=False,
            get_color="color",
        ),
        pdk.Layer(
            "ScatterplotLayer",
            payload["stops"],
            get_position="coordinates",
            get_radius=1,
            radius_units="common",
            radius_min_pixels=5,
            radius_max_pixels=5,
            get_fill_color="fill_color",
            get_line_color=[16, 34, 26, 180],
            line_width_min_pixels=1,
            stroked=True,
            pickable=True,
        ),
        pdk.Layer(
            "TextLayer",
            payload["stops"],
            get_position="coordinates",
            get_text="stop_name",
            get_size=13,
            get_color="label_color",
            get_alignment_baseline="'top'",
            get_pixel_offset=[0, 14],
        ),
        pdk.Layer(
            "IconLayer",
            payload["buses"],
            get_icon="icon_data",
            get_position="coordinates",
            get_size="icon_size",
            size_scale=1,
            size_units="common",
            size_min_pixels=26,
            size_max_pixels=30,
            pickable=True,
        ),
        pdk.Layer(
            "TextLayer",
            payload["buses"],
            get_position="coordinates",
            get_text="label",
            get_size=14,
            get_color="label_color",
            get_alignment_baseline="'center'",
            get_text_anchor="'middle'",
        ),
    ]

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(**payload["view"]),
        tooltip={"html": "{tooltip}", "style": {"backgroundColor": "#10221a", "color": "white"}},
        map_provider="carto",
        map_style="light",
    )
    st.pydeck_chart(deck, width="stretch", key=key)
