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
  --bg-a: #f7f7f4;
  --bg-b: #eaf0eb;
  --ink: #10221a;
  --muted: #5b6b62;
  --line: rgba(16, 34, 26, 0.12);
  --accent: #0f766e;
}

.stApp {
  background:
    radial-gradient(circle at 15% 10%, rgba(15,118,110,0.10), transparent 35%),
    radial-gradient(circle at 85% 20%, rgba(180,83,9,0.08), transparent 35%),
    linear-gradient(135deg, var(--bg-a), var(--bg-b));
  color: var(--ink);
  font-family: "Source Sans 3", sans-serif;
}

h1, h2, h3 {
  font-family: "Space Grotesk", sans-serif !important;
  letter-spacing: -0.02em;
  color: var(--ink);
}

p, li, .stCaption {
  color: var(--muted);
}

[data-testid="stMetric"] {
  border: 1px solid var(--line);
  border-radius: 14px;
  background: rgba(255,255,255,0.80);
  padding: 0.35rem 0.55rem;
}

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(16,34,26,0.96), rgba(26,48,38,0.94));
}
[data-testid="stSidebar"] * {
  color: #ecf5ee !important;
}

.stButton > button {
  border-radius: 999px;
  background: linear-gradient(135deg, #0f766e, #0ea5a0);
  border: 1px solid rgba(15,118,110,0.35);
  color: white;
  font-weight: 700;
}

[data-testid="stDataFrame"] {
  border-radius: 12px;
  overflow: hidden;
}

.block-divider {
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(16,34,26,0.22), transparent);
  margin: 0.75rem 0 1rem;
}

.story-card {
  border: 1px solid var(--line);
  border-radius: 18px;
  background: rgba(255,255,255,0.78);
  padding: 1rem 1rem 0.85rem;
  box-shadow: 0 12px 32px rgba(16, 34, 26, 0.06);
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
  background: rgba(15,118,110,0.10);
  border: 1px solid rgba(15,118,110,0.18);
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
