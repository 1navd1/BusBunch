from __future__ import annotations

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

@media (max-width: 768px) {
  [data-testid="stMetric"] { padding: 0.25rem 0.4rem; }
}
</style>
        """,
        unsafe_allow_html=True,
    )


def divider() -> None:
    st.markdown('<div class="block-divider"></div>', unsafe_allow_html=True)
