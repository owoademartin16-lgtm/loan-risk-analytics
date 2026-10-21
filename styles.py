"""Shared visual theme for the Loan Risk Analytics dashboard.

Defines the colour palette, the Plotly figure template and the small
amount of custom CSS used to give the app a consistent, professional
financial-analytics look.
"""

import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
NAVY = "#0f2b46"          # primary text / headings
MUTED = "#5b6b7c"         # secondary text
BLUE = "#1f5c99"          # primary accent (bars, highlights)
BLUE_DARK = "#174573"     # hover states, emphasis
TEAL = "#0f8a83"          # secondary accent (counts, paid loans)
RISK = "#b5443a"          # restrained red for default / risk figures
CARD_BG = "#ffffff"
PAGE_BG = "#f4f7fa"
BORDER = "#dfe6ee"
GRID = "#e9eef4"

FONT_FAMILY = "Helvetica Neue, Helvetica, Arial, sans-serif"

TEMPLATE_NAME = "loan_light"


# ---------------------------------------------------------------------------
# Plotly template
# ---------------------------------------------------------------------------
def build_plotly_template() -> go.layout.Template:
    """Create the shared Plotly template (light background, navy text)."""
    return go.layout.Template(
        layout=go.Layout(
            font=dict(family=FONT_FAMILY, size=13, color=NAVY),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            colorway=[BLUE, TEAL, BLUE_DARK, "#7aa6c2", RISK, "#c9a227"],
            xaxis=dict(
                gridcolor=GRID,
                zeroline=False,
                linecolor=BORDER,
                tickfont=dict(size=12, color=MUTED),
                title_font=dict(size=13, color=NAVY),
            ),
            yaxis=dict(
                gridcolor=GRID,
                zeroline=False,
                linecolor=BORDER,
                tickfont=dict(size=12, color=MUTED),
                title_font=dict(size=13, color=NAVY),
            ),
            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                font=dict(size=12, color=NAVY),
            ),
            hoverlabel=dict(
                bgcolor=CARD_BG,
                font_size=13,
                font_family=FONT_FAMILY,
                font_color=NAVY,
                bordercolor=BORDER,
            ),
            margin=dict(l=52, r=32, t=64, b=48),
        )
    )


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
_CUSTOM_CSS = f"""
<style>
    /* Page and typography */
    .stApp {{
        background: {PAGE_BG};
        font-family: {FONT_FAMILY};
    }}
    h1, h2, h3, h4 {{
        color: {NAVY} !important;
        font-family: {FONT_FAMILY};
        letter-spacing: -0.01em;
    }}
    p, li, label, .stMarkdown {{ color: {NAVY}; }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background: {CARD_BG};
        border-right: 1px solid {BORDER};
    }}
    [data-testid="stSidebar"] .side-title {{
        font-size: 21px;
        font-weight: 700;
        color: {NAVY};
        margin-bottom: 0;
        letter-spacing: -0.01em;
    }}
    [data-testid="stSidebar"] .side-subtitle {{
        font-size: 12.5px;
        color: {MUTED};
        margin-top: 2px;
    }}
    [data-testid="stSidebar"] .filter-heading {{
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: {MUTED};
        margin-top: 4px;
    }}

    /* KPI cards */
    .kpi-card {{
        background: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 16px 20px 18px 20px;
        box-shadow: 0 1px 2px rgba(15, 43, 70, 0.05);
        height: 100%;
    }}
    .kpi-label {{
        font-size: 11.5px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: {MUTED};
        margin-bottom: 6px;
    }}
    .kpi-value {{
        font-size: 27px;
        font-weight: 700;
        color: {NAVY};
        line-height: 1.15;
    }}
    .kpi-sub {{
        font-size: 12px;
        color: {MUTED};
        margin-top: 5px;
    }}

    /* Section notes under charts */
    .section-note {{
        font-size: 12.5px;
        color: {MUTED};
        margin-top: 2px;
    }}

    /* Portfolio summary block */
    .summary-block {{
        background: {CARD_BG};
        border: 1px solid {BORDER};
        border-left: 4px solid {BLUE};
        border-radius: 8px;
        padding: 14px 18px;
        font-size: 14px;
        line-height: 1.55;
        color: {NAVY};
    }}

    /* Prediction result banner */
    .pred-banner {{
        border-radius: 12px;
        padding: 20px 24px;
        color: #ffffff;
        box-shadow: 0 2px 10px rgba(15, 43, 70, 0.12);
        margin-bottom: 4px;
    }}
    .pred-banner .pred-verdict {{
        font-size: 23px;
        font-weight: 700;
        margin: 0 0 2px 0;
        letter-spacing: -0.01em;
    }}
    .pred-banner .pred-prob {{
        font-size: 15px;
        font-weight: 500;
        margin: 0;
        opacity: 0.95;
    }}
</style>
"""


def inject_css() -> None:
    """Inject the custom CSS (call once per Streamlit run)."""
    import streamlit as st

    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


def kpi_card(label: str, value: str, sub: str = "") -> str:
    """Return the HTML for a single KPI card."""
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f"{sub_html}"
        f"</div>"
    )


def summary_block(html: str) -> str:
    """Wrap arbitrary HTML in the summary-block container."""
    return f'<div class="summary-block">{html}</div>'
