"""Shared data loading, binning and chart helpers for the Loan Risk Analytics app.

Everything that is needed by more than one page lives here so the page
scripts stay short:

* :func:`load_data` / :func:`load_model` / :func:`load_features` — cached
  access to the three project assets (cached data, cached model).
* :func:`apply_filters` and the ``summary_*`` helpers — the sidebar filters
  and the grouped aggregations that feed the charts.
* the ``chart_*`` functions — one Plotly figure per dashboard chart, all
  sharing the same theme, colours and hover details.
* :func:`inject_styles` and :data:`STYLE_CSS` — the global stylesheet, and
  :func:`risk_assessment_html` — the colour-coded prediction card.

No cleaning, imputation or feature engineering happens here: the parquet
file is already clean and the model already contains its own preprocessing.
"""

from __future__ import annotations

import base64
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data" / "loans_clean.parquet"
MODEL_PATH = APP_DIR / "model" / "default_model.joblib"
FEATURES_PATH = APP_DIR / "model" / "features.joblib"

# Columns the app cannot work without.
REQUIRED_COLUMNS = [
    "Default",
    "loan_amnt",
    "fico_n",
    "dti_n",
    "purpose",
    "addr_state",
    "emp_length",
]

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
PRIMARY = "#2E86AB"      # main colour: fully paid / primary series
DEFAULT_RED = "#E4572E"  # default / defaulted series
TEXT = "#15304D"
MUTED = "#5B6B7C"
GRID = "#E4EAF0"
FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"

PAID_LABEL = "Fully Paid"
DEFAULT_LABEL = "Defaulted"

# Minimum number of loans a group needs before it appears in a ranking.

MIN_SAMPLE_SIZE = 100


# ---------------------------------------------------------------------------
# Global styling
# ---------------------------------------------------------------------------
# The stylesheet lives here (instead of in ``app.py``) so that every page can
# inject it: with ``st.navigation`` the main script's own elements are only
# rendered on the default page, so the CSS has to be injected from the page
# scripts (and from the sidebar) to be present everywhere.
STYLE_CSS = """
      :root {
        --brand: #2E86AB;
        --brand-dark: #1F6A8A;
        --brand-soft: #EAF3F8;
        --danger: #E4572E;
        --ink: #15304D;
        --muted: #5B6B7C;
        --line: #E4EAF0;
        --green: #22C55D;
        --amber: #F59E0B;
        --red: #EF4444;
      }
      .block-container { padding-top: 2.1rem; padding-bottom: 3rem; }
      h1, h2, h3 { color: var(--ink); letter-spacing: -0.01em; }
      h1 {
        font-size: 3rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.035em !important;
        line-height: 1.05 !important;
        color: var(--brand) !important;
        display: inline-block;
        border-bottom: 3px solid var(--brand);
        padding-bottom: 8px;
        margin-bottom: 6px;
      }
      h3 {
        font-size: 1.02rem !important;
        font-weight: 700 !important;
        color: var(--ink) !important;
        letter-spacing: -0.005em;
        padding-left: 11px !important;
        border-left: 4px solid var(--brand);
        margin: 0.1rem 0 0.2rem 0 !important;
      }
      section.main p { color: var(--muted); }
      section.main [data-testid="stCaptionContainer"] p { font-size: 0.86rem; }

      /* ---------------------------------------------------------------- KPI */
      div[data-testid="stMetric"] {
        position: relative;
        background: linear-gradient(180deg, #FFFFFF 0%, #FBFCFD 100%);
        border: 1px solid var(--line);
        border-left: 4px solid var(--brand);
        border-radius: 12px;
        padding: 14px 18px 12px 18px;
        box-shadow: 0 1px 3px rgba(21, 48, 77, 0.06);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        animation: kpiIn 0.45s cubic-bezier(0.22, 1, 0.36, 1) backwards;
      }
      div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 22px rgba(21, 48, 77, 0.10);
      }
      div[data-testid="stMetricLabel"] p {
        font-size: 0.76rem !important;
        font-weight: 700;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        color: var(--muted);
      }
      div[data-testid="stMetricValue"] {
        color: var(--ink);
        font-weight: 800;
        letter-spacing: -0.02em;
      }
      .st-key-kpi_row div[data-testid="stHorizontalBlock"] > div:nth-child(2)
        div[data-testid="stMetric"] { border-left-color: var(--danger); }
      .st-key-kpi_row div[data-testid="stHorizontalBlock"] > div:nth-child(3)
        div[data-testid="stMetric"] { border-left-color: var(--amber); }
      .st-key-kpi_row div[data-testid="stHorizontalBlock"] > div:nth-child(4)
        div[data-testid="stMetric"] { border-left-color: #7C6BD6; }

      /* KPI icon badge: a tinted chip with the glyph masked out of it. */
      .st-key-kpi_row div[data-testid="stMetric"]::before {
        content: "";
        position: absolute;
        top: 11px;
        right: 12px;
        width: 32px;
        height: 32px;
        border-radius: 10px;
        background: var(--kpi-soft);
      }
      .st-key-kpi_row div[data-testid="stMetric"]::after {
        content: "";
        position: absolute;
        top: 18px;
        right: 19px;
        width: 18px;
        height: 18px;
        background-color: var(--kpi-accent);
        -webkit-mask-image: var(--kpi-icon);
        mask-image: var(--kpi-icon);
        -webkit-mask-size: contain;
        mask-size: contain;
        -webkit-mask-repeat: no-repeat;
        mask-repeat: no-repeat;
        -webkit-mask-position: center;
        mask-position: center;
      }
      .st-key-kpi_row [data-testid="stHorizontalBlock"] > div:nth-child(2)
        div[data-testid="stMetric"] { animation-delay: 0.07s; }
      .st-key-kpi_row [data-testid="stHorizontalBlock"] > div:nth-child(3)
        div[data-testid="stMetric"] { animation-delay: 0.14s; }
      .st-key-kpi_row [data-testid="stHorizontalBlock"] > div:nth-child(4)
        div[data-testid="stMetric"] { animation-delay: 0.21s; }

      /* -------------------------------------------------------- Chart cards */
      [class*="st-key-card-"] {
        background: #FFFFFF;
        border: 1px solid var(--line) !important;
        border-radius: 14px;
        padding: 14px 16px 4px 16px;
        box-shadow: 0 1px 3px rgba(21, 48, 77, 0.06);
        transition: box-shadow 0.15s ease;
      }
      [class*="st-key-card-"]:hover {
        box-shadow: 0 10px 24px rgba(21, 48, 77, 0.09);
        transform: translateY(-2px);
      }

      /* ---------------------------------------------------------------- Tabs */
      div[data-testid="stTabs"] [role="tablist"] {
        position: relative;
        display: flex;
        gap: 0;
        background: #F2F6F9;
        border: 1px solid var(--line);
        border-radius: 999px;
        padding: 4px;
        box-shadow: inset 0 1px 3px rgba(21, 48, 77, 0.05);
      }
      /* The sliding pill behind the active tab. The position is carried by a
         custom property (Chromium does not re-check :has() on pseudo
         elements, but it does follow inherited custom properties). */
      div[data-testid="stTabs"] [role="tablist"] { --pill-x: 0px; }
      div[data-testid="stTabs"]
        [role="tablist"]:has([data-testid="stTab"]:nth-child(1)[aria-selected="true"])
        { --pill-x: 0px; }
      div[data-testid="stTabs"]
        [role="tablist"]:has([data-testid="stTab"]:nth-child(2)[aria-selected="true"])
        { --pill-x: 100%; }
      div[data-testid="stTabs"]
        [role="tablist"]:has([data-testid="stTab"]:nth-child(3)[aria-selected="true"])
        { --pill-x: 200%; }
      div[data-testid="stTabs"] [role="tablist"]::before {
        content: "";
        position: absolute;
        top: 4px;
        left: 4px;
        width: calc((100% - 8px) / 3);
        height: calc(100% - 8px);
        border-radius: 999px;
        background: linear-gradient(90deg, #2E86AB, #1F6A8A);
        box-shadow: 0 6px 14px rgba(46, 134, 171, 0.28);
        transform: translateX(var(--pill-x));
        transition: transform 0.34s cubic-bezier(0.4, 0, 0.2, 1);
        z-index: 0;
      }
      div[data-testid="stTabs"] .react-aria-SelectionIndicator { display: none; }
      div[data-testid="stTabs"] [data-testid="stTab"] {
        position: relative;
        z-index: 1;
        flex: 1 1 0;
        min-width: 0;
        border-radius: 999px;
        padding: 0.5rem 0.9rem;
        font-weight: 600;
        color: var(--muted);
        transition: color 0.2s ease;
      }
      div[data-testid="stTabs"] [data-testid="stTab"]:hover { color: var(--brand); }
      div[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"],
      div[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"]:hover
        { color: #FFFFFF; }
      div[data-testid="stTabs"] [data-testid="stTab"] p {
        color: inherit;
        font-weight: inherit;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }
      /* The icon in front of every tab label. */
      div[data-testid="stTabs"] [data-testid="stTab"] p::before {
        content: "";
        display: inline-block;
        width: 18px;
        height: 18px;
        margin-right: 8px;
        vertical-align: -3px;
        background-color: currentColor;
        -webkit-mask-image: var(--tab-icon);
        mask-image: var(--tab-icon);
        -webkit-mask-size: contain;
        mask-size: contain;
        -webkit-mask-repeat: no-repeat;
        mask-repeat: no-repeat;
        -webkit-mask-position: center;
        mask-position: center;
        transition: transform 0.25s ease;
      }
      div[data-testid="stTabs"] [data-testid="stTab"]:hover p::before {
        transform: scale(1.14) rotate(-5deg);
      }
      div[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"]:hover
        p::before { transform: scale(1.06); }

      /* --------------------------------------------------------- Panel motion */
      @keyframes panelIn1 {
        from { opacity: 0; transform: translateY(16px); }
        to { opacity: 1; transform: translateY(0); }
      }
      @keyframes panelIn2 {
        from { opacity: 0; transform: translateY(16px); }
        to { opacity: 1; transform: translateY(0); }
      }
      @keyframes panelIn3 {
        from { opacity: 0; transform: translateY(16px); }
        to { opacity: 1; transform: translateY(0); }
      }
      @keyframes cardIn1 {
        from { opacity: 0; transform: translateY(12px) scale(0.99); }
        to { opacity: 1; transform: translateY(0) scale(1); }
      }
      @keyframes cardIn2 {
        from { opacity: 0; transform: translateY(12px) scale(0.99); }
        to { opacity: 1; transform: translateY(0) scale(1); }
      }
      @keyframes cardIn3 {
        from { opacity: 0; transform: translateY(12px) scale(0.99); }
        to { opacity: 1; transform: translateY(0) scale(1); }
      }
      @keyframes kpiIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
      }
      /* Switching tabs changes the animation name, so the motion replays. */
      div[data-testid="stTabs"] { --panel-anim: panelIn1; --card-anim: cardIn1; }
      div[data-testid="stTabs"]:has([data-testid="stTab"]:nth-child(1)[aria-selected="true"])
        { --panel-anim: panelIn1; --card-anim: cardIn1; }
      div[data-testid="stTabs"]:has([data-testid="stTab"]:nth-child(2)[aria-selected="true"])
        { --panel-anim: panelIn2; --card-anim: cardIn2; }
      div[data-testid="stTabs"]:has([data-testid="stTab"]:nth-child(3)[aria-selected="true"])
        { --panel-anim: panelIn3; --card-anim: cardIn3; }
      div[data-testid="stTabs"] [data-testid="stTabPanel"] {
        animation: var(--panel-anim) 0.42s cubic-bezier(0.22, 1, 0.36, 1) backwards;
      }
      div[data-testid="stTabs"] [data-testid="stTabPanel"] [class*="st-key-card-"] {
        animation: var(--card-anim) 0.46s cubic-bezier(0.22, 1, 0.36, 1) backwards;
      }
      div[data-testid="stTabs"] [data-testid="stHorizontalBlock"] > div:nth-child(2)
        [class*="st-key-card-"] { animation-delay: 0.09s; }
      div[data-testid="stTabs"] [data-testid="stHorizontalBlock"] > div:nth-child(3)
        [class*="st-key-card-"] { animation-delay: 0.18s; }

      /* --------------------------------------------------------------- Form */
      div[data-testid="stForm"] button[data-testid="stBaseButton-primaryFormSubmit"] {
        background: linear-gradient(90deg, #2E86AB, #1F6A8A);
        border: none;
        border-radius: 10px;
        font-weight: 700;
        letter-spacing: 0.01em;
        padding: 0.5rem 1.4rem;
        box-shadow: 0 6px 16px rgba(46, 134, 171, 0.28);
      }
      div[data-testid="stForm"] button[data-testid="stBaseButton-primaryFormSubmit"]:hover {
        box-shadow: 0 8px 20px rgba(46, 134, 171, 0.38);
      }
      div[data-testid="stWidgetLabel"] p {
        font-weight: 600;
        color: var(--ink);
      }

      /* ---------------------------------------------------- Page subtitle */
      .page-sub {
        margin: 2px 0 18px 0;
        padding: 12px 16px;
        background: linear-gradient(90deg, #F1F7FB 0%, rgba(255, 255, 255, 0) 92%);
        border-left: 4px solid var(--brand);
        border-radius: 0 10px 10px 0;
        color: var(--muted);
        font-size: 0.95rem;
        line-height: 1.55;
      }
      .page-sub b { color: var(--ink); font-weight: 700; }
      /* ------------------------------------------------------- Risk card */
      .risk-card {
        font-family: inherit;
        border: 1px solid var(--line);
        border-radius: 16px;
        background: #FFFFFF;
        box-shadow: 0 8px 24px rgba(21, 48, 77, 0.08);
        overflow: hidden;
        margin: 0.2rem 0 0.8rem 0;
      }
      .risk-head {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 16px 22px;
        border-bottom: 1px solid var(--line);
        background: linear-gradient(90deg, #F6FAFC 0%, #FFFFFF 100%);
      }
      .risk-dot {
        width: 11px;
        height: 11px;
        border-radius: 50%;
        display: inline-block;
        flex: 0 0 auto;
      }
      .risk-head-title {
        font-size: 1.12rem;
        font-weight: 800;
        color: var(--ink);
        letter-spacing: -0.02em;
      }
      .risk-chip {
        margin-left: auto;
        padding: 6px 16px;
        border-radius: 999px;
        color: #FFFFFF;
        font-weight: 700;
        font-size: 0.9rem;
        letter-spacing: 0.01em;
        white-space: nowrap;
      }
      .risk-hero {
        display: flex;
        align-items: center;
        gap: 28px;
        flex-wrap: wrap;
        padding: 20px 22px 6px 22px;
      }
      .risk-prob-label {
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 2px;
      }
      .risk-prob-value {
        font-size: 2.75rem;
        font-weight: 800;
        line-height: 1.05;
        letter-spacing: -0.03em;
      }
      .risk-verdict-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--ink);
      }
      .risk-verdict-text { color: var(--muted); font-size: 0.95rem; }
      .risk-meter { padding: 14px 22px 2px 22px; }
      .risk-track {
        position: relative;
        display: flex;
        height: 14px;
        border-radius: 999px;
        background: #EEF2F6;
      }
      .risk-seg { height: 100%; }
      .risk-seg:first-child { border-radius: 999px 0 0 999px; }
      .risk-seg:last-child { border-radius: 0 999px 999px 0; }
      .risk-marker {
        position: absolute;
        top: -6px;
        width: 4px;
        height: 26px;
        border-radius: 2px;
        background: var(--ink);
        transform: translateX(-50%);
        box-shadow: 0 0 0 3px #FFFFFF;
      }
      .risk-scale {
        display: flex;
        justify-content: space-between;
        font-size: 0.76rem;
        color: var(--muted);
        margin-top: 9px;
      }
      .risk-body { padding: 14px 22px 18px 22px; }
      .risk-subhead {
        font-weight: 700;
        color: var(--ink);
        margin-bottom: 10px;
      }
      .risk-legend { list-style: none; margin: 0; padding: 0; }
      .risk-legend li {
        display: flex;
        gap: 10px;
        align-items: flex-start;
        margin-bottom: 7px;
        color: var(--ink);
        font-size: 0.94rem;
        line-height: 1.45;
      }
      .risk-legend li .risk-dot { margin-top: 6px; }
      .risk-foot {
        padding: 14px 22px;
        font-weight: 600;
        font-size: 0.95rem;
        border-top: 1px solid var(--line);
      }

      /* ------------------------------------------------------- Sidebar nav */
      [data-testid="stSidebarNavLink"] {
        border-radius: 10px;
        border: 1px solid var(--line);
        background: #FFFFFF;
        padding: 10px 14px;
        margin-bottom: 6px;
        font-weight: 600;
        color: var(--ink);
        box-shadow: 0 1px 2px rgba(21, 48, 77, 0.05);
        transition: transform 0.15s ease, box-shadow 0.15s ease,
          border-color 0.15s ease, background 0.15s ease;
      }
      [data-testid="stSidebarNavLink"] span[label] { color: inherit; }
      [data-testid="stSidebarNavLink"]:hover {
        border-color: var(--brand);
        color: var(--brand);
        box-shadow: 0 6px 14px rgba(46, 134, 171, 0.18);
        transform: translateY(-1px);
      }
      [data-testid="stSidebarNavLink"][aria-current="page"] {
        background: linear-gradient(90deg, #2E86AB, #1F6A8A);
        border-color: #1F6A8A;
        color: #FFFFFF;
        box-shadow: 0 6px 16px rgba(46, 134, 171, 0.30);
      }
      [data-testid="stSidebarNavLink"][aria-current="page"]:hover {
        color: #FFFFFF;
      }

      /* ------------------------------------------------------------ Sidebar */
      .brand-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: var(--ink);
        letter-spacing: -0.02em;
      }
      .brand-subtitle {
        font-size: 0.78rem;
        color: var(--muted);
        margin-top: 2px;
      }
      .filter-heading {
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--muted);
        margin: 0 0 6px 0;
      }
      hr { border-color: var(--line); }
"""

# ---------------------------------------------------------------------------
# Icons
# ---------------------------------------------------------------------------
# The icons are inline SVG (Material Design paths) that are handed to the
# browser as CSS masks. A mask only uses the alpha channel of the graphic, so
# a glyph always picks up the colour of the element it sits in — which means
# the tab icons turn white with the active tab on their own.

KPI_ICONS = {
    "total_loans": (
        '<path d="M12 2.4 22 7.9 12 13.4 2 7.9 12 2.4z"/>'
        '<path d="M2.6 11.8 12 17l9.4-5.2 1.7 1.1L12 18.6 1.9 12.9 2.6 11.8z"/>'
    ),
    "defaulted_loans": (
        '<path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z"/>'
    ),
    "default_rate": (
        '<path d="M11.99 2C6.47 2 2 6.48 2 12s4.47 10 9.99 10C17.52 22 22 17.52 '
        '22 12S17.52 2 11.99 2zm0 18c-4.42 0-8-3.58-8-8s3.58-8 8-8 8 3.58 8 8-3.58 '
        '8-8 8z"/>'
    ),
    "avg_amount": (
        '<path d="M21 18v1c0 1.1-.9 2-2 2H5c-1.11 0-2-.9-2-2V5c0-1.1.89-2 2-2h14c1.1 0 '
        '2 .9 2 2v1h-9c-1.11 0-2 .9-2 2v8c0 1.1.89 2 2 2h9zm-9-2h10V8H12v8zm4-2.5c-.83 '
        '0-1.5-.67-1.5-1.5s.67-1.5 1.5-1.5 1.5.67 1.5 1.5-.67 1.5-1.5 1.5z"/>'
    ),
}

TAB_ICONS = {
    "overview": '<path d="M3 13h8V3H3v10zm0 8h8v-6H3v6zm10 0h8V11h-8v10zm0-18v6h8V3h-8z"/>',
    "credit_risk": (
        '<path d="M20 4H4c-1.11 0-1.99.89-1.99 2L2 18c0 1.11.89 2 2 2h16c1.11 0 '
        '2-.89 2-2V6c0-1.11-.89-2-2-2zm0 14H4v-6h16v6zm0-10H4V6h16v2z"/>'
    ),
    "geography": (
        '<path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 '
        '9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 '
        '2.5z"/>'
    ),
}


def _icon_uri(markup: str) -> str:
    """Turn an inline SVG into a CSS ``url(...)`` that can be used as a mask."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
        f'fill="currentColor">{markup}</svg>'
    )
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f'url("data:image/svg+xml;base64,{encoded}")'


ICON_CSS = f"""
      /* ------------------------------------------------------- KPI icon colours */
      .st-key-kpi_row [data-testid="stHorizontalBlock"] > div:nth-child(1)
        {{ --kpi-accent: #2E86AB; --kpi-soft: #EAF3F8;
           --kpi-icon: {_icon_uri(KPI_ICONS["total_loans"])}; }}
      .st-key-kpi_row [data-testid="stHorizontalBlock"] > div:nth-child(2)
        {{ --kpi-accent: #E4572E; --kpi-soft: #FCEDE7;
           --kpi-icon: {_icon_uri(KPI_ICONS["defaulted_loans"])}; }}
      .st-key-kpi_row [data-testid="stHorizontalBlock"] > div:nth-child(3)
        {{ --kpi-accent: #F59E0B; --kpi-soft: #FDF3E2;
           --kpi-icon: {_icon_uri(KPI_ICONS["default_rate"])}; }}
      .st-key-kpi_row [data-testid="stHorizontalBlock"] > div:nth-child(4)
        {{ --kpi-accent: #7C6BD6; --kpi-soft: #EFEDFB;
           --kpi-icon: {_icon_uri(KPI_ICONS["avg_amount"])}; }}

      /* ---------------------------------------------------------- Tab icon glyphs */
      div[data-testid="stTabs"] [role="tablist"] > [data-testid="stTab"]:nth-child(1)
        {{ --tab-icon: {_icon_uri(TAB_ICONS["overview"])}; }}
      div[data-testid="stTabs"] [role="tablist"] > [data-testid="stTab"]:nth-child(2)
        {{ --tab-icon: {_icon_uri(TAB_ICONS["credit_risk"])}; }}
      div[data-testid="stTabs"] [role="tablist"] > [data-testid="stTab"]:nth-child(3)
        {{ --tab-icon: {_icon_uri(TAB_ICONS["geography"])}; }}
"""


def page_subtitle(html: str) -> None:
    """Render a styled subtitle line underneath the page title."""
    st.markdown(f'<div class="page-sub">{html}</div>', unsafe_allow_html=True)


def inject_styles() -> None:
    """Inject the global stylesheet into the current Streamlit page."""
    st.html(f"<style>{STYLE_CSS}{ICON_CSS}</style>")

# ---------------------------------------------------------------------------
# Band definitions
# ---------------------------------------------------------------------------
AMOUNT_BINS = [0, 5_000, 10_000, 15_000, 20_000, 25_000, 30_000, np.inf]
AMOUNT_LABELS = ["0-5k", "5k-10k", "10k-15k", "15k-20k", "20k-25k", "25k-30k", "30k+"]

FICO_BINS = [-np.inf, 580, 620, 660, 700, 740, np.inf]
FICO_LABELS = ["Below 580", "580-619", "620-659", "660-699", "700-739", "740+"]

DTI_BINS = [0, 10, 20, 30, 40, 50, np.inf]
DTI_LABELS = ["0-10", "10-20", "20-30", "30-40", "40-50", "50+"]

# Natural ordering of the employment-length field.
EMP_LENGTH_ORDER = [
    "< 1 year",
    "1 year",
    "2 years",
    "3 years",
    "4 years",
    "5 years",
    "6 years",
    "7 years",
    "8 years",
    "9 years",
    "10+ years",
    "Missing",
]

# The raw dataset marks "not reported" employment length with "NI".
EMP_LENGTH_RAW_MISSING = "NI"
EMP_LENGTH_MISSING_LABEL = "Missing"

STATE_NAMES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "DC": "District of Columbia", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii",
    "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine",
    "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska",
    "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
    "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island",
    "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas",
    "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
}


def state_name(abbreviation: str) -> str:
    """Full state name for an abbreviation, falling back to the abbreviation."""
    return STATE_NAMES.get(str(abbreviation).strip().upper(), str(abbreviation))


# ---------------------------------------------------------------------------
# Cached loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading loan data...")
def load_data() -> pd.DataFrame:
    """Read the cleaned loan data from parquet and validate its structure.

    The only derived column is ``issue_year``, which is needed to chart the
    default rate over time; no other transformation is applied.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Data file not found: {DATA_PATH}. Expected the cleaned loan "
            "dataset at data/loans_clean.parquet."
        )

    df = pd.read_parquet(DATA_PATH)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"{DATA_PATH.name} is missing required column(s): {missing}. "
            f"Available columns: {list(df.columns)}."
        )

    issued = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df["issue_year"] = issued.dt.year.astype("Int16")
    return df


@st.cache_resource(show_spinner="Loading the default model...")
def load_model():
    """Load the trained scikit-learn pipeline (preprocessor + random forest)."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. Expected the trained "
            "pipeline at model/default_model.joblib."
        )
    return joblib.load(MODEL_PATH)


@st.cache_resource(show_spinner="Loading feature list...")
def load_features() -> list[str]:
    """The exact column list the model was trained on."""
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURES_PATH}. Expected the model's "
            "training columns at model/features.joblib."
        )
    features = joblib.load(FEATURES_PATH)
    if not isinstance(features, (list, tuple)) or not features:
        raise ValueError(
            f"{FEATURES_PATH.name} should contain a non-empty list of column "
            f"names, got {type(features).__name__}."
        )
    return list(features)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
def apply_filters(
    df: pd.DataFrame,
    purposes: list[str] | None = None,
    year_range: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """Return the rows matching the active sidebar filters.

    An empty purpose selection means "all purposes".
    """
    mask = pd.Series(True, index=df.index)

    if purposes:
        mask &= df["purpose"].isin(purposes)

    if year_range is not None:
        start, end = year_range
        years = pd.to_numeric(df["issue_year"], errors="coerce")
        mask &= years.between(start, end)

    return df[mask]


# ---------------------------------------------------------------------------
# Grouped summaries
# ---------------------------------------------------------------------------
def summary_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Loan count, defaulted count and default rate for every value of a column."""
    grouped = (
        df.groupby(column, dropna=False)
        .agg(total_loans=("Default", "size"), defaulted=("Default", "sum"))
        .reset_index()
    )
    grouped["default_rate"] = np.where(
        grouped["total_loans"] > 0,
        grouped["defaulted"] / grouped["total_loans"] * 100.0,
        np.nan,
    )
    return grouped


def summary_by_bands(
    df: pd.DataFrame,
    column: str,
    bins: list,
    labels: list[str],
) -> pd.DataFrame:
    """Default-rate summary for a numeric field split into bands."""
    banded = df.assign(
        band=pd.cut(df[column], bins=bins, labels=labels, include_lowest=True)
    )
    summary = summary_by(banded, "band")
    summary["band"] = summary["band"].astype(str)
    # Keep the natural band order and drop bands without any loans.
    summary["order"] = pd.Categorical(summary["band"], categories=labels, ordered=True)
    summary = summary[summary["total_loans"] > 0].sort_values("order")
    return summary.drop(columns="order").reset_index(drop=True)


def summary_by_emp_length(df: pd.DataFrame) -> pd.DataFrame:
    """Default rate by employment length, in natural order with Missing last."""
    work = df.assign(
        emp_length=df["emp_length"].replace(
            {EMP_LENGTH_RAW_MISSING: EMP_LENGTH_MISSING_LABEL}
        )
    )
    summary = summary_by(work, "emp_length")
    order = {label: i for i, label in enumerate(EMP_LENGTH_ORDER)}
    summary["order"] = summary["emp_length"].map(order)
    summary = summary[summary["order"].notna()].sort_values("order")
    return summary.drop(columns="order").reset_index(drop=True)


def summary_by_purpose(df: pd.DataFrame, min_loans: int = MIN_SAMPLE_SIZE) -> pd.DataFrame:
    """Default rate by loan purpose, highest first, for purposes seen often enough."""
    summary = summary_by(df, "purpose")
    summary = summary[summary["total_loans"] >= min_loans]
    return summary.sort_values("default_rate", ascending=False).reset_index(drop=True)


def summary_top_states(
    df: pd.DataFrame,
    min_loans: int = MIN_SAMPLE_SIZE,
    n: int = 10,
) -> pd.DataFrame:
    """The ``n`` states with the highest default rate (minimum ``min_loans`` loans)."""
    summary = summary_by(df, "addr_state")
    summary = summary[summary["total_loans"] >= min_loans]
    summary = summary.sort_values("default_rate", ascending=False).head(n).reset_index(drop=True)
    summary["state_name"] = summary["addr_state"].map(state_name)
    return summary


def summary_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Default rate by loan issue year."""
    work = df.dropna(subset=["issue_year"])
    summary = (
        work.groupby("issue_year")
        .agg(total_loans=("Default", "size"), defaulted=("Default", "sum"))
        .reset_index()
    )
    summary["issue_year"] = summary["issue_year"].astype(int)
    summary["default_rate"] = (
        summary["defaulted"] / summary["total_loans"] * 100.0
    )
    return summary.sort_values("issue_year").reset_index(drop=True)


def portfolio_kpis(df: pd.DataFrame) -> dict:
    """Headline figures for the KPI row."""
    n = len(df)
    return {
        "total_loans": n,
        "defaulted_loans": int(df["Default"].sum()) if n else 0,
        "default_rate": float(df["Default"].mean() * 100.0) if n else float("nan"),
        "avg_loan_amount": float(df["loan_amnt"].mean()) if n else float("nan"),
    }


# ---------------------------------------------------------------------------
# Plotly theme
# ---------------------------------------------------------------------------
def _build_template() -> go.layout.Template:
    return go.layout.Template(
        layout=go.Layout(
            font=dict(family=FONT, size=13, color=TEXT),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            colorway=[PRIMARY, DEFAULT_RED, "#7FB2C7", "#F2A65A", "#4B7F52"],
            xaxis=dict(
                gridcolor=GRID, zeroline=False, linecolor=GRID,
                tickfont=dict(size=12, color=MUTED), title_font=dict(size=12.5, color=TEXT),
            ),
            yaxis=dict(
                gridcolor=GRID, zeroline=False, linecolor=GRID,
                tickfont=dict(size=12, color=MUTED), title_font=dict(size=12.5, color=TEXT),
            ),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=12, color=TEXT)),
            hoverlabel=dict(
                bgcolor="#FFFFFF", font_size=13, font_family=FONT, font_color=TEXT,
                bordercolor=GRID,
            ),
            margin=dict(l=56, r=28, t=56, b=46),
        )
    )


TEMPLATE_NAME = "loan_default_dashboard"
pio.templates[TEMPLATE_NAME] = _build_template()


def _finish(fig: go.Figure, *, height: int = 380, legend: bool = False) -> go.Figure:
    """Apply the shared template and layout to a figure."""
    fig.update_layout(
        template=TEMPLATE_NAME,
        height=height,
        showlegend=legend,
        bargap=0.3,
        hovermode="closest",
    )
    return fig


def _percent_labels(rates) -> list[str]:
    """Format a series of default rates as percentage labels."""
    return [
        "—" if pd.isna(rate) else f"{rate:.1f}%"
        for rate in rates
    ]


def _rate_hovertemplate(category_label: str, category_axis: str = "x") -> str:
    """Hover template shared by the default-rate charts.

    ``category_axis`` is the axis that holds the category labels: ``"x"``
    for the vertical charts and ``"y"`` for the horizontal ones.
    """
    return (
        f"{category_label}: %{{{category_axis}}}\n"
        "Default rate: %{customdata[0]:.2f}%\n"
        "Total loans: %{customdata[1]:,}\n"
        "Defaulted: %{customdata[2]:,}"
        "<extra></extra>"
    )


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def chart_outcomes(df: pd.DataFrame) -> go.Figure:
    """Distribution of loan outcomes: fully paid versus defaulted."""
    total = len(df)
    paid = int((df["Default"] == 0).sum())
    defaulted = int((df["Default"] == 1).sum())
    rate = (defaulted / total * 100.0) if total else float("nan")

    summary = pd.DataFrame(
        {
            "Outcome": [PAID_LABEL, DEFAULT_LABEL],
            "Loans": [paid, defaulted],
            "Colour": [PRIMARY, DEFAULT_RED],
        }
    )
    summary["share"] = summary["Loans"] / total * 100.0 if total else np.nan
    summary["default_rate"] = [rate, rate]

    fig = px.bar(
        summary,
        x="Outcome",
        y="Loans",
        color="Outcome",
        color_discrete_map={PAID_LABEL: PRIMARY, DEFAULT_LABEL: DEFAULT_RED},
        text=summary["Loans"].map(lambda v: f"{v:,}"),
        custom_data=["default_rate", "Loans", "share"],
        labels={"Loans": "Number of loans", "Outcome": ""},
    )
    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            "%{x}<br>Loans: %{customdata[1]:,}<br>"
            "Share of selection: %{customdata[2]:.2f}%<br>"
            "Default rate of selection: %{customdata[0]:.2f}%"
            "<extra></extra>"
        ),
    )
    top = max(summary["Loans"].max(), 1)
    fig.update_yaxes(range=[0, top * 1.18])
    fig.update_layout(showlegend=False)
    return _finish(fig, height=380)


def chart_default_rate_bars(summary: pd.DataFrame, x_col: str = "band") -> go.Figure:
    """Vertical bars of default rate per band, with percentage labels."""
    fig = px.bar(
        summary,
        x=x_col,
        y="default_rate",
        text=_percent_labels(summary["default_rate"]),
        color_discrete_sequence=[PRIMARY],
        custom_data=["default_rate", "total_loans", "defaulted"],
        labels={"default_rate": "Default rate (%)", x_col: ""},
    )
    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
        hovertemplate=_rate_hovertemplate(x_col.replace("_", " ").title()),
    )
    top = float(np.nanmax(summary["default_rate"])) if len(summary) else 1.0
    fig.update_yaxes(range=[0, max(top * 1.25, 1.0)])
    return _finish(fig, height=380)


def chart_horizontal_rates(
    summary: pd.DataFrame,
    y_col: str,
    category_label: str,
) -> go.Figure:
    """Horizontal bars of default rate for a ranked categorical field."""
    fig = px.bar(
        summary,
        x="default_rate",
        y=y_col,
        orientation="h",
        text=_percent_labels(summary["default_rate"]),
        color_discrete_sequence=[PRIMARY],
        custom_data=["default_rate", "total_loans", "defaulted"],
        labels={"default_rate": "Default rate (%)", y_col: ""},
    )
    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
        hovertemplate=_rate_hovertemplate(category_label, category_axis="y"),
    )
    top = float(np.nanmax(summary["default_rate"])) if len(summary) else 1.0
    fig.update_xaxes(range=[0, max(top * 1.28, 1.0)])
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return _finish(fig, height=380)


def chart_year_trend(summary: pd.DataFrame) -> go.Figure:
    """Default rate by loan issue year."""
    fig = px.line(
        summary,
        x="issue_year",
        y="default_rate",
        markers=True,
        color_discrete_sequence=[PRIMARY],
        custom_data=["default_rate", "total_loans", "defaulted"],
        labels={"issue_year": "Loan issue year", "default_rate": "Default rate (%)"},
    )
    fig.update_traces(
        line=dict(width=2.5),
        marker=dict(size=7),
        hovertemplate=(
            "Issue year: %{x}<br>"
            "Default rate: %{customdata[0]:.2f}%<br>"
            "Total loans: %{customdata[1]:,}<br>"
            "Defaulted: %{customdata[2]:,}"
            "<extra></extra>"
        ),
    )
    top = float(np.nanmax(summary["default_rate"])) if len(summary) else 1.0
    fig.update_yaxes(range=[0, max(top * 1.2, 1.0)])
    return _finish(fig, height=380)


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------
def raw_feature_importance(model) -> pd.DataFrame:
    """Aggregate the random forest's encoded importances back to raw columns.

    Returns a DataFrame with the columns ``feature`` and ``importance``,
    sorted from most to least important.
    """
    preprocessor = model.named_steps["preprocessor"]
    forest = model.named_steps["model"]

    encoded_names = list(preprocessor.get_feature_names_out())
    importances = np.asarray(forest.feature_importances_, dtype=float)
    if len(encoded_names) != len(importances):
        raise ValueError(
            "The preprocessor produced "
            f"{len(encoded_names)} feature names but the forest has "
            f"{len(importances)} importances; they cannot be matched."
        )

    totals: dict[str, float] = {}
    for name, value in zip(encoded_names, importances):
        _, raw = name.split("__", 1)
        base = raw
        for column in preprocessor.transformers:
            if len(column) == 3 and column[2] and isinstance(column[2], (list, tuple)):
                for candidate in column[2]:
                    if raw == candidate or raw.startswith(f"{candidate}_"):
                        base = candidate
                        break
        totals[base] = totals.get(base, 0.0) + float(value)

    frame = pd.DataFrame(
        {"feature": list(totals.keys()), "importance": list(totals.values())}
    )
    return frame.sort_values("importance", ascending=False).reset_index(drop=True)


def column_defaults(df: pd.DataFrame) -> dict[str, object]:
    """Default value per column: the median for numbers, the mode otherwise."""
    defaults: dict[str, object] = {}
    for column in df.columns:
        series = df[column].dropna()
        if pd.api.types.is_numeric_dtype(series):
            defaults[column] = float(series.median()) if len(series) else 0.0
        else:
            counts = series.value_counts()
            defaults[column] = counts.index[0] if len(counts) else ""
    return defaults


def column_options(df: pd.DataFrame, column: str) -> list:
    """Sorted unique values of a categorical column, as plain Python objects."""
    return sorted(df[column].dropna().unique().tolist(), key=lambda v: str(v))


def column_limits(df: pd.DataFrame, column: str) -> tuple[float, float]:
    """Observed minimum and maximum of a numeric column."""
    series = df[column].dropna()
    if not len(series):
        return (0.0, 1.0)
    return (float(series.min()), float(series.max()))


# ---------------------------------------------------------------------------
# Risk bands
# ---------------------------------------------------------------------------
# The three bands used everywhere in the app: the colour, the wording and the
# boundaries.  Green = safe, amber = watch it, red = act now.
RISK_LOW = 0.20
RISK_HIGH = 0.50

RISK_GREEN = "#22C55D"
RISK_AMBER = "#F59E0B"
RISK_RED = "#EF4444"

RISK_BANDS = [
    {
        "label": "Low risk",
        "colour": RISK_GREEN,
        "low": 0.0,
        "high": RISK_LOW,
        "legend": "comfortably safe, routine monitoring",
        "verdict": "Likely to be repaid",
        "description": "Low likelihood of default",
        "foot_bg": "#EAF9F0",
        "foot_fg": "#14663A",
    },
    {
        "label": "Medium risk",
        "colour": RISK_AMBER,
        "low": RISK_LOW,
        "high": RISK_HIGH,
        "legend": "uncertain, worth a preventive check-in",
        "verdict": "Worth a closer look",
        "description": "Moderate likelihood of default",
        "foot_bg": "#FEF6E7",
        "foot_fg": "#8A5300",
    },
    {
        "label": "High risk",
        "colour": RISK_RED,
        "low": RISK_HIGH,
        "high": 1.0,
        "legend": "act now, escalate for manual review",
        "verdict": "Likely to default",
        "description": "High likelihood of default",
        "foot_bg": "#FDECEC",
        "foot_fg": "#8C1D1D",
    },
]


def risk_profile(probability: float) -> dict:
    """Describe a default probability: band, colour, wording and footer."""
    for band in RISK_BANDS:
        if probability < band["high"]:
            return dict(band)
    return dict(RISK_BANDS[-1])


def risk_label(probability: float) -> tuple[str, str, str]:
    """Map a default probability to (label, hex colour, description)."""
    band = risk_profile(probability)
    return band["label"], band["colour"], band["description"]


def _band_range_text(band: dict) -> str:
    """Human readable boundaries of a band, e.g. 'below 20%' or '20% to 50%'."""
    low = band["low"] * 100
    high = band["high"] * 100
    if band["low"] == 0.0:
        return f"below {high:.0f}%"
    if band["high"] >= 1.0:
        return f"above {low:.0f}%"
    return f"{low:.0f}% to {high:.0f}%"


def risk_assessment_html(probability: float) -> str:
    """Build the coloured risk-assessment card shown after a prediction.

    The layout mirrors a simple assessment summary: a headline with the
    verdict and the probability, a segmented risk meter with a marker at the
    predicted probability, an explanation of the three bands and a coloured
    summary strip at the bottom.
    """
    band = risk_profile(probability)
    percent = probability * 100

    segments = "".join(
        '<div class="risk-seg" style="width:{:.0f}%;background:{};"></div>'.format(
            (item["high"] - item["low"]) * 100, item["colour"]
        )
        for item in RISK_BANDS
    )
    legend = "".join(
        '<li><span class="risk-dot" style="background:{};"></span>'
        "<span><b>{}</b> ({})&nbsp;&middot;&nbsp;{}</span></li>".format(
            item["colour"], item["label"], _band_range_text(item), item["legend"]
        )
        for item in RISK_BANDS
    )

    return f"""
<div class="risk-card">
  <div class="risk-head">
    <span class="risk-dot" style="background:{band['colour']};"></span>
    <span class="risk-head-title">Default Risk Assessment</span>
    <span class="risk-chip" style="background:{band['colour']};">{band['label']}</span>
  </div>

  <div class="risk-hero">
    <div class="risk-prob">
      <div class="risk-prob-label">Predicted probability of default</div>
      <div class="risk-prob-value" style="color:{band['colour']};">{percent:.2f}%</div>
    </div>
    <div class="risk-verdict">
      <div class="risk-verdict-title">{band['verdict']}</div>
      <div class="risk-verdict-text">{band['description']}</div>
    </div>
  </div>

  <div class="risk-meter">
    <div class="risk-track">
      {segments}
      <div class="risk-marker" style="left:{percent:.2f}%;"></div>
    </div>
    <div class="risk-scale">
      <span>0%</span><span>{RISK_LOW * 100:.0f}%</span>
      <span>{RISK_HIGH * 100:.0f}%</span><span>100%</span>
    </div>
  </div>

  <div class="risk-body">
    <div class="risk-subhead">What the risk levels mean</div>
    <ul class="risk-legend">{legend}</ul>
  </div>

  <div class="risk-foot" style="background:{band['foot_bg']};color:{band['foot_fg']};">
    This application currently shows a {band['label'].lower()} estimated default risk.
  </div>
</div>
""".strip()
