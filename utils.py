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
PRIMARY = "#148553"      # main colour: fully paid / primary series
SECONDARY = "#8CB9A5"    # muted green: the second series (defaulted)
TEXT = "#111111"
MUTED = "#7A7F87"
GRID = "#E4E4E6"
FONT = "Inter, 'Helvetica Neue', Helvetica, Arial, sans-serif"

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
      @import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap");

      :root {
        --brand: #148553;
        --brand-dark: #0F7A4C;
        --brand-soft: #E4F2EC;
        --brand-line: #BFE3D0;
        --danger: #D64545;
        --ink: #111111;
        --muted: #7A7F87;
        --line: #E4E4E6;
        --tile: #F3F3F3;
        --page: #DDDDDF;
        --green: #148553;
        --amber: #E0A200;
        --red: #D64545;
      }

      /* Typography: one clean grotesque across the whole app. */
      html, body, [class*="css"], [data-testid="stApp"], [data-testid="stApp"] p,
      [data-testid="stApp"] h1, [data-testid="stApp"] h2, [data-testid="stApp"] h3,
      [data-testid="stApp"] span, [data-testid="stApp"] div, [data-testid="stApp"] li,
      [data-testid="stApp"] label, button, input, select, textarea {
        font-family: "Inter", "Helvetica Neue", Helvetica, Arial, sans-serif !important;
      }

      /* App frame: a white sheet floating on a soft grey backdrop. */
      [data-testid="stAppViewContainer"] { background: var(--page); }
      section.main, [data-testid="stMain"] { background: transparent; }
      .block-container {
        background: #FFFFFF;
        border-radius: 18px;
        box-shadow: 0 18px 46px rgba(17, 17, 17, 0.13);
        padding: 2.1rem 2.4rem 2.8rem 2.4rem;
        margin: 1.1rem 0 1.1rem 0;
        max-width: 1200px;
      }
      h1, h2, h3 { color: var(--ink); letter-spacing: -0.02em; }
      h1 {
        font-size: 2.15rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.035em !important;
        line-height: 1.15 !important;
        color: var(--ink) !important;
        margin: 0 0 2px 0 !important;
      }
      h3 {
        font-size: 0.98rem !important;
        font-weight: 700 !important;
        color: var(--ink) !important;
        padding-left: 0 !important;
        border-left: none !important;
        margin: 0 0 0.25rem 0 !important;
        letter-spacing: -0.012em;
      }
      [data-testid="stMain"] p { color: var(--muted); }
      [data-testid="stMain"] [data-testid="stCaptionContainer"] p { font-size: 0.84rem; }
      [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: var(--muted);
        font-size: 0.8rem;
        line-height: 1.55;
      }

      /* ---------------------------------------------------------------- KPI */
      div[data-testid="stMetric"] {
        position: relative;
        background: var(--tile);
        border: none;
        border-radius: 12px;
        padding: 16px 18px 14px 18px;
        box-shadow: none;
        transition: transform 0.15s ease, background 0.15s ease;
        animation: kpiIn 0.45s cubic-bezier(0.22, 1, 0.36, 1) backwards;
      }
      div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        background: #EFEFEF;
      }
      [data-testid="stMetricLabel"] p {
        font-size: 0.71rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.09em !important;
        text-transform: uppercase;
        color: var(--muted) !important;
      }
      div[data-testid="stMetricValue"] {
        color: var(--ink) !important;
        font-weight: 700 !important;
        letter-spacing: -0.03em;
        font-size: 1.85rem;
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
        border-radius: 12px;
        padding: 16px 18px 6px 18px;
        box-shadow: none;
        transition: box-shadow 0.15s ease, transform 0.15s ease;
      }
      [class*="st-key-card-"]:hover {
        box-shadow: 0 10px 24px rgba(17, 17, 17, 0.06);
        transform: translateY(-2px);
      }

      /* ---------------------------------------------------------------- Tabs */
      div[data-testid="stTabs"] [role="tablist"] {
        position: relative;
        display: flex;
        gap: 0;
        background: var(--tile);
        border: none;
        border-radius: 10px;
        padding: 4px;
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
        box-sizing: border-box;
        border-radius: 8px;
        background: #FFFFFF;
        border: 1px solid var(--line);
        box-shadow: 0 2px 6px rgba(17, 17, 17, 0.07);
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
        border-radius: 8px;
        padding: 0.5rem 0.9rem;
        font-weight: 600;
        color: var(--muted);
        transition: color 0.2s ease;
      }
      div[data-testid="stTabs"] [data-testid="stTab"]:hover { color: var(--ink); }
      div[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"],
      div[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"]:hover
        { color: var(--ink); font-weight: 700; }
      div[data-testid="stTabs"] [data-testid="stTab"] p {
        color: inherit;
        font-weight: inherit;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

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
        background: var(--brand);
        border: none;
        border-radius: 8px;
        font-weight: 600;
        letter-spacing: 0;
        padding: 0.5rem 1.4rem;
        box-shadow: none;
        transition: background 0.15s ease;
      }
      div[data-testid="stForm"] button[data-testid="stBaseButton-primaryFormSubmit"]:hover {
        background: var(--brand-dark);
        color: #FFFFFF;
      }
      div[data-testid="stWidgetLabel"] p {
        font-weight: 600;
        color: var(--ink);
      }

      /* ---------------------------------------------------- Page subtitle */
      .page-sub {
        margin: 12px 0 22px 0;
        padding: 14px 18px;
        background: var(--tile);
        border: none;
        border-radius: 12px;
        color: var(--muted);
        font-size: 0.95rem;
        line-height: 1.6;
      }
      .page-sub b { color: var(--ink); font-weight: 700; }
      /* ------------------------------------------------------- Risk card */
      .risk-card {
        font-family: inherit;
        border: 1px solid var(--line);
        border-radius: 14px;
        background: #FFFFFF;
        box-shadow: none;
        overflow: hidden;
        margin: 0.2rem 0 0.8rem 0;
      }
      .risk-head {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 16px 22px;
        border-bottom: 1px solid var(--line);
        background: #FFFFFF;
      }
      .risk-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
        flex: 0 0 auto;
      }
      .risk-head-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--ink);
        letter-spacing: -0.02em;
      }
      .risk-chip {
        margin-left: auto;
        padding: 6px 16px;
        border-radius: 999px;
        color: #FFFFFF;
        font-weight: 600;
        font-size: 0.88rem;
        letter-spacing: 0;
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
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 2px;
      }
      .risk-prob-value {
        font-size: 2.6rem;
        font-weight: 700;
        line-height: 1.05;
        letter-spacing: -0.03em;
      }
      .risk-verdict-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--ink);
      }
      .risk-verdict-text { color: var(--muted); font-size: 0.93rem; }
      .risk-meter { padding: 14px 22px 2px 22px; }
      .risk-track {
        position: relative;
        display: flex;
        height: 12px;
        border-radius: 999px;
        background: #ECECEC;
      }
      .risk-seg { height: 100%; }
      .risk-seg:first-child { border-radius: 999px 0 0 999px; }
      .risk-seg:last-child { border-radius: 0 999px 999px 0; }
      .risk-marker {
        position: absolute;
        top: -6px;
        width: 4px;
        height: 24px;
        border-radius: 2px;
        background: var(--ink);
        transform: translateX(-50%);
        box-shadow: 0 0 0 3px #FFFFFF;
      }
      .risk-scale {
        display: flex;
        justify-content: space-between;
        font-size: 0.74rem;
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
        font-size: 0.92rem;
        line-height: 1.45;
      }
      .risk-legend li .risk-dot { margin-top: 6px; }
      .risk-foot {
        padding: 14px 22px;
        font-weight: 600;
        font-size: 0.93rem;
        border-top: 1px solid var(--line);
      }

      /* ------------------------------------------------------- Sidebar nav */
      [data-testid="stSidebarNavLink"] {
        border-radius: 8px;
        border: 1px solid var(--line);
        background: #FFFFFF;
        padding: 9px 12px;
        margin-bottom: 4px;
        font-weight: 600;
        color: var(--ink);
        box-shadow: none;
        transition: transform 0.15s ease, border-color 0.15s ease,
          background 0.15s ease, color 0.15s ease;
      }
      [data-testid="stSidebarNavLink"] span[label] { color: inherit; }
      [data-testid="stSidebarNavLink"]:hover {
        border-color: #C9C9CC;
        background: #FAFAFA;
        color: var(--ink);
        transform: none;
      }
      [data-testid="stSidebarNavLink"][aria-current="page"] {
        background: var(--brand-soft);
        border-color: var(--brand-line);
        color: var(--brand-dark);
        font-weight: 700;
        box-shadow: none;
      }
      [data-testid="stSidebarNavLink"][aria-current="page"]:hover {
        color: var(--brand-dark);
      }

      /* ------------------------------------------------------------ Sidebar */
      section[data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid var(--line);
      }
      .brand-title {
        font-size: 1.08rem;
        font-weight: 700;
        color: var(--ink);
        letter-spacing: -0.02em;
      }
      .brand-subtitle {
        font-size: 0.76rem;
        color: var(--muted);
        margin-top: 2px;
      }
      .filter-heading {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--muted);
        margin: 0 0 6px 0;
      }
      hr { border-color: var(--line); }
"""

def page_subtitle(html: str) -> None:
    """Render a styled subtitle line underneath the page title."""
    st.markdown(f'<div class="page-sub">{html}</div>', unsafe_allow_html=True)


def inject_styles() -> None:
    """Inject the global stylesheet into the current Streamlit page."""
    st.html(f"<style>{STYLE_CSS}</style>")

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
            font=dict(family=FONT, size=12.5, color="#333333"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            colorway=[PRIMARY, SECONDARY, "#0F7A4C", "#C7E9D8", "#5B8C74"],
            xaxis=dict(
                showgrid=False, zeroline=False, linecolor=GRID,
                tickfont=dict(size=11.5, color=MUTED), title_font=dict(size=12, color="#333333"),
            ),
            yaxis=dict(
                showgrid=False, zeroline=False, linecolor=GRID,
                tickfont=dict(size=11.5, color=MUTED), title_font=dict(size=12, color="#333333"),
            ),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11.5, color=TEXT)),
            hoverlabel=dict(
                bgcolor="#FFFFFF", font_size=12.5, font_family=FONT, font_color=TEXT,
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
            "Colour": [PRIMARY, SECONDARY],
        }
    )
    summary["share"] = summary["Loans"] / total * 100.0 if total else np.nan
    summary["default_rate"] = [rate, rate]

    fig = px.bar(
        summary,
        x="Outcome",
        y="Loans",
        color="Outcome",
        color_discrete_map={PAID_LABEL: PRIMARY, DEFAULT_LABEL: SECONDARY},
        text=summary["Loans"].map(lambda v: f"{v:,}"),
        custom_data=["default_rate", "Loans", "share"],
        labels={"Loans": "Number of loans", "Outcome": ""},
    )
    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
        textfont=dict(color=TEXT, size=12),
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
        textfont=dict(color=TEXT, size=12),
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
        textfont=dict(color=TEXT, size=12),
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

RISK_GREEN = "#148553"
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
        "foot_bg": "#E4F2EC",
        "foot_fg": "#0F7A4C",
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
