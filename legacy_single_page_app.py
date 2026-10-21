"""Loan Risk Analytics — interactive Streamlit dashboard.

Launch with:  streamlit run app.py

The charts mirror the analysis in ``Loan Risk Analytics.ipynb``; every
figure and KPI reacts to the sidebar filters.
"""

import numpy as np
import streamlit as st

import styles
from dashboard import (
    render_credit_risk,
    render_explorer,
    render_geography,
    render_overview,
    render_time_trends,
)
from loan_data import apply_filters, load_dataset
from prediction_page import render_predictor

st.set_page_config(page_title="Loan Risk Analytics", layout="wide")
styles.inject_css()

SECTIONS = {
    "Portfolio Overview": render_overview,
    "Credit Risk Analysis": render_credit_risk,
    "Loan Purpose and Geography": render_geography,
    "Time Trends": render_time_trends,
    "Data Explorer": render_explorer,
    "Default Risk Predictor": render_predictor,
}

FILTER_WIDGET_KEYS = ("f_purpose", "f_state", "f_loan", "f_fico", "f_dti", "f_date")


def reset_filters() -> None:
    """Callback: clear every filter widget back to its default."""
    for key in FILTER_WIDGET_KEYS:
        st.session_state.pop(key, None)


# ---------------------------------------------------------------------------
# Load data (cached) with clear error messages
# ---------------------------------------------------------------------------
try:
    DF, REPORT = load_dataset()
except (FileNotFoundError, ValueError) as exc:
    st.error("The loan dataset could not be loaded.")
    st.code(str(exc))
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="side-title">Loan Risk Analytics</div>'
        '<div class="side-subtitle">Loan default risk monitoring · Jun 2007 – Dec 2018</div>',
        unsafe_allow_html=True,
    )
    section = st.radio("Section", list(SECTIONS), label_visibility="collapsed")

    st.markdown('<div class="filter-heading">Filters</div>', unsafe_allow_html=True)

    purpose_options = DF["purpose"].value_counts().index.tolist()
    state_options = DF["addr_state"].value_counts().index.tolist()

    purposes = st.multiselect(
        "Loan purpose (empty = all)", purpose_options, key="f_purpose"
    )
    states = st.multiselect(
        "State (empty = all)", state_options, key="f_state"
    )

    loan_min, loan_max = int(DF["loan_amnt"].min()), int(DF["loan_amnt"].max())
    loan_range = st.slider(
        "Loan amount ($)", loan_min, loan_max, (loan_min, loan_max),
        step=500, key="f_loan", format="$%d",
    )

    fico_min = int(np.floor(DF["fico_n"].min()))
    fico_max = int(np.ceil(DF["fico_n"].max()))
    fico_range = st.slider(
        "FICO score", fico_min, fico_max, (fico_min, fico_max), key="f_fico"
    )

    dti_max = int(np.ceil(DF["dti_n"].max()))
    dti_range = st.slider(
        "Debt-to-income (DTI)", 0, dti_max, (0, dti_max), key="f_dti"
    )

    date_min = REPORT["date_min"].date()
    date_max = REPORT["date_max"].date()
    date_range = st.date_input(
        "Loan issue date range", (date_min, date_max),
        min_value=date_min, max_value=date_max, key="f_date",
    )

    st.button("Reset filters", on_click=reset_filters, width="stretch")

    with st.expander("Data quality notes"):
        st.markdown(
            f"""
- Source file: `{REPORT['file']}`
- Rows loaded: {REPORT['rows_loaded']:,}
- Unique loan ids: {REPORT['unique_ids']:,} ({REPORT['duplicate_ids']:,} duplicate ids)
- Target `Default` validated as binary 0/1 ({REPORT['unexpected_target_values']:,} unexpected values excluded)
- Issue dates parsed: {REPORT['rows_loaded'] - REPORT['unparsed_dates']:,} of {REPORT['rows_loaded']:,} ({REPORT['unparsed_dates']:,} unparsed)
- Date span: {REPORT['date_min']:%b %Y} – {REPORT['date_max']:%b %Y}
- Missing loan titles filled with "Unknown": {REPORT['title_missing']:,}
"""
        )

    st.caption(
        "Source: LC loans granting model dataset (Zenodo). "
        "Default = 1 is a defaulted loan, 0 is fully paid."
    )

# ---------------------------------------------------------------------------
# Apply filters and render the selected section
# ---------------------------------------------------------------------------
# st.date_input can return fewer than two dates while the user is editing;
# in that case the full date range is used.
active_dates = date_range if isinstance(date_range, (list, tuple)) and len(date_range) == 2 else None

filtered = apply_filters(DF, purposes, states, loan_range, fico_range, dti_range, active_dates)

st.title(section)
st.caption("All figures below reflect the current sidebar filter selection.")

if filtered.empty:
    st.warning(
        "No loans match the current filters. Broaden the selection or use "
        "'Reset filters' in the sidebar."
    )
    st.stop()

SECTIONS[section](filtered, DF)
