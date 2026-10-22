"""Loan Risk Analytics — multi-page Streamlit dashboard.

Launch with::

    streamlit run app.py

Pages
-----
``Dashboard``
    KPI summary and the eight default-rate charts, driven by the sidebar
    filters (loan purpose and loan issue year).
``Predict Default``
    Score a single loan application with the trained model and get a
    green / amber / red risk assessment.

Shared code lives in :mod:`utils`; the pages themselves are in ``pages/``.
"""

from __future__ import annotations

import streamlit as st

# Force light theme only - no dark mode
st.set_page_config(
    page_title="Loan Risk Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject CSS to force light backgrounds
st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .stApp {
        background-color: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #E7EBEA !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

import utils

utils.inject_styles()


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
nav = st.navigation(
    [
        st.Page("pages/dashboard.py", title="Dashboard", default=True),
        st.Page("pages/predict_default.py", title="Predict Default"),
    ]
)

# ---------------------------------------------------------------------------
# Sidebar: branding and the filters shared by the dashboard charts
# ---------------------------------------------------------------------------
with st.sidebar:
    # The stylesheet is injected from the sidebar as well, because the main
    # script's own elements are not rendered when a page other than the
    # default one is selected.
    utils.inject_styles()
    st.markdown(
        '<div class="brand-title">Loan Risk Analytics</div>'
        '<div class="brand-subtitle">Loan Default Dashboard · 2007–2018</div>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown('<div class="filter-heading">Filters</div>', unsafe_allow_html=True)

    try:
        loans = utils.load_data()
        purpose_options = sorted(loans["purpose"].dropna().unique().tolist())
        year_options = sorted(loans["issue_year"].dropna().astype(int).unique().tolist())
    except Exception as exc:  # noqa: BLE001 - surfaced to the user
        st.error(f"The filters are unavailable: {exc}")
    else:
        st.multiselect(
            "Loan purpose",
            purpose_options,
            key="filter_purposes",
        )
        st.slider(
            "Loan issue year",
            year_options[0],
            year_options[-1],
            (year_options[0], year_options[-1]),
            key="filter_years",
        )
        st.caption(
            "The filters narrow the results shown on the Dashboard page."
        )

# ---------------------------------------------------------------------------
# Render the selected page
# ---------------------------------------------------------------------------
nav.run()
