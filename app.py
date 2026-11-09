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

import numpy as np
import streamlit as st

# Force light theme only - no dark mode
st.set_page_config(
    page_title="Loan Risk Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Force light mode everywhere, prevent theme switching
st.markdown(
    """
    <style>
    :root {
        color-scheme: light only !important;
    }
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .stApp,
    [data-testid="stHeader"], section.main {
        background-color: #FFFFFF !important;
        color-scheme: light only !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #E7EBEA !important;
    }
    /* Force all text to be visible in light mode */
    * {
        color-scheme: light only !important;
    }
    /* Hide theme selector if it exists */
    [data-testid="stThemeSelector"], [data-testid="baseButton-headerNoPadding"] {
        display: none !important;
    }
    </style>
    <script>
    // Force light mode
    document.documentElement.setAttribute('data-theme', 'light');
    localStorage.setItem('stActiveTheme', '"Light"');
    </script>
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
        amount_low = int(np.floor(loans["loan_amnt"].min()))
        amount_high = int(np.ceil(loans["loan_amnt"].max()))
        fico_low = int(np.floor(loans["fico_n"].min()))
        fico_high = int(np.ceil(loans["fico_n"].max()))
        dti_low = int(np.floor(loans["dti_n"].min()))
        dti_high = int(np.ceil(loans["dti_n"].max()))
        home_options = sorted(loans["home_ownership_n"].dropna().unique().tolist())
        state_options = sorted(loans["addr_state"].dropna().unique().tolist())
        emp_present = set(loans["emp_length"].dropna().astype(str).unique())
        emp_options = [
            label for label in utils.EMP_LENGTH_ORDER
            if label in emp_present
            or (label == utils.EMP_LENGTH_MISSING_LABEL
                and utils.EMP_LENGTH_RAW_MISSING in emp_present)
        ]
    except Exception as exc:  # noqa: BLE001 - surfaced to the user
        st.error(f"The filters are unavailable: {exc}")
    else:
        with st.expander("Loan details", expanded=True):
            st.multiselect(
                "Loan purpose",
                purpose_options,
                key="filter_purposes",
                placeholder="All purposes",
            )
            st.slider(
                "Loan issue year",
                year_options[0],
                year_options[-1],
                (year_options[0], year_options[-1]),
                key="filter_years",
            )
            st.slider(
                "Loan amount ($)",
                amount_low,
                amount_high,
                (amount_low, amount_high),
                step=500,
                key="filter_amount",
            )
        with st.expander("Borrower profile", expanded=False):
            st.slider(
                "FICO score",
                fico_low,
                fico_high,
                (fico_low, fico_high),
                key="filter_fico",
            )
            st.slider(
                "Debt-to-income ratio",
                dti_low,
                dti_high,
                (dti_low, dti_high),
                key="filter_dti",
            )
            st.multiselect(
                "Employment length",
                emp_options,
                key="filter_emp",
                placeholder="All lengths",
            )
            st.multiselect(
                "Home ownership",
                home_options,
                key="filter_home",
                placeholder="All types",
            )
        with st.expander("Location", expanded=False):
            st.multiselect(
                "State",
                state_options,
                key="filter_states",
                placeholder="All states",
            )
        st.caption(
            "Every slicer narrows all of the figures on the Dashboard page."
        )

# ---------------------------------------------------------------------------
# Render the selected page
# ---------------------------------------------------------------------------
nav.run()
