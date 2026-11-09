"""Loan Repayment Risk — dashboard and prediction app.

Launch with::

    streamlit run app.py

The analysis behind it is ``Loan Repayment Risk Analysis Report.ipynb``: 9,578
LendingClub loans scored for whether they will **not be fully paid**.  The
dashboard reproduces the notebook's figures and slicers; the prediction page
uses the notebook's best model, the tuned XGBoost pipeline.

Pages
-----
``Dashboard``
    KPI summary and the report's figures, driven by nine slicers.
``Predict Default``
    Score a single loan application with the tuned XGBoost model.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Loan Repayment Risk",
    page_icon="◧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Force light theme only - no dark mode.
st.markdown(
    """
    <style>
    :root { color-scheme: light only !important; }
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .stApp,
    [data-testid="stHeader"], section.main {
        background-color: #FFFFFF !important;
        color-scheme: light only !important;
    }
    section[data-testid="stSidebar"] { background-color: #E7EBEA !important; }
    * { color-scheme: light only !important; }
    [data-testid="stThemeSelector"], [data-testid="baseButton-headerNoPadding"] {
        display: none !important;
    }
    </style>
    <script>
    document.documentElement.setAttribute('data-theme', 'light');
    localStorage.setItem('stActiveTheme', '"Light"');
    </script>
    """,
    unsafe_allow_html=True,
)

import loan_report
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
# Sidebar: branding and the slicers shared by the dashboard figures
# ---------------------------------------------------------------------------
with st.sidebar:
    utils.inject_styles()
    st.markdown(
        '<div class="brand-title">Loan Repayment Risk</div>'
        '<div class="brand-subtitle">LendingClub book · 9,578 loans</div>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown('<div class="filter-heading">Slicers</div>', unsafe_allow_html=True)

    try:
        loans = loan_report.load_data()
        purpose_options = sorted(loans["purpose"].dropna().unique().tolist())
        fico_low = int(np.floor(loans["fico"].min()))
        fico_high = int(np.ceil(loans["fico"].max()))
        dti_low = int(np.floor(loans["dti"].min()))
        dti_high = int(np.ceil(loans["dti"].max()))
        rate_low = float(np.floor((loans["int.rate"] * 100).min() * 2) / 2)
        rate_high = float(np.ceil((loans["int.rate"] * 100).max() * 2) / 2)
        installment_low = int(np.floor(loans["installment"].min()))
        installment_high = int(np.ceil(loans["installment"].max()))
        inquiry_options = sorted(loans["inq.last.6mths"].dropna().unique().astype(int).tolist())
        delinquency_options = sorted(loans["delinq.2yrs"].dropna().unique().astype(int).tolist())
        record_options = sorted(loans["pub.rec"].dropna().unique().astype(int).tolist())
    except Exception as exc:  # noqa: BLE001 - surfaced to the user
        st.error(f"The slicers are unavailable: {exc}")
    else:
        with st.expander("Loan", expanded=True):
            st.multiselect(
                "Loan purpose",
                purpose_options,
                key="filter_purposes",
                placeholder="All purposes",
            )
            st.slider(
                "Interest rate (%)",
                rate_low,
                rate_high,
                (rate_low, rate_high),
                step=0.5,
                key="filter_rate",
            )
            st.slider(
                "Monthly installment ($)",
                installment_low,
                installment_high,
                (installment_low, installment_high),
                step=10,
                key="filter_installment",
            )
            st.selectbox(
                "Credit policy",
                ["All", "Yes", "No"],
                key="filter_policy",
            )
        with st.expander("Borrower", expanded=False):
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
                step=1,
                key="filter_dti",
            )
        with st.expander("Credit history", expanded=False):
            st.multiselect(
                "Inquiries · last 6 months",
                inquiry_options,
                key="filter_inq",
                placeholder="All counts",
            )
            st.multiselect(
                "Delinquencies · previous 2 years",
                delinquency_options,
                key="filter_delinq",
                placeholder="All counts",
            )
            st.multiselect(
                "Public records",
                record_options,
                key="filter_pubrec",
                placeholder="All counts",
            )

        st.caption("Every slicer narrows all of the dashboard figures.")


nav.run()
