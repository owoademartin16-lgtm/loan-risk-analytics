"""Dashboard page — KPI summary and the default-rate charts.

All of the data access, binning and figure building lives in :mod:`utils`,
so this module stays short: it wires the sidebar filters to the charts and
lays the figures out in a two-column grid.
"""

from __future__ import annotations

import streamlit as st

import utils

utils.inject_styles()


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.title("Loan Default Dashboard")
utils.page_subtitle(
    "<b>Default risk and loan performance</b> across the Lending Club book "
    "(2007&ndash;2018). Use the sidebar filters to explore the loan data."
)

try:
    loans = utils.load_data()
except Exception as exc:  # noqa: BLE001 - surfaced to the user
    st.error(f"Could not load the loan data: {exc}")
    st.stop()

# The filters themselves live in app.py so that they keep their state while
# the user moves between pages.
purposes = st.session_state.get("filter_purposes") or []
year_range = st.session_state.get("filter_years")

filtered = utils.apply_filters(loans, purposes, year_range)

if filtered.empty:
    st.warning(
        "No loans match the current filters. Widen the purpose selection or "
        "the issue-year range in the sidebar."
    )
    st.stop()

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
kpis = utils.portfolio_kpis(filtered)
with st.container(key="kpi_row"):
    col_loans, col_defaulted, col_rate, col_amount = st.columns(4)
    col_loans.metric("Total loans", f"{kpis['total_loans']:,}")
    col_defaulted.metric("Defaulted loans", f"{kpis['defaulted_loans']:,}")
    col_rate.metric("Overall default rate", f"{kpis['default_rate']:.2f}%")
    col_amount.metric("Average loan amount", f"${kpis['avg_loan_amount']:,.0f}")

st.caption(
    f"Showing {kpis['total_loans']:,} of {len(loans):,} loans"
    + (f" · purposes: {', '.join(purposes)}" if purposes else " · all purposes")
    + (f" · issued {year_range[0]}–{year_range[1]}" if year_range else "")
)

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
overview, credit, geography = st.tabs(
    ["Overview", "Credit Risk", "Purpose & Geography"]
)

with overview:
    left, right = st.columns(2)
    with left:
        with st.container(border=True, key="card-outcomes"):
            st.subheader("Outcome Distribution")
            st.plotly_chart(
                utils.chart_outcomes(filtered),
                use_container_width=True,
                config={"displayModeBar": False},
            )
    with right:
        with st.container(border=True, key="card-year"):
            st.subheader("Default Rate By Issue Year")
            st.plotly_chart(
                utils.chart_year_trend(utils.summary_by_year(filtered)),
                use_container_width=True,
                config={"displayModeBar": False},
            )

with credit:
    top_left, top_right = st.columns(2)
    with top_left:
        with st.container(border=True, key="card-amount"):
            st.subheader("Default Rate By Loan Amount Band")
            st.plotly_chart(
                utils.chart_default_rate_bars(
                    utils.summary_by_bands(
                        filtered, "loan_amnt", utils.AMOUNT_BINS, utils.AMOUNT_LABELS
                    )
                ),
                use_container_width=True,
                config={"displayModeBar": False},
            )
    with top_right:
        with st.container(border=True, key="card-fico"):
            st.subheader("Default Rate By FICO Band")
            st.plotly_chart(
                utils.chart_default_rate_bars(
                    utils.summary_by_bands(
                        filtered, "fico_n", utils.FICO_BINS, utils.FICO_LABELS
                    )
                ),
                use_container_width=True,
                config={"displayModeBar": False},
            )

    bottom_left, bottom_right = st.columns(2)
    with bottom_left:
        with st.container(border=True, key="card-dti"):
            st.subheader("Default Rate By DTI Band")
            st.plotly_chart(
                utils.chart_default_rate_bars(
                    utils.summary_by_bands(
                        filtered, "dti_n", utils.DTI_BINS, utils.DTI_LABELS
                    )
                ),
                use_container_width=True,
                config={"displayModeBar": False},
            )
    with bottom_right:
        with st.container(border=True, key="card-emplength"):
            st.subheader("Default Rate By Employment Length")
            st.plotly_chart(
                utils.chart_default_rate_bars(
                    utils.summary_by_emp_length(filtered), x_col="emp_length"
                ),
                use_container_width=True,
                config={"displayModeBar": False},
            )

with geography:
    left, right = st.columns(2)
    with left:
        with st.container(border=True, key="card-purpose"):
            st.subheader("Default Rate By Loan Purpose")
            st.plotly_chart(
                utils.chart_horizontal_rates(
                    utils.summary_by_purpose(filtered), "purpose", "Purpose"
                ),
                use_container_width=True,
                config={"displayModeBar": False},
            )
    with right:
        with st.container(border=True, key="card-states"):
            st.subheader("Top 10 States By Default Rate")
            st.plotly_chart(
                utils.chart_horizontal_rates(
                    utils.summary_top_states(filtered), "state_name", "State"
                ),
                use_container_width=True,
                config={"displayModeBar": False},
            )
