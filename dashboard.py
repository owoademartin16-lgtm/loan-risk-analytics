"""Page renderers for the Loan Risk Analytics dashboard.

Each function receives the filtered DataFrame (and the full DataFrame for
comparison notes) and renders one dashboard section using the charts that
mirror ``Loan Risk Analytics.ipynb``.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import charts
import styles
from loan_data import (
    AMOUNT_BAND_BINS,
    AMOUNT_BAND_LABELS,
    DTI_BAND_BINS,
    DTI_BAND_LABELS,
    EXPLORER_EXPORT_CAP,
    FICO_BAND_BINS,
    FICO_BAND_LABELS,
    MIN_SAMPLE_SIZE,
    band_summary,
    fmt_int,
    fmt_money,
    fmt_pct,
    group_summary,
    monthly_summary,
    portfolio_kpis,
    top_states,
)

COLUMN_RENAMES = {
    "band": "Group",
    "addr_state": "State",
    "total_loans": "Loans",
    "defaulted": "Defaulted",
    "default_rate": "Default rate",
    "issue_month": "Issue month",
}

TABLE_FORMATS = {
    "Loans": "{:,.0f}",
    "Defaulted": "{:,.0f}",
    "Default rate": "{:.2f}%",
    "Issue month": "%b %Y",
}


def _summary_table(data: pd.DataFrame, renames: dict | None = None) -> pd.io.formats.style.Styler:
    """Format a grouped summary the way the notebook displays its tables."""
    names = {**COLUMN_RENAMES, **(renames or {})}
    out = data.rename(columns=names)
    return out.style.format(TABLE_FORMATS, na_rep="—").hide(axis="index")


def _kpi_row(metrics: list[tuple[str, str, str]]) -> None:
    cols = st.columns(3, gap="medium")
    for col, (label, value, sub) in zip(cols, metrics):
        with col:
            st.markdown(styles.kpi_card(label, value, sub), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# A. Portfolio Overview (chart: notebook cell 15)
# ---------------------------------------------------------------------------
def render_overview(df: pd.DataFrame, full: pd.DataFrame) -> None:
    kpis = portfolio_kpis(df)
    full_kpis = portfolio_kpis(full)
    filtered = len(df) < len(full)
    sub = "Full portfolio" if not filtered else f"Portfolio: {fmt_int(full_kpis['total_loans'])} loans"

    st.markdown("#### Key metrics")
    _kpi_row(
        [
            ("Total loans", fmt_int(kpis["total_loans"]), "Unique loan ids" if not filtered else sub),
            ("Total loan amount", fmt_money(kpis["total_amount"]), "Sum of loan_amnt"),
            ("Average loan amount", fmt_money(kpis["avg_amount"]), ""),
        ]
    )
    st.write("")
    _kpi_row(
        [
            ("Defaulted loans", fmt_int(kpis["defaults"]), "Loans with Default = 1"),
            ("Default rate", fmt_pct(kpis["default_rate"]), "" if not filtered else f"Portfolio: {fmt_pct(full_kpis['default_rate'])}"),
            ("Average FICO score", f"{kpis['avg_fico']:.0f}", "" if not filtered else f"Portfolio: {full_kpis['avg_fico']:.0f}"),
        ]
    )

    st.markdown("#### Distribution of Loan Outcomes")
    st.caption("Notebook chart: count of fully paid (Default = 0) versus defaulted (Default = 1) loans.")
    st.plotly_chart(charts.outcome_distribution(df), width="stretch")

    purposes = df["purpose"].value_counts()
    top_purpose = purposes.index[0] if len(purposes) else "—"
    top_share = purposes.iloc[0] / len(df) * 100 if len(purposes) else float("nan")
    date_span = (
        f"{df['issue_d'].min():%b %Y} – {df['issue_d'].max():%b %Y}"
        if df["issue_d"].notna().any()
        else "no dated loans in selection"
    )
    summary_html = (
        f"<strong>Filtered portfolio summary:</strong> {fmt_int(kpis['total_loans'])} loans "
        f"issued {date_span}, totalling {fmt_money(kpis['total_amount'])}. "
        f"Average loan {fmt_money(kpis['avg_amount'])}; average FICO {kpis['avg_fico']:.0f}. "
        f"Default rate {fmt_pct(kpis['default_rate'])}"
        + (f" (full portfolio: {fmt_pct(full_kpis['default_rate'])}). " if filtered else ". ")
        + f"Largest purpose: <strong>{top_purpose}</strong> ({top_share:.1f}% of loans in selection)."
    )
    st.markdown(styles.summary_block(summary_html), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# B. Credit Risk Analysis (charts: notebook cells 17, 19, 16, 23)
# ---------------------------------------------------------------------------
def render_credit_risk(df: pd.DataFrame, _full: pd.DataFrame) -> None:
    st.caption(
        "The figures below are descriptive associations within the current selection — "
        "they do not establish causation. Every percentage uses the loan count of its own group as the denominator."
    )

    sections = [
        (
            "Default Rate by FICO Score Band",
            band_summary(df, "fico_n", FICO_BAND_BINS, FICO_BAND_LABELS),
            "Notebook bands. Observed FICO values run from 612 to 847.5, so the lowest bands can be empty "
            "or very small — they are kept for consistency with the notebook.",
        ),
        (
            "Default Rate by Debt-to-Income Ratio",
            band_summary(df, "dti_n", DTI_BAND_BINS, DTI_BAND_LABELS),
            "Notebook bands. DTI values reach extreme outliers (up to 999 in the source data), which fall into the 50+ band.",
        ),
        (
            "Default Rate by Loan Amount",
            band_summary(df, "loan_amnt", AMOUNT_BAND_BINS, AMOUNT_BAND_LABELS),
            None,
        ),
        (
            "Loan Default Rate by Employment Length",
            group_summary(df, "emp_length").assign(
                band=lambda t: t["emp_length"].astype(str)
            ),
            "Sorted by default rate, as in the notebook. 'NI' is the dataset's own not-indicated category.",
        ),
    ]

    for title, summary, note in sections:
        st.markdown(f"#### {title}")
        chart_col, table_col = st.columns([7, 5], gap="large")
        with chart_col:
            if title.startswith("Loan Default Rate by Employment"):
                st.plotly_chart(charts.default_rate_by_emp_length(summary), width="stretch")
            else:
                st.plotly_chart(charts.default_rate_bars(summary), width="stretch")
        with table_col:
            st.dataframe(_summary_table(summary[["band", "total_loans", "defaulted", "default_rate"]]), width="stretch")
        if note:
            st.caption(note)
        st.write("")


# ---------------------------------------------------------------------------
# C. Loan Purpose and Geographic Analysis (charts: notebook cells 18, 20)
# ---------------------------------------------------------------------------
def render_geography(df: pd.DataFrame, _full: pd.DataFrame) -> None:
    st.caption(
        f"Minimum-sample threshold: groups with fewer than {MIN_SAMPLE_SIZE} loans in the current "
        "selection are excluded from the default-rate charts and rankings below."
    )

    st.markdown("#### Default Rate by Loan Purpose")
    purpose = group_summary(df, "purpose")
    purpose_ranked = (
        purpose[purpose["total_loans"] >= MIN_SAMPLE_SIZE]
        .sort_values("default_rate", ascending=False)
    )
    chart_col, table_col = st.columns([7, 5], gap="large")
    with chart_col:
        if purpose_ranked.empty:
            st.info("No loan purpose has at least the minimum number of loans in the current selection.")
        else:
            st.plotly_chart(
                charts.default_rate_barh(purpose_ranked, "purpose", category_label="Purpose"),
                width="stretch",
            )
    with table_col:
        st.dataframe(
            _summary_table(
                purpose_ranked[["purpose", "total_loans", "defaulted", "default_rate"]],
                {"purpose": "Purpose"},
            ),
            width="stretch",
        )
    st.write("")

    st.markdown("#### 10 States with the Highest Loan Default Rates")
    states = top_states(df)
    chart_col, table_col = st.columns([7, 5], gap="large")
    with chart_col:
        if states.empty:
            st.info("No state reaches the minimum loan count in the current selection.")
        else:
            st.plotly_chart(
                charts.default_rate_barh(states, "addr_state", category_label="State"),
                width="stretch",
            )
    with table_col:
        st.dataframe(
            _summary_table(states[["addr_state", "total_loans", "defaulted", "default_rate"]]),
            width="stretch",
        )


# ---------------------------------------------------------------------------
# D. Time Trends (chart: notebook cell 21)
# ---------------------------------------------------------------------------
def render_time_trends(df: pd.DataFrame, _full: pd.DataFrame) -> None:
    monthly = monthly_summary(df)
    if monthly.empty:
        st.info("No dated loans in the current selection — adjust the issue-date filter.")
        return

    st.info(
        "Cohort-maturity limitation: loans issued in recent months have had less time to default, "
        "so their outcome histories are incomplete. Differences between issue-month cohorts are not, "
        "by themselves, evidence of a change in underlying credit risk."
    )

    st.markdown("#### Loan Default Rate by Issue Month")
    st.plotly_chart(charts.default_rate_by_month(monthly), width="stretch")

    with st.expander("Show monthly summary (last 12 issue months)"):
        st.dataframe(
            _summary_table(monthly.tail(12)[["issue_month", "total_loans", "defaulted", "default_rate"]]),
            width="stretch",
        )


# ---------------------------------------------------------------------------
# E. Data Explorer
# ---------------------------------------------------------------------------
def render_explorer(df: pd.DataFrame, _full: pd.DataFrame) -> None:
    display_columns = [
        "id", "issue_d", "loan_amnt", "fico_n", "dti_n", "revenue",
        "emp_length", "purpose", "home_ownership_n", "addr_state",
        "zip_code", "Default", "title",
    ]
    options = st.columns([4, 3, 2], gap="medium")
    with options[0]:
        selected_columns = st.multiselect("Columns", display_columns, default=display_columns)
    with options[1]:
        search = st.text_input("Search loan title", placeholder="e.g. debt consolidation")
    with options[2]:
        row_limit = st.number_input("Rows displayed", min_value=50, max_value=2000, value=200, step=50)

    view = df
    if search:
        view = view[view["title"].astype("string").str.contains(search, case=False, na=False)]

    st.caption(f"{len(view):,} filtered records match · showing the first {min(int(row_limit), len(view)):,} below")
    if view.empty:
        st.info("No records match the current search and filters.")
        return

    table = view.loc[:, selected_columns].head(int(row_limit)).copy()
    table["issue_d"] = table["issue_d"].dt.strftime("%b %Y")
    st.dataframe(table, width="stretch", hide_index=True, height=460)

    export = view.loc[:, selected_columns].head(EXPLORER_EXPORT_CAP)
    st.download_button(
        "Download filtered records (CSV)",
        data=export.to_csv(index=False).encode("utf-8"),
        file_name="loan_records_filtered.csv",
        mime="text/csv",
    )
    if len(view) > EXPLORER_EXPORT_CAP:
        st.caption(
            f"Export limited to the first {EXPLORER_EXPORT_CAP:,} of {len(view):,} filtered records "
            "to keep the download responsive."
        )
