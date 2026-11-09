"""Dashboard page — the notebook's figures, driven by the sidebar slicers.

Every chart here is rebuilt with the exact logic of ``Loan Risk Analytics.ipynb``:
the same bins, band labels, ordering, bar colours, percentage labels and axis
headroom.  The sidebar slicers live in ``app.py`` and keep their state while
you move between pages.
"""

from __future__ import annotations

import streamlit as st

import utils

utils.inject_styles()

CHART_CONFIG = {"displayModeBar": False}


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.title("Loan Default Dashboard")
utils.page_subtitle(
    "<b>Every figure is reproduced from the analysis notebook</b> — the same "
    "bins, ordering, colours and labels — computed live on the LendingClub "
    "book (2007&ndash;2018). Use the slicers in the sidebar to narrow the view."
)

try:
    loans = utils.load_data()
except Exception as exc:  # noqa: BLE001 - surfaced to the user
    st.error(f"Could not load the loan data: {exc}")
    st.stop()

# ---------------------------------------------------------------------------
# Slicers (state created in app.py)
# ---------------------------------------------------------------------------
purposes = st.session_state.get("filter_purposes") or []
year_range = st.session_state.get("filter_years")
amount_range = st.session_state.get("filter_amount")
fico_range = st.session_state.get("filter_fico")
dti_range = st.session_state.get("filter_dti")
home = st.session_state.get("filter_home") or []
emp = st.session_state.get("filter_emp") or []
states = st.session_state.get("filter_states") or []

filtered = utils.apply_filters(
    loans,
    purposes=purposes,
    year_range=year_range,
    amount_range=amount_range,
    fico_range=fico_range,
    dti_range=dti_range,
    home_ownership=home,
    emp_length=emp,
    states=states,
)

if filtered.empty:
    st.warning(
        "No loans match the current slicers. Widen one of the filters in the "
        "sidebar to bring the figures back."
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

active = []
if purposes:
    active.append(f"{len(purposes)} purpose{'s' if len(purposes) > 1 else ''}")
if year_range:
    active.append(f"{year_range[0]}–{year_range[1]}")
if amount_range:
    active.append(f"${amount_range[0]:,}–${amount_range[1]:,}")
if fico_range:
    active.append(f"FICO {fico_range[0]}–{fico_range[1]}")
if dti_range:
    active.append(f"DTI {dti_range[0]}–{dti_range[1]}")
if home:
    active.append(f"{len(home)} home type{'s' if len(home) > 1 else ''}")
if emp:
    active.append(f"{len(emp)} employment length{'s' if len(emp) > 1 else ''}")
if states:
    active.append(f"{len(states)} state{'s' if len(states) > 1 else ''}")

st.caption(
    f"Showing {kpis['total_loans']:,} of {len(loans):,} loans"
    + (" · " + " · ".join(active) if active else " · all loans")
)

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
overview, profile, credit, geography, model_tab = st.tabs(
    ["Overview", "Loan profile", "Credit score", "Purpose & geography", "Model"]
)

with overview:
    left, right = st.columns(2)
    with left:
        with st.container(border=True, key="card-outcomes"):
            st.plotly_chart(
                utils.chart_notebook_outcomes(filtered),
                width="stretch",
                config=CHART_CONFIG,
            )
    with right:
        with st.container(border=True, key="card-year"):
            st.plotly_chart(
                utils.chart_notebook_year(utils.notebook_year_summary(filtered)),
                width="stretch",
                config=CHART_CONFIG,
            )

with profile:
    top_left, top_right = st.columns(2)
    with top_left:
        with st.container(border=True, key="card-amount"):
            st.plotly_chart(
                utils.chart_notebook_vertical(
                    utils.notebook_amount_summary(filtered),
                    "Default Rate by Loan Amount",
                    "Loan Amount Range",
                ),
                width="stretch",
                config=CHART_CONFIG,
            )
    with top_right:
        with st.container(border=True, key="card-dti"):
            st.plotly_chart(
                utils.chart_notebook_vertical(
                    utils.notebook_dti_summary(filtered),
                    "Default Rate by Debt-to-Income Ratio",
                    "DTI Band",
                    angle=0,
                ),
                width="stretch",
                config=CHART_CONFIG,
            )
    with st.container(border=True, key="card-emplength"):
        st.plotly_chart(
            utils.chart_notebook_employment(utils.notebook_emp_summary(filtered)),
            width="stretch",
            config=CHART_CONFIG,
        )
        st.caption("NI = employment length not indicated, exactly as labelled in the notebook.")

with credit:
    left, right = st.columns(2)
    with left:
        with st.container(border=True, key="card-fico"):
            st.plotly_chart(
                utils.chart_notebook_vertical(
                    utils.notebook_fico_summary(filtered),
                    "Default Rate by FICO Score Band",
                    "FICO Score Band",
                ),
                width="stretch",
                config=CHART_CONFIG,
            )
    with right:
        with st.container(border=True, key="card-fico-groups"):
            st.plotly_chart(
                utils.chart_notebook_fico_groups(
                    utils.notebook_fico_group_summary(filtered)
                ),
                width="stretch",
                config=CHART_CONFIG,
            )

with geography:
    left, right = st.columns(2)
    with left:
        with st.container(border=True, key="card-purpose"):
            st.plotly_chart(
                utils.chart_notebook_horizontal(
                    utils.notebook_purpose_summary(filtered),
                    "Default Rate by Loan Purpose",
                    "Loan Purpose",
                ),
                width="stretch",
                config=CHART_CONFIG,
            )
    with right:
        with st.container(border=True, key="card-states"):
            st.plotly_chart(
                utils.chart_notebook_horizontal(
                    utils.notebook_state_summary(filtered),
                    "10 States with the Highest Loan Default Rates",
                    "State",
                ),
                width="stretch",
                config=CHART_CONFIG,
            )

with model_tab:
    st.subheader("Model comparison · notebook test set")
    comparison = utils.NOTEBOOK_MODEL_RESULTS.copy()
    st.dataframe(
        comparison.style.format("{:.4f}").highlight_max(
            axis=0, color="#DCEFE6"
        ),
        width="stretch",
    )
    st.info(
        "**Random Forest is the selected model.** It has the best F1-score "
        "(0.3889) and recall (0.6036) of the three models, so it is the one "
        "used on the **Predict Default** page. CatBoost reports the highest "
        "accuracy and ROC-AUC, but at its default 0.5 threshold it flags only "
        "0.26% of loans as defaults, which is why its F1 collapses to 0.0128."
    )

    chart_col, table_col = st.columns([1.45, 1])
    with chart_col:
        with st.container(border=True, key="card-thresholds"):
            st.plotly_chart(
                utils.chart_notebook_thresholds(),
                width="stretch",
                config=CHART_CONFIG,
            )
    with table_col:
        st.markdown("**CatBoost threshold sweep · notebook cell 41**")
        st.dataframe(
            utils.NOTEBOOK_THRESHOLD_RESULTS.style.format(
                {"Threshold": "{:.2f}", "Accuracy": "{:.4f}",
                 "Precision": "{:.4f}", "Recall": "{:.4f}",
                 "F1_Score": "{:.4f}"}
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "The best F1-score is at a 0.20 threshold (0.3938) — 0.296 "
            "precision against 0.592 recall."
        )

    if "feature_importance_chart" not in st.session_state:
        try:
            model = utils.load_model()
            st.session_state["feature_importance_chart"] = (
                utils.chart_notebook_importance(model)
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"The feature-importance figure is unavailable: {exc}")

    if "feature_importance_chart" in st.session_state:
        with st.container(border=True, key="card-importance"):
            st.plotly_chart(
                st.session_state["feature_importance_chart"],
                width="stretch",
                config=CHART_CONFIG,
            )
        st.caption(
            "FICO score (35.4%), debt-to-income (21.3%) and loan amount "
            "(15.4%) carry 72% of the Random Forest's importance."
        )
