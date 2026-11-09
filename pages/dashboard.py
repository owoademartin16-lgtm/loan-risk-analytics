"""Dashboard page — the report's figures, driven by the sidebar slicers.

Every chart is rebuilt with the exact logic of
``Loan Repayment Risk Analysis Report.ipynb``: the same bins and labels, the
same bar colour per figure, the same titles, ordering, percentage labels and
axis headroom.
"""

from __future__ import annotations

import streamlit as st

import loan_report
import utils

utils.inject_styles()

CHART_CONFIG = {"displayModeBar": False}


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.title("Loan Repayment Risk Dashboard")
utils.page_subtitle(
    "<b>Every figure is reproduced from the report notebook</b> — the same "
    "bins, ordering, colours and labels — on 9,578 LendingClub loans. "
    "Use the slicers in the sidebar to narrow the book."
)

try:
    loans = loan_report.load_data()
except Exception as exc:  # noqa: BLE001 - surfaced to the user
    st.error(f"Could not load the loan data: {exc}")
    st.stop()

# ---------------------------------------------------------------------------
# Slicers (state created in app.py)
# ---------------------------------------------------------------------------
purposes = st.session_state.get("filter_purposes") or []
fico_range = st.session_state.get("filter_fico")
dti_range = st.session_state.get("filter_dti")
rate_range = st.session_state.get("filter_rate")
installment_range = st.session_state.get("filter_installment")
inquiries = st.session_state.get("filter_inq") or []
delinquencies = st.session_state.get("filter_delinq") or []
public_records = st.session_state.get("filter_pubrec") or []
credit_policy = st.session_state.get("filter_policy") or "All"

filtered = loan_report.apply_filters(
    loans,
    purposes=purposes,
    fico_range=fico_range,
    dti_range=dti_range,
    rate_range=rate_range,
    installment_range=installment_range,
    inquiries=inquiries,
    delinquencies=delinquencies,
    public_records=public_records,
    credit_policy=credit_policy,
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
kpis = loan_report.portfolio_kpis(filtered)
with st.container(key="kpi_row"):
    col_loans, col_problems, col_rate, col_interest = st.columns(4)
    col_loans.metric("Total loans", f"{kpis['total_loans']:,}")
    col_problems.metric("Not fully paid", f"{kpis['problem_loans']:,}")
    col_rate.metric("Non-full repayment rate", f"{kpis['problem_rate']:.2f}%")
    col_interest.metric("Average interest rate", f"{kpis['avg_interest']:.2f}%")

active = []
if purposes:
    active.append(f"{len(purposes)} purpose{'s' if len(purposes) > 1 else ''}")
if fico_range:
    active.append(f"FICO {fico_range[0]}–{fico_range[1]}")
if dti_range:
    active.append(f"DTI {dti_range[0]}–{dti_range[1]}")
if rate_range:
    active.append(f"rate {rate_range[0]:.1f}–{rate_range[1]:.1f}%")
if installment_range:
    active.append(f"installment ${installment_range[0]:,}–${installment_range[1]:,}")
if inquiries:
    active.append(f"{len(inquiries)} inquir{'y' if len(inquiries) == 1 else 'ies'}")
if delinquencies:
    active.append(f"{len(delinquencies)} delinquenc{'y' if len(delinquencies) == 1 else 'ies'}")
if public_records:
    active.append(f"{len(public_records)} public record group{'s' if len(public_records) > 1 else ''}")
if credit_policy != "All":
    active.append(f"credit policy {credit_policy.lower()}")

st.caption(
    f"Showing {kpis['total_loans']:,} of {len(loans):,} loans"
    + (" · " + " · ".join(active) if active else " · all loans")
)

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
overview, borrower, history, installments, model_tab = st.tabs(
    ["Overview", "Borrower", "Credit history", "Installments", "Model"]
)

with overview:
    left, right = st.columns(2)
    with left:
        with st.container(border=True, key="card-purpose"):
            st.plotly_chart(
                loan_report.chart_purpose(filtered),
                width="stretch",
                config=CHART_CONFIG,
            )
    with right:
        with st.container(border=True, key="card-interest"):
            st.plotly_chart(
                loan_report.chart_interest(filtered),
                width="stretch",
                config=CHART_CONFIG,
            )

with borrower:
    left, right = st.columns(2)
    with left:
        with st.container(border=True, key="card-fico"):
            st.plotly_chart(
                loan_report.chart_fico(filtered),
                width="stretch",
                config=CHART_CONFIG,
            )
    with right:
        with st.container(border=True, key="card-dti"):
            st.plotly_chart(
                loan_report.chart_dti(filtered),
                width="stretch",
                config=CHART_CONFIG,
            )

with history:
    with st.container(border=True, key="card-inquiries"):
        st.plotly_chart(
            loan_report.chart_inquiries(filtered),
            width="stretch",
            config=CHART_CONFIG,
        )
    left, right = st.columns(2)
    with left:
        with st.container(border=True, key="card-delinq"):
            st.plotly_chart(
                loan_report.chart_delinquencies(filtered),
                width="stretch",
                config=CHART_CONFIG,
            )
    with right:
        with st.container(border=True, key="card-pubrec"):
            st.plotly_chart(
                loan_report.chart_public_records(filtered),
                width="stretch",
                config=CHART_CONFIG,
            )

with installments:
    with st.container(border=True, key="card-installments"):
        st.plotly_chart(
            loan_report.chart_installments(filtered),
            width="stretch",
            config=CHART_CONFIG,
        )
    st.caption(
        "Ten equal-width bins over the monthly installment, exactly as "
        "`pd.cut(df['installment'], bins=10)` in the notebook."
    )

with model_tab:
    try:
        evaluation = loan_report.evaluate_models()
    except Exception as exc:  # noqa: BLE001
        st.error(f"The model figures are unavailable: {exc}")
    else:
        st.subheader("Model comparison · notebook test set")
        comparison = evaluation["comparison"].copy()
        st.dataframe(
            comparison.style.format("{:.3f}").highlight_max(
                axis=0, color="#DCEFE6"
            ),
            width="stretch",
        )
        st.info(
            "**The tuned XGBoost is the selected model.** After "
            "`RandomizedSearchCV` it has the best recall (0.567), F1 (0.337) "
            "and PR-AUC (0.285) of the three models, so it is the model used "
            "on the **Predict Default** page. The comparison table is "
            "recomputed from the notebook's deterministic 80/20 split."
        )

        st.subheader("Confusion matrices")
        matrix_tabs = st.tabs(list(evaluation["matrices"].keys()))
        for tab, (name, matrix) in zip(matrix_tabs, evaluation["matrices"].items()):
            with tab:
                with st.container(border=True, key=f"card-confusion-{name}"):
                    st.plotly_chart(
                        loan_report.chart_confusion_matrix(
                            matrix, f"{name} Confusion Matrix"
                        ),
                        width="stretch",
                        config=CHART_CONFIG,
                    )

        st.subheader("Top 15 features · tuned XGBoost")
        with st.container(border=True, key="card-importance"):
            st.plotly_chart(
                loan_report.chart_importance(evaluation["importance"]),
                width="stretch",
                config=CHART_CONFIG,
            )
        st.caption(
            "Feature importance after dropping the notebook's "
            "`numeric__` / `categorical__` prefixes — the same figure as cell 43."
        )
