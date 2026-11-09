"""Predict Default page — score a loan with the report's best model.

The form is built from the 13 features the tuned XGBoost pipeline was trained
on in ``Loan Repayment Risk Analysis Report.ipynb`` (cell 25).  Income is
entered in dollars and log-transformed, because the model consumes
``log.annual.inc``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import streamlit as st

import loan_report
import utils

utils.inject_styles()

st.title("Predict Default")
utils.page_subtitle(
    "<b>Score a hypothetical loan application</b> with the best model from the "
    "report — the tuned XGBoost — and get a clear repayment-risk assessment."
)

try:
    loans = loan_report.load_data()
    model = loan_report.load_model()
except Exception as exc:  # noqa: BLE001 - surfaced to the user
    st.error(f"Could not load the model assets: {exc}")
    st.stop()

# Model card: the notebook's tuned scores.
auc_col, recall_col, pr_col = st.columns(3)
auc_col.metric("Selected model", "Tuned XGBoost")
recall_col.metric("ROC-AUC · test", "0.667")
pr_col.metric("PR-AUC · test", "0.285")
st.caption(
    "Best recall (0.567) and F1 (0.337) of the three models compared in the "
    "report. The probability is a relative risk score, not a book-wide "
    "non-repayment rate."
)

# ---------------------------------------------------------------------------
# Defaults and limits from the training data
# ---------------------------------------------------------------------------
defaults = {c: loans[c].median() for c in loan_report.NUMERIC_FEATURES}
defaults["purpose"] = loans["purpose"].mode().iloc[0]


def limits(column: str) -> tuple[float, float]:
    return float(loans[column].min()), float(loans[column].max())


def median(column: str) -> float:
    return float(loans[column].median())


annual_low, annual_high = (
    float(np.exp(limits("log.annual.inc")[0])),
    float(np.exp(limits("log.annual.inc")[1])),
)

# ---------------------------------------------------------------------------
# Application form
# ---------------------------------------------------------------------------
st.subheader("Loan application")

with st.container(border=True, key="card-form"):
    with st.form("loan_repayment_form"):
        left, middle, right = st.columns(3)

        with left:
            purpose = st.selectbox(
                "Loan purpose",
                sorted(loans["purpose"].dropna().unique().tolist()),
                index=sorted(loans["purpose"].dropna().unique().tolist()).index(
                    defaults["purpose"]
                ),
            )
            credit_policy = st.selectbox("Credit policy met", ["Yes", "No"])
            interest_pct = st.number_input(
                "Interest rate (%)",
                min_value=round((loans["int.rate"].min()) * 100, 1),
                max_value=round((loans["int.rate"].max()) * 100, 1),
                value=round(float(defaults["int.rate"]) * 100, 2),
                step=0.1,
                format="%.2f",
            )
            installment = st.number_input(
                "Monthly installment ($)",
                min_value=math.floor(limits("installment")[0]),
                max_value=math.ceil(limits("installment")[1]),
                value=int(round(defaults["installment"])),
                step=10,
            )
            annual_income = st.number_input(
                "Annual income ($)",
                min_value=int(round(annual_low)),
                max_value=int(round(annual_high)),
                value=int(round(np.exp(defaults["log.annual.inc"]))),
                step=1000,
            )

        with middle:
            fico = st.number_input(
                "FICO score",
                min_value=math.floor(limits("fico")[0]),
                max_value=math.ceil(limits("fico")[1]),
                value=int(round(defaults["fico"])),
                step=1,
            )
            dti = st.number_input(
                "Debt-to-income ratio",
                min_value=math.floor(limits("dti")[0]),
                max_value=math.ceil(limits("dti")[1]),
                value=int(round(defaults["dti"])),
                step=1,
            )
            days_credit = st.number_input(
                "Days with credit line",
                min_value=math.floor(limits("days.with.cr.line")[0]),
                max_value=math.ceil(limits("days.with.cr.line")[1]),
                value=int(round(defaults["days.with.cr.line"])),
                step=100,
            )

        with right:
            revol_bal = st.number_input(
                "Revolving balance ($)",
                min_value=math.floor(limits("revol.bal")[0]),
                max_value=math.ceil(limits("revol.bal")[1]),
                value=int(round(defaults["revol.bal"])),
                step=1000,
            )
            revol_util = st.number_input(
                "Revolving line utilisation (%)",
                min_value=0.0,
                max_value=float(math.ceil(limits("revol.util")[1])),
                value=float(round(defaults["revol.util"], 1)),
                step=1.0,
            )
            inquiries = st.number_input(
                "Credit inquiries · last 6 months",
                min_value=0,
                max_value=math.ceil(limits("inq.last.6mths")[1]),
                value=int(round(defaults["inq.last.6mths"])),
                step=1,
            )
            delinquencies = st.number_input(
                "Delinquencies · previous 2 years",
                min_value=0,
                max_value=math.ceil(limits("delinq.2yrs")[1]),
                value=int(round(defaults["delinq.2yrs"])),
                step=1,
            )
            public_records = st.number_input(
                "Public records",
                min_value=0,
                max_value=math.ceil(limits("pub.rec")[1]),
                value=int(round(defaults["pub.rec"])),
                step=1,
            )

        submitted = st.form_submit_button("Predict repayment risk", type="primary")

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
if submitted:
    row = pd.DataFrame(
        [
            {
                "credit.policy": 1 if credit_policy == "Yes" else 0,
                "purpose": purpose,
                "int.rate": interest_pct / 100.0,
                "installment": float(installment),
                "log.annual.inc": float(np.log(annual_income)),
                "dti": float(dti),
                "fico": float(fico),
                "days.with.cr.line": float(days_credit),
                "revol.bal": float(revol_bal),
                "revol.util": float(revol_util),
                "inq.last.6mths": float(inquiries),
                "delinq.2yrs": float(delinquencies),
                "pub.rec": float(public_records),
            }
        ],
        columns=loan_report.FEATURE_COLUMNS,
    )

    try:
        probabilities = model.predict_proba(row)[0]
    except Exception as exc:  # noqa: BLE001 - surfaced to the user
        st.error(f"The model could not score this application: {exc}")
    else:
        classes = list(model.classes_)
        index = classes.index(1) if 1 in classes else len(classes) - 1
        probability = float(probabilities[index])

        st.divider()
        st.markdown(utils.risk_assessment_html(probability), unsafe_allow_html=True)

        st.caption(
            "Top drivers in the report: credit policy (21.0%), interest rate "
            "(11.9%), recent inquiries (7.4%) and FICO score (6.9%)."
        )
