"""Predict Default page — score a single loan application with the model.

The form is built from ``model/features.joblib`` (the exact columns the
model was trained on).  The most influential features are asked for; the
remaining ones are filled with the median (numeric) or the mode
(categorical) taken from the data, exactly as the specification requires.
"""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

import utils

utils.inject_styles()

# ---------------------------------------------------------------------------
# Human readable labels / input configuration per feature
# ---------------------------------------------------------------------------
FEATURE_LABELS = {
    "loan_amnt": "Loan amount ($)",
    "fico_n": "FICO score",
    "dti_n": "Debt-to-income ratio (%)",
    "emp_length": "Employment length",
    "purpose": "Loan purpose",
    "home_ownership_n": "Home ownership",
    "addr_state": "State",
}

INPUT_STEPS = {"loan_amnt": 250.0, "fico_n": 1.0, "dti_n": 1.0}

# A feature has to explain at least this share of the model's total
# importance before it is worth asking the user about it.
IMPORTANCE_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.title("Predict Default")
utils.page_subtitle(
    "<b>Score a hypothetical loan application</b> against the trained default "
    "model and get a clear risk assessment."
)

try:
    loans = utils.load_data()
    model = utils.load_model()
    features = utils.load_features()
except Exception as exc:  # noqa: BLE001 - surfaced to the user
    st.error(f"Could not load the model assets: {exc}")
    st.stop()

missing = [feature for feature in features if feature not in loans.columns]
if missing:
    st.error(
        f"The data file does not contain the model input column(s) {missing}. "
        "The model and the data are out of sync."
    )
    st.stop()

# The importance of each input only depends on the model, so it is computed
# once per session instead of on every rerun.
if "feature_importance" not in st.session_state:
    st.session_state["feature_importance"] = utils.raw_feature_importance(model)

importance = st.session_state["feature_importance"]
share = importance["importance"] / importance["importance"].sum()
asked = list(importance.loc[share >= IMPORTANCE_THRESHOLD, "feature"])
asked = [feature for feature in features if feature in asked]

defaults = utils.column_defaults(loans)

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
st.subheader("Loan application")

with st.container(border=True, key="card-form"):
    with st.form("loan_application_form"):
        values: dict[str, object] = {}
        columns = st.columns(3)

        for position, feature in enumerate(asked):
            with columns[position % len(columns)]:
                label = FEATURE_LABELS.get(feature, feature.replace("_", " ").title())
                series = loans[feature].dropna()

                if pd.api.types.is_numeric_dtype(series):
                    low, high = utils.column_limits(loans, feature)
                    step = INPUT_STEPS.get(feature, max(1.0, (high - low) / 100.0))
                    if step >= 1:
                        low, high = float(math.floor(low)), float(math.ceil(high))
                    values[feature] = st.number_input(
                        label,
                        min_value=float(low),
                        max_value=float(high),
                        value=float(defaults[feature]),
                        step=float(step),
                        format="%.2f" if feature == "dti_n" else "%.0f",
                    )
                else:
                    options = utils.column_options(loans, feature)
                    default_value = defaults[feature]
                    index = options.index(default_value) if default_value in options else 0
                    values[feature] = st.selectbox(label, options, index=index)

        submitted = st.form_submit_button("Predict default risk", type="primary")

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
if submitted:
    row_values: dict[str, object] = {}
    for feature in features:
        row_values[feature] = values.get(feature, defaults.get(feature))

    row = pd.DataFrame([row_values], columns=features)
    for feature in features:
        if pd.api.types.is_numeric_dtype(loans[feature]):
            row[feature] = row[feature].astype(float)
        else:
            row[feature] = row[feature].astype(str)

    try:
        probabilities = model.predict_proba(row)[0]
    except Exception as exc:  # noqa: BLE001 - surfaced to the user
        st.error(f"The model could not score this application: {exc}")
    else:
        classes = list(model.classes_)
        index = classes.index(1) if 1 in classes else len(classes) - 1
        probability = float(probabilities[index])

        st.divider()
        st.markdown(
            utils.risk_assessment_html(probability),
            unsafe_allow_html=True,
        )
