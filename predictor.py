"""Default-risk prediction using the Random Forest saved by the notebook.

Loads ``random_forest_loan_model.pkl`` — the full sklearn pipeline
(imputation -> one-hot -> RandomForestClassifier) trained in
``Loan Risk Analytics.ipynb`` — and scores a single loan application.

The model is never retrained here. It takes the seven raw feature columns
directly and handles preprocessing internally, so no manual encoding is
required.

Held-out metrics are recomputed once against the notebook's own
chronological 80/20 split so the app reports the model's true, measured
performance rather than a hard-coded figure.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.io as pio
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import styles

# Register the shared template regardless of import order (charts.py also
# does this; pio.templates is a global registry).
pio.templates[styles.TEMPLATE_NAME] = styles.build_plotly_template()

APP_DIR = Path(__file__).resolve().parent
MODEL_FILENAME = "random_forest_loan_model.pkl"

# Raw feature columns the pipeline expects, in the notebook's training order.
FEATURES = [
    "loan_amnt",
    "fico_n",
    "dti_n",
    "emp_length",
    "purpose",
    "home_ownership_n",
    "addr_state",
]

# Human labels and dropdown options, taken from the dataset's observed values.
EMP_LENGTH_OPTIONS = [
    "< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years",
    "6 years", "7 years", "8 years", "9 years", "10+ years", "NI",
]
PURPOSE_OPTIONS = [
    "car", "credit_card", "debt_consolidation", "educational",
    "home_improvement", "house", "major_purchase", "medical", "moving",
    "other", "renewable_energy", "small_business", "vacation", "wedding",
]
HOME_OPTIONS = ["MORTGAGE", "OWN", "RENT", "OTHER"]
STATE_OPTIONS = [
    "AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "HI",
    "IA", "ID", "IL", "IN", "KS", "KY", "LA", "MA", "MD", "ME", "MI", "MN",
    "MO", "MS", "MT", "NC", "ND", "NE", "NH", "NJ", "NM", "NV", "NY", "OH",
    "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VA", "VT", "WA",
    "WI", "WV", "WY",
]

# Risk bands for the predicted default probability.
BAND_LOW = 0.30
BAND_HIGH = 0.60


def find_model() -> Path:
    """Locate the saved model file (project folder or the notebook folder)."""
    candidates = [
        APP_DIR / MODEL_FILENAME,
        Path.home() / "Desktop" / "Python Script" / MODEL_FILENAME,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    names = " or ".join(f"'{p.name}'" for p in candidates)
    raise FileNotFoundError(
        f"Model file not found. Place {MODEL_FILENAME} in the project folder "
        f"(searched: {', '.join(str(p) for p in candidates)})."
    )


@st.cache_resource(show_spinner="Loading the default-risk model…")
def load_model():
    """Load the saved sklearn pipeline once per app process."""
    return joblib.load(find_model())


def predict_default(model, loan_amnt, fico_n, dti_n, emp_length,
                    purpose, home_ownership, state) -> float:
    """Return P(default) for one loan application.

    The seven raw values are passed straight to the pipeline, which applies
    the imputation and one-hot encoding learned during training.
    """
    row = pd.DataFrame(
        [{
            "loan_amnt": float(loan_amnt),
            "fico_n": float(fico_n),
            "dti_n": float(dti_n),
            "emp_length": emp_length,
            "purpose": purpose,
            "home_ownership_n": home_ownership,
            "addr_state": state,
        }],
        columns=FEATURES,
    )
    return float(model.predict_proba(row)[0, 1])


def risk_band(probability: float) -> tuple[str, str]:
    """Map a default probability to a (label, hex colour) risk band."""
    if probability < BAND_LOW:
        return "Low risk", "#0f8a83"   # TEAL
    if probability <= BAND_HIGH:
        return "Moderate risk", "#b07d1a"  # amber
    return "High risk", "#b5443a"      # RISK


@st.cache_data(show_spinner="Computing held-out model metrics…")
def held_out_metrics(_model, csv_path: str) -> dict:
    """Recompute real test metrics on the notebook's chronological 20% holdout.

    The split mirrors ``Loan Risk Analytics.ipynb`` (first 80% train, last
    20% test, in row order). Nothing is hard-coded.
    """
    df = pd.read_csv(csv_path, low_memory=False, usecols=FEATURES + ["Default"])
    split = int(len(df) * 0.80)
    test = df.iloc[split:]
    X_test, y_test = test[FEATURES], test["Default"]

    y_prob = _model.predict_proba(X_test)[:, 1]
    y_pred = _model.predict(X_test)
    return {
        "test_size": int(len(y_test)),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "actual_rate": float(y_test.mean() * 100),
    }


def build_gauge(probability: float, colour: str):
    """A Plotly half-gauge showing the predicted default probability."""
    import plotly.graph_objects as go

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 40, "color": styles.NAVY}},
            title={
                "text": "Predicted default probability",
                "font": {"size": 14, "color": styles.MUTED},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": styles.MUTED,
                },
                "bar": {"color": colour, "thickness": 0.35},
                "bgcolor": "rgba(0,0,0,0)",
                "steps": [
                    {"range": [0, BAND_LOW * 100], "color": "#e9f4f3"},
                    {"range": [BAND_LOW * 100, BAND_HIGH * 100], "color": "#faf3e3"},
                    {"range": [BAND_HIGH * 100, 100], "color": "#f9ecea"},
                ],
                "threshold": {
                    "line": {"color": styles.NAVY, "width": 2},
                    "thickness": 0.85,
                    "value": probability * 100,
                },
            },
        )
    )
    fig.update_layout(
        template=styles.TEMPLATE_NAME,
        height=300,
        margin=dict(l=40, r=40, t=90, b=10),
    )
    return fig
