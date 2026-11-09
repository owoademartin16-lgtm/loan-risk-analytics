"""Data, figures and model for the Loan Repayment Risk Analysis Report.

Everything in this module mirrors ``Loan Repayment Risk Analysis Report.ipynb``:
the same dataset (9,578 LendingClub loans), the same bins and category labels,
the same bar colours, titles, ordering and axis headroom, and the same saved
best model (the tuned XGBoost pipeline).

The notebook is the source of truth for the figure values; the helpers below
recompute them from the raw CSV so the dashboard can react to its slicers.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "loan_data.csv"
MODEL_PATH = BASE_DIR / "model" / "repayment_xgb_model.pkl"

TARGET = "not.fully.paid"
PROBLEM_LABEL = "Not Fully Paid"
PAID_LABEL = "Fully Paid"

# The notebook's colours: one per figure.
COLOUR_PURPOSE = "#1f77b4"
COLOUR_INTEREST = "#2ca02c"
COLOUR_FICO = "#d62728"
COLOUR_DTI = "#ff7f0e"
COLOUR_INQUIRY = "#9467bd"
COLOUR_DELINQ = "#8c564b"
COLOUR_PUBREC = "#bcbd22"
COLOUR_INSTALLMENT = "#1f77b4"

INTEREST_BINS = [0, 8, 10, 12, 14, 16, 30]
INTEREST_LABELS = ["Below 8%", "8–10%", "10–12%", "12–14%", "14–16%", "Above 16%"]

FICO_BINS = [0, 600, 650, 700, 750, 850]
FICO_LABELS = ["Below 600", "600–649", "650–699", "700–749", "750+"]

DTI_BINS = [-np.inf, 5, 10, 15, 20, 25, 100]
DTI_LABELS = ["0–5", "5–10", "10–15", "15–20", "20–25", "Above 25"]

FEATURE_COLUMNS = [
    "credit.policy",
    "purpose",
    "int.rate",
    "installment",
    "log.annual.inc",
    "dti",
    "fico",
    "days.with.cr.line",
    "revol.bal",
    "revol.util",
    "inq.last.6mths",
    "delinq.2yrs",
    "pub.rec",
]
CATEGORICAL_FEATURES = ["purpose"]
NUMERIC_FEATURES = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_FEATURES]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading the loan data...")
def load_data() -> pd.DataFrame:
    """Read the report's CSV and add the notebook's grouping columns."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    missing = [c for c in [*FEATURE_COLUMNS, TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"{DATA_PATH.name} is missing columns: {missing}")

    # Cell 10, 12, 13 and 18 add these bands to the frame once.  Fixing the
    # edges here keeps the labels stable when the slicers change the rows.
    df["interest_rate_group"] = pd.cut(
        df["int.rate"] * 100,
        bins=INTEREST_BINS,
        labels=INTEREST_LABELS,
        include_lowest=True,
    )
    df["fico_group"] = pd.cut(
        df["fico"], bins=FICO_BINS, labels=FICO_LABELS, right=False
    )
    df["dti_group"] = pd.cut(
        df["dti"], bins=DTI_BINS, labels=DTI_LABELS, right=False
    )
    df["installment_group"] = pd.cut(df["installment"], bins=10)

    return df


@st.cache_resource(show_spinner="Loading the best model...")
def load_model():
    """The tuned XGBoost pipeline saved by the report notebook."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


# ---------------------------------------------------------------------------
# Slicers
# ---------------------------------------------------------------------------
def apply_filters(
    df: pd.DataFrame,
    purposes: list[str] | None = None,
    fico_range: tuple[int, int] | None = None,
    dti_range: tuple[float, float] | None = None,
    rate_range: tuple[float, float] | None = None,
    installment_range: tuple[float, float] | None = None,
    inquiries: list[int] | None = None,
    delinquencies: list[int] | None = None,
    public_records: list[int] | None = None,
    credit_policy: str | None = None,
) -> pd.DataFrame:
    """AND every slicer, like a Power BI report."""
    mask = pd.Series(True, index=df.index)

    if purposes:
        mask &= df["purpose"].isin(purposes)
    if fico_range is not None:
        mask &= df["fico"].between(*fico_range)
    if dti_range is not None:
        mask &= df["dti"].between(*dti_range)
    if rate_range is not None:
        mask &= (df["int.rate"] * 100).between(*rate_range)
    if installment_range is not None:
        mask &= df["installment"].between(*installment_range)
    if inquiries:
        mask &= df["inq.last.6mths"].isin(inquiries)
    if delinquencies:
        mask &= df["delinq.2yrs"].isin(delinquencies)
    if public_records:
        mask &= df["pub.rec"].isin(public_records)
    if credit_policy in {"Yes", "No"}:
        mask &= df["credit.policy"] == (1 if credit_policy == "Yes" else 0)

    return df[mask]


# ---------------------------------------------------------------------------
# Notebook summaries
# ---------------------------------------------------------------------------
def _rate_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    grouped = (
        df.groupby(group_col, observed=False)
        .agg(total_loans=(TARGET, "size"), problem_loans=(TARGET, "sum"))
        .reset_index()
    )
    grouped["problem_rate"] = np.where(
        grouped["total_loans"] > 0,
        grouped["problem_loans"] / grouped["total_loans"] * 100.0,
        np.nan,
    )
    return grouped


def summary_purpose(df: pd.DataFrame) -> pd.DataFrame:
    """Cell 9: rates by purpose, sorted ascending (lowest at the bottom)."""
    data = _rate_summary(df, "purpose")
    return data.sort_values("problem_rate", ascending=True, na_position="last")


def summary_interest(df: pd.DataFrame) -> pd.DataFrame:
    """Cell 10: rates by interest-rate band."""
    data = _rate_summary(df, "interest_rate_group")
    data["interest_rate_group"] = data["interest_rate_group"].astype(str)
    data["order"] = pd.Categorical(
        data["interest_rate_group"], categories=INTEREST_LABELS, ordered=True
    )
    return data.sort_values("order").drop(columns="order")


def summary_fico(df: pd.DataFrame) -> pd.DataFrame:
    """Cell 12: rates by FICO band (the notebook keeps empty bands)."""
    data = _rate_summary(df, "fico_group")
    data["fico_group"] = data["fico_group"].astype(str)
    data["order"] = pd.Categorical(
        data["fico_group"], categories=FICO_LABELS, ordered=True
    )
    return data.sort_values("order").drop(columns="order")


def summary_dti(df: pd.DataFrame) -> pd.DataFrame:
    """Cell 13: rates by debt-to-income band."""
    data = _rate_summary(df, "dti_group")
    data["dti_group"] = data["dti_group"].astype(str)
    data["order"] = pd.Categorical(
        data["dti_group"], categories=DTI_LABELS, ordered=True
    )
    return data.sort_values("order").drop(columns="order")


def summary_inquiries(df: pd.DataFrame) -> pd.DataFrame:
    """Cell 15: rates by credit inquiries in the last six months."""
    data = _rate_summary(df, "inq.last.6mths").sort_values("inq.last.6mths")
    data["inq.last.6mths"] = data["inq.last.6mths"].astype(int).astype(str)
    return data


def summary_delinquencies(df: pd.DataFrame) -> pd.DataFrame:
    """Cell 16: rates by previous delinquencies."""
    data = _rate_summary(df, "delinq.2yrs")
    data["delinq.2yrs"] = data["delinq.2yrs"].astype(int).astype(str)
    return data


def summary_public_records(df: pd.DataFrame) -> pd.DataFrame:
    """Cell 17: rates by number of public records."""
    data = _rate_summary(df, "pub.rec")
    data["pub.rec"] = data["pub.rec"].astype(int).astype(str)
    return data


def summary_installments(df: pd.DataFrame) -> pd.DataFrame:
    """Cell 18: rates by the notebook's ten equal-width installment bins."""
    data = _rate_summary(df, "installment_group")
    data["installment_group"] = data["installment_group"].astype(str)
    return data


# ---------------------------------------------------------------------------
# Figures — same colour, labels, titles and ranges as the notebook
# ---------------------------------------------------------------------------
PLOT_HEIGHT = 500
PLOT_MARGIN = dict(l=70, r=40, t=64, b=64)


def _rate_label() -> str:
    return "Loans Not Fully Paid (%)"


def _finish(fig: go.Figure, title: str, x_title: str, y_title: str,
            *, tickangle: int = 0) -> go.Figure:
    fig.update_layout(
        title=dict(
            text=title,
            x=0,
            xanchor="left",
            xref="paper",
            font=dict(
                family="Inter, Helvetica Neue, sans-serif",
                size=17,
                color="#1F2A37",
            ),
            pad=dict(l=2, t=2),
        ),
        xaxis_title=x_title,
        yaxis_title=y_title,
        template="plotly_white",
        height=PLOT_HEIGHT,
        margin=PLOT_MARGIN,
        font=dict(family="Inter, Helvetica Neue, sans-serif", size=12, color="#2A2F36"),
        xaxis_tickangle=tickangle,
        showlegend=False,
        bargap=0.4,
        barcornerradius=3,
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            bordercolor="#D6DBE1",
            font=dict(
                family="Inter, Helvetica Neue, sans-serif",
                size=12,
                color="#1F2A37",
            ),
        ),
    )
    # Tableau-style axes: labels and titles only, no grid lines, no tick marks
    # and a single quiet baseline at zero.
    axis_style = dict(
        showgrid=False,
        showline=False,
        ticks="",
        zeroline=True,
        zerolinecolor="#DDE2E8",
        zerolinewidth=1,
        automargin=True,
        tickfont=dict(
            family="Inter, Helvetica Neue, sans-serif", size=11, color="#5F6B7A"
        ),
        title_font=dict(
            family="Inter, Helvetica Neue, sans-serif", size=12, color="#5F6B7A"
        ),
    )
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    return fig


def _max_rate(data: pd.DataFrame) -> float:
    values = pd.to_numeric(data["problem_rate"], errors="coerce").dropna()
    return float(values.max()) * 1.15 if len(values) else 1.0


def chart_vertical(data: pd.DataFrame, x_col: str, title: str, x_title: str,
                   colour: str, *, tickangle: int = 0) -> go.Figure:
    """A vertical notebook bar chart with outside percentage labels."""
    fig = px.bar(
        data,
        x=x_col,
        y="problem_rate",
        text="problem_rate",
        title=title,
        labels={x_col: x_title, "problem_rate": _rate_label()},
        custom_data=["total_loans"],
    )
    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        textfont=dict(
            family="Inter, Helvetica Neue, sans-serif", size=11, color="#3F4A56"
        ),
        marker_color=colour,
        marker_line_width=0,
        opacity=0.95,
        cliponaxis=False,
        hovertemplate=(
            f"{x_title}: %{{x}}<br>"
            "Loans not fully paid: %{y:.2f}%<br>"
            "Total loans: %{customdata[0]:,}<extra></extra>"
        ),
    )
    fig.update_yaxes(range=[0, _max_rate(data)])
    return _finish(fig, title, x_title, _rate_label(), tickangle=tickangle)


def chart_horizontal(data: pd.DataFrame, title: str, y_title: str,
                     colour: str) -> go.Figure:
    """A horizontal notebook bar chart (purpose)."""
    fig = px.bar(
        data,
        x="problem_rate",
        y="purpose",
        orientation="h",
        text="problem_rate",
        title=title,
        labels={"problem_rate": _rate_label(), "purpose": y_title},
        custom_data=["total_loans"],
    )
    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        textfont=dict(
            family="Inter, Helvetica Neue, sans-serif", size=11, color="#3F4A56"
        ),
        marker_color=colour,
        marker_line_width=0,
        opacity=0.95,
        cliponaxis=False,
        hovertemplate=(
            f"{y_title}: %{{y}}<br>"
            "Loans not fully paid: %{x:.2f}%<br>"
            "Total loans: %{customdata[0]:,}<extra></extra>"
        ),
    )
    fig.update_xaxes(range=[0, _max_rate(data)])
    fig = _finish(fig, title, _rate_label(), y_title)
    fig.update_layout(margin=dict(l=110, r=40, t=64, b=64))
    return fig


def chart_purpose(df: pd.DataFrame) -> go.Figure:
    """Cell 9."""
    return chart_horizontal(
        summary_purpose(df),
        "Non-Full Repayment Rate by Loan Purpose",
        "Loan Purpose",
        COLOUR_PURPOSE,
    )


def chart_interest(df: pd.DataFrame) -> go.Figure:
    """Cell 10."""
    return chart_vertical(
        summary_interest(df),
        "interest_rate_group",
        "Non-Full Repayment Rate by Interest Rate",
        "Interest Rate",
        COLOUR_INTEREST,
    )


def chart_fico(df: pd.DataFrame) -> go.Figure:
    """Cell 12."""
    return chart_vertical(
        summary_fico(df),
        "fico_group",
        "Non-Full Repayment Rate by FICO Score",
        "FICO Score Group",
        COLOUR_FICO,
    )


def chart_dti(df: pd.DataFrame) -> go.Figure:
    """Cell 13."""
    return chart_vertical(
        summary_dti(df),
        "dti_group",
        "Non-Full Repayment Rate by Debt-to-Income Ratio",
        "Debt-to-Income Ratio",
        COLOUR_DTI,
    )


def chart_inquiries(df: pd.DataFrame) -> go.Figure:
    """Cell 15."""
    return chart_vertical(
        summary_inquiries(df),
        "inq.last.6mths",
        "Non-Full Repayment Rate by Recent Credit Inquiries",
        "Credit Inquiries in the Last 6 Months",
        COLOUR_INQUIRY,
        tickangle=-45,
    )


def chart_delinquencies(df: pd.DataFrame) -> go.Figure:
    """Cell 16."""
    return chart_vertical(
        summary_delinquencies(df),
        "delinq.2yrs",
        "Non-Full Repayment Rate by Previous Delinquencies",
        "Delinquencies in the Previous 2 Years",
        COLOUR_DELINQ,
        tickangle=-45,
    )


def chart_public_records(df: pd.DataFrame) -> go.Figure:
    """Cell 17."""
    return chart_vertical(
        summary_public_records(df),
        "pub.rec",
        "Non-Full Repayment Rate by Public Records",
        "Number of Public Records",
        COLOUR_PUBREC,
    )


def chart_installments(df: pd.DataFrame) -> go.Figure:
    """Cell 18."""
    return chart_vertical(
        summary_installments(df),
        "installment_group",
        "Non-Full Repayment Rate by Monthly Installment",
        "Monthly Installment Range",
        COLOUR_INSTALLMENT,
        tickangle=-35,
    )


# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------
def portfolio_kpis(df: pd.DataFrame) -> dict:
    total = len(df)
    problems = int(df[TARGET].sum())
    return {
        "total_loans": total,
        "problem_loans": problems,
        "problem_rate": (problems / total * 100.0) if total else float("nan"),
        "avg_interest": float(df["int.rate"].mean() * 100.0) if total else float("nan"),
        "avg_fico": float(df["fico"].mean()) if total else float("nan"),
    }


# ---------------------------------------------------------------------------
# Model: comparison, confusion matrices, feature importance
# ---------------------------------------------------------------------------
def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def _metrics(y_true, predictions, probabilities) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Recall": recall_score(y_true, predictions, zero_division=0),
        "F1 Score": f1_score(y_true, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, probabilities),
        "PR-AUC (Average Precision)": average_precision_score(y_true, probabilities),
    }


@st.cache_resource(show_spinner="Reproducing the notebook's model results...")
def evaluate_models() -> dict:
    """Rebuild the notebook's test split, models and recorded scores.

    The split, preprocessing, Decision Tree, XGBoost and tuned XGBoost are all
    deterministic, so the figures and the comparison table match the notebook
    exactly (including the tuned model's [[1057, 552], [133, 174]] matrix).
    """
    df = load_data()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    decision_tree = Pipeline(
        [
            ("preprocessor", _preprocessor()),
            (
                "classifier",
                DecisionTreeClassifier(class_weight="balanced", random_state=42),
            ),
        ]
    ).fit(X_train, y_train)

    xgboost = Pipeline(
        [
            ("preprocessor", _preprocessor()),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    scale_pos_weight=scale_pos_weight,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    ).fit(X_train, y_train)

    tuned = load_model()

    rows, matrices = [], {}
    for name, model in (
        ("Decision Tree", decision_tree),
        ("XGBoost", xgboost),
        ("Tuned XGBoost", tuned),
    ):
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        rows.append({"Model": name, **_metrics(y_test, predictions, probabilities)})
        matrices[name] = confusion_matrix(y_test, predictions).tolist()

    fitted = tuned.named_steps["preprocessor"]
    classifier = tuned.named_steps["classifier"]
    importance = pd.DataFrame(
        {
            "Feature": [
                name.replace("numeric__", "").replace("categorical__", "")
                for name in fitted.get_feature_names_out()
            ],
            "Importance": classifier.feature_importances_,
        }
    ).sort_values("Importance", ascending=False).head(15)

    return {
        "comparison": pd.DataFrame(rows).set_index("Model"),
        "matrices": matrices,
        "importance": importance,
        "test_rows": len(y_test),
    }


def chart_confusion_matrix(matrix: list[list[int]], title: str) -> go.Figure:
    """The notebook's confusion matrix as a labelled heat map."""
    labels = [PAID_LABEL, PROBLEM_LABEL]
    values = np.asarray(matrix)
    annotations = [
        dict(
            x=col, y=row, text=f"{values[row, col]:,}", showarrow=False,
            font=dict(color="#FFFFFF" if values[row, col] > values.max() * 0.55 else "#14181D"),
        )
        for row in range(2) for col in range(2)
    ]
    fig = go.Figure(
        go.Heatmap(
            z=values,
            x=labels,
            y=labels,
            colorscale="Blues",
            showscale=False,
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Loans: %{z:,}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Predicted label",
        yaxis_title="True label",
        template="plotly_white",
        height=420,
        margin=dict(l=90, r=30, t=64, b=64),
        font=dict(family="Inter, Helvetica Neue, sans-serif", size=12, color="#2A2F36"),
        annotations=annotations,
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def chart_importance(importance: pd.DataFrame) -> go.Figure:
    """Cell 43: top 15 features, notebook order and Blues gradient."""
    data = importance.sort_values("Importance", ascending=True).copy()
    normalised = data["Importance"] / data["Importance"].max()
    colours = px.colors.sample_colorscale("Blues", normalised.tolist())
    fig = go.Figure(
        go.Bar(
            x=data["Importance"],
            y=data["Feature"],
            orientation="h",
            marker_color=colours,
            hovertemplate="%{y}<br>Importance: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Top 15 Features Influencing Loan Repayment Predictions",
        xaxis_title="Feature Importance",
        yaxis_title="Feature",
        template="plotly_white",
        height=620,
        margin=dict(l=210, r=34, t=64, b=62),
        font=dict(family="Inter, Helvetica Neue, sans-serif", size=12, color="#2A2F36"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    return fig
