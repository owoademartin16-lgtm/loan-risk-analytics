"""Data loading, validation, filtering and summary helpers.

The dataset is the Lending Club-style loan file used by
``Loan Risk Analytics.ipynb`` (1,347,681 rows). Loading is cached with
``st.cache_resource`` so the CSV is read only once per app process.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent

DATASET_CANDIDATES = (
    "loan_risk_analytics.csv",
    "LC_loans_granting_model_dataset.csv",
)

REQUIRED_COLUMNS = [
    "id",
    "issue_d",
    "revenue",
    "dti_n",
    "loan_amnt",
    "fico_n",
    "experience_c",
    "emp_length",
    "purpose",
    "home_ownership_n",
    "addr_state",
    "zip_code",
    "Default",
    "title",
]

DTYPES = {
    "id": "int64",
    "issue_d": "string",
    "revenue": "float64",
    "dti_n": "float32",
    "loan_amnt": "int32",
    "fico_n": "float32",
    "experience_c": "int8",
    "emp_length": "category",
    "purpose": "category",
    "home_ownership_n": "category",
    "addr_state": "category",
    "zip_code": "category",
    "Default": "int8",
    "title": "category",
}

# Band definitions — identical to the notebook so dashboard and notebook agree.
FICO_BAND_BINS = [0, 580, 620, 660, 700, 740, 850]
FICO_BAND_LABELS = ["Below 580", "580–619", "620–659", "660–699", "700–739", "740+"]

DTI_BAND_BINS = [0, 10, 20, 30, 40, 50, np.inf]
DTI_BAND_LABELS = ["0–10", "10–20", "20–30", "30–40", "40–50", "50+"]

AMOUNT_BAND_BINS = [0, 5000, 10000, 15000, 20000, 25000, 30000, np.inf]
AMOUNT_BAND_LABELS = [
    "$0–5k", "$5k–10k", "$10k–15k", "$15k–20k",
    "$20k–25k", "$25k–30k", "$30k+",
]

# Minimum number of loans required for a group to enter risk rankings.
MIN_SAMPLE_SIZE = 100

# Cap for the Data Explorer CSV export so the browser stays responsive.
EXPLORER_EXPORT_CAP = 50_000


# ---------------------------------------------------------------------------
# Dataset discovery and loading
# ---------------------------------------------------------------------------
def find_dataset() -> Path:
    """Locate the loan CSV in the project folder (or via env var)."""
    env_path = os.environ.get("LOAN_DATASET_PATH")
    if env_path:
        candidate = Path(env_path).expanduser()
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(
            f"LOAN_DATASET_PATH is set to '{env_path}' but that file does not exist."
        )

    search_dirs = [APP_DIR, APP_DIR / "data"]
    for directory in search_dirs:
        for name in DATASET_CANDIDATES:
            candidate = directory / name
            if candidate.is_file():
                return candidate

    names = " or ".join(f"'{name}'" for name in DATASET_CANDIDATES)
    raise FileNotFoundError(
        f"Dataset file not found. Place the CSV in the project folder as {names} "
        f"(searched: {', '.join(str(d) for d in search_dirs)}), "
        "or set the LOAN_DATASET_PATH environment variable to the file location."
    )


@st.cache_resource(show_spinner="Loading and validating loan dataset (1.35M rows)…")
def load_dataset() -> tuple[pd.DataFrame, dict]:
    """Load the CSV with memory-efficient dtypes and validate its structure.

    Returns the DataFrame plus a report dictionary describing the checks
    that were performed (row counts, id uniqueness, date parsing, …).
    """
    path = find_dataset()

    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = [col for col in REQUIRED_COLUMNS if col not in header]
    if missing:
        raise ValueError(
            f"The dataset at '{path}' is missing required columns: {missing}. "
            f"Expected columns: {REQUIRED_COLUMNS}."
        )

    # Only read the required columns; extra columns such as 'desc'
    # (91% missing in the source file) are not loaded at all.
    df = pd.read_csv(
        path,
        usecols=REQUIRED_COLUMNS,
        dtype={col: DTYPES[col] for col in REQUIRED_COLUMNS},
        low_memory=False,
    )

    report: dict = {"file": str(path.name), "rows_loaded": int(len(df))}

    # --- Target column -----------------------------------------------------
    valid_target = df["Default"].isin([0, 1])
    report["unexpected_target_values"] = int((~valid_target).sum())
    if not valid_target.all():
        df = df[valid_target].copy()

    # --- Loan identifier ---------------------------------------------------
    report["unique_ids"] = int(df["id"].nunique())
    report["duplicate_ids"] = int(df["id"].duplicated().sum())

    # --- Issue date --------------------------------------------------------
    parsed = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    if parsed.isna().mean() > 0.01:  # fallback if the expected format fails
        parsed = pd.to_datetime(df["issue_d"], errors="coerce")
    report["unparsed_dates"] = int(parsed.isna().sum())
    df["issue_d"] = parsed
    df["issue_month"] = parsed.dt.to_period("M").dt.to_timestamp()
    df["issue_year"] = parsed.dt.year.astype("Int16")
    report["date_min"] = parsed.min()
    report["date_max"] = parsed.max()

    # --- Light cleanup (no rows dropped) -----------------------------------
    report["title_missing"] = int(df["title"].isna().sum())
    for col in ("title", "zip_code"):
        if df[col].isna().any():
            if "Unknown" not in df[col].cat.categories:
                df[col] = df[col].cat.add_categories(["Unknown"])
            df[col] = df[col].fillna("Unknown")

    report["rows_final"] = int(len(df))
    return df, report


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
def apply_filters(
    df: pd.DataFrame,
    purposes: list[str],
    states: list[str],
    loan_range: tuple[int, int],
    fico_range: tuple[int, int],
    dti_range: tuple[int, int],
    date_range: tuple | None,
) -> pd.DataFrame:
    """Return the rows matching every active sidebar filter."""
    mask = pd.Series(True, index=df.index)

    if purposes:  # empty selection means "all"
        mask &= df["purpose"].isin(purposes)
    if states:
        mask &= df["addr_state"].isin(states)

    mask &= df["loan_amnt"].between(loan_range[0], loan_range[1])
    mask &= df["fico_n"].between(fico_range[0], fico_range[1])
    mask &= df["dti_n"].between(dti_range[0], dti_range[1])

    if date_range is not None:
        start, end = date_range
        mask &= df["issue_d"].between(pd.Timestamp(start), pd.Timestamp(end))

    return df[mask]


# ---------------------------------------------------------------------------
# KPIs and grouped summaries
# ---------------------------------------------------------------------------
def portfolio_kpis(df: pd.DataFrame) -> dict:
    """Core portfolio figures (loan counts use rows; ids are unique)."""
    n = len(df)
    return {
        "total_loans": n,
        "total_amount": float(df["loan_amnt"].sum()) if n else 0.0,
        "avg_amount": float(df["loan_amnt"].mean()) if n else float("nan"),
        "defaults": int(df["Default"].sum()),
        "default_rate": float(df["Default"].mean() * 100) if n else float("nan"),
        "avg_fico": float(df["fico_n"].mean()) if n else float("nan"),
    }


def group_summary(
    df: pd.DataFrame,
    col: str,
    categories: list | None = None,
) -> pd.DataFrame:
    """Loan count, defaulted count and default rate for every value of ``col``.

    Groups with no loans keep a NaN rate (never division by zero). When
    ``categories`` is given the result is reindexed to that exact order.
    """
    grouped = (
        df.groupby(col, dropna=False, observed=True)["Default"]
        .agg(total_loans="size", defaulted="sum")
        .reset_index()
    )
    grouped["default_rate"] = np.where(
        grouped["total_loans"] > 0,
        grouped["defaulted"] / grouped["total_loans"] * 100,
        np.nan,
    )
    if categories is not None:
        grouped = (
            grouped.set_index(col)
            .reindex(categories)
            .rename_axis(col)
            .reset_index()
        )
        grouped[["total_loans", "defaulted"]] = (
            grouped[["total_loans", "defaulted"]].fillna(0).astype(int)
        )
    return grouped


def band_summary(
    df: pd.DataFrame,
    col: str,
    bins: list,
    labels: list,
) -> pd.DataFrame:
    """Default-rate summary for numeric bands (FICO, DTI, loan amount)."""
    banded = df.assign(band=pd.cut(df[col], bins=bins, labels=labels, include_lowest=True))
    return group_summary(banded, "band", categories=labels)


def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Default rate by loan issue month (loans without a date are excluded)."""
    dated = df.dropna(subset=["issue_month"])
    return (
        dated.groupby("issue_month")["Default"]
        .agg(total_loans="size", defaulted="sum")
        .assign(default_rate=lambda t: t["defaulted"] / t["total_loans"] * 100)
        .reset_index()
    )


def top_states(
    df: pd.DataFrame,
    min_loans: int = MIN_SAMPLE_SIZE,
    n: int = 10,
) -> pd.DataFrame:
    """The ``n`` states with the highest default rate (min. ``min_loans`` loans)."""
    summary = group_summary(df, "addr_state")
    qualified = summary[summary["total_loans"] >= min_loans]
    return qualified.sort_values("default_rate", ascending=False).head(n)


# ---------------------------------------------------------------------------
# Display formatting
# ---------------------------------------------------------------------------
def fmt_money(value: float) -> str:
    """Whole dollars with thousands separators (no abbreviations)."""
    return "—" if pd.isna(value) else f"${value:,.0f}"


def fmt_int(value: float) -> str:
    return "—" if pd.isna(value) else f"{value:,.0f}"


def fmt_pct(value: float) -> str:
    return "—" if pd.isna(value) else f"{value:.2f}%"
