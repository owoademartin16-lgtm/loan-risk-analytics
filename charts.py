"""Plotly figures mirroring the charts from ``Loan Risk Analytics.ipynb``.

Every function takes either the filtered DataFrame or a pre-computed
summary table and returns a ``plotly.graph_objects.Figure`` styled with
the shared ``loan_light`` template. The eight charts correspond one-to-one
to the matplotlib charts in the notebook (see README.md for the mapping).
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

import styles

# Register the shared light template once at import time.
pio.templates[styles.TEMPLATE_NAME] = styles.build_plotly_template()


def _finalize(
    fig: go.Figure,
    *,
    height: int = 420,
    x_title: str | None = None,
    y_title: str | None = None,
    legend: bool = False,
) -> go.Figure:
    """Apply the shared template and layout to a figure."""
    fig.update_layout(
        template=styles.TEMPLATE_NAME,
        height=height,
        showlegend=legend,
        bargap=0.28,
    )
    if x_title:
        fig.update_xaxes(title_text=x_title)
    if y_title:
        fig.update_yaxes(title_text=y_title, rangemode="tozero")
    return fig


# ---------------------------------------------------------------------------
# 1. Distribution of loan outcomes (notebook cell 15)
# ---------------------------------------------------------------------------
def outcome_distribution(df: pd.DataFrame) -> go.Figure:
    counts = df["Default"].value_counts().reindex([0, 1]).fillna(0).astype(int)
    labels = ["Fully Paid", "Defaulted"]
    total = int(counts.sum())
    shares = counts / total * 100 if total else counts * 0.0

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=counts.values,
            marker_color=[styles.TEAL, styles.RISK],
            text=[f"{v:,}" for v in counts.values],
            textposition="outside",
            customdata=shares.values,
            hovertemplate=(
                "%{x}<br>Loans: %{y:,}<br>Share of selection: %{customdata:.2f}%"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        yaxis_range=[0, max(counts.values.max() * 1.2, 1)],
        margin=dict(t=48),
    )
    return _finalize(fig, height=400, y_title="Number of loans")


# ---------------------------------------------------------------------------
# 2–4. Default rate by band (notebook cells 16, 17, 19)
# ---------------------------------------------------------------------------
def default_rate_bars(
    summary: pd.DataFrame,
    x_col: str = "band",
    *,
    title: str | None = None,
) -> go.Figure:
    """Vertical bars of default rate per band with loan counts in the hover.

    ``summary`` comes from ``loan_data.band_summary`` and contains the
    columns ``total_loans``, ``defaulted`` and ``default_rate``.
    """
    text = [
        None if pd.isna(rate) else f"{rate:.1f}%"
        for rate in summary["default_rate"]
    ]
    fig = go.Figure(
        go.Bar(
            x=summary[x_col].astype(str),
            y=summary["default_rate"],
            marker_color=styles.BLUE,
            text=text,
            textposition="outside",
            customdata=summary[["total_loans", "defaulted"]],
            hovertemplate=(
                f"{x_col}: %{{x}}<br>"
                "Default rate: %{y:.2f}%<br>"
                "Loans: %{customdata[0]:,}<br>"
                "Defaulted: %{customdata[1]:,}"
                "<extra></extra>"
            ),
        )
    )
    return _finalize(fig, height=400, x_title=title, y_title="Default rate (%)")


# ---------------------------------------------------------------------------
# 5. Default rate by employment length (notebook cell 23)
# ---------------------------------------------------------------------------
def default_rate_by_emp_length(summary: pd.DataFrame) -> go.Figure:
    """Employment-length bars, sorted by default rate (as in the notebook)."""
    ordered = summary.dropna(subset=["band"]).sort_values(
        "default_rate", ascending=False
    )
    text = [f"{rate:.1f}%" for rate in ordered["default_rate"]]
    fig = go.Figure(
        go.Bar(
            x=ordered["band"].astype(str),
            y=ordered["default_rate"],
            marker_color=styles.BLUE,
            text=text,
            textposition="outside",
            customdata=ordered[["total_loans", "defaulted"]],
            hovertemplate=(
                "Employment length: %{x}<br>"
                "Default rate: %{y:.2f}%<br>"
                "Loans: %{customdata[0]:,}<br>"
                "Defaulted: %{customdata[1]:,}"
                "<extra></extra>"
            ),
        )
    )
    fig.update_xaxes(tickangle=-30)
    return _finalize(fig, height=420, y_title="Default rate (%)")


# ---------------------------------------------------------------------------
# 6–7. Horizontal default-rate bars (notebook cells 18, 20)
# ---------------------------------------------------------------------------
def default_rate_barh(
    summary: pd.DataFrame,
    cat_col: str,
    *,
    category_label: str,
) -> go.Figure:
    """Horizontal bars of default rate; ``summary`` must already be sorted."""
    fig = go.Figure(
        go.Bar(
            x=summary["default_rate"],
            y=summary[cat_col].astype(str),
            orientation="h",
            marker_color=styles.BLUE,
            text=[f"{rate:.1f}%" for rate in summary["default_rate"]],
            textposition="outside",
            customdata=summary[["total_loans", "defaulted"]],
            hovertemplate=(
                f"{category_label}: %{{y}}<br>"
                "Default rate: %{x:.2f}%<br>"
                "Loans: %{customdata[0]:,}<br>"
                "Defaulted: %{customdata[1]:,}"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return _finalize(fig, height=460, x_title="Default rate (%)", y_title=category_label)


# ---------------------------------------------------------------------------
# 8. Default rate by issue month (notebook cell 21)
# ---------------------------------------------------------------------------
def default_rate_by_month(summary: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=summary["issue_month"],
            y=summary["default_rate"],
            mode="lines+markers",
            line=dict(color=styles.BLUE, width=2),
            marker=dict(size=4, color=styles.BLUE),
            customdata=summary["total_loans"],
            hovertemplate=(
                "Issue month: %b %Y<br>"
                "Default rate: %{y:.2f}%<br>"
                "Loans issued: %{customdata:,}"
                "<extra></extra>"
            ),
        )
    )
    return _finalize(fig, height=440, x_title="Loan issue month", y_title="Default rate (%)")
