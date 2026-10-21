"""Default Risk Predictor page.

Scores a single loan application with the Random Forest saved by the
notebook and explains the result. The sidebar filters do not apply here —
this page predicts for one loan you configure.
"""

from __future__ import annotations

import streamlit as st

import styles
from predictor import (
    EMP_LENGTH_OPTIONS,
    HOME_OPTIONS,
    PURPOSE_OPTIONS,
    STATE_OPTIONS,
    build_gauge,
    held_out_metrics,
    load_model,
    predict_default,
    risk_band,
)


def render_predictor(_filtered, full) -> None:
    """Render the prediction page (signature matches the other sections)."""
    st.caption(
        "Score one loan application with the Random Forest trained in "
        "``Loan Risk Analytics.ipynb``. The sidebar filters do not affect "
        "this page — it predicts for a single loan you configure below."
    )

    try:
        model = load_model()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    # ---- Input form ------------------------------------------------------
    with st.container(border=True):
        st.subheader("Loan terms")
        c1, c2, c3 = st.columns(3)
        with c1:
            loan_amnt = st.slider(
                "Loan amount ($)", 500, 40000, 10000, step=500, format="$%d",
                key="p_amnt",
            )
        with c2:
            fico_n = st.slider(
                "FICO score", 612.0, 847.5, 690.0, step=2.5, format="%.1f",
                key="p_fico",
            )
        with c3:
            dti_n = st.slider(
                "Debt-to-income (DTI)", 0.0, 99.0, 20.0, step=0.5, format="%.1f",
                key="p_dti",
            )

    with st.container(border=True):
        st.subheader("Borrower profile")
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            emp_length = st.selectbox("Employment length", EMP_LENGTH_OPTIONS,
                                      index=len(EMP_LENGTH_OPTIONS) - 2)
        with p2:
            purpose = st.selectbox("Loan purpose", PURPOSE_OPTIONS,
                                   index=PURPOSE_OPTIONS.index("debt_consolidation"))
        with p3:
            home = st.selectbox("Home ownership", HOME_OPTIONS, index=0)
        with p4:
            state = st.selectbox("State", STATE_OPTIONS,
                                 index=STATE_OPTIONS.index("CA"))

    assess = st.button("Assess default risk", type="primary", width="stretch")

    # ---- Result ----------------------------------------------------------
    if assess:
        probability = predict_default(
            model, loan_amnt, fico_n, dti_n, emp_length, purpose, home, state
        )
        band, colour = risk_band(probability)
        verdict = "Likely to default" if probability > 0.5 else "Likely to be repaid"

        st.divider()
        st.subheader("Risk assessment")
        st.markdown(
            f'<div class="pred-banner" style="background:{colour};">'
            f'<p class="pred-verdict">{verdict}</p>'
            f'<p class="pred-prob">Predicted default probability '
            f"{probability * 100:.1f}% · {band}</p></div>",
            unsafe_allow_html=True,
        )

        gcol, scol = st.columns([5, 4], gap="large")
        with gcol:
            st.plotly_chart(build_gauge(probability, colour), width="stretch")
        with scol:
            st.markdown("**What the risk levels mean**")
            st.markdown(
                f"- **Low risk** (below {30}%) — repayments historically "
                "reliable; routine monitoring."
                f"\n- **Moderate risk** (30–60%) — elevated; worth a closer "
                "affordability review."
                f"\n- **High risk** (above 60%) — strongly associated with "
                "default in the training data; treat with caution."
            )
            st.markdown(
                f"**Applicant:** {int(loan_amnt):,} · FICO {fico_n:.1f} · "
                f"DTI {dti_n:.1f} · {emp_length} · {purpose.replace('_', ' ')} "
                f"· {home.title()} · {state}"
            )

        if probability > 0.5:
            st.warning(
                "This application carries characteristics the model associates "
                "with default. Consider a fuller affordability check before "
                "approving."
            )
        else:
            st.success(
                "This application scores in the model's lower-risk range."
            )

        st.caption(
            "This is a historical association learned from past loans, not a "
            "guarantee or financial advice. It does not establish causation."
        )

    # ---- Model performance ------------------------------------------------
    st.divider()
    with st.expander("About this model"):
        from loan_data import find_dataset

        try:
            metrics = held_out_metrics(model, str(find_dataset()))
        except Exception:
            metrics = None

        st.markdown(
            "**Model**  \nRandom Forest Classifier (200 trees, max depth 16), "
            "saved from ``Loan Risk Analytics.ipynb``.  \n"
            "**Input**  \nloan amount, FICO score, DTI, employment length, "
            "purpose, home ownership, state.  \n"
            "**Purpose**  \nEstimate a loan's probability of default "
            "(``Default = 1``)."
        )
        if metrics:
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
            m2.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            m3.metric("Precision", f"{metrics['precision']:.3f}")
            m4.metric("Recall", f"{metrics['recall']:.3f}")
            m5.metric("F1", f"{metrics['f1']:.3f}")
            st.caption(
                f"Measured on the held-out chronological test split of "
                f"{metrics['test_size']:,} loans (the most recent 20% of "
                f"rows), which the model never saw during training. Actual "
                f"default rate in that split: {metrics['actual_rate']:.1f}%."
            )
        else:
            st.caption(
                "Held-out metrics unavailable (the training CSV could not be "
                "read at prediction time)."
            )
