"""Nigerian Loan Analytics Dashboard.

White background with sidebar and KPI cards.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Nigerian Loan Analytics",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Palette
NAVY = "#0f2b46"
MUTED = "#5b6b7c"
BLUE = "#1f5c99"
TEAL = "#0f8a83"
AMBER = "#c9a227"
RISK = "#b5443a"
GREEN = "#1f7a4d"
CARD_BG = "#ffffff"
PAGE_BG = "#ffffff"
BORDER = "#dfe6ee"
GRID = "#e9eef4"

FONT_FAMILY = "Helvetica Neue, Helvetica, Arial, sans-serif"

_CUSTOM_CSS = f"""
<style>
    .stApp {{ background: #ffffff; font-family: {FONT_FAMILY}; }}
    html, body, [class*="css"] {{ font-family: {FONT_FAMILY}; }}
    h1 {{ color: {NAVY} !important; font-family: {FONT_FAMILY}; font-weight: 700 !important; font-size: 1.8rem !important; }}
    h2, h3, h4 {{ color: {NAVY} !important; font-family: {FONT_FAMILY}; font-weight: 700 !important; }}
    p, li, label {{ color: {NAVY}; }}
    [data-testid="stSidebar"] {{ background: #eef2f7; border-right: 1px solid {BORDER}; }}
    [data-testid="stSidebar"] * {{ color: #000000 !important; }}
    .block-container {{ padding-top: 1rem !important; padding-bottom: 2rem !important; }}
    [data-testid="stHeader"] {{ background: transparent !important; height: 2.5rem !important; }}
    .kpi-card {{ background: {CARD_BG}; border: 1px solid {BORDER}; border-radius: 10px; padding: 16px 20px; box-shadow: 0 1px 2px rgba(15,43,70,0.05); height: 100%; }}
    .kpi-label {{ font-size: 10.5px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.07em; color: {MUTED}; margin-bottom: 6px; }}
    .kpi-value {{ font-size: 24px; font-weight: 700; color: {NAVY}; line-height: 1.15; }}
    .kpi-sub {{ font-size: 12px; color: {MUTED}; margin-top: 5px; }}
    [data-testid="stVerticalBlockBorderWrapper"] {{ background: {CARD_BG}; border: 1px solid {BORDER}; border-radius: 10px; padding: 16px 18px 12px 18px; box-shadow: 0 1px 2px rgba(15,43,70,0.05); }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 8px; border-bottom: 2px solid {BORDER}; }}
    .stTabs [data-baseweb="tab"] {{ color: {MUTED}; font-weight: 600; }}
    .stTabs [aria-selected="true"] {{ color: {NAVY} !important; border-bottom: 2px solid {BLUE}; }}
</style>
"""
st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)

@st.cache_data
def load_data():
    data_path = Path(__file__).parent / "data" / "nigerian_loan_data.csv"
    if not data_path.exists():
        # Try alternative names
        for name in ["nigerian_loans.csv", "loans_nigeria.csv"]:
            p = Path(__file__).parent / "data" / name
            if p.exists():
                data_path = p
                break
    df = pd.read_csv(data_path)
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("### Filters")
    if 'loan_status' in df.columns:
        statuses = ['All'] + sorted(df['loan_status'].dropna().unique().tolist())
        status_filter = st.selectbox("Loan Status", statuses)
    else:
        status_filter = 'All'
    if 'purpose' in df.columns:
        purposes = ['All'] + sorted(df['purpose'].dropna().unique().tolist())
        purpose_filter = st.selectbox("Loan Purpose", purposes)
    else:
        purpose_filter = 'All'
    st.markdown("---")
    st.markdown("**Nigerian Loan Portfolio**")
    st.caption("Dashboard")

# Apply filters
df_filtered = df.copy()
if status_filter != 'All' and 'loan_status' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['loan_status'] == status_filter]
if purpose_filter != 'All' and 'purpose' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['purpose'] == purpose_filter]

# Header
st.title("Nigerian Loan Analytics Dashboard")
st.caption("Loan Portfolio Performance • Risk Analysis")

# KPIs
col1, col2, col3, col4 = st.columns(4)
with col1:
    total_loans = len(df_filtered)
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Total Loans</div><div class="kpi-value">{total_loans:,}</div></div>', unsafe_allow_html=True)
with col2:
    if 'loan_amount' in df_filtered.columns:
        total_amount = df_filtered['loan_amount'].sum()
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Total Loan Amount</div><div class="kpi-value">₦{total_amount:,.0f}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Total Loan Amount</div><div class="kpi-value">-</div></div>', unsafe_allow_html=True)
with col3:
    if 'loan_status' in df_filtered.columns:
        defaulted = df_filtered[df_filtered['loan_status'].str.contains('Default|Charged|Bad', case=False, na=False)].shape[0]
        default_rate = (defaulted / total_loans * 100) if total_loans > 0 else 0
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Default Rate</div><div class="kpi-value">{default_rate:.2f}%</div><div class="kpi-sub">{defaulted:,} defaulted loans</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Default Rate</div><div class="kpi-value">-</div></div>', unsafe_allow_html=True)
with col4:
    if 'interest_rate' in df_filtered.columns:
        avg_rate = df_filtered['interest_rate'].mean()
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Avg Interest Rate</div><div class="kpi-value">{avg_rate:.2f}%</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Avg Interest Rate</div><div class="kpi-value">-</div></div>', unsafe_allow_html=True)

st.markdown("---")

# Charts
tab1, tab2, tab3 = st.tabs(["Overview", "Risk Analysis", "Portfolio"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        if 'loan_status' in df_filtered.columns:
            status_counts = df_filtered['loan_status'].value_counts().reset_index()
            status_counts.columns = ['Status', 'Count']
            fig = px.bar(status_counts, x='Status', y='Count', title='Loan Status Distribution',
                         color_discrete_sequence=[BLUE])
            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        if 'purpose' in df_filtered.columns:
            purpose_counts = df_filtered['purpose'].value_counts().head(10).reset_index()
            purpose_counts.columns = ['Purpose', 'Count']
            fig = px.pie(purpose_counts, values='Count', names='Purpose', title='Top 10 Loan Purposes',
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)
    
    if 'loan_amount' in df_filtered.columns:
        fig = px.histogram(df_filtered, x='loan_amount', nbins=30, title='Loan Amount Distribution',
                          color_discrete_sequence=[BLUE])
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        if 'loan_amount' in df_filtered.columns and 'loan_status' in df_filtered.columns:
            df_risk = df_filtered.copy()
            df_risk['is_default'] = df_risk['loan_status'].str.contains('Default|Charged|Bad', case=False, na=False)
            fig = px.box(df_risk, x='is_default', y='loan_amount', title='Loan Amount by Default Status',
                         labels={'is_default': 'Defaulted', 'loan_amount': 'Loan Amount (₦)'},
                         color_discrete_sequence=[BLUE, RISK])
            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        if 'interest_rate' in df_filtered.columns and 'loan_status' in df_filtered.columns:
            df_risk = df_filtered.copy()
            df_risk['is_default'] = df_risk['loan_status'].str.contains('Default|Charged|Bad', case=False, na=False)
            fig = px.box(df_risk, x='is_default', y='interest_rate', title='Interest Rate by Default Status',
                         labels={'is_default': 'Defaulted', 'interest_rate': 'Interest Rate (%)'},
                         color_discrete_sequence=[BLUE, RISK])
            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Loan Portfolio Summary")
    summary_cols = []
    for col in ['loan_amount', 'interest_rate', 'term', 'monthly_payment']:
        if col in df_filtered.columns:
            summary_cols.append(col)
    if summary_cols:
        st.dataframe(df_filtered[summary_cols].describe().round(2), use_container_width=True)
    
    st.subheader("Recent Loans")
    display_cols = ['loan_id', 'loan_amount', 'purpose', 'interest_rate', 'loan_status'] if all(c in df_filtered.columns for c in ['loan_id', 'loan_amount', 'purpose', 'interest_rate', 'loan_status']) else df_filtered.columns[:8]
    st.dataframe(df_filtered[display_cols].head(100), use_container_width=True)

st.markdown("---")
st.caption("Nigerian Loan Analytics Dashboard | Data-driven insights for credit risk management")
