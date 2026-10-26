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
BORDER = "#dfe6ee"
GRID = "#e9eef4"
FONT_FAMILY = "Helvetica Neue, Helvetica, Arial, sans-serif"

_CUSTOM_CSS = f"""
<style>
    .stApp {{ background: #ffffff; font-family: {FONT_FAMILY}; }}
    html, body, [class*="css"] {{ font-family: {FONT_FAMILY}; }}
    h1 {{ color: {NAVY} !important; font-weight: 700 !important; font-size: 1.8rem !important; }}
    h2, h3, h4 {{ color: {NAVY} !important; font-weight: 700 !important; }}
    p, li, label {{ color: {NAVY}; }}
    [data-testid="stSidebar"] {{ background: #eef2f7; border-right: 1px solid {BORDER}; }}
    [data-testid="stSidebar"] * {{ color: #000000 !important; }}
    .block-container {{ padding-top: 1rem !important; padding-bottom: 2rem !important; }}
    [data-testid="stHeader"] {{ background: transparent !important; height: 2.5rem !important; }}
    .kpi-card {{ background: #ffffff; border: 1px solid {BORDER}; border-radius: 10px; padding: 16px 20px; box-shadow: 0 1px 2px rgba(15,43,70,0.05); height: 100%; }}
    .kpi-label {{ font-size: 10.5px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.07em; color: {MUTED}; margin-bottom: 6px; }}
    .kpi-value {{ font-size: 24px; font-weight: 700; color: {NAVY}; line-height: 1.15; }}
    .kpi-sub {{ font-size: 12px; color: {MUTED}; margin-top: 5px; }}
    [data-testid="stVerticalBlockBorderWrapper"] {{ background: #ffffff; border: 1px solid {BORDER}; border-radius: 10px; padding: 16px 18px 12px 18px; box-shadow: 0 1px 2px rgba(15,43,70,0.05); }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 8px; border-bottom: 2px solid {BORDER}; }}
    .stTabs [data-baseweb="tab"] {{ color: {MUTED}; font-weight: 600; }}
    .stTabs [aria-selected="true"] {{ color: {NAVY} !important; border-bottom: 2px solid {BLUE}; }}
</style>
"""
st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)

@st.cache_data
def load_data():
    data_path = Path(__file__).parent / "data" / "loans_clean.parquet"
    df = pd.read_parquet(data_path)
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("### Filters")
    if 'purpose' in df.columns:
        purposes = ['All'] + sorted(df['purpose'].dropna().unique().tolist())
        purpose_filter = st.selectbox("Loan Purpose", purposes)
    else:
        purpose_filter = 'All'
    if 'home_ownership_n' in df.columns:
        homes = ['All'] + sorted(df['home_ownership_n'].dropna().unique().tolist())
        home_filter = st.selectbox("Home Ownership", homes)
    else:
        home_filter = 'All'
    st.markdown("---")
    st.markdown("**Nigerian Loan Portfolio**")

# Apply filters
df_filtered = df.copy()
if purpose_filter != 'All' and 'purpose' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['purpose'] == purpose_filter]
if home_filter != 'All' and 'home_ownership_n' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['home_ownership_n'] == home_filter]

# Header
st.title("Nigerian Loan Analytics Dashboard")
st.caption("Loan Portfolio Performance • Credit Risk Analysis")

# KPIs - check notebook figures
col1, col2, col3, col4 = st.columns(4)
with col1:
    total_loans = len(df_filtered)
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Total Loans</div><div class="kpi-value">{total_loans:,}</div></div>', unsafe_allow_html=True)
with col2:
    total_amount = df_filtered['loan_amnt'].sum() if 'loan_amnt' in df_filtered.columns else 0
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Total Loan Amount</div><div class="kpi-value">₦{total_amount:,.0f}</div></div>', unsafe_allow_html=True)
with col3:
    if 'Default' in df_filtered.columns:
        defaulted = int(df_filtered['Default'].sum())
        default_rate = (defaulted / total_loans * 100) if total_loans > 0 else 0
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Default Rate</div><div class="kpi-value">{default_rate:.2f}%</div><div class="kpi-sub">{defaulted:,} defaulted loans</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Default Rate</div><div class="kpi-value">-</div></div>', unsafe_allow_html=True)
with col4:
    avg_fico = df_filtered['fico_n'].mean() if 'fico_n' in df_filtered.columns else 0
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Avg FICO Score</div><div class="kpi-value">{avg_fico:.0f}</div></div>', unsafe_allow_html=True)

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["Overview", "Risk Analysis", "Portfolio"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        if 'purpose' in df_filtered.columns:
            purpose_counts = df_filtered['purpose'].value_counts().head(10).reset_index()
            purpose_counts.columns = ['Purpose', 'Count']
            fig = px.bar(purpose_counts, x='Purpose', y='Count', title='Loans by Purpose',
                         color_discrete_sequence=[BLUE])
            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', xaxis_title="", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        if 'Default' in df_filtered.columns:
            default_counts = df_filtered['Default'].value_counts().reset_index()
            default_counts.columns = ['Default', 'Count']
            default_counts['Status'] = default_counts['Default'].map({0: 'Non-Default', 1: 'Default'})
            fig = px.pie(default_counts, values='Count', names='Status', title='Default Distribution',
                         color_discrete_sequence=[TEAL, RISK])
            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)
    
    if 'loan_amnt' in df_filtered.columns:
        fig = px.histogram(df_filtered, x='loan_amnt', nbins=40, title='Loan Amount Distribution',
                          color_discrete_sequence=[BLUE])
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', xaxis_title="Loan Amount (₦)", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        if 'fico_n' in df_filtered.columns and 'Default' in df_filtered.columns:
            df_risk = df_filtered.copy()
            df_risk['Status'] = df_risk['Default'].map({0: 'Non-Default', 1: 'Default'})
            fig = px.box(df_risk, x='Status', y='fico_n', title='FICO Score by Default Status',
                         color='Status', color_discrete_map={'Non-Default': TEAL, 'Default': RISK})
            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', xaxis_title="", yaxis_title="FICO Score")
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        if 'dti_n' in df_filtered.columns and 'Default' in df_filtered.columns:
            df_risk = df_filtered.copy()
            df_risk['Status'] = df_risk['Default'].map({0: 'Non-Default', 1: 'Default'})
            fig = px.box(df_risk, x='Status', y='dti_n', title='DTI by Default Status',
                         color='Status', color_discrete_map={'Non-Default': TEAL, 'Default': RISK})
            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', xaxis_title="", yaxis_title="DTI")
            st.plotly_chart(fig, use_container_width=True)
    
    if 'purpose' in df_filtered.columns and 'Default' in df_filtered.columns:
        purpose_default = df_filtered.groupby('purpose')['Default'].agg(['sum', 'count']).reset_index()
        purpose_default.columns = ['Purpose', 'Defaults', 'Total']
        purpose_default['Default Rate (%)'] = (purpose_default['Defaults'] / purpose_default['Total'] * 100).round(2)
        purpose_default = purpose_default.sort_values('Default Rate (%)', ascending=False).head(10)
        fig = px.bar(purpose_default, x='Purpose', y='Default Rate (%)', title='Default Rate by Loan Purpose (Top 10)',
                     color_discrete_sequence=[RISK])
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Portfolio Summary")
    summary_data = []
    if 'loan_amnt' in df_filtered.columns:
        summary_data.append({'Metric': 'Average Loan Amount', 'Value': f"₦{df_filtered['loan_amnt'].mean():,.0f}"})
    if 'fico_n' in df_filtered.columns:
        summary_data.append({'Metric': 'Average FICO Score', 'Value': f"{df_filtered['fico_n'].mean():.0f}"})
    if 'dti_n' in df_filtered.columns:
        summary_data.append({'Metric': 'Average DTI', 'Value': f"{df_filtered['dti_n'].mean():.2f}"})
    if 'revenue' in df_filtered.columns:
        summary_data.append({'Metric': 'Average Revenue', 'Value': f"₦{df_filtered['revenue'].mean():,.0f}"})
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
    
    st.subheader("Loan Data Sample")
    display_cols = [c for c in ['id', 'loan_amnt', 'purpose', 'fico_n', 'dti_n', 'home_ownership_n', 'Default'] if c in df_filtered.columns]
    st.dataframe(df_filtered[display_cols].head(100), use_container_width=True)

st.markdown("---")
st.caption("Nigerian Loan Analytics Dashboard | Powered by Lending Club Data")
