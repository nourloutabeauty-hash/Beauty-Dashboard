import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import datetime
import io
import re
import os
from collections import OrderedDict

# ================================================================================================
# SUPPRESS WARNINGS
# ================================================================================================
warnings.filterwarnings('ignore')

# ================================================================================================
# PAGE CONFIG
# ================================================================================================
st.set_page_config(
    page_title="💄 Beauty Search Analytics",
    layout="wide",
    page_icon="✨",
    initial_sidebar_state="expanded"
)

# ================================================================================================
# UNIFIED UTILITY FUNCTIONS (NO DUPLICATES)
# ================================================================================================

def format_number(num):
    """Unified number formatting with K/M suffix"""
    if pd.isna(num):
        return "0"
    num = float(num)
    if num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    else:
        return f"{num:,.0f}"

def format_percentage(num):
    """Unified percentage formatting"""
    if pd.isna(num):
        return "0.0%"
    return f"{float(num):.1f}%"

# ================================================================================================
# UNIFIED CSS (LOADED ONCE)
# ================================================================================================

@st.cache_resource
def load_css():
    """Load CSS once and cache it"""
    st.markdown("""
    <style>
    /* Global styling */
    .main {
        background: linear-gradient(135deg, #FFF0F5 0%, #FFE4E9 100%);
    }
    
    /* Header styling */
    .main-header {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(45deg, #AD1457, #D81B60, #F06292);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #C2185B;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 600;
    }
    
    /* KPI Card */
    .kpi-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #FFF5F7 100%);
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(216, 27, 96, 0.15);
        transition: transform 0.3s ease;
        border: 2px solid rgba(240, 98, 146, 0.2);
        height: 150px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .kpi-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 12px 35px rgba(216, 27, 96, 0.25);
    }
    
    .kpi-value {
        font-size: 2.5rem;
        font-weight: 900;
        background: linear-gradient(45deg, #AD1457, #D81B60);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 10px 0;
    }
    
    .kpi-label {
        color: #EC407A;
        font-size: 1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .kpi-icon {
        font-size: 2rem;
        margin-bottom: 5px;
    }
    
    /* Mini Metric Card */
    .mini-metric {
        background: linear-gradient(135deg, #EC407A 0%, #F06292 50%, #F48FB1 100%);
        padding: 18px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(236, 64, 122, 0.25);
        transition: transform 0.3s ease;
        height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        border: 2px solid rgba(255, 255, 255, 0.3);
    }
    
    .mini-metric:hover {
        transform: translateY(-6px) scale(1.02);
        box-shadow: 0 12px 35px rgba(236, 64, 122, 0.35);
    }
    
    .mini-metric .value {
        font-size: 1.8rem;
        font-weight: 900;
        color: #FFFFFF;
        margin-bottom: 6px;
        text-shadow: 1px 1px 3px rgba(173, 20, 87, 0.3);
    }
    
    .mini-metric .label {
        font-size: 0.95rem;
        color: #FFF0F5;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    
    .mini-metric .icon {
        font-size: 1.4rem;
        color: #FFFFFF;
        margin-bottom: 8px;
    }
    
    /* Insight Box */
    .insight-box {
        background: linear-gradient(135deg, #FCE4EC 0%, #F8BBD0 100%);
        padding: 20px;
        border-left: 6px solid #EC407A;
        border-radius: 12px;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(236, 64, 122, 0.15);
        transition: transform 0.3s ease;
    }
    
    .insight-box:hover {
        transform: translateX(8px);
        box-shadow: 0 6px 25px rgba(236, 64, 122, 0.25);
    }
    
    .insight-box h4 {
        color: #AD1457;
        margin-bottom: 10px;
        font-weight: 700;
    }
    
    .insight-box p {
        color: #C2185B;
        line-height: 1.6;
    }
    
    .insight-box ul li {
        color: #880E4F;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        background: linear-gradient(135deg, #FCE4EC 0%, #F8BBD0 100%);
        padding: 15px;
        border-radius: 15px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 55px;
        border-radius: 12px;
        padding: 15px 20px;
        font-weight: 700;
        background: linear-gradient(135deg, #FFFFFF 0%, #FFF5F7 100%);
        color: #C2185B;
        border: 2px solid rgba(236, 64, 122, 0.2);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #EC407A 0%, #F06292 100%);
        color: #FFFFFF !important;
        border-color: #D81B60;
        box-shadow: 0 4px 15px rgba(236, 64, 122, 0.3);
    }
    
    /* Table styling */
    .beauty-table {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(216, 27, 96, 0.12);
        border-radius: 12px;
        overflow: hidden;
    }
    
    .beauty-table thead th {
        background: linear-gradient(135deg, #D81B60 0%, #EC407A 100%);
        color: white;
        font-weight: 700;
        padding: 12px;
        text-align: center;
    }
    
    .beauty-table tbody td {
        text-align: center;
        padding: 10px 12px;
        border: 1px solid #F8BBD0;
    }
    
    .beauty-table tbody tr:nth-child(even) {
        background-color: #FFF5F7;
    }
    
    .beauty-table tbody tr:hover {
        background-color: #FCE4EC;
        transition: background-color 0.2s;
    }
    
    /* Download button */
    div.stDownloadButton > button {
        background: linear-gradient(135deg, #D81B60 0%, #EC407A 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.5rem 1rem !important;
        border-radius: 0.5rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(216, 27, 96, 0.3) !important;
    }
    
    div.stDownloadButton > button:hover {
        background: linear-gradient(135deg, #AD1457 0%, #D81B60 100%) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Data source indicator */
    .data-source-badge {
        background: linear-gradient(135deg, #4CAF50 0%, #66BB6A 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        display: inline-block;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(76, 175, 80, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)

# Load CSS once
load_css()

# ================================================================================================
# UNIFIED DATA LOADING (CACHED)
# ================================================================================================

@st.cache_data(show_spinner=False)
def load_data_from_file(file_path):
    """Load data from local file path"""
    try:
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path, engine='openpyxl')
        else:
            df = pd.read_csv(file_path)
        
        # Fix unhashable types
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].apply(
                lambda x: str(x) if isinstance(x, (list, dict, tuple)) else x
            )
        
        return df
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        return None

@st.cache_data(show_spinner=False)
def load_data_from_upload(file):
    """Load data from uploaded file"""
    try:
        if file.name.endswith('.xlsx'):
            df = pd.read_excel(file, engine='openpyxl')
        else:
            df = pd.read_csv(file, low_memory=False)
        
        # Fix unhashable types
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].apply(
                lambda x: str(x) if isinstance(x, (list, dict, tuple)) else x
            )
        
        return df
    except Exception as e:
        st.error(f"❌ Error loading uploaded file: {e}")
        return None

# ================================================================================================
# UNIFIED DATA PROCESSING (CACHED)
# ================================================================================================

@st.cache_data(show_spinner=False)
def process_data(_df):
    """Unified data processing - all transformations in one place"""
    df = _df.copy()
    
    # Optimize dtypes
    categorical_cols = ['Department', 'Category', 'Sub Category', 'Class', 'Brand']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    # Convert dates
    if 'start_date' in df.columns:
        df['Date'] = pd.to_datetime(df['start_date'], errors='coerce')
    else:
        df['Date'] = pd.NaT
    
    # Numeric columns
    numeric_cols = ['count', 'Clicks', 'Conversions', 'Click Through Rate', 
                   'Converion Rate', 'averageClickPosition']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Unified column mapping
    df['Counts'] = df['count'] if 'count' in df.columns else 0
    df['clicks'] = df['Clicks'] if 'Clicks' in df.columns else 0
    df['conversions'] = df['Conversions'] if 'Conversions' in df.columns else 0
    
    # Calculate rates (unified)
    df['CTR'] = df['Click Through Rate'] * 100 if 'Click Through Rate' in df.columns else 0
    df['CR'] = df['Converion Rate'] * 100 if 'Converion Rate' in df.columns else 0
    
    # Time features (unified)
    df['Month'] = df['Date'].dt.strftime('%B %Y')
    df['Month_Short'] = df['Date'].dt.strftime('%b %Y')
    df['month_sort'] = df['Date'].dt.to_period('M')
    df['Year'] = df['Date'].dt.year
    df['Quarter'] = df['Date'].dt.quarter
    
    # Query features
    if 'search' in df.columns:
        df['normalized_query'] = df['search'].astype(str)
        df['query_length'] = df['normalized_query'].str.len()
    
    # Optimize memory
    df = optimize_memory(df)
    
    return df

def optimize_memory(df):
    """Unified memory optimization"""
    # Downcast integers
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    # Downcast floats
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    return df

# ================================================================================================
# UNIFIED METRICS CALCULATION (CACHED)
# ================================================================================================

@st.cache_data(show_spinner=False)
def calculate_all_metrics(_df):
    """Calculate ALL metrics once - no redundant calculations"""
    metrics = {}
    
    # Basic metrics
    metrics['total_searches'] = int(_df['Counts'].sum())
    metrics['total_clicks'] = int(_df['clicks'].sum())
    metrics['total_conversions'] = int(_df['conversions'].sum())
    metrics['avg_ctr'] = _df['CTR'].mean()
    metrics['avg_cr'] = _df['CR'].mean()
    metrics['unique_queries'] = _df['search'].nunique() if 'search' in _df.columns else 0
    
    # Top performers
    if 'Category' in _df.columns:
        cat_counts = _df.groupby('Category')['Counts'].sum()
        if not cat_counts.empty:
            metrics['top_category'] = cat_counts.idxmax()
            metrics['top_category_volume'] = int(cat_counts.max())
    
    if 'Brand' in _df.columns:
        brand_counts = _df[_df['Brand'] != 'Other'].groupby('Brand')['Counts'].sum()
        if not brand_counts.empty:
            metrics['top_brand'] = brand_counts.idxmax()
            metrics['top_brand_volume'] = int(brand_counts.max())
    
    # Monthly data (unified)
    if 'month_sort' in _df.columns and 'Month_Short' in _df.columns:
        monthly = _df.groupby(['month_sort', 'Month_Short']).agg({
            'Counts': 'sum',
            'clicks': 'sum',
            'conversions': 'sum',
            'CTR': 'mean',
            'CR': 'mean'
        }).reset_index()
        monthly = monthly.sort_values('month_sort').reset_index(drop=True)
        metrics['monthly_data'] = monthly
    
    # Category performance (unified)
    if 'Category' in _df.columns:
        category_perf = _df.groupby('Category').agg({
            'Counts': 'sum',
            'clicks': 'sum',
            'conversions': 'sum',
            'CTR': 'mean',
            'CR': 'mean'
        }).reset_index()
        category_perf['share'] = (category_perf['Counts'] / metrics['total_searches'] * 100).round(2)
        category_perf = category_perf.sort_values('Counts', ascending=False)
        metrics['category_data'] = category_perf
    
    # Brand performance (unified)
    if 'Brand' in _df.columns:
        brand_perf = _df[_df['Brand'] != 'Other'].groupby('Brand').agg({
            'Counts': 'sum',
            'clicks': 'sum',
            'conversions': 'sum',
            'CTR': 'mean',
            'CR': 'mean'
        }).reset_index()
        brand_perf = brand_perf.sort_values('Counts', ascending=False).head(10)
        metrics['brand_data'] = brand_perf
    
    # Top queries (unified)
    if 'search' in _df.columns:
        top_queries = _df.groupby('search').agg({
            'Counts': 'sum',
            'clicks': 'sum',
            'conversions': 'sum',
            'CTR': 'mean',
            'CR': 'mean'
        }).reset_index()
        top_queries['share'] = (top_queries['Counts'] / metrics['total_searches'] * 100).round(2)
        top_queries = top_queries.sort_values('Counts', ascending=False).head(50)
        metrics['top_queries'] = top_queries
    
    return metrics

# ================================================================================================
# UNIFIED VISUALIZATION FUNCTIONS
# ================================================================================================

def create_monthly_chart(monthly_data):
    """Unified monthly trend chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=monthly_data['Month_Short'],
        y=monthly_data['Counts'],
        name='Searches',
        marker=dict(
            color=monthly_data['Counts'],
            colorscale=[[0, '#FCE4EC'], [0.5, '#F06292'], [1, '#D81B60']],
            line=dict(color='#AD1457', width=2)
        ),
        text=monthly_data['Counts'].apply(format_number),
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Searches: %{y:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        plot_bgcolor='rgba(252, 228, 236, 0.3)',
        paper_bgcolor='rgba(255, 255, 255, 0)',
        font=dict(color='#AD1457', family='Segoe UI'),
        height=400,
        showlegend=False,
        xaxis=dict(title='Month', showgrid=False),
        yaxis=dict(title='Search Volume', showgrid=True, gridcolor='#FCE4EC'),
        margin=dict(t=20, b=20, l=20, r=20)
    )
    
    return fig

def create_bar_chart(data, x_col, y_col, title, color_col=None):
    """Unified horizontal bar chart"""
    if color_col is None:
        color_col = y_col
    
    fig = px.bar(
        data,
        x=y_col,
        y=x_col,
        orientation='h',
        color=color_col,
        color_continuous_scale=[[0, '#FCE4EC'], [0.5, '#F06292'], [1, '#D81B60']],
        text=data[y_col].apply(format_number),
        title=f'<b style="color:#D81B60;">{title}</b>'
    )
    
    fig.update_traces(
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Volume: %{x:,.0f}<extra></extra>'
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(252, 228, 236, 0.3)',
        paper_bgcolor='rgba(255, 255, 255, 0)',
        font=dict(color='#AD1457', family='Segoe UI'),
        height=400,
        showlegend=False,
        xaxis=dict(title='Search Volume', showgrid=True, gridcolor='#FCE4EC'),
        yaxis=dict(title='', showgrid=False),
        margin=dict(t=40, b=20, l=20, r=20)
    )
    
    return fig

def display_styled_table(df, title=None, download_filename=None):
    """Unified table display with styling"""
    if title:
        st.markdown(f'<h3 style="color: #C2185B;">{title}</h3>', unsafe_allow_html=True)
    
    # Create styled HTML table
    html = '<div style="overflow-x: auto;"><table class="beauty-table"><thead><tr>'
    
    # Headers
    for col in df.columns:
        html += f'<th>{col}</th>'
    html += '</tr></thead><tbody>'
    
    # Rows
    for _, row in df.iterrows():
        html += '<tr>'
        for val in row:
            html += f'<td>{val}</td>'
        html += '</tr>'
    html += '</tbody></table></div>'
    
    st.markdown(html, unsafe_allow_html=True)
    
    # Download button
    if download_filename:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"📥 Download {download_filename}",
            data=csv,
            file_name=download_filename,
            mime="text/csv"
        )

# ================================================================================================
# MAIN APP
# ================================================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">💄 Beauty Search Analytics Dashboard ✨</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced Performance Analytics • Search Insights • Beauty Data Intelligence</p>', 
                unsafe_allow_html=True)
    
    # ================================================================================================
    # DATA SOURCE SELECTION
    # ================================================================================================
    
    st.sidebar.title("📁 Data Source")
    
    # Default file path
    DEFAULT_FILE = "Sep Beauty Rearranged Clusters.xlsx"
    
    # Check if default file exists
    default_exists = os.path.exists(DEFAULT_FILE)
    
    df_raw = None
    source_name = ""
    
    # Optional file uploader
    uploaded_file = st.sidebar.file_uploader(
        "📤 Upload custom file (optional)",
        type=['csv', 'xlsx'],
        help="Leave empty to use default file"
    )
    
    if uploaded_file:
        # Use uploaded file
        with st.spinner('💄 Loading uploaded file...'):
            df_raw = load_data_from_upload(uploaded_file)
            source_name = f"Uploaded: {uploaded_file.name}"
            
            if df_raw is not None:
                st.sidebar.markdown(
                    '<div class="data-source-badge">✅ Custom File Loaded</div>',
                    unsafe_allow_html=True
                )
    
    elif default_exists:
        # Use default file
        with st.spinner('💄 Loading default data...'):
            df_raw = load_data_from_file(DEFAULT_FILE)
            source_name = f"Default: {DEFAULT_FILE}"
            
            if df_raw is not None:
                st.sidebar.markdown(
                    '<div class="data-source-badge">✅ Default File Loaded</div>',
                    unsafe_allow_html=True
                )
    
    else:
        st.error(f"❌ Default file '{DEFAULT_FILE}' not found. Please upload a file.")
        st.stop()
    
    # Check if data loaded successfully
    if df_raw is None or df_raw.empty:
        st.error("❌ Failed to load data. Please check your data source.")
        st.stop()
    
    # Process data (ONCE)
    with st.spinner('🔄 Processing data...'):
        df = process_data(df_raw)
        metrics = calculate_all_metrics(df)
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Data Summary")
    st.sidebar.success(f"""
    **Source:** {source_name}
    
    **Loaded:**
    - Rows: {len(df):,}
    - Searches: {format_number(metrics['total_searches'])}
    - Clicks: {format_number(metrics['total_clicks'])}
    - Conversions: {format_number(metrics['total_conversions'])}
    """)
    
    # Reload button
    if st.sidebar.button("🔄 Reload Data"):
        st.cache_data.clear()
        st.rerun()
    
    # ================================================================================================
    # OVERVIEW TAB
    # ================================================================================================
    
    st.markdown("## 💄 Performance Overview")
    
    # Data source indicator
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 20px;">
        <span style="background: linear-gradient(135deg, #FCE4EC 0%, #F8BBD0 100%); 
                     padding: 8px 20px; border-radius: 20px; color: #AD1457; 
                     font-weight: 600; font-size: 0.9rem; border: 2px solid #F06292;">
            📂 Data Source: {source_name}
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    # KPI Cards (using pre-calculated metrics)
    col1, col2, col3, col4, col5 = st.columns(5)
    
    kpi_data = [
        (col1, "🔍", metrics['total_searches'], "Total Searches"),
        (col2, "✨", metrics['total_clicks'], "Total Clicks"),
        (col3, "💖", metrics['total_conversions'], "Conversions"),
        (col4, "📈", format_percentage(metrics['avg_ctr']), "Avg CTR"),
        (col5, "🎯", format_percentage(metrics['avg_cr']), "Avg CR")
    ]
    
    for col, icon, value, label in kpi_data:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-value">{format_number(value) if isinstance(value, (int, float)) else value}</div>
                <div class="kpi-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Monthly Trend Chart
    st.markdown("### 📅 Monthly Search Trends")
    if 'monthly_data' in metrics:
        fig_monthly = create_monthly_chart(metrics['monthly_data'])
        st.plotly_chart(fig_monthly, use_container_width=True)
    
    # Category and Brand Charts
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("### 📦 Top Categories")
        if 'category_data' in metrics:
            fig_cat = create_bar_chart(
                metrics['category_data'].head(10),
                'Category',
                'Counts',
                'Top Categories by Volume'
            )
            st.plotly_chart(fig_cat, use_container_width=True)
    
    with col_right:
        st.markdown("### 🏷️ Top Brands")
        if 'brand_data' in metrics:
            fig_brand = create_bar_chart(
                metrics['brand_data'].head(10),
                'Brand',
                'Counts',
                'Top Brands by Volume'
            )
            st.plotly_chart(fig_brand, use_container_width=True)
    
    # Insights Section
    st.markdown("---")
    st.markdown("## 💡 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'top_category' in metrics:
            st.markdown(f"""
            <div class="insight-box">
                <h4>🎯 Top Performing Category</h4>
                <p><strong>{metrics['top_category']}</strong> leads with <strong>{format_number(metrics['top_category_volume'])}</strong> searches</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if 'top_brand' in metrics:
            st.markdown(f"""
            <div class="insight-box">
                <h4>⭐ Leading Brand</h4>
                <p><strong>{metrics['top_brand']}</strong> dominates with <strong>{format_number(metrics['top_brand_volume'])}</strong> searches</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Performance Summary
    st.markdown(f"""
    <div class="insight-box">
        <h4>📊 Performance Summary</h4>
        <p>
            • <strong>{format_number(metrics['unique_queries'])}</strong> unique search queries<br>
            • Average CTR of <strong>{format_percentage(metrics['avg_ctr'])}</strong><br>
            • Average CR of <strong>{format_percentage(metrics['avg_cr'])}</strong><br>
            • <strong>{format_number(metrics['total_conversions'])}</strong> total conversions
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Top Queries Table
    st.markdown("---")
    st.markdown("## 🔍 Top Queries Analysis")
    
    if 'top_queries' in metrics:
        # Format for display
        display_df = metrics['top_queries'].copy()
        display_df['Counts'] = display_df['Counts'].apply(lambda x: format_number(int(x)))
        display_df['clicks'] = display_df['clicks'].apply(lambda x: format_number(int(x)))
        display_df['conversions'] = display_df['conversions'].apply(lambda x: format_number(int(x)))
        display_df['CTR'] = display_df['CTR'].apply(lambda x: f"{x:.1f}%")
        display_df['CR'] = display_df['CR'].apply(lambda x: f"{x:.1f}%")
        display_df['share'] = display_df['share'].apply(lambda x: f"{x:.1f}%")
        
        # Rename columns
        display_df = display_df.rename(columns={
            'search': 'Query',
            'Counts': 'Volume',
            'clicks': 'Clicks',
            'conversions': 'Conversions',
            'share': 'Share %'
        })
        
        display_styled_table(
            display_df[['Query', 'Volume', 'Share %', 'CTR', 'CR', 'Clicks', 'Conversions']],
            title="Top 50 Search Queries",
            download_filename="top_queries.csv"
        )

if __name__ == "__main__":
    main()
