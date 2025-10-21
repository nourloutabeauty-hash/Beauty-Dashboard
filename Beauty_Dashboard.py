import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter, defaultdict, OrderedDict
import re, os, logging
from datetime import datetime
import pytz
from fuzzywuzzy import fuzz
from plotly.subplots import make_subplots
from uuid import uuid4
import hashlib
import warnings
import io

# ✅ SUPPRESS WARNINGS
warnings.filterwarnings('ignore')

# ✅ STREAMLIT & PANDAS PERFORMANCE CONFIG
try:
    st.set_option('deprecation.showPyplotGlobalUse', False)
except:
    pass

pd.set_option('mode.chained_assignment', None)
pd.set_option('compute.use_bottleneck', True)
pd.set_option('compute.use_numexpr', True)

# ✅ PLOTLY PERFORMANCE
try:
    import plotly.io as pio
    pio.templates.default = "plotly_white"
except:
    pass

# Optional packages
try:
    from st_aggrid import AgGrid, GridOptionsBuilder
    AGGRID_OK = True
except:
    AGGRID_OK = False

try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    WORDCLOUD_OK = True
except:
    WORDCLOUD_OK = False

# ✅ PANDAS COPY-ON-WRITE
os.environ['PANDAS_COPY_ON_WRITE'] = '1'

# ========================================
# 🚀 UTILITY FUNCTIONS (SHARED ACROSS ALL TABS)
# ========================================

def format_number(num):
    """Format numbers with K/M suffix"""
    if num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    else:
        return f"{num:,.0f}"

def format_percentage(num):
    """Format percentages with 1 decimal place"""
    return f"{num:.1f}%"

def extract_keywords(text: str):
    """Extract words (Arabic & Latin & numbers)"""
    if not isinstance(text, str):
        return []
    tokens = re.findall(r'[\u0600-\u06FF\w%+\-]+', text)
    return [t.strip().lower() for t in tokens if len(t.strip()) > 0]

def generate_csv_ultra(df):
    """Enhanced CSV generator"""
    try:
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
        return csv_buffer.getvalue()
    except Exception as e:
        return df.to_csv(index=False)

# ========================================
# 🚀 CORE DATA PROCESSING (CACHED)
# ========================================

@st.cache_data(persist="disk", show_spinner=False, max_entries=5)
def load_excel_ultra_fast(file_input):
    """
    ✅ STREAMLIT CLOUD COMPATIBLE - Handles both uploaded files and file paths
    """
    try:
        # Determine input type
        if hasattr(file_input, 'name'):
            # It's an uploaded file object (from st.file_uploader)
            file_name = file_input.name
            
            if file_name.endswith('.csv'):
                try:
                    df_csv = pd.read_csv(file_input, low_memory=False, encoding='utf-8')
                except:
                    file_input.seek(0)
                    df_csv = pd.read_csv(file_input, low_memory=False, encoding='utf-8-sig')
                sheets = {'queries_clustered': df_csv}
            else:
                # Excel file
                file_input.seek(0)
                sheets = pd.read_excel(file_input, sheet_name=None, engine='openpyxl')
        
        elif isinstance(file_input, (str, bytes, os.PathLike)):
            # It's a file path string
            if str(file_input).endswith('.csv'):
                df_csv = pd.read_csv(file_input, low_memory=False, encoding='utf-8')
                sheets = {'queries_clustered': df_csv}
            else:
                sheets = pd.read_excel(file_input, sheet_name=None, engine='openpyxl')
        
        else:
            raise ValueError(f"Unsupported file input type: {type(file_input)}")
        
        # ✅ FIX UNHASHABLE TYPES
        for sheet_name, df in sheets.items():
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].apply(
                    lambda x: str(x) if isinstance(x, (list, dict, tuple)) else x
                )
        
        return sheets
        
    except Exception as e:
        st.error(f"❌ Load error: {e}")
        import traceback
        st.error(traceback.format_exc())
        raise

@st.cache_data(show_spinner=False, max_entries=3)
def prepare_queries_df_ultra(df):
    """🎯 SINGLE UNIFIED DATA PREPARATION FUNCTION - Preserves ALL logic"""
    
    # 🚀 SMART SAMPLING for large datasets
    if len(df) > 100000:
        df = smart_sampling(df, max_rows=50000)
        st.info(f"📊 Dataset sampled to {len(df):,} rows for performance")
    
    # ✅ Work with copy to avoid modifying cached data
    df = df.copy()
    
    # ✅ FIX UNHASHABLE TYPES
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].apply(
            lambda x: str(x) if isinstance(x, (list, dict, tuple)) else x
        )
    
    # 🚀 BATCH NUMERIC CONVERSIONS
    numeric_cols = ['count', 'Clicks', 'Conversions']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 🚀 COLUMN MAPPING
    column_mapping = {
        'count': 'Counts',
        'Clicks': 'clicks', 
        'Conversions': 'conversions'
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df[new_col] = df[old_col]
        elif new_col not in df.columns:
            df[new_col] = 0

    # 🚀 DATE PROCESSING
    if 'start_date' in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df['start_date']):
            df['Date'] = df['start_date']
        else:
            df['Date'] = pd.to_datetime(df['start_date'], unit='D', origin='1899-12-30', errors='coerce')
    else:
        df['Date'] = pd.NaT

    # 🚀 NUMPY VECTORIZATION FOR METRICS
    counts = df['Counts'].values
    clicks = df['clicks'].values
    conversions = df['conversions'].values
    
    df['ctr'] = np.divide(clicks * 100, counts, 
                         out=np.zeros_like(clicks, dtype=np.float32), 
                         where=counts!=0)
    
    df['cr'] = np.divide(conversions * 100, counts, 
                        out=np.zeros_like(conversions, dtype=np.float32), 
                        where=counts!=0)
    
    # Classical CR
    if 'classical_cr' in df.columns:
        classical_cr = pd.to_numeric(df['classical_cr'], errors='coerce').fillna(0)
        df['classical_cr'] = classical_cr * 100 if classical_cr.max() <= 1 else classical_cr
    else:
        df['classical_cr'] = df['cr']
    
    # 🚀 CATEGORICAL COLUMNS (INCLUDING SUB CLASS)
    categorical_mapping = {
        'Brand': 'brand',
        'Category': 'category', 
        'Sub Category': 'sub_category',
        'Department': 'department',
        'Class': 'class',
        'Sub Class': 'sub_class'
    }
    
    for orig_col, new_col in categorical_mapping.items():
        if orig_col in df.columns:
            df[new_col] = df[orig_col].astype('category')
        elif new_col not in df.columns:
            df[new_col] = pd.Categorical([''])
    
    # 🚀 QUERY PROCESSING
    if 'search' in df.columns:
        df['normalized_query'] = df['search'].astype(str)
    else:
        df['normalized_query'] = df.iloc[:, 0].astype(str)
    
    df['query_length'] = df['normalized_query'].str.len().astype('uint16')
    
    # 🚀 TIME BUCKETS
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.strftime('%B %Y')
    df['month_short'] = df['Date'].dt.strftime('%b')
    df['day_of_week'] = df['Date'].dt.day_name()
    
    # 🚀 KEYWORDS (LAZY - only when needed)
    df['keywords'] = df['normalized_query'].apply(extract_keywords)
    
    # 🚀 ADDITIONAL FIELDS
    df['revenue'] = 0
    df['brand_ar'] = ''
    
    # 🚀 OPTIONAL COLUMNS
    optional_cols = ['underperforming', 'averageClickPosition', 'cluster_id']
    for col in optional_cols:
        if col in df.columns:
            if col == 'averageClickPosition':
                df['average_click_position'] = df[col]
    
    # 🚀 MEMORY OPTIMIZATION
    df = optimize_memory_ultra(df)
    
    return df

@st.cache_data(show_spinner=False)
def smart_sampling(df, max_rows=50000):
    """Intelligent sampling preserving high-value rows"""
    if len(df) <= max_rows:
        return df
    
    try:
        # Keep high-value rows
        high_value_mask = df['Clicks'] > df['Clicks'].quantile(0.8)
        high_value = df[high_value_mask]
        remaining = df[~high_value_mask]
        
        sample_size = max_rows - len(high_value)
        if sample_size > 0 and len(remaining) > 0:
            sampled = remaining.sample(n=min(sample_size, len(remaining)), random_state=42)
            result = pd.concat([high_value, sampled], ignore_index=True)
        else:
            result = high_value.head(max_rows)
            
        return result
    except:
        return df.sample(n=max_rows, random_state=42).reset_index(drop=True)

def optimize_memory_ultra(df):
    """ULTRA memory optimization"""
    
    # 🚀 SMART INTEGER DOWNCASTING
    for col in df.select_dtypes(include=['int64']).columns:
        col_max = df[col].max()
        col_min = df[col].min()
        
        if col_min >= 0:
            if col_max < 255:
                df[col] = df[col].astype('uint8')
            elif col_max < 65535:
                df[col] = df[col].astype('uint16')
            elif col_max < 4294967295:
                df[col] = df[col].astype('uint32')
    
    # 🚀 FLOAT32 OPTIMIZATION
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    
    # 🚀 CATEGORY OPTIMIZATION
    for col in df.select_dtypes(include=['object']).columns:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio < 0.5:
            df[col] = df[col].astype('category')
    
    return df

# 🚀 ULTRA-FAST KEYWORD EXTRACTION
_keyword_pattern = re.compile(r'[\u0600-\u06FF\w%+\-]+', re.IGNORECASE)

@st.cache_data(show_spinner=False)
def extract_keywords_ultra_fast(text_series):
    """Vectorized keyword extraction"""
    if len(text_series) > 1000:
        sample_series = text_series.sample(n=1000, random_state=42)
    else:
        sample_series = text_series
    
    keywords = sample_series.str.findall(_keyword_pattern).apply(
        lambda x: [token.lower() for token in x if token.strip()]
    )
    return keywords

# ========================================
# 🚀 SESSION STATE MANAGEMENT (UNIFIED)
# ========================================

def init_session_state():
    """🎯 UNIFIED session state - single source of truth"""
    defaults = {
        'data_loaded': False,
        'queries': None,  # ✅ CURRENT filtered data (used by ALL tabs)
        'sheets': None,
        'original_queries': None,  # ✅ UNFILTERED backup for reset
        'filters_applied': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ========================================
# 🎨 STYLING & UI COMPONENTS
# ========================================

def display_styled_table(df, title=None, download_filename=None, max_rows=None, 
                        align="center", scrollable=False, max_height="600px", 
                        wrap_text=True, max_cell_width="300px"):
    """Display styled table with beauty theme"""
    if df is None or df.empty:
        st.warning("⚠️ No data available")
        return
    
    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception as e:
            st.error(f"❌ Cannot convert to DataFrame: {e}")
            return
    
    display_df = df.head(max_rows) if max_rows else df.copy()
    
    if title:
        st.markdown(f'<h3 style="color: #C2185B; margin-bottom: 10px;">{title}</h3>', 
                   unsafe_allow_html=True)
    
    white_space = "normal" if wrap_text else "nowrap"
    
    html = f'''
    <style>
        .beauty-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 4px 15px rgba(216, 27, 96, 0.12);
            border-radius: 12px;
            overflow: hidden;
        }}
        .beauty-table thead tr {{
            background: linear-gradient(135deg, #D81B60 0%, #EC407A 100%);
            color: #FFFFFF;
            text-align: center;
            font-weight: bold;
        }}
        .beauty-table th {{
            padding: 14px;
            white-space: {white_space};
            max-width: {max_cell_width};
        }}
        .beauty-table tbody tr:nth-child(odd) {{
            background-color: #FCE4EC;
        }}
        .beauty-table tbody tr:nth-child(even) {{
            background-color: #FFFFFF;
        }}
        .beauty-table tbody tr:hover {{
            background-color: #F8BBD0 !important;
            cursor: pointer;
        }}
        .beauty-table td {{
            padding: 12px;
            text-align: {align} !important;
            color: #AD1457;
            white-space: {white_space};
            max-width: {max_cell_width};
        }}
    </style>
    <table class="beauty-table">
        <thead><tr>{''.join(f'<th>{col}</th>' for col in display_df.columns)}</tr></thead>
        <tbody>
    '''
    
    for _, row in display_df.iterrows():
        html += '<tr>' + ''.join(f'<td>{val if pd.notna(val) else ""}</td>' for val in row) + '</tr>'
    
    html += '</tbody></table>'
    
    st.markdown(html, unsafe_allow_html=True)
    
    if max_rows and len(df) > max_rows:
        st.caption(f"📊 Showing {max_rows} of {len(df)} rows")
    
    if download_filename:
        csv = generate_csv_ultra(df)
        st.download_button(
            label=f"📥 Download {download_filename}",
            data=csv,
            file_name=download_filename,
            mime="text/csv",
            key=f"download_{download_filename}_{id(df)}"
        )

def get_table_theme(theme_name="beauty"):
    """Get pre-defined color themes for tables"""
    themes = {
        "beauty": {"align": "center"},
        "pink": {"align": "center"},
        "rose": {"align": "center"},
    }
    return themes.get(theme_name, themes["beauty"])

# ========================================
# 🚀 PAGE CONFIGURATION
# ========================================

st.set_page_config(
    page_title="💄 Beauty Care — Ultimate Search Analytics", 
    layout="wide", 
    page_icon="✨",
    initial_sidebar_state="expanded"
)

init_session_state()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ YOUR EXACT CSS - NO CHANGES
st.markdown("""
<style>
/* [YOUR ENTIRE CSS BLOCK - KEPT EXACTLY AS IS] */
body {
    font-family: 'Segoe UI', 'Arial', sans-serif;
    background: linear-gradient(135deg, #FFF0F5 0%, #FFE4E9 100%);
}

.sidebar .sidebar-content {
    background: linear-gradient(135deg, #D81B60 0%, #F06292 100%);
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 8px 25px rgba(216, 27, 96, 0.2);
}
.sidebar .sidebar-content h1, .sidebar .sidebar-content * {
    color: #FFFFFF !important;
}

.main-header {
    font-size: 2.8rem;
    font-weight: 900;
    background: linear-gradient(45deg, #AD1457, #D81B60, #F06292);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin-bottom: 0.3rem;
    text-shadow: 2px 2px 4px rgba(173, 20, 87, 0.1);
}

.sub-header {
    font-size: 1.2rem;
    color: #C2185B;
    text-align: center;
    margin-bottom: 1.5rem;
    font-weight: 600;
}

.welcome-box {
    background: linear-gradient(135deg, #FCE4EC 0%, #F8BBD0 50%, #F48FB1 100%);
    padding: 25px;
    border-radius: 15px;
    margin-bottom: 25px;
    box-shadow: 0 6px 20px rgba(216, 27, 96, 0.15);
    text-align: center;
    border: 2px solid rgba(240, 98, 146, 0.3);
}
.welcome-box h2 {
    color: #AD1457;
    font-size: 2rem;
    margin-bottom: 12px;
    font-weight: 800;
}
.welcome-box p {
    color: #C2185B;
    font-size: 1.1rem;
    line-height: 1.6;
}

.kpi {
    background: linear-gradient(135deg, #FFFFFF 0%, #FFF5F7 100%);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0 8px 25px rgba(216, 27, 96, 0.12);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    border: 2px solid rgba(240, 98, 146, 0.2);
}
.kpi:hover {
    transform: translateY(-8px);
    box-shadow: 0 12px 35px rgba(216, 27, 96, 0.2);
    border-color: rgba(240, 98, 146, 0.4);
}
.kpi .value {
    font-size: 2rem;
    font-weight: 900;
    background: linear-gradient(45deg, #AD1457, #D81B60);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.kpi .label {
    color: #EC407A;
    font-size: 1rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.insight-box {
    background: linear-gradient(135deg, #FCE4EC 0%, #F8BBD0 100%) !important;
    padding: 20px;
    border-left: 6px solid #EC407A !important;
    border-radius: 12px;
    margin-bottom: 20px;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    box-shadow: 0 4px 15px rgba(236, 64, 122, 0.15);
}
.insight-box:hover {
    transform: translateX(8px);
    box-shadow: 0 6px 25px rgba(236, 64, 122, 0.25) !important;
    background: linear-gradient(135deg, #F8BBD0 0%, #F48FB1 100%) !important;
}
.insight-box h4 {
    margin: 0 0 10px 0;
    color: #AD1457 !important;
    font-weight: 700;
    font-size: 1.1rem;
}
.insight-box p {
    margin: 8px 0;
    color: #C2185B !important;
    line-height: 1.6;
}
.insight-box ul {
    margin: 10px 0;
    padding-left: 20px;
    color: #C2185B !important;
}
.insight-box ul li {
    margin: 6px 0;
    color: #880E4F !important;
}
.insight-box ul li strong {
    color: #AD1457 !important;
}
.insight-box em {
    color: #AD1457 !important;
    font-style: italic;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 15px;
    background: linear-gradient(135deg, #FCE4EC 0%, #F8BBD0 100%);
    padding: 15px;
    border-radius: 15px;
    box-shadow: inset 0 2px 8px rgba(216, 27, 96, 0.1);
}
.stTabs [data-baseweb="tab"] {
    height: 55px;
    border-radius: 12px;
    padding: 15px 20px;
    font-weight: 700;
    background: linear-gradient(135deg, #FFFFFF 0%, #FFF5F7 100%);
    color: #C2185B;
    border: 2px solid rgba(236, 64, 122, 0.2);
    transition: all 0.3s ease;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #EC407A 0%, #F06292 100%);
    color: #FFFFFF !important;
    border-color: #D81B60;
    box-shadow: 0 4px 15px rgba(236, 64, 122, 0.3);
}
.stTabs [data-baseweb="tab"]:hover {
    background: linear-gradient(135deg, #FCE4EC 0%, #F48FB1 100%);
    color: #AD1457;
    border-color: #EC407A;
    transform: translateY(-2px);
}

.footer {
    text-align: center;
    padding: 20px 0;
    color: #EC407A;
    font-size: 1rem;
    margin-top: 30px;
    border-top: 3px solid #F06292;
    background: linear-gradient(135deg, #FFF5F7 0%, #FCE4EC 100%);
    border-radius: 15px 15px 0 0;
}

.dataframe, .stDataFrame {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 6px 20px rgba(216, 27, 96, 0.1);
}
.stDataFrame th {
    background: linear-gradient(135deg, #FCE4EC 0%, #F48FB1 100%) !important;
    color: #AD1457 !important;
    font-weight: 700 !important;
}

.mini-metric {
    background: linear-gradient(135deg, #EC407A 0%, #F06292 50%, #F48FB1 100%);
    padding: 18px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0 8px 25px rgba(236, 64, 122, 0.25);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    height: 120px;
    display: flex;
    flex-direction: column;
    justify-center;
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

::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-track {
    background: #FCE4EC;
    border-radius: 10px;
}
::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #EC407A, #F06292);
    border-radius: 10px;
}

div[data-testid="stMarkdownContainer"] table {
    width: 100%;
    border-collapse: collapse;
    margin: 20px 0;
    box-shadow: 0 4px 15px rgba(216, 27, 96, 0.12);
    border-radius: 12px;
    overflow: hidden;
}
div[data-testid="stMarkdownContainer"] table thead th {
    text-align: center !important;
    background: linear-gradient(135deg, #D81B60 0%, #EC407A 100%) !important;
    color: white !important;
    font-weight: 700 !important;
    padding: 12px !important;
}
div[data-testid="stMarkdownContainer"] table tbody tr:nth-child(even) {
    background-color: #FFF5F7 !important;
}
div[data-testid="stMarkdownContainer"] table tbody tr:hover {
    background-color: #FCE4EC !important;
}

div.stDownloadButton > button {
    background: linear-gradient(135deg, #D81B60 0%, #EC407A 100%) !important;
    color: white !important;
    border: none !important;
    padding: 0.5rem 1rem !important;
    border-radius: 0.5rem !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 12px rgba(216, 27, 96, 0.3) !important;
}
div.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #AD1457 0%, #D81B60 100%) !important;
    transform: translateY(-2px) !important;
}

.stButton > button {
    background: linear-gradient(135deg, #EC407A 0%, #F06292 100%) !important;
    color: white !important;
    border: 2px solid rgba(255, 255, 255, 0.3) !important;
    border-radius: 12px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(236, 64, 122, 0.3) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #D81B60 0%, #EC407A 100%) !important;
    transform: translateY(-2px) !important;
}

.stSelectbox > div > div, .stMultiSelect > div > div {
    background: linear-gradient(135deg, #FFFFFF 0%, #FFF5F7 100%) !important;
    border: 2px solid #F8BBD0 !important;
    border-radius: 10px !important;
}
.stSelectbox > div > div:hover, .stMultiSelect > div > div:hover {
    border-color: #EC407A !important;
}

.stTextInput > div > div > input {
    background: linear-gradient(135deg, #FFFFFF 0%, #FFF5F7 100%) !important;
    border: 2px solid #F8BBD0 !important;
    border-radius: 10px !important;
    color: #C2185B !important;
}
.stTextInput > div > div > input:focus {
    border-color: #EC407A !important;
    box-shadow: 0 0 0 2px rgba(236, 64, 122, 0.2) !important;
}
</style>
""", unsafe_allow_html=True)


# ========================================
# 🚀 DATA LOADING (OPTIMIZED FOR STREAMLIT CLOUD)
# ========================================

import os
import gc  # Garbage collector for memory management

st.sidebar.title("📁 Upload Data")
upload = st.sidebar.file_uploader("Upload Excel (multi-sheet) or CSV (queries)", type=['xlsx','csv'])

# ✅ Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.queries = None
    st.session_state.original_queries = None
    st.session_state.sheets = None
    st.session_state.filters_applied = False

# ========================================
# 🎯 OPTIMIZED LOADING FUNCTION
# ========================================

@st.cache_data(show_spinner=False, ttl=3600)  # Cache for 1 hour
def load_excel_optimized(file_path_or_upload, is_upload=False):
    """
    Load Excel with memory optimization
    Returns: (sheets_dict, main_queries_df)
    """
    try:
        # Read Excel file
        if is_upload:
            sheets = pd.read_excel(
                file_path_or_upload, 
                sheet_name=None, 
                engine='openpyxl',
                dtype_backend='numpy_nullable'  # More memory efficient
            )
        else:
            sheets = pd.read_excel(
                file_path_or_upload, 
                sheet_name=None, 
                engine='openpyxl',
                dtype_backend='numpy_nullable'
            )
        
        # Find main sheet
        sheet_names = list(sheets.keys())
        preferred = ['queries_clustered', 'queries_dedup', 'queries']
        main_sheet = next((pref for pref in preferred if pref in sheets), sheet_names[0])
        
        # Get main queries
        raw_queries = sheets[main_sheet]
        
        # Process queries
        queries = prepare_queries_df_ultra(raw_queries)
        
        # Clean up
        del raw_queries
        gc.collect()
        
        return sheets, queries, main_sheet
        
    except Exception as e:
        raise Exception(f"Error loading Excel: {str(e)}")

# ========================================
# 📂 LOAD DATA WITH PROGRESS
# ========================================

if not st.session_state.data_loaded:
    
    # Create progress indicators
    progress_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Determine source
            if upload is not None:
                status_text.text("📤 Loading uploaded file...")
                progress_bar.progress(10)
                
                # Load uploaded file
                sheets, queries, main_sheet = load_excel_optimized(upload, is_upload=True)
                source_msg = f"📤 Using uploaded file: {upload.name}"
                
            else:
                status_text.text("📂 Loading default file from repository...")
                progress_bar.progress(10)
                
                default_path = "Sep Beauty Rearranged Clusters.xlsx"
                
                # Check if file exists
                if not os.path.exists(default_path):
                    st.warning("📁 Default file not found. Please upload your Excel or CSV file.")
                    st.info("**Expected file:** `Sep Beauty Rearranged Clusters.xlsx`")
                    st.stop()
                
                # Load default file
                sheets, queries, main_sheet = load_excel_optimized(default_path, is_upload=False)
                source_msg = "📂 Using default file from repository"
            
            progress_bar.progress(60)
            status_text.text("🔄 Processing data...")
            
            # Optimize memory: Convert object columns to category where appropriate
            for col in queries.select_dtypes(include=['object']).columns:
                if queries[col].nunique() < len(queries) * 0.5:  # If less than 50% unique
                    queries[col] = queries[col].astype('category')
            
            progress_bar.progress(80)
            status_text.text("💾 Caching data...")
            
            # Store in session state
            st.session_state.queries = queries
            st.session_state.original_queries = queries.copy()
            st.session_state.sheets = sheets
            st.session_state.data_loaded = True
            
            progress_bar.progress(100)
            status_text.text("✅ Complete!")
            
            # Show success message
            st.sidebar.success(source_msg)
            st.success(f"✅ Loaded **{len(queries):,} queries** from sheet **'{main_sheet}'**")
            
            # Clean up progress indicators
            import time
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
            
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            
            st.error(f"❌ **Loading Error:** {str(e)}")
            
            # Show detailed error in expander
            with st.expander("🔍 Show Technical Details"):
                import traceback
                st.code(traceback.format_exc())
            
            # Provide troubleshooting tips
            st.warning("""
            **Troubleshooting Tips:**
            1. Check if file exists in repository
            2. Verify file is not corrupted
            3. Try uploading file manually
            4. Check Streamlit Cloud logs for memory issues
            """)
            
            st.stop()

# ========================================
# ✅ USE CACHED DATA
# ========================================

queries = st.session_state.queries
sheets = st.session_state.sheets

# Load summary sheets (lazy loading)
brand_summary = sheets.get('brand_summary', None)
category_summary = sheets.get('category_summary', None)
subcategory_summary = sheets.get('subcategory_summary', None)
generic_type = sheets.get('generic_type', None)

# ========================================
# 🔄 RELOAD & DEBUG OPTIONS
# ========================================

st.sidebar.markdown("---")

# Reload button
if st.sidebar.button("🔄 Reload Data"):
    # Clear cache and session state
    st.cache_data.clear()
    st.session_state.data_loaded = False
    gc.collect()
    st.rerun()

# Show data info
if st.sidebar.checkbox("📊 Show Data Info"):
    st.sidebar.info(f"""
    **📈 Data Loaded:**
    - **Queries:** {len(queries):,}
    - **Sheets:** {len(sheets)}
    - **Main Sheet:** {list(sheets.keys())[0] if sheets else 'N/A'}
    - **Memory:** {queries.memory_usage(deep=True).sum() / 1024**2:.2f} MB
    """)

# Debug mode
if st.sidebar.checkbox("🐛 Debug Mode"):
    st.sidebar.warning("**Debug Info:**")
    st.sidebar.code(f"""
Current Dir: {os.getcwd()}
Files: {os.listdir('.')}
Excel Exists: {os.path.exists('Sep Beauty Rearranged Clusters.xlsx')}
Session State: {list(st.session_state.keys())}
    """)

st.markdown("---")



# ========================================
# 🚀 FILTERS (FIXED)
# ========================================

st.sidebar.header("🔎 Filters")

# ✅ SAFETY CHECK: Ensure data is loaded
if st.session_state.original_queries is None or st.session_state.original_queries.empty:
    st.sidebar.warning("⚠️ Please upload data first")
    st.stop()

# ✅ CACHED FILTER OPTIONS
@st.cache_data(show_spinner=False)
def get_filter_options(df, col):
    """Cache filter options"""
    try:
        if df is None or df.empty or col not in df.columns:
            return []
        return sorted(df[col].dropna().astype(str).unique().tolist())
    except:
        return []

# ✅ DATE FILTER (with safety checks)
@st.cache_data(show_spinner=False)
def get_date_range(df):
    """Cache date range"""
    try:
        if df is None or df.empty or 'Date' not in df.columns:
            return []
        valid_dates = df['Date'].dropna()
        if len(valid_dates) == 0:
            return []
        min_date = valid_dates.min()
        max_date = valid_dates.max()
        return [min_date, max_date] if pd.notna(min_date) and pd.notna(max_date) else []
    except:
        return []

default_dates = get_date_range(st.session_state.original_queries)

if default_dates:
    date_range = st.sidebar.date_input("📅 Date Range", value=default_dates)
else:
    st.sidebar.info("📅 No date information available")
    date_range = None

# ✅ MULTI-SELECT FILTERS
filter_config = {
    'brand': ('🏷', 'Brand(s)'),
    'department': ('🏬', 'Department(s)'),
    'category': ('📦', 'Category(ies)'),
    'sub_category': ('🧴', 'Sub Category(ies)'),
    'class': ('🎯', 'Class(es)'),  # ✅ FIXED: lowercase 'class'
    'sub_class': ('🎨', 'Sub Class(es)')
}

filters = {}
for col, (emoji, label) in filter_config.items():
    opts = get_filter_options(st.session_state.original_queries, col)
    if opts:  # ✅ Only show filter if options exist
        filters[col] = st.sidebar.multiselect(f"{emoji} {label}", options=opts, default=opts)
    else:
        filters[col] = []

# ✅ TEXT FILTER
text_filter = st.sidebar.text_input("🔍 Filter queries by text")

# ✅ FILTER BUTTONS
st.sidebar.markdown("---")
col1, col2 = st.sidebar.columns(2)

with col1:
    apply_filters = st.button("🔄 Apply Filters", type="primary")

with col2:
    reset_filters = st.button("🗑️ Reset")

# ✅ APPLY FILTERS (UPDATES SESSION STATE)
if reset_filters:
    st.session_state.queries = st.session_state.original_queries.copy()
    st.session_state.filters_applied = False
    st.rerun()

elif apply_filters:
    filtered_queries = st.session_state.original_queries.copy()
    
    # Date filter
    if date_range and isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = date_range
        if 'Date' in filtered_queries.columns:
            filtered_queries = filtered_queries[
                (filtered_queries['Date'] >= pd.to_datetime(start_date)) & 
                (filtered_queries['Date'] <= pd.to_datetime(end_date))
            ]
    
    # Multi-select filters
    for col, selected in filters.items():
        if col in filtered_queries.columns and selected:
            all_opts = get_filter_options(st.session_state.original_queries, col)
            if len(selected) < len(all_opts):  # Only filter if not all selected
                filtered_queries = filtered_queries[filtered_queries[col].astype(str).isin(selected)]
    
    # Text filter
    if text_filter and 'normalized_query' in filtered_queries.columns:
        filtered_queries = filtered_queries[
            filtered_queries['normalized_query'].str.contains(re.escape(text_filter), case=False, na=False)
        ]
    
    # ✅ UPDATE SESSION STATE
    st.session_state.queries = filtered_queries
    st.session_state.filters_applied = True
    st.rerun()

# ✅ FILTER STATUS
if st.session_state.filters_applied:
    original_count = len(st.session_state.original_queries)
    current_count = len(st.session_state.queries)
    reduction_pct = ((original_count - current_count) / original_count) * 100 if original_count > 0 else 0
    st.sidebar.success(f"✅ Filters Applied - {current_count:,} rows ({reduction_pct:.1f}% filtered)")
else:
    st.sidebar.info(f"📊 No filters applied - {len(st.session_state.queries):,} rows")

st.sidebar.markdown(f"**📊 Current rows:** {len(st.session_state.queries):,}")

# ✅ DEBUG INFO
if st.sidebar.checkbox("🔍 Debug Info", value=False):
    st.sidebar.write("**Session State:**")
    st.sidebar.write(f"- Filters applied: {st.session_state.filters_applied}")
    st.sidebar.write(f"- Original rows: {len(st.session_state.original_queries):,}")
    st.sidebar.write(f"- Current rows: {len(st.session_state.queries):,}")
    st.sidebar.write(f"- Available columns: {list(st.session_state.queries.columns)[:10]}")



# ========================================
# 🚀 OPTIMIZED FILTERS (UPDATES SESSION STATE)
# ========================================

st.sidebar.header("🔎 Filters")

# ✅ SAFETY CHECK: Stop if data not loaded
if st.session_state.original_queries is None or st.session_state.original_queries.empty:
    st.sidebar.warning("⚠️ Please load data first")
    st.stop()

# ✅ CACHED FILTER OPTIONS
@st.cache_data(show_spinner=False)
def get_filter_options(df, col):
    """Cache filter options"""
    try:
        if df is None or df.empty:  # ✅ Added safety check
            return []
        if col not in df.columns:
            return []
        return sorted(df[col].dropna().astype(str).unique().tolist())
    except Exception as e:
        return []

# ✅ DATE FILTER
@st.cache_data(show_spinner=False)
def get_date_range(df):
    """Cache date range"""
    try:
        if df is None or df.empty:  # ✅ Added safety check
            return []
        min_date = df['Date'].min()
        max_date = df['Date'].max()
        return [min_date, max_date] if pd.notna(min_date) and pd.notna(max_date) else []
    except:
        return []

default_dates = get_date_range(st.session_state.original_queries)
date_range = st.sidebar.date_input("📅 Date Range", value=default_dates)

# ✅ MULTI-SELECT FILTERS
filter_config = {
    'brand': ('🏷', 'Brand(s)'),
    'department': ('🏬', 'Department(s)'),
    'category': ('📦', 'Category(ies)'),
    'sub_category': ('🧴', 'Sub Category(ies)'),
    'Class': ('🎯', 'Class(es)'),  # ✅ Fixed: Capital 'C' to match your data
    'sub_class': ('🎨', 'Sub Class(es)')
}

filters = {}
for col, (emoji, label) in filter_config.items():
    opts = get_filter_options(st.session_state.original_queries, col)
    filters[col] = st.sidebar.multiselect(f"{emoji} {label}", options=opts, default=opts)

# ✅ TEXT FILTER
text_filter = st.sidebar.text_input("🔍 Filter queries by text")

# ✅ FILTER BUTTONS
st.sidebar.markdown("---")
col1, col2 = st.sidebar.columns(2)

with col1:
    apply_filters = st.button("🔄 Apply Filters", type="primary")

with col2:
    reset_filters = st.button("🗑️ Reset")

# ✅ APPLY FILTERS (UPDATES SESSION STATE)
if reset_filters:
    st.session_state.queries = st.session_state.original_queries.copy()
    st.session_state.filters_applied = False
    st.rerun()

elif apply_filters:
    queries = st.session_state.original_queries.copy()
    
    # Date filter
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = date_range
        queries = queries[(queries['Date'] >= pd.to_datetime(start_date)) & 
                         (queries['Date'] <= pd.to_datetime(end_date))]
    
    # Multi-select filters
    for col, selected in filters.items():
        if col in queries.columns and selected:
            all_opts = get_filter_options(st.session_state.original_queries, col)
            if len(selected) < len(all_opts):  # ✅ Only filter if not all selected
                queries = queries[queries[col].astype(str).isin(selected)]
    
    # Text filter
    if text_filter:
        queries = queries[queries['normalized_query'].str.contains(re.escape(text_filter), case=False, na=False)]
    
    # ✅ UPDATE SESSION STATE (ALL TABS WILL SEE THIS)
    st.session_state.queries = queries
    st.session_state.filters_applied = True
    st.rerun()

# ✅ FILTER STATUS
if st.session_state.filters_applied:
    original_count = len(st.session_state.original_queries)
    current_count = len(st.session_state.queries)
    reduction_pct = ((original_count - current_count) / original_count) * 100 if original_count > 0 else 0
    st.sidebar.success(f"✅ Filters Applied - {current_count:,} rows ({reduction_pct:.1f}% filtered)")
else:
    st.sidebar.info(f"📊 No filters applied - {len(st.session_state.queries):,} rows")

st.sidebar.markdown(f"**📊 Current rows:** {len(st.session_state.queries):,}")

# ✅ DEBUG INFO
if st.sidebar.checkbox("🔍 Debug Info", value=False):
    st.sidebar.write("**Session State:**")
    st.sidebar.write(f"- Filters applied: {st.session_state.filters_applied}")
    st.sidebar.write(f"- Original rows: {len(st.session_state.original_queries):,}")
    st.sidebar.write(f"- Current rows: {len(st.session_state.queries):,}")
    st.sidebar.write(f"- Memory: {st.session_state.queries.memory_usage(deep=True).sum() / 1024**2:.1f} MB")


# ----------------- Welcome Message -----------------
st.markdown("""
<div class="welcome-box">
    <h2>💄 Welcome to Beauty Care Analytics! ✨</h2>
    <p>Discover beauty trends, skincare insights, and cosmetic performance data. Navigate through categories, analyze beauty searches, and unlock actionable insights for optimal beauty strategies!</p>
</div>
""", unsafe_allow_html=True)

# ----------------- KPI cards -----------------
st.markdown('<div class="main-header">💄 Beauty Care & Cosmetics — Advanced Analytics Hub</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Explore Beauty Care & Cosmetics search patterns and skincare insights with <b>data-driven beauty analytics</b></div>', unsafe_allow_html=True)

# ✅ OPTIMIZED: Direct calculation from session state (no function overhead)
total_counts = int(st.session_state.queries['Counts'].sum())
total_clicks = int(st.session_state.queries['clicks'].sum())
total_conversions = int(st.session_state.queries['conversions'].sum())
overall_ctr = (total_clicks / total_counts * 100) if total_counts > 0 else 0
overall_cr = (total_conversions / total_counts * 100) if total_counts > 0 else 0

# ✅ SIMPLIFIED: Direct KPI display (no container complexity)
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(f"<div class='kpi'><div class='value'>{format_number(total_counts)}</div><div class='label'>💄 Total Searches</div></div>", unsafe_allow_html=True)
with c2:
    st.markdown(f"<div class='kpi'><div class='value'>{format_number(total_clicks)}</div><div class='label'>✨ Total Clicks</div></div>", unsafe_allow_html=True)
with c3:
    st.markdown(f"<div class='kpi'><div class='value'>{format_number(total_conversions)}</div><div class='label'>💖 Total Conversions</div></div>", unsafe_allow_html=True)
with c4:
    st.markdown(f"<div class='kpi'><div class='value'>{overall_ctr:.1f}%</div><div class='label'>📈 Overall CTR</div></div>", unsafe_allow_html=True)
with c5:
    st.markdown(f"<div class='kpi'><div class='value'>{overall_cr:.1f}%</div><div class='label'>💕 Overall CR</div></div>", unsafe_allow_html=True)

# ✅ REMOVED: Redundant sidebar update functions (already in filters section)

# ----------------- Tabs -----------------
tab_overview, tab_search, tab_brand, tab_category, tab_subcat, tab_class, tab_generic, tab_time, tab_pivot, tab_insights, tab_export = st.tabs([
    "💄 Overview","🔍 Search Analysis","🏷 Brand","📦 Category","🧴 Subcategory","🎯 Class","💊 Generic Type",
    "⏰ Time Analysis","📊 Pivot Builder","💡 Insights","⬇ Export"
])

# ----------------- Overview Tab -----------------
with tab_overview:
    st.header("💄 Overview & Insights")
    st.markdown("Discover beauty performance patterns. ✨ Based on **data** (e.g., millions of beauty searches across categories).")

    # ✅ OPTIMIZED: Use session state directly (no reassignment)
    queries = st.session_state.queries

    # 🎨 HERO HEADER
    st.markdown("""
    <div style="
        text-align: center; 
        padding: 3rem 2rem; 
        background: linear-gradient(135deg, #FCE4EC 0%, #F8BBD0 50%, #F48FB1 100%); 
        border-radius: 20px; 
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(216, 27, 96, 0.15);
        border: 1px solid rgba(240, 98, 146, 0.2);
    ">
        <h1 style="
            color: #AD1457; 
            margin: 0; 
            font-size: 3rem; 
            text-shadow: 2px 2px 8px rgba(173, 20, 87, 0.2);
            font-weight: 700;
            letter-spacing: -1px;
        ">
            💄 Beauty Care & Cosmetics Analytics ✨
        </h1>
        <p style="
            color: #C2185B; 
            margin: 1rem 0 0 0; 
            font-size: 1.3rem;
            font-weight: 300;
            opacity: 0.9;
        ">
            Advanced Performance Analytics • Search Insights • Beauty Data Intelligence
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ========================================
    # 📊 MONTHLY ANALYSIS
    # ========================================
    st.markdown("## 💕 Monthly Analysis Overview")
    col_table, col_chart = st.columns([1, 2])

    with col_table:
        st.markdown("### 📋 Monthly Searches Table")

        # ✅ OPTIMIZED: Direct groupby without temp variables
        monthly_counts = (queries
            .assign(month_sort=queries['Date'].dt.to_period('M'))
            .assign(month_display=queries['Date'].dt.strftime('%B %Y'))
            .groupby(['month_sort', 'month_display'], observed=True)['Counts']
            .sum()
            .reset_index()
            .sort_values('month_sort')
            .drop(columns=['month_sort'])
            .rename(columns={'month_display': 'Date'})
        )

        if not monthly_counts.empty:
            total_all_months = monthly_counts['Counts'].sum()
            monthly_counts['Percentage'] = (monthly_counts['Counts'] / total_all_months * 100).round(1)
            
            # ✅ OPTIMIZED: Format in single pass
            display_monthly = monthly_counts.copy()
            display_monthly['Counts'] = display_monthly['Counts'].apply(lambda x: format_number(int(x)))
            display_monthly['Percentage'] = display_monthly['Percentage'].apply(lambda x: f"{x}%")
            display_monthly = display_monthly.rename(columns={'Date': 'Month'})
            
            display_styled_table(
                df=display_monthly,
                align="center",
                scrollable=True,
                max_height="600px"
            )
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #D81B60 0%, #F06292 100%); 
                        padding: 15px; border-radius: 10px; color: white; margin: 10px 0; text-align: center;">
                <strong>💄 Total: {format_number(int(total_all_months))} searches across {len(monthly_counts)} months</strong>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No monthly Beauty Care data available")

    with col_chart:
        st.markdown("### 📈 Monthly Trends Visualization")
        
        if not monthly_counts.empty and len(monthly_counts) >= 2:
            fig = px.bar(monthly_counts, x='Date', y='Counts',
                        title='<b style="color:#D81B60; font-size:16px;">Monthly Search Trends 💄</b>',
                        labels={'Date': '<i>Month</i>', 'Counts': '<b>Searches</b>'},
                        color='Counts',
                        color_continuous_scale=['#FCE4EC', '#F06292', '#D81B60'],
                        template='plotly_white',
                        text=monthly_counts['Counts'].astype(str))
                
            fig.update_traces(
                texttemplate='%{text}<br>%{customdata:.1f}%',
                customdata=monthly_counts['Percentage'],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Searches: %{y:,.0f}<br>Share: %{customdata:.1f}%<extra></extra>'
            )
            
            fig.update_layout(
                plot_bgcolor='rgba(252, 228, 236, 0.95)',
                paper_bgcolor='rgba(248, 187, 208, 0.8)',
                font=dict(color='#AD1457', family='Segoe UI'),
                title_x=0.5,
                xaxis=dict(showgrid=True, gridcolor='#FCE4EC', linecolor='#D81B60', linewidth=2),
                yaxis=dict(showgrid=True, gridcolor='#FCE4EC', linecolor='#D81B60', linewidth=2),
                bargap=0.2,
                barcornerradius=8,
                height=400,
                margin=dict(l=40, r=40, t=80, b=40)
            )
            
            peak_month = monthly_counts.loc[monthly_counts['Counts'].idxmax(), 'Date']
            peak_value = monthly_counts['Counts'].max()
            fig.add_annotation(
                x=peak_month, y=peak_value,
                text=f"🏆 Peak: {peak_value:,.0f}",
                showarrow=True,
                arrowhead=3,
                arrowcolor='#D81B60',
                ax=0, ay=-40,
                font=dict(size=12, color='#D81B60', family='Segoe UI', weight='bold')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📅 Add more date range for Beauty Care trends visualization")

    st.markdown("---")

    # ========================================
    # 🔍 TOP QUERIES ANALYSIS
    # ========================================
    st.markdown("## 🔍 Top Queries Analysis")
    
    top_n_queries = st.selectbox(
        "📊 Select number of top queries to display:",
        options=[10, 25, 50, 100, 200, 500],
        index=2,
        help="Choose how many top queries you want to analyze"
    )

    if queries.empty or 'Counts' not in queries.columns:
        st.warning("No valid data available for top beauty queries.")
    else:
        # ✅ OPTIMIZED: Simpler cache key
        filter_cache_key = f"{queries.shape}_{top_n_queries}_{st.session_state.get('filters_applied', False)}"
        
        # ✅ OPTIMIZED: Static month names (no dynamic lookup)
        month_names = OrderedDict([
            ('2025-06', 'June 2025'),
            ('2025-07', 'July 2025'),
            ('2025-08', 'August 2025')
        ])

        @st.cache_data(ttl=300, show_spinner=False)
        def compute_top_queries(_df, month_names_dict, cache_key, top_n=50):
            """🚀 OPTIMIZED: Compute top N queries with monthly breakdown"""
            if _df.empty:
                return pd.DataFrame(), []
            
            # ✅ OPTIMIZED: Single groupby operation
            grouped = _df.groupby('search', observed=True).agg({
                'Counts': 'sum',
                'clicks': 'sum',
                'conversions': 'sum'
            }).reset_index()
            
            topN_queries = grouped.nlargest(top_n, 'Counts')['search'].tolist()
            topN_data = _df[_df['search'].isin(topN_queries)].copy()
            
            unique_months = sorted(topN_data['month'].unique()) if 'month' in topN_data.columns else []
            
            result_data = []
            for query in topN_queries:
                query_data = topN_data[topN_data['search'] == query]
                
                total_counts = int(query_data['Counts'].sum())
                total_clicks = int(query_data['clicks'].sum())
                total_conversions = int(query_data['conversions'].sum())
                overall_ctr = (total_clicks / total_counts * 100) if total_counts > 0 else 0
                overall_cr = (total_conversions / total_counts * 100) if total_counts > 0 else 0
                
                row = {
                    'Query': query,
                    'Total Volume': total_counts,
                    'Share %': (total_counts / _df['Counts'].sum()) * 100,
                    'Overall CTR': overall_ctr,
                    'Overall CR': overall_cr,
                    'Total Clicks': total_clicks,
                    'Total Conversions': total_conversions
                }
                
                # ✅ OPTIMIZED: Vectorized monthly calculations
                for month in unique_months:
                    month_display = month_names_dict.get(month, month)
                    month_data = query_data[query_data['month'] == month]
                    
                    if not month_data.empty:
                        month_counts = int(month_data['Counts'].sum())
                        month_clicks = int(month_data['clicks'].sum())
                        month_conversions = int(month_data['conversions'].sum())
                        
                        row[f'{month_display} Vol'] = month_counts
                        row[f'{month_display} CTR'] = (month_clicks / month_counts * 100) if month_counts > 0 else 0
                        row[f'{month_display} CR'] = (month_conversions / month_counts * 100) if month_counts > 0 else 0
                    else:
                        row[f'{month_display} Vol'] = 0
                        row[f'{month_display} CTR'] = 0
                        row[f'{month_display} CR'] = 0
                
                result_data.append(row)
            
            result_df = pd.DataFrame(result_data).sort_values('Total Volume', ascending=False).reset_index(drop=True)
            return result_df, unique_months

        topN, unique_months = compute_top_queries(queries, month_names, filter_cache_key, top_n_queries)

        if not topN.empty:
            # ✅ FILTER STATUS
            if st.session_state.get('filters_applied', False):
                st.info(f"🔍 **Filtered Results**: Showing Top {top_n_queries} from {len(queries):,} filtered queries")
            else:
                st.info(f"📊 **All Data**: Showing Top {top_n_queries} from {len(queries):,} total queries")

            # ✅ OPTIMIZED: Format and style in single pass
            display_topN = topN.copy()
            
            # Format volume columns
            volume_cols = ['Total Volume'] + [f'{month_names.get(m, m)} Vol' for m in sorted(unique_months)]
            for col in volume_cols:
                if col in display_topN.columns:
                    display_topN[col] = display_topN[col].apply(lambda x: format_number(int(x)))
            
            # Format clicks and conversions
            if 'Total Clicks' in display_topN.columns:
                display_topN['Total Clicks'] = display_topN['Total Clicks'].apply(lambda x: format_number(int(x)))
            if 'Total Conversions' in display_topN.columns:
                display_topN['Total Conversions'] = display_topN['Total Conversions'].apply(lambda x: format_number(int(x)))
            
            # ✅ OPTIMIZED: Simplified highlighting
            def highlight_performance(styled_df):
                """✅ OPTIMIZED: Faster highlighting logic"""
                styles = pd.DataFrame('', index=styled_df.index, columns=styled_df.columns)
                
                if len(unique_months) < 2:
                    return styles
                
                sorted_months = sorted(unique_months)
                
                for i in range(1, len(sorted_months)):
                    current_month = month_names.get(sorted_months[i], sorted_months[i])
                    prev_month = month_names.get(sorted_months[i-1], sorted_months[i-1])
                    
                    current_ctr_col = f'{current_month} CTR'
                    prev_ctr_col = f'{prev_month} CTR'
                    current_cr_col = f'{current_month} CR'
                    prev_cr_col = f'{prev_month} CR'
                    
                    # ✅ CTR highlighting (GREEN for growth)
                    if current_ctr_col in topN.columns and prev_ctr_col in topN.columns:
                        for idx in topN.index:
                            current_ctr = topN.loc[idx, current_ctr_col]
                            prev_ctr = topN.loc[idx, prev_ctr_col]
                            
                            if pd.notnull(current_ctr) and pd.notnull(prev_ctr) and prev_ctr > 0:
                                change_pct = ((current_ctr - prev_ctr) / prev_ctr) * 100
                                if change_pct > 10:
                                    styles.loc[idx, current_ctr_col] = 'background-color: rgba(76, 175, 80, 0.4); color: #1B5E20; font-weight: bold;'
                                elif change_pct < -10:
                                    styles.loc[idx, current_ctr_col] = 'background-color: rgba(244, 67, 54, 0.4); color: #B71C1C; font-weight: bold;'
                                elif change_pct > 5:
                                    styles.loc[idx, current_ctr_col] = 'background-color: rgba(76, 175, 80, 0.2); color: #2E7D32; font-weight: 600;'
                                elif change_pct < -5:
                                    styles.loc[idx, current_ctr_col] = 'background-color: rgba(244, 67, 54, 0.2); color: #C62828; font-weight: 600;'
                    
                    # ✅ CR highlighting (GREEN for growth)
                    if current_cr_col in topN.columns and prev_cr_col in topN.columns:
                        for idx in topN.index:
                            current_cr = topN.loc[idx, current_cr_col]
                            prev_cr = topN.loc[idx, prev_cr_col]
                            
                            if pd.notnull(current_cr) and pd.notnull(prev_cr) and prev_cr > 0:
                                change_pct = ((current_cr - prev_cr) / prev_cr) * 100
                                if change_pct > 10:
                                    styles.loc[idx, current_cr_col] = 'background-color: rgba(76, 175, 80, 0.4); color: #1B5E20; font-weight: bold;'
                                elif change_pct < -10:
                                    styles.loc[idx, current_cr_col] = 'background-color: rgba(244, 67, 54, 0.4); color: #B71C1C; font-weight: bold;'
                                elif change_pct > 5:
                                    styles.loc[idx, current_cr_col] = 'background-color: rgba(76, 175, 80, 0.2); color: #2E7D32; font-weight: 600;'
                                elif change_pct < -5:
                                    styles.loc[idx, current_cr_col] = 'background-color: rgba(244, 67, 54, 0.2); color: #C62828; font-weight: 600;'
                
                return styles
            
            styled_topN = display_topN.style.apply(highlight_performance, axis=None)
            
            styled_topN = styled_topN.set_properties(**{
                'text-align': 'center',
                'vertical-align': 'middle',
                'font-size': '11px',
                'padding': '4px'
            }).set_table_styles([
                {'selector': 'th', 'props': [
                    ('text-align', 'center'), 
                    ('background-color', '#FCE4EC'), 
                    ('color', '#AD1457'), 
                    ('font-weight', 'bold')
                ]},
                {'selector': 'td', 'props': [('text-align', 'center')]},
                {'selector': 'tbody tr:nth-child(even)', 'props': [('background-color', '#FFF0F5')]}
            ])
            
            format_dict = {
                'Share %': '{:.1f}%',
                'Overall CTR': '{:.1f}%',
                'Overall CR': '{:.1f}%'
            }
            
            for month in unique_months:
                month_display = month_names.get(month, month)
                format_dict[f'{month_display} CTR'] = '{:.1f}%'
                format_dict[f'{month_display} CR'] = '{:.1f}%'
            
            styled_topN = styled_topN.format(format_dict)
            
            html_content = styled_topN.to_html(index=False, escape=False)
            
            st.markdown(
                f"""
                <div style="height: 600px; overflow-y: auto; overflow-x: auto; border: 1px solid #ddd;">
                    {html_content}
                </div>
                """,
                unsafe_allow_html=True
            )

            # ✅ LEGEND
            st.markdown("""
            <div style="background: rgba(216, 27, 96, 0.1); padding: 12px; border-radius: 8px; margin: 15px 0;">
                <h4 style="margin: 0 0 8px 0; color: #AD1457;">💄 Comparison Guide:</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">
                    <div>📈 <strong style="background-color: rgba(76, 175, 80, 0.4); padding: 2px 6px; border-radius: 4px; color: #1B5E20;">Dark Green</strong> = >10% improvement</div>
                    <div>📈 <strong style="background-color: rgba(76, 175, 80, 0.2); padding: 2px 6px; border-radius: 4px; color: #2E7D32;">Light Green</strong> = 5-10% improvement</div>
                    <div>📉 <strong style="background-color: rgba(244, 67, 54, 0.4); padding: 2px 6px; border-radius: 4px; color: #B71C1C;">Dark Red</strong> = >10% decline</div>
                    <div>📉 <strong style="background-color: rgba(244, 67, 54, 0.2); padding: 2px 6px; border-radius: 4px; color: #C62828;">Light Red</strong> = 5-10% decline</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ✅ SUMMARY METRICS
            st.markdown("---")
            
            col1, col2, col3, col4 = st.columns(4)
            
            metrics = [
                (col1, "💄", len(topN), "Total Queries"),
                (col2, "🔍", format_number(topN['Total Volume'].sum()), "Total Search Volume"),
                (col3, "✨", format_number(topN['Total Clicks'].sum()), "Total Clicks"),
                (col4, "💖", format_number(topN['Total Conversions'].sum()), "Total Conversions")
            ]
            
            for col, icon, value, label in metrics:
                with col:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #EC407A 0%, #F06292 50%, #F48FB1 100%);
                                padding: 18px; border-radius: 15px; text-align: center; color: white;
                                box-shadow: 0 8px 25px rgba(236, 64, 122, 0.25); height: 120px;
                                display: flex; flex-direction: column; justify-content: center;">
                        <div style="font-size: 2em; margin-bottom: 6px;">{icon}</div>
                        <div style="font-size: 1.8rem; font-weight: 900; margin-bottom: 6px;">{value}</div>
                        <div style="font-size: 0.95rem; font-weight: 600;">{label}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # ✅ DOWNLOAD
            st.markdown("<br>", unsafe_allow_html=True)
            
            csv = generate_csv_ultra(topN)
            filter_suffix = "_filtered" if st.session_state.get('filters_applied', False) else "_all"
            
            col_download = st.columns([1, 2, 1])
            with col_download[1]:
                st.download_button(
                    label="📥 Download Queries CSV",
                    data=csv,
                    file_name=f"top_{top_n_queries}_beauty_queries{filter_suffix}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    st.markdown("---")

    # ========================================
    # 💕 PERFORMANCE SNAPSHOT
    # ========================================
    st.subheader("💕 Performance Snapshot")

    colM1, colM2, colM3, colM4 = st.columns(4)
    
    avg_ctr = queries['ctr'].mean() if 'ctr' in queries.columns else 0
    avg_cr = queries['cr'].mean() if 'cr' in queries.columns else 0
    unique_queries = queries['search'].nunique()
    
    with colM1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #EC407A 0%, #F06292 50%, #F48FB1 100%);
                    padding: 18px; border-radius: 15px; text-align: center; color: white;
                    box-shadow: 0 8px 25px rgba(236, 64, 122, 0.25); height: 120px;
                    display: flex; flex-direction: column; justify-content: center;">
            <span style="font-size: 2em;">💄</span>
            <div style="font-size: 1.8rem; font-weight: 900; margin: 6px 0;">{avg_ctr:.1f}%</div>
            <div style="font-size: 0.95rem; font-weight: 600;">Avg CTR (All)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with colM2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #EC407A 0%, #F06292 50%, #F48FB1 100%);
                    padding: 18px; border-radius: 15px; text-align: center; color: white;
                    box-shadow: 0 8px 25px rgba(236, 64, 122, 0.25); height: 120px;
                    display: flex; flex-direction: column; justify-content: center;">
            <span style="font-size: 2em;">💖</span>
            <div style="font-size: 1.8rem; font-weight: 900; margin: 6px 0;">{avg_cr:.1f}%</div>
            <div style="font-size: 0.95rem; font-weight: 600;">Avg CR (All)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with colM3:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #EC407A 0%, #F06292 50%, #F48FB1 100%);
                    padding: 18px; border-radius: 15px; text-align: center; color: white;
                    box-shadow: 0 8px 25px rgba(236, 64, 122, 0.25); height: 120px;
                    display: flex; flex-direction: column; justify-content: center;">
            <span style="font-size: 2em;">🔍</span>
            <div style="font-size: 1.8rem; font-weight: 900; margin: 6px 0;">{format_number(unique_queries)}</div>
            <div style="font-size: 0.95rem; font-weight: 600;">Unique Queries</div>
        </div>
        """, unsafe_allow_html=True)
    
    with colM4:
        if 'category' in queries.columns:
            cat_counts = queries.groupby('category', observed=True)['Counts'].sum()
            top_cat = cat_counts.idxmax()
            top_cat_share = (cat_counts.max() / total_counts * 100) if total_counts > 0 else 0
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #EC407A 0%, #F06292 50%, #F48FB1 100%);
                        padding: 18px; border-radius: 15px; text-align: center; color: white;
                        box-shadow: 0 8px 25px rgba(236, 64, 122, 0.25); height: 120px;
                        display: flex; flex-direction: column; justify-content: center;">
                <span style="font-size: 2em;">✨</span>
                <div style="font-size: 1.8rem; font-weight: 900; margin: 6px 0;">{format_number(int(cat_counts.max()))} ({top_cat_share:.1f}%)</div>
                <div style="font-size: 0.95rem; font-weight: 600;">Top Category ({top_cat})</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ========================================
    # 🏷 BRAND & CATEGORY SNAPSHOT
    # ========================================
    st.subheader("🏷 Brand & Category Snapshot")
    g1, g2 = st.columns(2)
    
    with g1:
        if 'brand' in queries.columns:
            brand_perf = (queries[queries['brand'] != 'Other']
                         .groupby('brand', observed=True)
                         .agg({'Counts': 'sum', 'clicks': 'sum', 'conversions': 'sum'})
                         .reset_index())
            
            brand_perf['share'] = (brand_perf['Counts'] / total_counts * 100).round(2)
            
            if not brand_perf.empty:
                fig = px.bar(brand_perf.sort_values('Counts', ascending=False).head(10), 
                            x='brand', y='Counts',
                            title='<b style="color:#D81B60; font-size:18px;">Top Brands by Search Volume</b>',
                            labels={'brand': '<i>Brand</i>', 'Counts': '<b>Search Volume</b>'},
                            color='conversions',
                            color_continuous_scale=['#FCE4EC', '#F06292', '#D81B60'],
                            template='plotly_white',
                            hover_data=['share', 'conversions'])
                
                fig.update_traces(
                    texttemplate='%{y:,.0f}',
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>Volume: %{y:,.0f}<br>Share: %{customdata[0]:.1f}%<br>Conversions: %{customdata[1]:,.0f}<extra></extra>'
                )

                fig.update_layout(
                    plot_bgcolor='rgba(252, 228, 236, 0.95)',
                    paper_bgcolor='rgba(248, 187, 208, 0.8)',
                    font=dict(color='#AD1457', family='Segoe UI'),
                    title_x=0,
                    xaxis=dict(showgrid=True, gridcolor='#FCE4EC', linecolor='#D81B60', linewidth=2),
                    yaxis=dict(showgrid=True, gridcolor='#FCE4EC', linecolor='#D81B60', linewidth=2),
                    bargap=0.2,
                    barcornerradius=8
                )

                top_brand = brand_perf.loc[brand_perf['Counts'].idxmax(), 'brand']
                top_count = brand_perf['Counts'].max()
                fig.add_annotation(
                    x=top_brand, y=top_count,
                    text=f"🏆 Peak: {top_count:,.0f}",
                    showarrow=True,
                    arrowhead=3,
                    arrowcolor='#D81B60',
                    ax=0, ay=-30,
                    font=dict(size=12, color='#D81B60', family='Segoe UI', weight='bold')
                )

                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("🏷 Brand column not found in the dataset.")

    with g2:
        if 'category' in queries.columns:
            cat_perf = (queries
                       .groupby('category', observed=True)
                       .agg({'Counts': 'sum', 'clicks': 'sum', 'conversions': 'sum'})
                       .reset_index())
            
            cat_perf['share'] = (cat_perf['Counts'] / total_counts * 100).round(2)
            cat_perf['cr'] = (cat_perf['conversions'] / cat_perf['Counts'] * 100).round(2)
            
            display_cat_perf = cat_perf.copy()
            display_cat_perf['Counts'] = display_cat_perf['Counts'].apply(lambda x: format_number(int(x)))
            display_cat_perf['clicks'] = display_cat_perf['clicks'].apply(lambda x: format_number(int(x)))
            display_cat_perf['conversions'] = display_cat_perf['conversions'].apply(lambda x: format_number(int(x)))
            display_cat_perf['share'] = display_cat_perf['share'].apply(lambda x: f"{x:.1f}%")
            display_cat_perf['cr'] = display_cat_perf['cr'].apply(lambda x: f"{x:.1f}%")
            
            display_cat_perf = display_cat_perf.rename(columns={
                'category': 'Category',
                'Counts': 'Search Volume',
                'share': 'Market Share',
                'clicks': 'Total Clicks',
                'conversions': 'Conversions',
                'cr': 'Conversion Rate'
            })
            
            sorted_cat_perf = display_cat_perf.sort_values('Search Volume', ascending=False, key=lambda x: x.str.replace('K', '').str.replace('M', '').astype(float)).head(10).reset_index(drop=True)
            
            display_styled_table(
                df=sorted_cat_perf,
                download_filename="beauty_categories_performance.csv",
                max_rows=10,
                align="center"
            )
        else:
            st.info("✨ Category column not found in the dataset.")

    # ========================================
    # 💄 INSIGHTS & RECOMMENDATIONS
    # ========================================
    st.markdown("---")
    st.subheader("💄 Insights & Recommendations")

    # ✅ OPTIMIZED: Calculate insights from current data
    top_categories = queries.groupby('category', observed=True)['conversions'].sum().sort_values(ascending=False).head(3) if 'category' in queries.columns else pd.Series()
    top_brands = queries[queries['brand'] != 'Other'].groupby('brand', observed=True)['conversions'].sum().sort_values(ascending=False).head(3) if 'brand' in queries.columns else pd.Series()
    
    monthly_trend = queries.groupby('month', observed=True)['Counts'].sum().sort_index() if 'month' in queries.columns else pd.Series()
    mom_growth = 0
    trend_emoji = "➡️"
    if len(monthly_trend) >= 2:
        latest_month_vol = monthly_trend.iloc[-1]
        prev_month_vol = monthly_trend.iloc[-2]
        mom_growth = ((latest_month_vol - prev_month_vol) / prev_month_vol * 100) if prev_month_vol > 0 else 0
        trend_emoji = "📈" if mom_growth > 0 else "📉"

    insight_col1, insight_col2 = st.columns(2)

    with insight_col1:
        if not top_categories.empty:
            st.markdown(f"""
            <div class="insight-box">
                <h4>💕 Top Performing Categories</h4>
                <p><strong>Leading Categories:</strong></p>
                <ul>
                    <li><strong>{top_categories.index[0]}</strong>: {format_number(int(top_categories.iloc[0]))} conversions</li>
                    <li><strong>{top_categories.index[1]}</strong>: {format_number(int(top_categories.iloc[1]))} conversions</li>
                    <li><strong>{top_categories.index[2]}</strong>: {format_number(int(top_categories.iloc[2]))} conversions</li>
                </ul>
                <p style="margin-top: 10px; font-size: 0.9em; color: #AD1457;">
                    💡 <em>Focus marketing efforts on these high-conversion categories for optimal ROI.</em>
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="insight-box">
            <h4>💖 Search Volume Trends {trend_emoji}</h4>
            <p><strong>Month-over-Month Growth:</strong> {mom_growth:+.1f}%</p>
            <p>Search interest is <strong>{"increasing" if mom_growth > 0 else "decreasing" if mom_growth < 0 else "stable"}</strong> with <strong>{format_number(total_counts)}</strong> total searches.</p>
            <p style="margin-top: 10px; font-size: 0.9em; color: #AD1457;">
                💡 <em>{"Capitalize on growing demand with expanded inventory and targeted campaigns." if mom_growth > 0 else "Focus on conversion optimization and customer retention strategies."}</em>
            </p>
        </div>
        """, unsafe_allow_html=True)

    with insight_col2:
        if not top_brands.empty:
            st.markdown(f"""
            <div class="insight-box">
                <h4>✨ Brand Performance Leaders</h4>
                <p><strong>Top Converting Brands:</strong></p>
                <ul>
                    <li><strong>{top_brands.index[0]}</strong>: {format_number(int(top_brands.iloc[0]))} conversions</li>
                    <li><strong>{top_brands.index[1]}</strong>: {format_number(int(top_brands.iloc[1]))} conversions</li>
                    <li><strong>{top_brands.index[2]}</strong>: {format_number(int(top_brands.iloc[2]))} conversions</li>
                </ul>
                <p style="margin-top: 10px; font-size: 0.9em; color: #AD1457;">
                    💡 <em>Optimize product placement and inventory for these high-performing brands.</em>
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        high_perf_pct = len(queries[(queries['ctr'] > avg_ctr) & (queries['cr'] > avg_cr)]) / len(queries) * 100 if len(queries) > 0 else 0
        
        st.markdown(f"""
        <div class="insight-box">
            <h4>🔍 Performance Metrics</h4>
            <p><strong>Overall Performance:</strong></p>
            <ul>
                <li>Average CTR: <strong>{avg_ctr:.2f}%</strong></li>
                <li>Average CR: <strong>{avg_cr:.2f}%</strong></li>
                <li>High Performers: <strong>{high_perf_pct:.1f}%</strong> of queries</li>
            </ul>
            <p style="margin-top: 10px; font-size: 0.9em; color: #AD1457;">
                💡 <em>{"Strong performance across key metrics. Continue current strategies." if high_perf_pct > 30 else "Opportunity to improve CTR/CR through A/B testing and content optimization."}</em>
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ✅ ACTIONABLE RECOMMENDATIONS
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📋 Actionable Recommendations")

    rec_col1, rec_col2, rec_col3 = st.columns(3)

    with rec_col1:
        st.markdown(f"""
        <div class="insight-box" style="border-left: 4px solid #D81B60;">
            <h4>🎯 Immediate Actions</h4>
            <ul style="font-size: 0.9em;">
                <li>Optimize high-volume, low-conversion queries</li>
                <li>Increase inventory for {top_categories.index[0] if not top_categories.empty else "top category"}</li>
                <li>Feature {top_brands.index[0] if not top_brands.empty else "top brand"} prominently</li>
                <li>A/B test product images and descriptions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with rec_col2:
        st.markdown("""
        <div class="insight-box" style="border-left: 4px solid #F06292;">
            <h4>📊 Analytics Focus</h4>
            <ul style="font-size: 0.9em;">
                <li>Monitor CTR trends weekly</li>
                <li>Track seasonal patterns</li>
                <li>Analyze customer journey for low-CR queries</li>
                <li>Set up conversion funnel tracking</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with rec_col3:
        st.markdown("""
        <div class="insight-box" style="border-left: 4px solid #EC407A;">
            <h4>💡 Growth Strategies</h4>
            <ul style="font-size: 0.9em;">
                <li>Expand product range in growing categories</li>
                <li>Launch targeted campaigns for high-volume queries</li>
                <li>Implement personalized recommendations</li>
                <li>Optimize mobile shopping experience</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

# ----------------- Search Analysis Tab (ULTRA-OPTIMIZED) -----------------
with tab_search:
    
    # ✅ OPTIMIZED: Single master keyword dictionary (no function wrapper needed)
    MASTER_KEYWORDS = {
        'CLARY': {'variations': ['clary', 'کلاری', 'کلری', 'کلارا', 'clara'], 'threshold': 80, 'min_length': 5},
        'EUCERIN': {'variations': ['eucerin', 'یوسرین', 'یوسیرین', 'یوسری', 'یوسیرن', 'euc'], 'threshold': 75, 'min_length': 3},
        'QV': {'variations': ['qv', 'کیو فی', 'کیوفی', 'کیو', 'کیو وی'], 'threshold': 85, 'min_length': 2},
        'BIODERMA': {'variations': ['bioderma', 'بیودیرما', 'بایودیرما', 'بیودرما'], 'threshold': 75, 'min_length': 7},
        'ACM': {'variations': ['acm', 'ای سی ام', 'ایسیام'], 'threshold': 85, 'min_length': 3},
        'CETAPHIL': {'variations': ['cetaphil', 'سیتافیل', 'ستافیل', 'سیتافی', 'ceta'], 'threshold': 75, 'min_length': 3},
        'CERAVE': {'variations': ['cerave', 'سیرافی', 'سیراف', 'سرافی', 'cera'], 'threshold': 75, 'min_length': 3},
        'LOCA': {'variations': ['loca', 'لوکا', 'لوکه', 'لکا'], 'threshold': 85, 'min_length': 4},
        'COLLAGEN': {'variations': ['collagen', 'کولاجین', 'کولاجن', 'كولاجين', 'کلاجین'], 'threshold': 80, 'min_length': 4},
        'LA ROCHE POSAY': {'variations': ['la roche posay', 'la roche', 'لاروش', 'لاروش بوزیه', 'laroche'], 'threshold': 70, 'min_length': 3},
        'NYX': {'variations': ['nyx', 'نیکس', 'نکس'], 'excluded_terms': ['nuxe', 'نوکس'], 'threshold': 85, 'min_length': 3},
        'VITAYES': {'variations': ['vitayes', 'فیتایس', 'ویتایس'], 'threshold': 80, 'min_length': 7},
        'DERCOS': {'variations': ['dercos', 'دیرکوس', 'درکوس'], 'threshold': 80, 'min_length': 6},
        'MAYBELLINE': {'variations': ['maybelline', 'میبیلین', 'میبلین', 'مایبلین'], 'threshold': 75, 'min_length': 7},
        'MONDAY HAIRCARE': {'variations': ['monday haircare', 'monday', 'مون دای', 'موندای'], 'threshold': 75, 'min_length': 6},
        'LOREAL': {'variations': ['loreal', "l'oreal", 'لوریال'], 'threshold': 70, 'min_length': 6},
        'ELVIVE': {'variations': ['elvive', 'الفیف'], 'threshold': 80, 'min_length': 6},
        'LACABINE': {'variations': ['lacabine', 'لاکابین', 'لاکبین'], 'threshold': 80, 'min_length': 7},
        'SKIN1004': {'variations': ['skin1004', 'skin 1004', 'سکین 1004', '1004'], 'threshold': 75, 'min_length': 4},
        'BYPHASSE': {'variations': ['byphasse', 'بای فاس', 'بایفاس'], 'threshold': 80, 'min_length': 7}
    }
    
    # ✅ OPTIMIZED: Simplified emoji mapping (static, no function)
    EMOJI_MAP = {
        'serum': '💧', 'cream': '🧴', 'mask': '🎭', 'cleanser': '🧼', 'toner': '💦',
        'shampoo': '🧴', 'conditioner': '💆', 'perfume': '🌸', 'lipstick': '💄',
        'foundation': '💅', 'la roche': '🏥', 'cerave': '💙', 'bioderma': '💚',
        'eucerin': '🧪', 'vichy': '💎', 'loreal': '✨', 'cetaphil': '🌊', 'qv': '🩹'
    }
    
    # ✅ OPTIMIZED: Lightweight keyword extraction (no regex compilation)
    def extract_keywords(text, min_length=2):
        """Fast keyword extraction without heavy regex"""
        if not isinstance(text, str) or len(text) < min_length:
            return []
        return [w.strip() for w in text.lower().split() if len(w.strip()) >= min_length]
    
    # ✅ OPTIMIZED: Basic similarity (no fuzzywuzzy dependency)
    def similarity(s1, s2):
        """Fast string similarity without external libs"""
        s1, s2 = s1.lower().strip(), s2.lower().strip()
        if s1 == s2: return 100
        if s1 in s2 or s2 in s1:
            return int((min(len(s1), len(s2)) / max(len(s1), len(s2))) * 90)
        intersection = len(set(s1) & set(s2))
        union = len(set(s1) | set(s2))
        return int((intersection / union) * 80) if union > 0 else 0
    
    # ✅ OPTIMIZED: Single cached function for all keyword analysis
    @st.cache_data(ttl=1800, show_spinner=False)
    def analyze_keywords(_df, master_dict, cache_key):
        """Ultra-fast keyword analysis with fuzzy matching"""
        from collections import defaultdict
        
        if _df.empty:
            return pd.DataFrame()
        
        # Step 1: Extract keywords from queries
        keyword_data = defaultdict(lambda: {'counts': 0, 'clicks': 0, 'conversions': 0, 'queries': []})
        
        for _, row in _df.iterrows():
            query = str(row.get('search', ''))
            if not query:
                continue
            
            keywords = extract_keywords(query)
            for kw in keywords:
                keyword_data[kw]['counts'] += row.get('Counts', 0)
                keyword_data[kw]['clicks'] += row.get('clicks', 0)
                keyword_data[kw]['conversions'] += row.get('conversions', 0)
                keyword_data[kw]['queries'].append(query)
        
        # Step 2: Fuzzy match to master keywords
        grouped = defaultdict(lambda: {'counts': 0, 'clicks': 0, 'conversions': 0, 'variations': [], 'queries': []})
        
        for kw, data in keyword_data.items():
            if len(kw) < 3:
                continue
            
            best_match = None
            best_score = 0
            
            for master, info in master_dict.items():
                if len(kw) < info.get('min_length', 3):
                    continue
                
                # Check exclusions
                if any(ex in kw for ex in info.get('excluded_terms', [])):
                    continue
                
                # Check variations
                for var in info['variations']:
                    score = similarity(kw, var)
                    if score >= info['threshold'] and score > best_score:
                        best_score = score
                        best_match = master
            
            group_key = best_match if best_match else kw
            grouped[group_key]['counts'] += data['counts']
            grouped[group_key]['clicks'] += data['clicks']
            grouped[group_key]['conversions'] += data['conversions']
            grouped[group_key]['variations'].append(kw)
            grouped[group_key]['queries'].extend(data['queries'])
        
        # Step 3: Convert to DataFrame
        results = []
        for keyword, data in grouped.items():
            if data['counts'] == 0:
                continue
            
            ctr = (data['clicks'] / data['counts'] * 100) if data['counts'] > 0 else 0
            cr = (data['conversions'] / data['counts'] * 100) if data['counts'] > 0 else 0
            
            results.append({
                'keyword': keyword,
                'total_counts': data['counts'],
                'total_clicks': data['clicks'],
                'total_conversions': data['conversions'],
                'avg_ctr': round(ctr, 2),
                'cr': round(cr, 2),
                'variations_count': len(set(data['variations'])),
                'unique_queries': len(set(data['queries'])),
                'variations': list(set(data['variations']))[:50],  # Limit to 50
                'example_queries': list(set(data['queries']))[:5]
            })
        
        df = pd.DataFrame(results)
        return df.sort_values('total_counts', ascending=False).reset_index(drop=True) if not df.empty else pd.DataFrame()
    
    # ✅ OPTIMIZED: Lightweight chart creation
    @st.cache_data(ttl=1800, show_spinner=False)
    def create_length_chart(_df):
        """Fast histogram creation"""
        if _df.empty or 'query_length' not in _df.columns:
            return None
        
        fig = px.histogram(_df, x='query_length', nbins=30,
                          title='<b style="color:#C2185B;">Query Length Distribution</b>',
                          color_discrete_sequence=['#F48FB1'])
        fig.update_layout(
            plot_bgcolor='rgba(253,242,248,0.95)',
            paper_bgcolor='rgba(252,228,236,0.8)',
            font=dict(color='#880E4F'),
            height=400
        )
        return fig
    
    # ================================================================================================
    # 🎨 MAIN EXECUTION
    # ================================================================================================
    
    # Hero header
    st.markdown("""
    <div style="text-align: center; padding: 3rem 2rem; background: linear-gradient(135deg, #FCE4EC 0%, #F8BBD0 50%, #F48FB1 100%); 
                border-radius: 20px; margin-bottom: 2rem; box-shadow: 0 8px 32px rgba(194, 24, 91, 0.15);">
        <h1 style="color: #880E4F; margin: 0; font-size: 3rem; font-weight: 700;">💄 Beauty Care Intelligence Hub 💅</h1>
        <p style="color: #C2185B; margin: 1rem 0 0 0; font-size: 1.3rem;">Advanced Brand Matching • Performance Analytics • Beauty Insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ✅ OPTIMIZED: Direct access to session state
    queries = st.session_state.queries
    
    # Calculate keyword performance ONCE
    start_time = datetime.now()
    
    with st.spinner("🔍 Analyzing beauty keywords..."):
        # Create cache key from filter state
        filter_key = f"{queries.shape}_{st.session_state.get('filters_applied', False)}"
        kw_perf_df = analyze_keywords(queries, MASTER_KEYWORDS, filter_key)
    
    processing_time = (datetime.now() - start_time).total_seconds()
    
    # ✅ DISPLAY RESULTS
    if not kw_perf_df.empty:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_keywords = len(kw_perf_df)
        total_volume = kw_perf_df['total_counts'].sum()
        avg_ctr = kw_perf_df['avg_ctr'].mean()
        avg_cr = kw_perf_df['cr'].mean()
        
        metrics = [
            (col1, "💄", format_number(total_keywords), "Keyword Groups"),
            (col2, "🔍", format_number(int(total_volume)), "Total Volume"),
            (col3, "✨", f"{avg_ctr:.1f}%", "Avg CTR"),
            (col4, "💖", f"{avg_cr:.1f}%", "Avg CR")
        ]
        
        for col, icon, value, label in metrics:
            with col:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #FCE4EC 0%, #F8BBD0 100%); padding: 1.5rem; 
                            border-radius: 12px; text-align: center; border: 2px solid #E91E63;">
                    <div style="font-size: 2rem;">{icon}</div>
                    <div style="font-size: 1.8rem; color: #880E4F; font-weight: bold; margin: 0.5rem 0;">{value}</div>
                    <div style="color: #C2185B; font-weight: 600;">{label}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Top 4 keywords
        st.markdown("""
        <div style="background: linear-gradient(135deg, #C2185B 0%, #E91E63 100%); color: white; padding: 1.5rem; 
                    border-radius: 15px; margin: 2rem 0; text-align: center;">
            <h2 style="margin: 0;">🏆 Top Performing Grouped Keywords</h2>
        </div>
        """, unsafe_allow_html=True)
        
        top_4 = kw_perf_df.head(4)
        cols = st.columns(4)
        
        for idx, (_, row) in enumerate(top_4.iterrows()):
            keyword = row['keyword']
            emoji = next((v for k, v in EMOJI_MAP.items() if k in keyword.lower()), '💄')
            
            with cols[idx]:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #FCE4EC 0%, #F8BBD0 100%); padding: 1rem; 
                            border-radius: 10px; border: 2px solid #E91E63; text-align: center; min-height: 170px;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">{emoji}</div>
                    <div style="color: #880E4F; font-weight: bold; margin-bottom: 0.5rem;">{keyword[:18]}</div>
                    <div style="color: #C2185B; font-size: 1.5rem; font-weight: bold;">{format_number(int(row['total_counts']))}</div>
                    <div style="color: #E91E63; font-size: 0.9rem;">{row['variations_count']} variations</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Interactive table
        st.markdown("### 📊 Keyword Performance Table")
        
        num_keywords = st.slider("Number of keywords to display:", 10, min(300, len(kw_perf_df)), 50, 10)
        
        top_keywords = kw_perf_df.head(num_keywords).copy()
        top_keywords['share_pct'] = (top_keywords['total_counts'] / queries['Counts'].sum() * 100).round(2)
        
        # Format for display
        display_df = top_keywords.copy()
        display_df['total_counts'] = display_df['total_counts'].apply(lambda x: format_number(int(x)))
        display_df['total_clicks'] = display_df['total_clicks'].apply(lambda x: format_number(int(x)))
        display_df['total_conversions'] = display_df['total_conversions'].apply(lambda x: format_number(int(x)))
        display_df['share_pct'] = display_df['share_pct'].apply(lambda x: f"{x:.1f}%")
        display_df['avg_ctr'] = display_df['avg_ctr'].apply(lambda x: f"{x:.1f}%")
        display_df['cr'] = display_df['cr'].apply(lambda x: f"{x:.1f}%")
        
        display_df = display_df.rename(columns={
            'keyword': 'Keyword',
            'total_counts': 'Search Volume',
            'share_pct': 'Market Share',
            'total_clicks': 'Clicks',
            'total_conversions': 'Conversions',
            'avg_ctr': 'CTR',
            'cr': 'CR',
            'variations_count': 'Variations',
            'unique_queries': 'Unique Queries'
        })
        
        # Display with reusable function
        display_styled_table(
            df=display_df[['Keyword', 'Search Volume', 'Market Share', 'Clicks', 'Conversions', 'CTR', 'CR', 'Variations', 'Unique Queries']],
            title=f"Top {num_keywords} Keywords",
            download_filename=f"keywords_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            align="center",
            scrollable=True,
            max_height="600px"
        )
        
        # Performance summary
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #FCE4EC 0%, #F8BBD0 100%); padding: 1.5rem; border-radius: 12px; 
                    margin: 2rem 0; text-align: center;">
            <h4 style="color: #880E4F; margin: 0 0 1rem 0;">⚡ Analysis Complete</h4>
            <p style="color: #C2185B; margin: 0;">
                Processed <strong>{len(kw_perf_df):,}</strong> keyword groups in <strong>{processing_time:.2f}</strong> seconds
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Download button
        csv = generate_csv_ultra(top_keywords)
        st.download_button(
            label="📥 Download Keywords CSV",
            data=csv,
            file_name=f"beauty_keywords_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    else:
        st.warning("⚠️ No keyword data available")
    
    # Query length analysis
    st.markdown("---")
    st.markdown("### 📊 Query Length Analysis")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        fig = create_length_chart(queries)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if not queries.empty and 'query_length' in queries.columns:
            avg_length = queries['query_length'].mean()
            median_length = queries['query_length'].median()
            
            st.markdown(f"""
            <div style="background: #FCE4EC; padding: 1.5rem; border-radius: 12px; border-left: 4px solid #E91E63;">
                <h5 style="color: #880E4F; margin: 0 0 1rem 0;">📏 Length Insights</h5>
                <p style="color: #C2185B; margin: 0.3rem 0;"><strong>Average:</strong> {avg_length:.1f} chars</p>
                <p style="color: #C2185B; margin: 0.3rem 0;"><strong>Median:</strong> {median_length:.1f} chars</p>
            </div>
            """, unsafe_allow_html=True)

    


# ----------------- Brand Tab (ULTRA-OPTIMIZED) -----------------
with tab_brand:
    
    # ✅ OPTIMIZED: Single hero header (no duplicate CSS)
    st.markdown("""
    <div style="text-align: center; padding: 3rem 2rem; background: linear-gradient(135deg, #FCE4EC 0%, #F8BBD0 50%, #F48FB1 100%); 
                border-radius: 20px; margin-bottom: 2rem; box-shadow: 0 8px 32px rgba(233, 30, 99, 0.15);">
        <h1 style="color: #880E4F; margin: 0; font-size: 3rem; font-weight: 700;">💄 Brand Market Position 💄</h1>
        <p style="color: #C2185B; margin: 1rem 0 0 0; font-size: 1.3rem;">Advanced Brand Analytics • Market Intelligence • Competitive Insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ✅ OPTIMIZED: Direct access to session state
    queries = st.session_state.queries
    
    # ✅ OPTIMIZED: Find brand column (single pass)
    brand_column = next((col for col in ['brand', 'Brand', 'BRAND', 'Brand Name', 'brand_name'] if col in queries.columns), None)
    
    if not brand_column or not queries[brand_column].notna().any():
        st.error(f"❌ No brand data available. Available columns: {list(queries.columns)}")
        st.stop()
    
    # ✅ OPTIMIZED: Filter out "Other" (single operation)
    brand_queries = queries[
        (queries[brand_column].notna()) & 
        (~queries[brand_column].str.lower().isin(['other', 'others']))
    ]
    
    if brand_queries.empty:
        st.error("❌ No valid brand data after filtering.")
        st.stop()
    
    # ✅ OPTIMIZED: Single cached function for all brand calculations
    @st.cache_data(ttl=300, show_spinner=False)
    def analyze_brands(_df, brand_col, cache_key):
        """Ultra-fast brand analysis with all metrics"""
        if _df.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        # Calculate brand summary (aggregated across all months)
        bs_summary = _df.groupby(brand_col).agg({
            'Counts': 'sum',
            'clicks': 'sum',
            'conversions': 'sum'
        }).reset_index()
        
        bs_summary = bs_summary.rename(columns={brand_col: 'brand'})
        
        # Calculate metrics
        total_counts = bs_summary['Counts'].sum()
        bs_summary['share_pct'] = (bs_summary['Counts'] / total_counts * 100).round(2)
        bs_summary['ctr'] = ((bs_summary['clicks'] / bs_summary['Counts']) * 100).round(2)
        bs_summary['cr'] = ((bs_summary['conversions'] / bs_summary['Counts']) * 100).round(2)
        bs_summary['classic_cr'] = ((bs_summary['conversions'] / bs_summary['clicks']) * 100).fillna(0).round(2)
        
        # Calculate strategic metrics
        max_ctr = bs_summary['ctr'].max() if bs_summary['ctr'].max() > 0 else 1
        max_cr = bs_summary['cr'].max() if bs_summary['cr'].max() > 0 else 1
        
        bs_summary['market_strength'] = bs_summary['share_pct'] * bs_summary['ctr'] / 100
        bs_summary['efficiency_score'] = (bs_summary['conversions'] / bs_summary['Counts'] * 1000).fillna(0)
        bs_summary['growth_potential'] = (
            (100 - bs_summary['share_pct']) * 0.4 +
            (bs_summary['ctr'] / max_ctr * 100) * 0.3 +
            (bs_summary['cr'] / max_cr * 100) * 0.3
        )
        
        # Categorize positions
        median_strength = bs_summary['market_strength'].median()
        median_efficiency = bs_summary['efficiency_score'].median()
        
        def categorize(row):
            if row['market_strength'] >= median_strength and row['efficiency_score'] >= median_efficiency:
                return "🌟 Market Leaders"
            elif row['market_strength'] >= median_strength:
                return "📈 Volume Players"
            elif row['efficiency_score'] >= median_efficiency:
                return "💎 Efficiency Champions"
            return "🌸 Emerging Brands"
        
        bs_summary['position_category'] = bs_summary.apply(categorize, axis=1)
        
        # Monthly data (if needed for trends)
        if 'start_date' in _df.columns:
            _df_with_month = _df.copy()
            _df_with_month['month'] = pd.to_datetime(_df_with_month['start_date']).dt.to_period('M').astype(str)
            
            bs_monthly = _df_with_month.groupby([brand_col, 'month']).agg({
                'Counts': 'sum',
                'clicks': 'sum',
                'conversions': 'sum'
            }).reset_index()
            
            bs_monthly = bs_monthly.rename(columns={brand_col: 'brand'})
            bs_monthly['ctr'] = ((bs_monthly['clicks'] / bs_monthly['Counts']) * 100).round(2)
            bs_monthly['cr'] = ((bs_monthly['conversions'] / bs_monthly['Counts']) * 100).round(2)
        else:
            bs_monthly = pd.DataFrame()
        
        return bs_summary, bs_monthly
    
    # ✅ COMPUTE: All brand data ONCE
    filter_key = f"{brand_queries.shape}_{st.session_state.get('filters_applied', False)}"
    bs_summary, bs_monthly = analyze_brands(brand_queries, brand_column, filter_key)
    
    # ================================================================================================
    # 📊 MAIN VISUALIZATIONS
    # ================================================================================================
    
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.subheader("📈 Brand Performance Matrix")
        
        # Scatter plot
        num_scatter = st.slider("Brands in scatter:", 20, 100, 50, 10, key="scatter_brands")
        
        scatter_data = bs_summary.nlargest(num_scatter, 'Counts')
        
        fig_perf = px.scatter(
            scatter_data, x='Counts', y='ctr', size='clicks', color='cr',
            hover_name='brand',
            title='<b style="color:#C2185B;">💄 Brand Performance Matrix</b>',
            color_continuous_scale=['#FCE4EC', '#F48FB1', '#C2185B']
        )
        
        fig_perf.update_traces(
            hovertemplate='<b>%{hovertext}</b><br>Searches: %{x:,.0f}<br>CTR: %{y:.1f}%<br>Clicks: %{marker.size:,.0f}<br>CR: %{marker.color:.1f}%<extra></extra>'
        )
        
        fig_perf.update_layout(
            plot_bgcolor='rgba(252, 228, 236, 0.95)',
            paper_bgcolor='rgba(248, 187, 208, 0.3)',
            font=dict(color='#880E4F'),
            height=500
        )
        
        st.plotly_chart(fig_perf, use_container_width=True)
        
        # Trend analysis
        if not bs_monthly.empty and 'Date' in brand_queries.columns:
            st.subheader("📈 Brand Trend Analysis")
            
            top_5_brands = bs_summary.nlargest(5, 'Counts')['brand'].tolist()
            trend_data = bs_monthly[bs_monthly['brand'].isin(top_5_brands)].copy()
            
            if not trend_data.empty:
                trend_data['Date'] = pd.to_datetime(trend_data['month'] + '-01')
                
                fig_trend = px.line(
                    trend_data, x='Date', y='Counts', color='brand',
                    title='<b style="color:#C2185B;">💄 Top 5 Brands Monthly Trend</b>',
                    color_discrete_sequence=['#C2185B', '#E91E63', '#F48FB1', '#F8BBD0', '#FCE4EC'],
                    markers=True
                )
                
                fig_trend.update_layout(
                    plot_bgcolor='rgba(252, 228, 236, 0.95)',
                    paper_bgcolor='rgba(248, 187, 208, 0.3)',
                    font=dict(color='#880E4F'),
                    height=400
                )
                
                st.plotly_chart(fig_trend, use_container_width=True)
    
    with col_right:
        st.subheader("🌸 Brand Market Share")
        
        num_pie = st.slider("Brands in pie:", 5, 20, 10, 1, key="pie_brands")
        
        pie_data = bs_summary.nlargest(num_pie, 'Counts')
        
        fig_pie = px.pie(
            pie_data, names='brand', values='Counts',
            title=f'<b style="color:#C2185B;">💄 Top {len(pie_data)} Brands</b>',
            color_discrete_sequence=['#C2185B', '#E91E63', '#F48FB1', '#F8BBD0', '#FCE4EC'] * 4
        )
        
        fig_pie.update_layout(
            font=dict(color='#880E4F'),
            paper_bgcolor='rgba(248, 187, 208, 0.3)'
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Performance categories
        st.subheader("🎯 Performance Categories")
        
        bs_summary['perf_category'] = pd.cut(
            bs_summary['ctr'],
            bins=[0, 2, 5, 10, float('inf')],
            labels=['Emerging', 'Growing', 'Strong', 'Premium']
        )
        
        cat_counts = bs_summary['perf_category'].value_counts().reset_index()
        cat_counts.columns = ['Category', 'Count']
        
        fig_cat = px.bar(
            cat_counts, x='Category', y='Count',
            title='<b style="color:#C2185B;">💄 CTR Distribution</b>',
            color='Count',
            color_continuous_scale=['#FCE4EC', '#C2185B'],
            text='Count'
        )
        
        fig_cat.update_layout(
            plot_bgcolor='rgba(252, 228, 236, 0.95)',
            paper_bgcolor='rgba(248, 187, 208, 0.3)',
            font=dict(color='#880E4F'),
            height=300
        )
        
        st.plotly_chart(fig_cat, use_container_width=True)
    
    st.markdown("---")
    
    # ================================================================================================
    # 📊 TOP BRANDS TABLE
    # ================================================================================================
    
    st.subheader("🏆 Top Brands Performance")
    
    num_brands = st.slider("Brands to display:", 10, 50, 20, 5, key="table_brands")
    
    top_brands = bs_summary.nlargest(num_brands, 'Counts').copy()
    
    # Format for display
    display_df = top_brands.copy()
    display_df['Counts'] = display_df['Counts'].apply(lambda x: format_number(int(x)))
    display_df['clicks'] = display_df['clicks'].apply(lambda x: format_number(int(x)))
    display_df['conversions'] = display_df['conversions'].apply(lambda x: format_number(int(x)))
    display_df['share_pct'] = display_df['share_pct'].apply(lambda x: f"{x:.1f}%")
    display_df['ctr'] = display_df['ctr'].apply(lambda x: f"{x:.1f}%")
    display_df['cr'] = display_df['cr'].apply(lambda x: f"{x:.1f}%")
    display_df['classic_cr'] = display_df['classic_cr'].apply(lambda x: f"{x:.1f}%")
    
    display_df = display_df.rename(columns={
        'brand': 'Brand',
        'Counts': 'Search Volume',
        'share_pct': 'Market Share',
        'clicks': 'Clicks',
        'conversions': 'Conversions',
        'ctr': 'CTR',
        'cr': 'CR (Search)',
        'classic_cr': 'Classic CR'
    })
    
    # Display with reusable function
    display_styled_table(
        df=display_df[['Brand', 'Search Volume', 'Market Share', 'Clicks', 'Conversions', 'CTR', 'CR (Search)', 'Classic CR']],
        title=f"Top {num_brands} Brands",
        download_filename=f"brands_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        align="center",
        scrollable=True,
        max_height="600px"
    )
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = [
        (col1, "🏆", len(bs_summary), "Total Brands"),
        (col2, "🔍", format_number(int(bs_summary['Counts'].sum())), "Total Volume"),
        (col3, "💄", format_number(int(bs_summary['clicks'].sum())), "Total Clicks"),
        (col4, "💖", format_number(int(bs_summary['conversions'].sum())), "Conversions")
    ]
    
    for col, icon, value, label in metrics:
        with col:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #FCE4EC 0%, #F8BBD0 100%); padding: 1.5rem; 
                        border-radius: 12px; text-align: center; border: 2px solid #E91E63;">
                <div style="font-size: 2rem;">{icon}</div>
                <div style="font-size: 1.8rem; color: #880E4F; font-weight: bold; margin: 0.5rem 0;">{value}</div>
                <div style="color: #C2185B; font-weight: 600;">{label}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ================================================================================================
    # 🔥 BRAND-KEYWORD MATRIX
    # ================================================================================================
    
    st.subheader("🔥 Brand-Keyword Intelligence")
    
    # Aggregate matrix data
    matrix_data = brand_queries.groupby([brand_column, 'search']).agg({
        'Counts': 'sum',
        'clicks': 'sum',
        'conversions': 'sum'
    }).reset_index()
    
    matrix_data = matrix_data.rename(columns={brand_column: 'brand'})
    matrix_data['ctr'] = ((matrix_data['clicks'] / matrix_data['Counts']) * 100).round(2)
    matrix_data['cr'] = ((matrix_data['conversions'] / matrix_data['Counts']) * 100).round(2)
    
    # Brand selector
    available_brands = sorted(matrix_data['brand'].unique())
    selected_brand = st.selectbox(
        "🎯 Select Brand:",
        ['All Brands'] + available_brands,
        key="matrix_brand"
    )
    
    if selected_brand == 'All Brands':
        top_8_brands = bs_summary.nlargest(8, 'Counts')['brand'].tolist()
        filtered_matrix = matrix_data[matrix_data['brand'].isin(top_8_brands)]
        
        # Get top 12 searches
        top_searches = filtered_matrix.groupby('search')['Counts'].sum().nlargest(12).index.tolist()
        filtered_matrix = filtered_matrix[filtered_matrix['search'].isin(top_searches)]
        
        # Create heatmap
        heatmap_data = filtered_matrix.pivot(index='brand', columns='search', values='Counts').fillna(0)
        
        fig_matrix = px.imshow(
            heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            color_continuous_scale=['#FCE4EC', '#F48FB1', '#C2185B'],
            title='<b style="color:#C2185B;">💄 Brand-Search Heatmap</b>',
            aspect='auto'
        )
        
        fig_matrix.update_layout(
            plot_bgcolor='rgba(252, 228, 236, 0.95)',
            paper_bgcolor='rgba(248, 187, 208, 0.3)',
            font=dict(color='#880E4F'),
            height=600
        )
        
        st.plotly_chart(fig_matrix, use_container_width=True)
    
    else:
        brand_data = matrix_data[matrix_data['brand'] == selected_brand].nlargest(15, 'Counts')
        
        fig_brand = px.bar(
            brand_data, x='search', y='Counts',
            title=f'<b style="color:#C2185B;">💄 {selected_brand} - Top Searches</b>',
            color='cr',
            color_continuous_scale=['#FCE4EC', '#F48FB1', '#C2185B'],
            text='Counts'
        )
        
        fig_brand.update_traces(
            texttemplate='%{text}',
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Volume: %{y:,.0f}<br>CTR: %{customdata[0]:.1f}%<br>CR: %{marker.color:.1f}%<extra></extra>',
            customdata=brand_data[['ctr']].values,
            text=[format_number(x) for x in brand_data['Counts']]
        )
        
        fig_brand.update_layout(
            plot_bgcolor='rgba(252, 228, 236, 0.95)',
            paper_bgcolor='rgba(248, 187, 208, 0.3)',
            font=dict(color='#880E4F'),
            height=500,
            xaxis=dict(tickangle=45)
        )
        
        st.plotly_chart(fig_brand, use_container_width=True)
    
    st.markdown("---")
    
    # ================================================================================================
    # 🧠 STRATEGIC INSIGHTS
    # ================================================================================================
    
    st.subheader("🧠 Strategic Intelligence")
    
    tab1, tab2, tab3 = st.tabs(["🎯 Market Position", "🚀 Growth Opportunities", "💡 Competitive Intel"])
    
    with tab1:
        # Quadrant analysis
        fig_quad = px.scatter(
            bs_summary.head(30),
            x='market_strength', y='efficiency_score',
            size='Counts', color='position_category',
            hover_name='brand',
            title='<b style="color:#C2185B;">🎯 Market Position Quadrant</b>',
            color_discrete_map={
                "🌟 Market Leaders": "#C2185B",
                "📈 Volume Players": "#E91E63",
                "💎 Efficiency Champions": "#F48FB1",
                "🌸 Emerging Brands": "#F8BBD0"
            }
        )
        
        fig_quad.update_layout(
            plot_bgcolor='rgba(252, 228, 236, 0.95)',
            paper_bgcolor='rgba(248, 187, 208, 0.3)',
            font=dict(color='#880E4F'),
            height=500
        )
        
        st.plotly_chart(fig_quad, use_container_width=True)
    
    with tab2:
        # Growth opportunities
        high_opp = bs_summary[
            (bs_summary['growth_potential'] > bs_summary['growth_potential'].quantile(0.7)) &
            (bs_summary['share_pct'] < 10)
        ].nlargest(10, 'growth_potential')
        
        if not high_opp.empty:
            fig_opp = px.bar(
                high_opp, x='growth_potential', y='brand', orientation='h',
                title='<b style="color:#C2185B;">🚀 Top Growth Opportunities</b>',
                color='growth_potential',
                color_continuous_scale=['#FCE4EC', '#C2185B']
            )
            
            fig_opp.update_layout(
                plot_bgcolor='rgba(252, 228, 236, 0.95)',
                paper_bgcolor='rgba(248, 187, 208, 0.3)',
                font=dict(color='#880E4F'),
                height=500
            )
            
            st.plotly_chart(fig_opp, use_container_width=True)
    
    with tab3:
        # Competitive insights
        top_performer = bs_summary.loc[bs_summary['Counts'].idxmax()]
        efficiency_leader = bs_summary.loc[bs_summary['cr'].idxmax()]
        top_5_share = bs_summary.nlargest(5, 'Counts')['share_pct'].sum()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #FCE4EC 0%, #F8BBD0 100%); padding: 1.5rem; border-radius: 12px; border-left: 5px solid #E91E63;">
                <h4 style="color: #880E4F;">🎯 Market Intelligence</h4>
                <p><strong>Market Leader:</strong> {top_performer['brand']} ({top_performer['share_pct']:.1f}%)</p>
                <p><strong>Top 5 Share:</strong> {top_5_share:.1f}%</p>
                <p><strong>Total Brands:</strong> {len(bs_summary)}</p>
                <p><strong>Avg CTR:</strong> {bs_summary['ctr'].mean():.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #FCE4EC 0%, #F8BBD0 100%); padding: 1.5rem; border-radius: 12px; border-left: 5px solid #F48FB1;">
                <h4 style="color: #880E4F;">🏆 Performance Leaders</h4>
                <p><strong>Volume Leader:</strong> {top_performer['brand']}</p>
                <p><strong>Efficiency Leader:</strong> {efficiency_leader['brand']} ({efficiency_leader['cr']:.1f}%)</p>
                <p><strong>Best CTR:</strong> {bs_summary.loc[bs_summary['ctr'].idxmax(), 'brand']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Download section
    st.markdown("---")
    st.subheader("📥 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = generate_csv_ultra(bs_summary)
        st.download_button(
            "📊 Full Analysis",
            csv,
            f"brands_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col2:
        strategic = bs_summary[['brand', 'Counts', 'share_pct', 'ctr', 'cr', 'position_category', 'growth_potential']]
        csv2 = generate_csv_ultra(strategic)
        st.download_button(
            "🎯 Strategic Insights",
            csv2,
            f"strategic_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col3:
        csv3 = generate_csv_ultra(matrix_data)
        st.download_button(
            "🔥 Brand-Keyword Matrix",
            csv3,
            f"matrix_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            "text/csv",
            use_container_width=True
        )



# ----------------- Category Tab (ULTRA-OPTIMIZED - PINK THEME) -----------------
with tab_category:
    
    # ✅ PINK THEME: Hero header matching Brand/Search tabs
    st.markdown("""
    <div style="text-align: center; padding: 3rem 2rem; background: linear-gradient(135deg, #FCE4EC 0%, #F8BBD0 50%, #F48FB1 100%); 
                border-radius: 20px; margin-bottom: 2rem; box-shadow: 0 8px 32px rgba(233, 30, 99, 0.15); border: 1px solid rgba(233, 30, 99, 0.2);">
        <h1 style="color: #880E4F; margin: 0; font-size: 3rem; font-weight: 700; text-shadow: 2px 2px 8px rgba(136, 14, 79, 0.2);">
            💄 Category Performance Analysis 💄
        </h1>
        <p style="color: #C2185B; margin: 1rem 0 0 0; font-size: 1.3rem; font-weight: 300; opacity: 0.9;">
            Advanced Category Analytics • Performance Insights • Market Intelligence
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # ✅ OPTIMIZED: Direct access to session state
    queries = st.session_state.queries
    
    # ✅ OPTIMIZED: Find category column (single pass)
    category_column = next((col for col in ['category', 'Category', 'CATEGORY', 'Category Name', 'category_name', 'product_category'] if col in queries.columns), None)
    
    if not category_column or not queries[category_column].notna().any():
        st.error(f"❌ No category data available. Available columns: {list(queries.columns)}")
        st.stop()
    
    # ✅ OPTIMIZED: Filter out "Other" (single operation)
    category_queries = queries[
        (queries[category_column].notna()) & 
        (~queries[category_column].str.lower().isin(['other', 'others']))
    ]
    
    if category_queries.empty:
        st.error("❌ No valid category data after filtering.")
        st.stop()
    
    # ✅ OPTIMIZED: Single cached function for all category calculations
    @st.cache_data(ttl=300, show_spinner=False)
    def analyze_categories(_df, cat_col, cache_key):
        """Ultra-fast category analysis with all metrics"""
        if _df.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        # Calculate category summary (aggregated across all months)
        cs_summary = _df.groupby(cat_col).agg({
            'Counts': 'sum',
            'clicks': 'sum',
            'conversions': 'sum'
        }).reset_index()
        
        cs_summary = cs_summary.rename(columns={cat_col: 'category'})
        
        # Calculate metrics
        total_counts = cs_summary['Counts'].sum()
        cs_summary['share_pct'] = (cs_summary['Counts'] / total_counts * 100).round(2)
        cs_summary['ctr'] = ((cs_summary['clicks'] / cs_summary['Counts']) * 100).round(2)
        cs_summary['cr'] = ((cs_summary['conversions'] / cs_summary['Counts']) * 100).round(2)
        cs_summary['classic_cr'] = ((cs_summary['conversions'] / cs_summary['clicks']) * 100).fillna(0).round(2)
        
        # Performance categories
        cs_summary['perf_category'] = pd.cut(
            cs_summary['ctr'],
            bins=[0, 2, 5, 10, float('inf')],
            labels=['Emerging', 'Growing', 'Strong', 'Premium']
        )
        
        # Monthly data (if needed for trends)
        if 'start_date' in _df.columns:
            _df_with_month = _df.copy()
            _df_with_month['month'] = pd.to_datetime(_df_with_month['start_date']).dt.to_period('M').astype(str)
            
            cs_monthly = _df_with_month.groupby([cat_col, 'month']).agg({
                'Counts': 'sum',
                'clicks': 'sum',
                'conversions': 'sum'
            }).reset_index()
            
            cs_monthly = cs_monthly.rename(columns={cat_col: 'category'})
            cs_monthly['ctr'] = ((cs_monthly['clicks'] / cs_monthly['Counts']) * 100).round(2)
            cs_monthly['cr'] = ((cs_monthly['conversions'] / cs_monthly['Counts']) * 100).round(2)
        else:
            cs_monthly = pd.DataFrame()
        
        return cs_summary, cs_monthly
    
    # ✅ COMPUTE: All category data ONCE
    filter_key = f"{category_queries.shape}_{st.session_state.get('filters_applied', False)}"
    cs_summary, cs_monthly = analyze_categories(category_queries, category_column, filter_key)
    
    # ================================================================================================
    # 📊 MAIN VISUALIZATIONS (PINK THEME)
    # ================================================================================================
    
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.subheader("📈 Category Performance Matrix")
        
        # Scatter plot with PINK theme
        scatter_data = cs_summary.head(30)
        
        fig_perf = px.scatter(
            scatter_data, x='Counts', y='ctr', size='clicks', color='cr',
            hover_name='category',
            title='<b style="color:#C2185B;">💄 Category Performance Matrix</b>',
            color_continuous_scale=['#FCE4EC', '#F48FB1', '#C2185B']  # ✅ PINK GRADIENT
        )
        
        fig_perf.update_traces(
            hovertemplate='<b>%{hovertext}</b><br>Searches: %{customdata[0]}<br>CTR: %{y:.1f}%<br>Clicks: %{customdata[1]}<br>CR: %{marker.color:.1f}%<extra></extra>',
            customdata=[[format_number(row['Counts']), format_number(row['clicks'])] for _, row in scatter_data.iterrows()]
        )
        
        fig_perf.update_layout(
            plot_bgcolor='rgba(252, 228, 236, 0.95)',  # ✅ PINK BACKGROUND
            paper_bgcolor='rgba(248, 187, 208, 0.3)',
            font=dict(color='#880E4F', family='Segoe UI'),  # ✅ PINK TEXT
            height=500,
            xaxis=dict(showgrid=True, gridcolor='#FCE4EC', linecolor='#E91E63', linewidth=2),
            yaxis=dict(showgrid=True, gridcolor='#FCE4EC', linecolor='#E91E63', linewidth=2)
        )
        
        st.plotly_chart(fig_perf, use_container_width=True)
        
        # Bar charts with PINK theme
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            top_15 = cs_summary.nlargest(15, 'Counts')
            
            fig_counts = px.bar(
                top_15, x='category', y='Counts',
                title='<b style="color:#C2185B;">💄 Top 15 Categories</b>',
                color='Counts',
                color_continuous_scale=['#FCE4EC', '#C2185B'],  # ✅ PINK GRADIENT
                text='Counts'
            )
            
            fig_counts.update_traces(
                texttemplate='%{text}',
                textposition='outside',
                text=[format_number(x) for x in top_15['Counts']]
            )
            
            fig_counts.update_layout(
                plot_bgcolor='rgba(252, 228, 236, 0.95)',
                paper_bgcolor='rgba(248, 187, 208, 0.3)',
                font=dict(color='#880E4F', family='Segoe UI'),
                xaxis=dict(tickangle=45, showgrid=True, gridcolor='#FCE4EC'),
                yaxis=dict(showgrid=True, gridcolor='#FCE4EC'),
                height=400
            )
            
            st.plotly_chart(fig_counts, use_container_width=True)
        
        with col_chart2:
            top_cr = cs_summary.nlargest(15, 'cr')
            
            fig_cr = px.bar(
                top_cr, x='category', y='cr',
                title='<b style="color:#C2185B;">💖 Top 15 by CR</b>',
                color='cr',
                color_continuous_scale=['#F8BBD0', '#880E4F'],  # ✅ PINK GRADIENT
                text='cr'
            )
            
            fig_cr.update_traces(
                texttemplate='%{text:.1f}%',
                textposition='outside'
            )
            
            fig_cr.update_layout(
                plot_bgcolor='rgba(252, 228, 236, 0.95)',
                paper_bgcolor='rgba(248, 187, 208, 0.3)',
                font=dict(color='#880E4F', family='Segoe UI'),
                xaxis=dict(tickangle=45, showgrid=True, gridcolor='#FCE4EC'),
                yaxis=dict(showgrid=True, gridcolor='#FCE4EC'),
                height=400
            )
            
            st.plotly_chart(fig_cr, use_container_width=True)
    
    with col_right:
        st.subheader("💄 Category Market Share")
        
        pie_data = cs_summary.nlargest(10, 'Counts')
        
        fig_pie = px.pie(
            pie_data, names='category', values='Counts',
            title='<b style="color:#C2185B;">💄 Top 10 Categories</b>',
            color_discrete_sequence=['#C2185B', '#E91E63', '#F48FB1', '#F8BBD0', '#FCE4EC'] * 2  # ✅ PINK PALETTE
        )
        
        fig_pie.update_layout(
            font=dict(color='#880E4F', family='Segoe UI'),
            paper_bgcolor='rgba(248, 187, 208, 0.3)'
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Performance distribution with PINK theme
        cat_counts = cs_summary['perf_category'].value_counts().reset_index()
        cat_counts.columns = ['Performance Level', 'Count']
        
        fig_cat = px.bar(
            cat_counts, x='Performance Level', y='Count',
            title='<b style="color:#C2185B;">💄 CTR Distribution</b>',
            color='Count',
            color_continuous_scale=['#FCE4EC', '#C2185B'],  # ✅ PINK GRADIENT
            text='Count'
        )
        
        fig_cat.update_layout(
            plot_bgcolor='rgba(252, 228, 236, 0.95)',
            paper_bgcolor='rgba(248, 187, 208, 0.3)',
            font=dict(color='#880E4F', family='Segoe UI'),
            height=300,
            xaxis=dict(showgrid=True, gridcolor='#FCE4EC'),
            yaxis=dict(showgrid=True, gridcolor='#FCE4EC')
        )
        
        st.plotly_chart(fig_cat, use_container_width=True)
        
        # Trend analysis with PINK theme
        if not cs_monthly.empty and 'Date' in category_queries.columns:
            st.subheader("📈 Category Trends")
            
            top_5 = cs_summary.nlargest(5, 'Counts')['category'].tolist()
            trend_data = cs_monthly[cs_monthly['category'].isin(top_5)].copy()
            
            if not trend_data.empty:
                trend_data['Date'] = pd.to_datetime(trend_data['month'] + '-01')
                
                fig_trend = px.line(
                    trend_data, x='Date', y='Counts', color='category',
                    title='<b style="color:#C2185B;">💄 Top 5 Trends</b>',
                    color_discrete_sequence=['#C2185B', '#E91E63', '#F48FB1', '#F8BBD0', '#FCE4EC'],  # ✅ PINK PALETTE
                    markers=True
                )
                
                fig_trend.update_layout(
                    plot_bgcolor='rgba(252, 228, 236, 0.95)',
                    paper_bgcolor='rgba(248, 187, 208, 0.3)',
                    font=dict(color='#880E4F', family='Segoe UI'),
                    height=400,
                    xaxis=dict(showgrid=True, gridcolor='#FCE4EC'),
                    yaxis=dict(showgrid=True, gridcolor='#FCE4EC')
                )
                
                st.plotly_chart(fig_trend, use_container_width=True)
    
    st.markdown("---")
    
    # ================================================================================================
    # 📊 TOP CATEGORIES TABLE (PINK THEME)
    # ================================================================================================
    
    st.subheader("🏆 Top Categories Performance")
    
    num_categories = st.slider("Categories to display:", 10, 50, 20, 5, key="table_cats")
    
    top_categories = cs_summary.nlargest(num_categories, 'Counts').copy()
    
    # Format for display
    display_df = top_categories.copy()
    display_df['Counts'] = display_df['Counts'].apply(lambda x: format_number(int(x)))
    display_df['clicks'] = display_df['clicks'].apply(lambda x: format_number(int(x)))
    display_df['conversions'] = display_df['conversions'].apply(lambda x: format_number(int(x)))
    display_df['share_pct'] = display_df['share_pct'].apply(lambda x: f"{x:.1f}%")
    display_df['ctr'] = display_df['ctr'].apply(lambda x: f"{x:.1f}%")
    display_df['cr'] = display_df['cr'].apply(lambda x: f"{x:.1f}%")
    display_df['classic_cr'] = display_df['classic_cr'].apply(lambda x: f"{x:.1f}%")
    
    display_df = display_df.rename(columns={
        'category': 'Category',
        'Counts': 'Search Volume',
        'share_pct': 'Market Share',
        'clicks': 'Clicks',
        'conversions': 'Conversions',
        'ctr': 'CTR',
        'cr': 'CR (Search)',
        'classic_cr': 'Classic CR'
    })
    
    # Display with reusable function
    display_styled_table(
        df=display_df[['Category', 'Search Volume', 'Market Share', 'Clicks', 'Conversions', 'CTR', 'CR (Search)', 'Classic CR']],
        title=f"Top {num_categories} Categories",
        download_filename=f"categories_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        align="center",
        scrollable=True,
        max_height="600px"
    )
    
    # Summary metrics with PINK theme
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = [
        (col1, "💄", len(cs_summary), "Total Categories"),
        (col2, "🔍", format_number(int(cs_summary['Counts'].sum())), "Total Volume"),
        (col3, "💖", format_number(int(cs_summary['clicks'].sum())), "Total Clicks"),
        (col4, "✨", format_number(int(cs_summary['conversions'].sum())), "Conversions")
    ]
    
    for col, icon, value, label in metrics:
        with col:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #FCE4EC 0%, #F8BBD0 100%); padding: 1.5rem; 
                        border-radius: 12px; text-align: center; border: 2px solid #E91E63; 
                        box-shadow: 0 4px 15px rgba(233, 30, 99, 0.2);">
                <div style="font-size: 2rem;">{icon}</div>
                <div style="font-size: 1.8rem; color: #880E4F; font-weight: bold; margin: 0.5rem 0;">{value}</div>
                <div style="color: #C2185B; font-weight: 600;">{label}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ================================================================================================
    # 🔥 CATEGORY-KEYWORD MATRIX (PINK THEME)
    # ================================================================================================
    
    st.subheader("🔥 Category-Keyword Intelligence")
    
    # Aggregate matrix data
    matrix_data = category_queries.groupby([category_column, 'search']).agg({
        'Counts': 'sum',
        'clicks': 'sum',
        'conversions': 'sum'
    }).reset_index()
    
    matrix_data = matrix_data.rename(columns={category_column: 'category'})
    matrix_data['ctr'] = ((matrix_data['clicks'] / matrix_data['Counts']) * 100).round(2)
    matrix_data['cr'] = ((matrix_data['conversions'] / matrix_data['Counts']) * 100).round(2)
    matrix_data['classic_cr'] = ((matrix_data['conversions'] / matrix_data['clicks']) * 100).fillna(0).round(2)
    
    # Category selector with PINK theme
    st.markdown("""
    <div style="background: linear-gradient(135deg, #FCE4EC 0%, #F1F8E9 100%); border: 2px solid #E91E63; 
                border-radius: 15px; padding: 1.5rem; margin: 1rem 0; box-shadow: 0 4px 15px rgba(233, 30, 99, 0.2);">
        <h4 style="color: #880E4F; margin: 0 0 1rem 0; text-align: center;">🎯 Category Analysis Control Center</h4>
    </div>
    """, unsafe_allow_html=True)
    
    available_categories = sorted(matrix_data['category'].unique())
    selected_category = st.selectbox(
        "🎯 Select Category:",
        ['All Categories'] + available_categories,
        key="matrix_category"
    )
    
    if selected_category == 'All Categories':
        top_8_cats = cs_summary.nlargest(8, 'Counts')['category'].tolist()
        filtered_matrix = matrix_data[matrix_data['category'].isin(top_8_cats)]
        
        # Get top 12 searches
        top_searches = filtered_matrix.groupby('search')['Counts'].sum().nlargest(12).index.tolist()
        filtered_matrix = filtered_matrix[filtered_matrix['search'].isin(top_searches)]
        
        # Create heatmap with PINK theme
        heatmap_data = filtered_matrix.pivot(index='category', columns='search', values='Counts').fillna(0)
        
        # Create hover data
        ctr_data = filtered_matrix.pivot(index='category', columns='search', values='ctr').fillna(0)
        cr_data = filtered_matrix.pivot(index='category', columns='search', values='cr').fillna(0)
        classic_cr_data = filtered_matrix.pivot(index='category', columns='search', values='classic_cr').fillna(0)
        
        hover_text = []
        for i, cat in enumerate(heatmap_data.index):
            hover_row = []
            for j, search in enumerate(heatmap_data.columns):
                counts = heatmap_data.iloc[i, j]
                ctr = ctr_data.iloc[i, j]
                cr = cr_data.iloc[i, j]
                classic_cr = classic_cr_data.iloc[i, j]
                hover_row.append(
                    f"<b>{cat}</b><br>Search: {search}<br>Volume: {format_number(counts)}<br>" +
                    f"CTR: {ctr:.1f}%<br>CR: {cr:.1f}%<br>Classic CR: {classic_cr:.1f}%"
                )
            hover_text.append(hover_row)
        
        fig_matrix = px.imshow(
            heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            color_continuous_scale=['#FCE4EC', '#F48FB1', '#C2185B'],  # ✅ PINK GRADIENT
            title='<b style="color:#C2185B;">💄 Category-Search Heatmap</b>',
            aspect='auto'
        )
        
        fig_matrix.update_traces(
            hovertemplate='%{customdata}<extra></extra>',
            customdata=hover_text
        )
        
        fig_matrix.update_layout(
            plot_bgcolor='rgba(252, 228, 236, 0.95)',
            paper_bgcolor='rgba(248, 187, 208, 0.3)',
            font=dict(color='#880E4F', family='Segoe UI'),
            height=600,
            xaxis=dict(tickangle=45, showgrid=True, gridcolor='#FCE4EC'),
            yaxis=dict(showgrid=True, gridcolor='#FCE4EC')
        )
        
        st.plotly_chart(fig_matrix, use_container_width=True)
    
    else:
        cat_data = matrix_data[matrix_data['category'] == selected_category].nlargest(15, 'Counts')
        
        cr_option = st.radio(
            "Color by:",
            ['CR (Search)', 'Classic CR'],
            horizontal=True,
            key="cat_cr_option"
        )
        
        color_col = 'classic_cr' if cr_option == 'Classic CR' else 'cr'
        
        fig_cat = px.bar(
            cat_data, x='search', y='Counts',
            title=f'<b style="color:#C2185B;">💄 {selected_category} - Top Searches</b>',
            color=color_col,
            color_continuous_scale=['#FCE4EC', '#F48FB1', '#C2185B'],  # ✅ PINK GRADIENT
            text='Counts'
        )
        
        fig_cat.update_traces(
            texttemplate='%{text}',
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Volume: %{customdata[0]}<br>CTR: %{customdata[1]:.1f}%<br>CR: %{customdata[2]:.1f}%<br>Classic CR: %{customdata[3]:.1f}%<extra></extra>',
            customdata=[[format_number(row['Counts']), row['ctr'], row['cr'], row['classic_cr']] for _, row in cat_data.iterrows()],
            text=[format_number(x) for x in cat_data['Counts']]
        )
        
        fig_cat.update_layout(
            plot_bgcolor='rgba(252, 228, 236, 0.95)',
            paper_bgcolor='rgba(248, 187, 208, 0.3)',
            font=dict(color='#880E4F', family='Segoe UI'),
            height=500,
            xaxis=dict(tickangle=45, showgrid=True, gridcolor='#FCE4EC'),
            yaxis=dict(showgrid=True, gridcolor='#FCE4EC')
        )
        
        st.plotly_chart(fig_cat, use_container_width=True)
    
    st.markdown("---")
    
    # ================================================================================================
    # 🔑 TOP KEYWORDS PER CATEGORY (PINK THEME)
    # ================================================================================================
    
    st.subheader("🔑 Top Keywords per Category")
    
    num_keywords = st.selectbox(
        "Keywords to analyze:",
        [10, 15, 20, 25, 30, 50],
        index=0,
        key="num_keywords"
    )
    
    # Calculate keywords per category
    rows = []
    for cat, grp in category_queries.groupby(category_column):
        keyword_counts = {}
        
        for _, row in grp.iterrows():
            keywords_list = row.get('keywords', [])
            query_count = row['Counts']
            
            if isinstance(keywords_list, list):
                for kw in keywords_list:
                    keyword_counts[kw] = keyword_counts.get(kw, 0) + query_count
            elif pd.notna(keywords_list):
                search_term = row.get('normalized_query', '')
                if pd.notna(search_term):
                    for kw in str(search_term).lower().split():
                        keyword_counts[kw] = keyword_counts.get(kw, 0) + query_count
        
        # Get top N keywords
        top_kws = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:num_keywords]
        
        for kw, count in top_kws:
            rows.append({'category': cat, 'keyword': kw, 'count': count})
    
    df_ckw = pd.DataFrame(rows)
    
    if not df_ckw.empty:
        # Create summary
        summary_data = []
        
        for cat in df_ckw['category'].unique():
            cat_data = df_ckw[df_ckw['category'] == cat].nlargest(num_keywords, 'count')
            
            keywords_str = ' | '.join([f"{row['keyword']} ({format_number(row['count'])})" for _, row in cat_data.iterrows()])
            
            cat_total = cs_summary[cs_summary['category'] == cat]['Counts'].iloc[0] if len(cs_summary[cs_summary['category'] == cat]) > 0 else 0
            share_pct = (cat_total / cs_summary['Counts'].sum() * 100) if cs_summary['Counts'].sum() > 0 else 0
            
            summary_data.append({
                'Category': cat,
                f'Top {num_keywords} Keywords': keywords_str,
                'Total Volume': format_number(cat_total),
                'Market Share': f"{share_pct:.1f}%",
                'Unique Keywords': len(df_ckw[df_ckw['category'] == cat]),
                '_sort': cat_total
            })
        
        summary_data = sorted(summary_data, key=lambda x: x['_sort'], reverse=True)
        summary_df = pd.DataFrame(summary_data).drop('_sort', axis=1)
        
        display_styled_table(
            df=summary_df,
            title=f"Top {num_keywords} Keywords per Category",
            download_filename=f"category_keywords_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            scrollable=True,
            max_height="500px",
            align="center"
        )
        
        # Insights with PINK theme
        st.markdown("---")
        st.subheader("📊 Category Keyword Intelligence")
        
        col1, col2, col3 = st.columns(3)
        
        # Calculate insights
        cat_stats = {}
        for cat in df_ckw['category'].unique():
            cat_kw_data = df_ckw[df_ckw['category'] == cat]
            cat_total = cs_summary[cs_summary['category'] == cat]['Counts'].iloc[0] if len(cs_summary[cs_summary['category'] == cat]) > 0 else 0
            
            cat_stats[cat] = {
                'total_keywords': len(cat_kw_data),
                'total_count': cat_total,
                'share_pct': (cat_total / cs_summary['Counts'].sum() * 100) if cs_summary['Counts'].sum() > 0 else 0
            }
        
        most_diverse = max(cat_stats.items(), key=lambda x: x[1]['total_keywords'])
        highest_volume = max(cat_stats.items(), key=lambda x: x[1]['total_count'])
        
        with col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #E91E63 0%, #F48FB1 100%); padding: 1.5rem; 
                        border-radius: 12px; text-align: center; color: white; box-shadow: 0 6px 20px rgba(233, 30, 99, 0.3);">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🌟</div>
                <div style="font-size: 1.4rem; font-weight: bold; margin-bottom: 0.5rem;">{most_diverse[0][:20]}...</div>
                <div style="font-size: 1rem; opacity: 0.95;">Most Diverse Category</div>
                <div style="font-size: 0.9rem; margin-top: 0.5rem;">{most_diverse[1]['total_keywords']} unique keywords</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #C2185B 0%, #E91E63 100%); padding: 1.5rem; 
                        border-radius: 12px; text-align: center; color: white; box-shadow: 0 6px 20px rgba(194, 24, 91, 0.3);">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🚀</div>
                <div style="font-size: 1.4rem; font-weight: bold; margin-bottom: 0.5rem;">{highest_volume[0][:20]}...</div>
                <div style="font-size: 1rem; opacity: 0.95;">Highest Volume</div>
                <div style="font-size: 0.9rem; margin-top: 0.5rem;">{format_number(highest_volume[1]['total_count'])} searches</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            avg_keywords = sum(x['total_keywords'] for x in cat_stats.values()) / len(cat_stats)
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #880E4F 0%, #C2185B 100%); padding: 1.5rem; 
                        border-radius: 12px; text-align: center; color: white; box-shadow: 0 6px 20px rgba(136, 14, 79, 0.3);">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📊</div>
                <div style="font-size: 1.4rem; font-weight: bold; margin-bottom: 0.5rem;">{avg_keywords:.0f}</div>
                <div style="font-size: 1rem; opacity: 0.95;">Avg Keywords</div>
                <div style="font-size: 0.9rem; margin-top: 0.5rem;">per category</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Download section with PINK theme
    st.markdown("---")
    st.subheader("📥 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = generate_csv_ultra(cs_summary)
        st.download_button(
            "📊 Full Analysis",
            csv,
            f"categories_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col2:
        csv2 = generate_csv_ultra(matrix_data)
        st.download_button(
            "🔥 Category-Keyword Matrix",
            csv2,
            f"matrix_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col3:
        if not df_ckw.empty:
            csv3 = generate_csv_ultra(df_ckw)
            st.download_button(
                "🔑 Keywords Analysis",
                csv3,
                f"keywords_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                "text/csv",
                use_container_width=True
            )

    

# ----------------- Subcategory Tab (ULTRA-OPTIMIZED - PINK THEME) -----------------
with tab_subcat:
    # ✅ PINK THEME: Hero header matching other tabs
    st.markdown("""
    <div style="text-align: center; padding: 3rem 2rem; background: linear-gradient(135deg, #FCE4EC 0%, #F8BBD0 50%, #F48FB1 100%); 
                border-radius: 20px; margin-bottom: 2rem; box-shadow: 0 8px 32px rgba(233, 30, 99, 0.15); border: 1px solid rgba(233, 30, 99, 0.2);">
        <h1 style="color: #880E4F; margin: 0; font-size: 3rem; font-weight: 700; text-shadow: 2px 2px 8px rgba(136, 14, 79, 0.2);">
            💄 Subcategory Intelligence Hub 💄
        </h1>
        <p style="color: #C2185B; margin: 1rem 0 0 0; font-size: 1.3rem; font-weight: 300; opacity: 0.9;">
            Deep dive into subcategory performance and search trends
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # ✅ OPTIMIZED: Direct access to session state
    queries = st.session_state.queries
    
    # ✅ OPTIMIZED: Find subcategory column (single pass)
    subcategory_column = next((col for col in ['sub_category', 'Sub_Category', 'SUB_CATEGORY', 'subcategory', 'Subcategory', 'SUBCATEGORY', 'sub category', 'Sub Category'] if col in queries.columns), None)
    
    if not subcategory_column or not queries[subcategory_column].notna().any():
        st.error(f"❌ No subcategory data available. Available columns: {list(queries.columns)}")
        st.info("💡 Please ensure your dataset contains a subcategory column")
        st.stop()
    
    # ✅ OPTIMIZED: Filter and clean subcategory data
    subcategory_queries = queries[
        (queries[subcategory_column].notna()) & 
        (~queries[subcategory_column].str.lower().isin(['other', 'others', 'n/a', 'na', 'none', '']))
    ].copy()
    
    if subcategory_queries.empty:
        st.error("❌ No valid subcategory data after filtering.")
        st.stop()
    
    # ✅ OPTIMIZED: Single cached function for all subcategory calculations
    @st.cache_data(ttl=300, show_spinner=False)
    def analyze_subcategories(_df, subcat_col, cache_key):
        """Ultra-fast subcategory analysis with all metrics"""
        if _df.empty:
            return pd.DataFrame(), {}
        
        # Ensure numeric columns
        numeric_cols = ['Counts', 'clicks', 'conversions']
        for col in numeric_cols:
            if col in _df.columns:
                _df[col] = pd.to_numeric(_df[col], errors='coerce').fillna(0)
        
        # Calculate subcategory summary
        sc = _df.groupby(subcat_col).agg({
            'Counts': 'sum',
            'clicks': 'sum',
            'conversions': 'sum'
        }).reset_index()
        
        sc = sc.rename(columns={subcat_col: 'sub_category'})
        
        # Convert to integers
        sc['Counts'] = sc['Counts'].round().astype('int64')
        sc['clicks'] = sc['clicks'].round().astype('int64')
        sc['conversions'] = sc['conversions'].round().astype('int64')
        
        # Calculate metrics
        total_clicks = int(sc['clicks'].sum())
        total_conversions = int(sc['conversions'].sum())
        
        sc['ctr'] = ((sc['clicks'] / sc['Counts']) * 100).round(2)
        sc['classic_cvr'] = ((sc['conversions'] / sc['clicks']) * 100).fillna(0).round(2)
        sc['conversion_rate'] = ((sc['conversions'] / sc['Counts']) * 100).round(2)
        sc['click_share'] = (sc['clicks'] / total_clicks * 100) if total_clicks > 0 else 0
        sc['conversion_share'] = (sc['conversions'] / total_conversions * 100) if total_conversions > 0 else 0
        
        sc = sc.sort_values('Counts', ascending=False).reset_index(drop=True)
        
        # Calculate market metrics
        total_subcats = len(sc)
        top_5_conc = sc.head(5)['Counts'].sum() / sc['Counts'].sum() * 100 if total_subcats >= 5 else 100
        top_10_conc = sc.head(10)['Counts'].sum() / sc['Counts'].sum() * 100 if total_subcats >= 10 else 100
        
        # Gini and Herfindahl
        if len(sc) > 1:
            sorted_counts = sc['Counts'].sort_values()
            cumsum_counts = np.cumsum(sorted_counts)
            n = len(sc)
            gini = (2 * np.sum((np.arange(1, n + 1) * sorted_counts))) / (n * np.sum(sorted_counts)) - (n + 1) / n
            herfindahl = np.sum((sc['Counts'] / sc['Counts'].sum()) ** 2)
        else:
            gini = 0
            herfindahl = 1
        
        market_stats = {
            'total_subcats': total_subcats,
            'top_5_conc': top_5_conc,
            'top_10_conc': top_10_conc,
            'gini': gini,
            'herfindahl': herfindahl
        }
        
        return sc, market_stats
    
    # ✅ COMPUTE: All subcategory data ONCE
    filter_key = f"{subcategory_queries.shape}_{st.session_state.get('filters_applied', False)}"
    sc, market_stats = analyze_subcategories(subcategory_queries, subcategory_column, filter_key)
    
    # ================================================================================================
    # 📊 KEY METRICS (PINK THEME)
    # ================================================================================================
    
    st.subheader("💄 Subcategory Performance Overview")
    
    # Calculate key metrics
    total_searches = int(sc['Counts'].sum())
    total_clicks = int(sc['clicks'].sum())
    total_conversions = int(sc['conversions'].sum())
    avg_ctr = float(sc['ctr'].mean())
    avg_cr = float(sc['conversion_rate'].mean())
    top_subcat = sc.iloc[0]['sub_category'] if len(sc) > 0 else 'N/A'
    top_subcat_vol = int(sc.iloc[0]['Counts']) if len(sc) > 0 else 0
    top_conv_subcat = sc.nlargest(1, 'conversions')['sub_category'].iloc[0] if len(sc) > 0 else 'N/A'
    
    # First metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    metrics_row1 = [
        (col1, "💄", format_number(market_stats['total_subcats']), "Total Subcategories", "Active segments"),
        (col2, "🔍", format_number(total_searches), "Total Searches", "Across all subcategories"),
        (col3, "📈", f"{avg_ctr:.1f}%", "Average CTR", "Click-through rate"),
        (col4, "👑", top_subcat[:12] + "..." if len(top_subcat) > 12 else top_subcat, "Top Subcategory", f"{(top_subcat_vol/total_searches*100):.1f}% share")
    ]
    
    for col, icon, value, label, sublabel in metrics_row1:
        with col:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #FCE4EC 0%, #F8BBD0 100%); padding: 1.5rem; 
                        border-radius: 12px; text-align: center; border: 2px solid #E91E63; 
                        box-shadow: 0 4px 15px rgba(233, 30, 99, 0.2);">
                <div style="font-size: 2rem;">{icon}</div>
                <div style="font-size: 1.6rem; color: #880E4F; font-weight: bold; margin: 0.5rem 0;">{value}</div>
                <div style="color: #C2185B; font-weight: 600; font-size: 1.1rem;">{label}</div>
                <div style="color: #E91E63; font-size: 0.9rem; margin-top: 0.3rem;">{sublabel}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Second metrics row
    col5, col6, col7, col8 = st.columns(4)
    
    metrics_row2 = [
        (col5, "💖", f"{avg_cr:.1f}%", "Avg Conversion Rate", "Overall performance"),
        (col6, "🖱️", format_number(total_clicks), "Total Clicks", "Across all subcategories"),
        (col7, "✅", format_number(total_conversions), "Total Conversions", "Successful outcomes"),
        (col8, "🏆", top_conv_subcat[:12] + "..." if len(top_conv_subcat) > 12 else top_conv_subcat, "Conversion Leader", "Most conversions")
    ]
    
    for col, icon, value, label, sublabel in metrics_row2:
        with col:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #FCE4EC 0%, #F8BBD0 100%); padding: 1.5rem; 
                        border-radius: 12px; text-align: center; border: 2px solid #E91E63; 
                        box-shadow: 0 4px 15px rgba(233, 30, 99, 0.2);">
                <div style="font-size: 2rem;">{icon}</div>
                <div style="font-size: 1.6rem; color: #880E4F; font-weight: bold; margin: 0.5rem 0;">{value}</div>
                <div style="color: #C2185B; font-weight: 600; font-size: 1.1rem;">{label}</div>
                <div style="color: #E91E63; font-size: 0.9rem; margin-top: 0.3rem;">{sublabel}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ================================================================================================
    # 📊 TOP PERFORMERS (PINK THEME)
    # ================================================================================================
    
    st.subheader("🏆 Top 20 Subcategories Performance")
    
    display_count = min(20, len(sc))
    top_sc = sc.head(display_count).copy()
    
    # Combined chart
    fig_combined = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig_combined.add_trace(
        go.Bar(
            name='Search Volume',
            x=top_sc['sub_category'],
            y=top_sc['Counts'],
            marker_color='rgba(233, 30, 99, 0.7)',
            text=[format_number(int(x)) for x in top_sc['Counts']],
            textposition='outside'
        ),
        secondary_y=False
    )
    
    fig_combined.add_trace(
        go.Scatter(
            name='CTR %',
            x=top_sc['sub_category'],
            y=top_sc['ctr'],
            mode='lines+markers',
            line=dict(color='#FF6B35', width=3),
            marker=dict(size=8)
        ),
        secondary_y=True
    )
    
    fig_combined.add_trace(
        go.Scatter(
            name='Conversion Rate %',
            x=top_sc['sub_category'],
            y=top_sc['conversion_rate'],
            mode='lines+markers',
            line=dict(color='#9C27B0', width=3, dash='dash'),
            marker=dict(size=8)
        ),
        secondary_y=True
    )
    
    fig_combined.update_layout(
        title='<b style="color:#C2185B;">💄 Search Volume vs Performance</b>',
        plot_bgcolor='rgba(252, 228, 236, 0.95)',
        paper_bgcolor='rgba(248, 187, 208, 0.3)',
        font=dict(color='#880E4F', family='Segoe UI'),
        height=600,
        xaxis=dict(tickangle=45, showgrid=True, gridcolor='#FCE4EC'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig_combined.update_yaxes(title_text="<b>Search Volume</b>", secondary_y=False, showgrid=True, gridcolor='#FCE4EC')
    fig_combined.update_yaxes(title_text="<b>Performance Rate (%)</b>", secondary_y=True, showgrid=False)
    
    st.plotly_chart(fig_combined, use_container_width=True)
    
    # Bar chart
    fig_bar = px.bar(
        top_sc, x='sub_category', y='Counts',
        title=f'<b style="color:#C2185B;">💄 Top {display_count} by Volume</b>',
        color='Counts',
        color_continuous_scale=['#FCE4EC', '#F48FB1', '#C2185B'],
        text='Counts'
    )
    
    fig_bar.update_traces(
        texttemplate='%{text}',
        textposition='outside',
        text=[format_number(int(x)) for x in top_sc['Counts']]
    )
    
    fig_bar.update_layout(
        plot_bgcolor='rgba(252, 228, 236, 0.95)',
        paper_bgcolor='rgba(248, 187, 208, 0.3)',
        font=dict(color='#880E4F', family='Segoe UI'),
        height=500,
        xaxis=dict(tickangle=45, showgrid=True, gridcolor='#FCE4EC'),
        yaxis=dict(showgrid=True, gridcolor='#FCE4EC'),
        showlegend=False
    )
    
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Performance comparison
    fig_perf = go.Figure()
    
    fig_perf.add_trace(go.Bar(
        name='CTR %',
        x=top_sc['sub_category'],
        y=top_sc['ctr'],
        marker_color='#E91E63',
        text=[f'{x:.1f}%' for x in top_sc['ctr']],
        textposition='outside'
    ))
    
    fig_perf.add_trace(go.Bar(
        name='Conversion Rate %',
        x=top_sc['sub_category'],
        y=top_sc['conversion_rate'],
        marker_color='#F48FB1',
        text=[f'{x:.1f}%' for x in top_sc['conversion_rate']],
        textposition='outside'
    ))
    
    fig_perf.update_layout(
        title='<b style="color:#C2185B;">💄 CTR vs Conversion Rate</b>',
        barmode='group',
        plot_bgcolor='rgba(252, 228, 236, 0.95)',
        paper_bgcolor='rgba(248, 187, 208, 0.3)',
        font=dict(color='#880E4F', family='Segoe UI'),
        height=500,
        xaxis=dict(tickangle=45, showgrid=True, gridcolor='#FCE4EC'),
        yaxis=dict(title='Percentage (%)', showgrid=True, gridcolor='#FCE4EC')
    )
    
    st.plotly_chart(fig_perf, use_container_width=True)
    
    st.markdown("---")
    
    # ================================================================================================
    # 📊 MARKET SHARE ANALYSIS (PINK THEME)
    # ================================================================================================
    
    st.subheader("📊 Market Share & Distribution")
    
    col_pie, col_treemap = st.columns(2)
    
    with col_pie:
        pie_count = min(10, len(sc))
        top_market = sc.head(pie_count).copy()
        others_value = sc.iloc[pie_count:]['Counts'].sum() if len(sc) > pie_count else 0
        
        if others_value > 0:
            others_row = pd.DataFrame({'sub_category': ['Others'], 'Counts': [others_value]})
            pie_data = pd.concat([top_market[['sub_category', 'Counts']], others_row])
        else:
            pie_data = top_market[['sub_category', 'Counts']]
        
        fig_pie = px.pie(
            pie_data, values='Counts', names='sub_category',
            title=f'<b style="color:#C2185B;">💄 Top {pie_count} Market Share</b>',
            color_discrete_sequence=['#C2185B', '#E91E63', '#F48FB1', '#F8BBD0', '#FCE4EC'] * 3
        )
        
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(
            height=400,
            font=dict(color='#880E4F', family='Segoe UI'),
            paper_bgcolor='rgba(248, 187, 208, 0.3)'
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col_treemap:
        treemap_count = min(20, len(sc))
        fig_treemap = px.treemap(
            sc.head(treemap_count),
            path=['sub_category'],
            values='Counts',
            title=f'<b style="color:#C2185B;">💄 Volume Distribution (Top {treemap_count})</b>',
            color='ctr',
            color_continuous_scale=['#FCE4EC', '#F48FB1', '#C2185B']
        )
        
        fig_treemap.update_layout(
            height=400,
            font=dict(color='#880E4F', family='Segoe UI'),
            paper_bgcolor='rgba(248, 187, 208, 0.3)'
        )
        
        st.plotly_chart(fig_treemap, use_container_width=True)
    
    # Market metrics
    col1, col2, col3, col4 = st.columns(4)
    
    market_metrics = [
        (col1, "📊", f"{market_stats['gini']:.3f}", "Gini Coefficient", "Market concentration"),
        (col2, "📈", f"{market_stats['herfindahl']:.4f}", "Herfindahl Index", "Market dominance"),
        (col3, "🔝", f"{market_stats['top_5_conc']:.1f}%", "Top 5 Share", "Market concentration"),
        (col4, "💯", f"{market_stats['top_10_conc']:.1f}%", "Top 10 Share", "Market concentration")
    ]
    
    for col, icon, value, label, sublabel in market_metrics:
        with col:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #E91E63 0%, #F48FB1 100%); padding: 1.5rem; 
                        border-radius: 12px; text-align: center; color: white; box-shadow: 0 6px 20px rgba(233, 30, 99, 0.3);">
                <div style="font-size: 2rem;">{icon}</div>
                <div style="font-size: 1.6rem; font-weight: bold; margin: 0.5rem 0;">{value}</div>
                <div style="font-size: 1.1rem; opacity: 0.95;">{label}</div>
                <div style="font-size: 0.9rem; margin-top: 0.3rem; opacity: 0.9;">{sublabel}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Lorenz curve
    if len(sc) > 1:
        sorted_counts = sc['Counts'].sort_values()
        cumulative_counts = np.cumsum(sorted_counts) / sorted_counts.sum()
        cumulative_subcats = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
        
        fig_lorenz = go.Figure()
        
        fig_lorenz.add_trace(go.Scatter(
            x=cumulative_subcats, y=cumulative_counts,
            mode='lines', name='Actual Distribution',
            line=dict(color='#E91E63', width=3)
        ))
        
        fig_lorenz.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines', name='Perfect Equality',
            line=dict(color='#F48FB1', width=2, dash='dash')
        ))
        
        fig_lorenz.update_layout(
            title='<b style="color:#C2185B;">💄 Lorenz Curve - Market Distribution</b>',
            xaxis_title='Cumulative % of Subcategories',
            yaxis_title='Cumulative % of Search Volume',
            plot_bgcolor='rgba(252, 228, 236, 0.95)',
            paper_bgcolor='rgba(248, 187, 208, 0.3)',
            font=dict(color='#880E4F', family='Segoe UI'),
            height=500,
            xaxis=dict(showgrid=True, gridcolor='#FCE4EC'),
            yaxis=dict(showgrid=True, gridcolor='#FCE4EC')
        )
        
        st.plotly_chart(fig_lorenz, use_container_width=True)
    
    # ================================================================================================
    # 📥 DOWNLOAD OPTIONS
    # ================================================================================================
    
    st.markdown("---")
    st.subheader("📥 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = generate_csv_ultra(sc)
        st.download_button(
            "📊 Full Analysis",
            csv,
            f"subcategories_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col2:
        top_20_csv = generate_csv_ultra(top_sc)
        st.download_button(
            "🏆 Top 20 Performers",
            top_20_csv,
            f"top_20_subcats_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col3:
        market_data = pd.DataFrame({
            'Metric': ['Total Subcategories', 'Gini Coefficient', 'Herfindahl Index', 'Top 5 Share %', 'Top 10 Share %'],
            'Value': [market_stats['total_subcats'], market_stats['gini'], market_stats['herfindahl'], market_stats['top_5_conc'], market_stats['top_10_conc']]
        })
        market_csv = generate_csv_ultra(market_data)
        st.download_button(
            "📈 Market Metrics",
            market_csv,
            f"market_metrics_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            "text/csv",
            use_container_width=True
        )


                    

# ----------------- Class Tab (Pink Theme + Optimized) -----------------
with tab_class:
    # 🎨 PINK-THEMED HERO HEADER
    st.markdown("""
    <div style="
        text-align: center; 
        padding: 3rem 2rem; 
        background: linear-gradient(135deg, #FCE4EC 0%, #F8BBD0 50%, #F48FB1 100%); 
        border-radius: 20px; 
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(136, 14, 79, 0.15);
        border: 1px solid rgba(233, 30, 99, 0.2);
    ">
        <h1 style="
            color: #880E4F; 
            margin: 0; 
            font-size: 3rem; 
            text-shadow: 2px 2px 8px rgba(136, 14, 79, 0.2);
            font-weight: 700;
            letter-spacing: -1px;
        ">
            💄 Class Performance Analysis ✨
        </h1>
        <p style="
            color: #C2185B; 
            margin: 1rem 0 0 0; 
            font-size: 1.3rem;
            font-weight: 300;
            opacity: 0.9;
        ">
            Advanced Classification • Performance Analytics • Search Intelligence
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 🎨 PINK-THEMED CSS (Load once per session)
    if 'class_pink_css_loaded' not in st.session_state:
        st.markdown("""
        <style>
        .pink-class-metric {
            background: linear-gradient(135deg, #FCE4EC 0%, #F8BBD0 100%);
            padding: 20px; border-radius: 12px; text-align: center;
            box-shadow: 0 4px 15px rgba(194, 24, 91, 0.2);
            margin: 10px 0; border-left: 4px solid #E91E63;
        }
        
        .pink-class-insight {
            background: linear-gradient(135deg, #C2185B 0%, #F06292 100%);
            padding: 25px; border-radius: 15px; color: white;
            margin: 15px 0; box-shadow: 0 6px 20px rgba(194, 24, 91, 0.3);
        }
        
        .enhanced-pink-class-metric {
            background: linear-gradient(135deg, #E91E63 0%, #F48FB1 100%);
            padding: 25px; border-radius: 15px; text-align: center; color: white;
            box-shadow: 0 8px 32px rgba(233, 30, 99, 0.3); margin: 10px 0;
            min-height: 160px; display: flex; flex-direction: column; justify-content: center;
        }
        
        .enhanced-pink-class-metric .icon { font-size: 3em; margin-bottom: 10px; display: block; }
        .enhanced-pink-class-metric .value { font-size: 1.6em; font-weight: bold; margin-bottom: 8px; line-height: 1.2; }
        .enhanced-pink-class-metric .label { font-size: 1.1em; opacity: 0.95; font-weight: 600; margin-bottom: 6px; }
        .enhanced-pink-class-metric .sub-label { font-size: 1em; opacity: 0.9; font-weight: 500; line-height: 1.2; }
        
        .class-metric-card {
            background: linear-gradient(135deg, #FCE4EC 0%, #F8BBD0 100%);
            padding: 15px; border-radius: 10px; text-align: center;
            box-shadow: 0 4px 15px rgba(194, 24, 91, 0.2);
            margin: 5px; border-left: 4px solid #E91E63;
        }
        
        .class-metric-value { font-size: 1.4em; font-weight: bold; color: #880E4F; margin-bottom: 5px; }
        .class-metric-label { color: #C2185B; font-size: 0.9em; font-weight: 600; }
        
        .pink-class-performance-increase { background-color: rgba(233, 30, 99, 0.1) !important; }
        .pink-class-performance-decrease { background-color: rgba(244, 67, 54, 0.1) !important; }
        </style>
        """, unsafe_allow_html=True)
        st.session_state.class_pink_css_loaded = True
    
    # ✅ CHECK: Class column availability
    class_column = None
    possible_class_columns = ['Class', 'class', 'CLASS', 'Class Name', 'class_name', 'product_class']
    
    for col in possible_class_columns:
        if col in queries.columns:
            class_column = col
            break
    
    has_class_data = (class_column is not None and queries[class_column].notna().any())
    
    if not has_class_data:
        st.error(f"❌ No class data available. Available columns: {list(queries.columns)}")
        st.info("💡 Please ensure your dataset contains a class column")
        st.stop()
    
    # Filter out "Other" class
    class_queries = queries[
        (queries[class_column].notna()) & 
        (~queries[class_column].str.lower().isin(['other', 'others']))
    ]
    
    if class_queries.empty:
        st.error("❌ No valid class data available after filtering.")
        st.stop()
    
    st.markdown("---")
    
    # 🎯 MAIN LAYOUT
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.subheader("📈 Class Performance Matrix")
        
        # 📊 CALCULATE: Class metrics (cached via session state)
        @st.cache_data(ttl=1800, show_spinner=False)
        def compute_class_metrics(_df, class_col):
            """Compute aggregated class metrics"""
            cls = _df.groupby(class_col).agg({
                'Counts': 'sum',
                'clicks': 'sum', 
                'conversions': 'sum'
            }).reset_index()
            
            cls['clicks'] = cls['clicks'].round().astype(int)
            cls['conversions'] = cls['conversions'].round().astype(int)
            cls = cls.rename(columns={class_col: 'class'})
            
            # Calculate metrics
            cls['ctr'] = (cls['clicks'] / cls['Counts'] * 100).fillna(0)
            cls['cr'] = (cls['conversions'] / cls['Counts'] * 100).fillna(0)
            cls['classic_cr'] = (cls['conversions'] / cls['clicks'] * 100).fillna(0)
            
            total_counts = cls['Counts'].sum()
            cls['share_pct'] = (cls['Counts'] / total_counts * 100).round(2)
            
            return cls
        
        cls = compute_class_metrics(class_queries, class_column)
        
        # 📊 SCATTER PLOT: Performance matrix
        fig_class_perf = px.scatter(
            cls.head(30), 
            x='Counts', 
            y='ctr',
            size='clicks',
            color='cr',
            hover_name='class',
            title='<b style="color:#C2185B; font-size:18px;">💄 Class Performance Matrix: Volume vs CTR</b>',
            labels={'Counts': 'Total Searches', 'ctr': 'CTR (%)', 'cr': 'CR (%)'},
            color_continuous_scale=['#FCE4EC', '#F48FB1', '#C2185B'],
            template='plotly_white'
        )
        
        fig_class_perf.update_traces(
            hovertemplate='<b>%{hovertext}</b><br>' +
                         'Searches: %{customdata[0]}<br>' +
                         'CTR: %{y:.1f}%<br>' +
                         'Clicks: %{customdata[1]}<br>' +
                         'CR: %{marker.color:.1f}%<extra></extra>',
            customdata=[[format_number(row['Counts']), format_number(row['clicks'])] 
                       for _, row in cls.head(30).iterrows()]
        )
        
        fig_class_perf.update_layout(
            plot_bgcolor='rgba(252, 228, 236, 0.3)',
            paper_bgcolor='rgba(252, 228, 236, 0.1)',
            font=dict(color='#880E4F', family='Segoe UI'),
            title_x=0,
            xaxis=dict(showgrid=True, gridcolor='#F8BBD0', linecolor='#E91E63', linewidth=2),
            yaxis=dict(showgrid=True, gridcolor='#F8BBD0', linecolor='#E91E63', linewidth=2),
        )
        
        st.plotly_chart(fig_class_perf, use_container_width=True)
        
        # 📊 DUAL CHARTS: Volume & CR
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            fig_counts = px.bar(
                cls.sort_values('Counts', ascending=False).head(15), 
                x='class', 
                y='Counts',
                title='<b style="color:#C2185B;">💖 Searches by Class</b>',
                color='Counts',
                color_continuous_scale=['#FCE4EC', '#C2185B'],
                text='Counts'
            )
            
            fig_counts.update_traces(
                texttemplate='%{text}',
                textposition='outside',
                text=[format_number(x) for x in cls.sort_values('Counts', ascending=False).head(15)['Counts']]
            )
            
            fig_counts.update_layout(
                plot_bgcolor='rgba(252, 228, 236, 0.3)',
                paper_bgcolor='rgba(252, 228, 236, 0.1)',
                font=dict(color='#880E4F', family='Segoe UI'),
                xaxis=dict(tickangle=45, showgrid=True, gridcolor='#F8BBD0'),
                yaxis=dict(showgrid=True, gridcolor='#F8BBD0'),
                height=400
            )
            
            st.plotly_chart(fig_counts, use_container_width=True)
        
        with col_chart2:
            fig_cr = px.bar(
                cls.sort_values('cr', ascending=False).head(15), 
                x='class', 
                y='cr',
                title='<b style="color:#C2185B;">✨ Conversion Rate by Class</b>',
                color='cr',
                color_continuous_scale=['#F48FB1', '#880E4F'],
                text='cr'
            )
            
            fig_cr.update_traces(
                texttemplate='%{text:.1f}%',
                textposition='outside'
            )
            
            fig_cr.update_layout(
                plot_bgcolor='rgba(252, 228, 236, 0.3)',
                paper_bgcolor='rgba(252, 228, 236, 0.1)',
                font=dict(color='#880E4F', family='Segoe UI'),
                xaxis=dict(tickangle=45, showgrid=True, gridcolor='#F8BBD0'),
                yaxis=dict(showgrid=True, gridcolor='#F8BBD0'),
                height=400
            )
            
            st.plotly_chart(fig_cr, use_container_width=True)
        
        # 🏆 TOP CLASSES TABLE
        st.subheader("🏆 Top Class Performance")
        
        num_classes = st.slider(
            "Number of classes to display:", 
            min_value=10, 
            max_value=50, 
            value=20, 
            step=5,
            key="class_count_slider"
        )
        
        # Reuse month names from main dashboard
        if 'month_names' not in locals():
            month_names = get_dynamic_month_names(queries)
        
        # Ensure month column exists
        if 'month' not in queries.columns and 'start_date' in queries.columns:
            queries['month'] = pd.to_datetime(queries['start_date']).dt.to_period('M').astype(str)
        
        # 🚀 COMPUTE: Monthly class performance
        @st.cache_data(ttl=1800, show_spinner=False)
        def compute_class_monthly(_df, _cls, month_dict, num_cls, class_col):
            """Build monthly class performance table"""
            if _df.empty or _cls.empty:
                return pd.DataFrame(), []
            
            top_classes_list = _cls.nlargest(num_cls, 'Counts')['class'].tolist()
            top_data = _df[_df[class_col].isin(top_classes_list)].copy()
            
            unique_months = sorted(top_data['month'].dropna().unique(), key=lambda x: pd.to_datetime(x)) if 'month' in top_data.columns else []
            
            result_data = []
            for class_name in top_classes_list:
                class_data = top_data[top_data[class_col] == class_name]
                if class_data.empty:
                    continue
                
                total_counts = int(class_data['Counts'].sum())
                if total_counts == 0:
                    continue
                
                total_clicks = int(class_data['clicks'].sum())
                total_conversions = int(class_data['conversions'].sum())
                dataset_total = _df['Counts'].sum()
                share_pct = (total_counts / dataset_total * 100) if dataset_total > 0 else 0
                overall_ctr = (total_clicks / total_counts * 100) if total_counts > 0 else 0
                overall_cr = (total_conversions / total_counts * 100) if total_counts > 0 else 0
                
                row = {
                    'Class': class_name,
                    'Total Volume': total_counts,
                    'Share %': share_pct,
                    'Overall CTR': overall_ctr,
                    'Overall CR': overall_cr,
                    'Total Clicks': total_clicks,
                    'Total Conversions': total_conversions
                }
                
                for month in unique_months:
                    month_display = month_dict.get(month, month)
                    month_data = class_data[class_data['month'] == month]
                    
                    if not month_data.empty:
                        m_counts = int(month_data['Counts'].sum())
                        m_clicks = int(month_data['clicks'].sum())
                        m_conversions = int(month_data['conversions'].sum())
                        m_ctr = (m_clicks / m_counts * 100) if m_counts > 0 else 0
                        m_cr = (m_conversions / m_counts * 100) if m_counts > 0 else 0
                        
                        row[f'{month_display} Vol'] = m_counts
                        row[f'{month_display} CTR'] = m_ctr
                        row[f'{month_display} CR'] = m_cr
                    else:
                        row[f'{month_display} Vol'] = 0
                        row[f'{month_display} CTR'] = 0
                        row[f'{month_display} CR'] = 0
                
                result_data.append(row)
            
            result_df = pd.DataFrame(result_data)
            result_df = result_df.sort_values('Total Volume', ascending=False).reset_index(drop=True)
            return result_df[result_df['Total Volume'] > 0], unique_months
        
        top_classes_monthly, unique_months_cls = compute_class_monthly(
            queries, cls, month_names, num_classes, class_column
        )
        
        if not top_classes_monthly.empty:
            # Filter status
            unique_classes_count = queries[class_column].nunique()
            if st.session_state.get('filters_applied', False):
                st.info(f"🔍 **Filtered**: Top {num_classes} from {unique_classes_count:,} classes")
            else:
                st.info(f"📊 **All Data**: Top {num_classes} from {unique_classes_count:,} classes")
            
            # Column organization
            base_columns = ['Class', 'Total Volume', 'Share %', 'Overall CTR', 'Overall CR', 'Total Clicks', 'Total Conversions']
            sorted_months = sorted(unique_months_cls, key=lambda x: pd.to_datetime(x))
            
            volume_columns = [f'{month_names.get(m, m)} Vol' for m in sorted_months]
            ctr_columns = [f'{month_names.get(m, m)} CTR' for m in sorted_months]
            cr_columns = [f'{month_names.get(m, m)} CR' for m in sorted_months]
            
            ordered_columns = base_columns + volume_columns + ctr_columns + cr_columns
            existing_columns = [col for col in ordered_columns if col in top_classes_monthly.columns]
            top_classes_monthly = top_classes_monthly[existing_columns]
            
            # Format display
            display_classes = top_classes_monthly.copy()
            
            for col in ['Total Volume'] + volume_columns:
                if col in display_classes.columns:
                    display_classes[col] = display_classes[col].apply(lambda x: format_number(int(x)) if pd.notnull(x) else '0')
            
            if 'Total Clicks' in display_classes.columns:
                display_classes['Total Clicks'] = display_classes['Total Clicks'].apply(lambda x: format_number(int(x)))
            if 'Total Conversions' in display_classes.columns:
                display_classes['Total Conversions'] = display_classes['Total Conversions'].apply(lambda x: format_number(int(x)))
            
            # Styling function
            def highlight_pink_performance(df):
                """Pink-themed performance highlighting"""
                styles = pd.DataFrame('', index=df.index, columns=df.columns)
                
                if len(unique_months_cls) < 2:
                    return styles
                
                sorted_months_local = sorted(unique_months_cls, key=lambda x: pd.to_datetime(x))
                
                for i in range(1, len(sorted_months_local)):
                    current_month = month_names.get(sorted_months_local[i], sorted_months_local[i])
                    prev_month = month_names.get(sorted_months_local[i-1], sorted_months_local[i-1])
                    
                    for metric in ['CTR', 'CR']:
                        current_col = f'{current_month} {metric}'
                        prev_col = f'{prev_month} {metric}'
                        
                        if current_col in df.columns and prev_col in df.columns:
                            for idx in df.index:
                                current_val = df.loc[idx, current_col]
                                prev_val = df.loc[idx, prev_col]
                                
                                if pd.notnull(current_val) and pd.notnull(prev_val) and prev_val > 0:
                                    change_pct = ((current_val - prev_val) / prev_val) * 100
                                    if change_pct > 10:
                                        styles.loc[idx, current_col] = 'background-color: rgba(233, 30, 99, 0.3); color: #880E4F; font-weight: bold;'
                                    elif change_pct < -10:
                                        styles.loc[idx, current_col] = 'background-color: rgba(244, 67, 54, 0.3); color: #B71C1C; font-weight: bold;'
                                    elif abs(change_pct) > 5:
                                        color = 'rgba(233, 30, 99, 0.15)' if change_pct > 0 else 'rgba(244, 67, 54, 0.15)'
                                        styles.loc[idx, current_col] = f'background-color: {color};'
                
                for col in volume_columns:
                    if col in df.columns:
                        styles.loc[:, col] = 'background-color: rgba(194, 24, 91, 0.05);'
                
                return styles
            
            styled_classes = display_classes.style.apply(highlight_pink_performance, axis=None)
            
            styled_classes = styled_classes.set_properties(**{
                'text-align': 'center',
                'vertical-align': 'middle',
                'font-size': '11px',
                'padding': '4px',
                'line-height': '1.1'
            }).set_table_styles([
                {'selector': 'th', 'props': [('background-color', '#FCE4EC'), ('color', '#880E4F'), ('font-weight', 'bold'), ('text-align', 'center'), ('padding', '6px'), ('border', '1px solid #ddd'), ('font-size', '10px')]},
                {'selector': 'td', 'props': [('text-align', 'center'), ('padding', '4px'), ('border', '1px solid #ddd')]},
                {'selector': 'tbody tr:nth-child(even)', 'props': [('background-color', '#FFF0F5')]}
            ])
            
            format_dict = {'Share %': '{:.1f}%', 'Overall CTR': '{:.1f}%', 'Overall CR': '{:.1f}%'}
            for col in ctr_columns + cr_columns:
                if col in display_classes.columns:
                    format_dict[col] = '{:.1f}%'
            
            styled_classes = styled_classes.format(format_dict)
            
            # Display table
            html_content = styled_classes.to_html(index=False, escape=False)
            st.markdown(
                f'<div style="height: 600px; overflow-y: auto; overflow-x: auto; border: 1px solid #ddd; border-radius: 5px;">{html_content}</div>',
                unsafe_allow_html=True
            )
            
            # Legend
            st.markdown("""
            <div style="background: rgba(194, 24, 91, 0.1); padding: 12px; border-radius: 8px; margin: 15px 0;">
                <h4 style="margin: 0 0 8px 0; color: #880E4F;">💄 Performance Guide:</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">
                    <div>📈 <strong style="background-color: rgba(233, 30, 99, 0.3); padding: 2px 6px; border-radius: 4px; color: #880E4F;">Dark Pink</strong> = >10% improvement</div>
                    <div>📈 <strong style="background-color: rgba(233, 30, 99, 0.15); padding: 2px 6px; border-radius: 4px;">Light Pink</strong> = 5-10% improvement</div>
                    <div>📉 <strong style="background-color: rgba(244, 67, 54, 0.3); padding: 2px 6px; border-radius: 4px; color: #B71C1C;">Dark Red</strong> = >10% decline</div>
                    <div>📉 <strong style="background-color: rgba(244, 67, 54, 0.15); padding: 2px 6px; border-radius: 4px;">Light Red</strong> = 5-10% decline</div>
                    <div>💖 <strong style="background-color: rgba(194, 24, 91, 0.05); padding: 2px 6px; border-radius: 4px;">Pink Tint</strong> = Volume columns</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Download
            csv_classes = top_classes_monthly.to_csv(index=False)
            col_download = st.columns([1, 2, 1])
            with col_download[1]:
                filter_suffix = "_filtered" if st.session_state.get('filters_applied', False) else "_all"
                st.download_button(
                    label="📥 Download Classes CSV",
                    data=csv_classes,
                    file_name=f"top_{num_classes}_classes{filter_suffix}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="class_monthly_download"
                )
    
    with col_right:
        # 🥧 PIE CHART: Market share
        st.subheader("💄 Class Market Share")
        
        top_classes_pie = cls.nlargest(10, 'Counts')
        pink_colors = ['#C2185B', '#E91E63', '#F06292', '#F48FB1', '#F8BBD0', 
                       '#FCE4EC', '#AD1457', '#D81B60', '#EC407A', '#FF4081']
        
        fig_pie = px.pie(
            top_classes_pie, 
            names='class', 
            values='Counts',
            title='<b style="color:#C2185B;">💖 Market Distribution</b>',
            color_discrete_sequence=pink_colors
        )
        
        fig_pie.update_layout(
            font=dict(color='#880E4F', family='Segoe UI'),
            paper_bgcolor='rgba(252, 228, 236, 0.1)'
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # 📊 PERFORMANCE DISTRIBUTION
        st.subheader("✨ Performance Distribution")
        
        cls['performance_category'] = pd.cut(
            cls['ctr'], 
            bins=[0, 2, 5, 10, float('inf')], 
            labels=['Emerging (0-2%)', 'Growing (2-5%)', 'Strong (5-10%)', 'Premium (>10%)']
        )
        
        class_perf_counts = cls['performance_category'].value_counts().reset_index()
        class_perf_counts.columns = ['Performance Level', 'Count']
        
        fig_class_perf = px.bar(
            class_perf_counts, 
            x='Performance Level', 
            y='Count',
            title='<b style="color:#C2185B;">💄 CTR Performance Levels</b>',
            color='Count',
            color_continuous_scale=['#FCE4EC', '#C2185B'],
            text='Count'
        )
        
        fig_class_perf.update_traces(texttemplate='%{text}', textposition='outside')
        fig_class_perf.update_layout(
            plot_bgcolor='rgba(252, 228, 236, 0.3)',
            paper_bgcolor='rgba(252, 228, 236, 0.1)',
            font=dict(color='#880E4F', family='Segoe UI'),
            xaxis=dict(showgrid=True, gridcolor='#F8BBD0'),
            yaxis=dict(showgrid=True, gridcolor='#F8BBD0')
        )
        
        st.plotly_chart(fig_class_perf, use_container_width=True)
        
        # 📈 TREND ANALYSIS (Simplified)
        if 'Date' in queries.columns:
            st.subheader("📈 Class Trends")
            
            top_5_classes = cls.nlargest(5, 'Counts')['class'].tolist()
            trend_data = queries[
                (queries[class_column].isin(top_5_classes)) &
                (queries[class_column].notna())
            ].copy()
            
            if not trend_data.empty:
                trend_data['Date'] = pd.to_datetime(trend_data['Date'], errors='coerce')
                trend_data = trend_data.dropna(subset=['Date'])
                
                if not trend_data.empty:
                    if 'month' not in trend_data.columns:
                        trend_data['month'] = trend_data['Date'].dt.strftime('%Y-%m')
                    
                    monthly_trends_list = []
                    for class_name in top_5_classes:
                        class_data = trend_data[trend_data[class_column] == class_name]
                        if class_data.empty:
                            continue
                        
                        for month in sorted(class_data['month'].unique()):
                            month_data = class_data[class_data['month'] == month]
                            if not month_data.empty:
                                m_counts = int(month_data['Counts'].sum())
                                m_clicks = int(month_data['clicks'].sum())
                                m_conversions = int(month_data['conversions'].sum())
                                m_ctr = (m_clicks / m_counts * 100) if m_counts > 0 else 0
                                m_cr = (m_conversions / m_counts * 100) if m_counts > 0 else 0
                                
                                monthly_trends_list.append({
                                    'month': month, 'class': class_name,
                                    'Counts': m_counts, 'clicks': m_clicks,
                                    'conversions': m_conversions,
                                    'CTR': round(m_ctr, 2), 'CR': round(m_cr, 2)
                                })
                    
                    monthly_trends = pd.DataFrame(monthly_trends_list)
                    
                    if not monthly_trends.empty:
                        monthly_trends['Date'] = pd.to_datetime(monthly_trends['month'] + '-01')
                        monthly_trends = monthly_trends.sort_values(['Date', 'class'])
                        
                        # Metric selector
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            show_volume = st.checkbox("📊 Volume", value=True, key="vol_trend_class")
                        with col2:
                            show_ctr = st.checkbox("📈 CTR", value=False, key="ctr_trend_class")
                        with col3:
                            show_cr = st.checkbox("✨ CR", value=False, key="cr_trend_class")
                        
                        charts = []
                        if show_volume:
                            charts.append(('Volume', 'Counts', '💖 Monthly Search Volume'))
                        if show_ctr:
                            charts.append(('CTR (%)', 'CTR', '📈 Monthly CTR Trend'))
                        if show_cr:
                            charts.append(('CR (%)', 'CR', '✨ Monthly CR Trend'))
                        
                        if charts:
                            for metric_name, y_col, title in charts:
                                fig_trend = px.line(
                                    monthly_trends, x='Date', y=y_col, color='class',
                                    title=f'<b style="color:#C2185B;">{title}</b>',
                                    color_discrete_sequence=['#C2185B', '#E91E63', '#F06292', '#F48FB1', '#F8BBD0'],
                                    markers=True, line_shape='spline'
                                )
                                
                                fig_trend.update_layout(
                                    plot_bgcolor='rgba(252, 228, 236, 0.3)',
                                    paper_bgcolor='rgba(252, 228, 236, 0.1)',
                                    font=dict(color='#880E4F', family='Segoe UI', size=12),
                                    height=400,
                                    xaxis=dict(showgrid=True, gridcolor='#F8BBD0', title='<b>Month</b>'),
                                    yaxis=dict(showgrid=True, gridcolor='#F8BBD0', title=f'<b>{metric_name}</b>'),
                                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                                )
                                
                                st.plotly_chart(fig_trend, use_container_width=True)
                        else:
                            st.warning("Select at least one metric")
    
    st.markdown("---")
    
    # 🔥 CLASS-KEYWORD MATRIX (Simplified)
    st.subheader("🔥 Class-Keyword Intelligence")
    
    if 'search' in queries.columns:
        available_classes = sorted(class_queries[class_column].unique())
        class_options = ['All Classes'] + list(available_classes)
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #FCE4EC 0%, #F8BBD0 100%); border: 2px solid #E91E63; border-radius: 15px; padding: 1.5rem; margin: 1rem 0; box-shadow: 0 4px 15px rgba(233, 30, 99, 0.2);">
            <h4 style="color: #880E4F; margin: 0 0 1rem 0; text-align: center;">💄 Class Analysis Control Center</h4>
        </div>
        """, unsafe_allow_html=True)
        
        col_select, col_metrics = st.columns([2, 3])
        
        with col_select:
            selected_class = st.selectbox(
                "💄 Select Class:",
                options=class_options,
                index=0,
                key="class_selector"
            )
        
        with col_metrics:
            if selected_class != 'All Classes':
                class_metrics = cls[cls['class'] == selected_class].iloc[0] if not cls[cls['class'] == selected_class].empty else None
                
                if class_metrics is not None:
                    m1, m2, m3, m4, m5 = st.columns(5)
                    
                    with m1:
                        st.markdown(f'<div class="class-metric-card"><div class="class-metric-value">{format_number(class_metrics["Counts"])}</div><div class="class-metric-label">📊 Searches</div></div>', unsafe_allow_html=True)
                    with m2:
                        st.markdown(f'<div class="class-metric-card"><div class="class-metric-value">{class_metrics["ctr"]:.1f}%</div><div class="class-metric-label">📈 CTR</div></div>', unsafe_allow_html=True)
                    with m3:
                        st.markdown(f'<div class="class-metric-card"><div class="class-metric-value">{class_metrics["cr"]:.1f}%</div><div class="class-metric-label">✨ CR</div></div>', unsafe_allow_html=True)
                    with m4:
                        st.markdown(f'<div class="class-metric-card"><div class="class-metric-value">{class_metrics["classic_cr"]:.1f}%</div><div class="class-metric-label">🔄 Classic CR</div></div>', unsafe_allow_html=True)
                    with m5:
                        st.markdown(f'<div class="class-metric-card"><div class="class-metric-value">{class_metrics["share_pct"]:.1f}%</div><div class="class-metric-label">💖 Share</div></div>', unsafe_allow_html=True)
            else:
                total_searches = int(class_queries['Counts'].sum())
                total_clicks = int(class_queries['clicks'].sum())
                total_conversions = int(class_queries['conversions'].sum())
                overall_ctr = (total_clicks / total_searches * 100) if total_searches > 0 else 0
                overall_cr = (total_conversions / total_searches * 100) if total_searches > 0 else 0
                overall_classic_cr = (total_conversions / total_clicks * 100) if total_clicks > 0 else 0
                
                m1, m2, m3, m4, m5 = st.columns(5)
                with m1:
                    st.markdown(f'<div class="class-metric-card"><div class="class-metric-value">{format_number(total_searches)}</div><div class="class-metric-label">📊 Total</div></div>', unsafe_allow_html=True)
                with m2:
                    st.markdown(f'<div class="class-metric-card"><div class="class-metric-value">{overall_ctr:.1f}%</div><div class="class-metric-label">📈 CTR</div></div>', unsafe_allow_html=True)
                with m3:
                    st.markdown(f'<div class="class-metric-card"><div class="class-metric-value">{overall_cr:.1f}%</div><div class="class-metric-label">✨ CR</div></div>', unsafe_allow_html=True)
                with m4:
                    st.markdown(f'<div class="class-metric-card"><div class="class-metric-value">{overall_classic_cr:.1f}%</div><div class="class-metric-label">🔄 Classic CR</div></div>', unsafe_allow_html=True)
                with m5:
                    st.markdown(f'<div class="class-metric-card"><div class="class-metric-value">{format_number(total_clicks)}</div><div class="class-metric-label">💖 Clicks</div></div>', unsafe_allow_html=True)
        
        # Filter data
        if selected_class == 'All Classes':
            top_classes_matrix = cls.nlargest(8, 'Counts')['class'].tolist()
            filtered_data = class_queries[class_queries[class_column].isin(top_classes_matrix)]
            matrix_title = "Top Classes vs Search Terms"
        else:
            filtered_data = class_queries[class_queries[class_column] == selected_class]
            matrix_title = f"{selected_class} - Search Analysis"
        
        matrix_data = filtered_data[
            (filtered_data[class_column].notna()) & 
            (filtered_data['search'].notna()) &
            (~filtered_data['search'].str.lower().isin(['other', 'others']))
        ].copy()
        
        if not matrix_data.empty:
            if selected_class == 'All Classes':
                # Heatmap for all classes
                class_search_matrix = matrix_data.groupby([class_column, 'search']).agg({
                    'Counts': 'sum', 'clicks': 'sum', 'conversions': 'sum'
                }).reset_index()
                class_search_matrix = class_search_matrix.rename(columns={class_column: 'class'})
                
                class_search_matrix['ctr'] = (class_search_matrix['clicks'] / class_search_matrix['Counts'] * 100).round(2)
                class_search_matrix['cr'] = (class_search_matrix['conversions'] / class_search_matrix['Counts'] * 100).round(2)
                
                top_searches = matrix_data['search'].value_counts().head(12).index.tolist()
                class_search_matrix = class_search_matrix[class_search_matrix['search'].isin(top_searches)]
                
                heatmap_data = class_search_matrix.pivot(index='class', columns='search', values='Counts').fillna(0)
                
                if not heatmap_data.empty:
                    fig_matrix = px.imshow(
                        heatmap_data.values,
                        labels=dict(x="Search Terms", y="Classes", color="Counts"),
                        x=heatmap_data.columns, y=heatmap_data.index,
                        color_continuous_scale=['#FCE4EC', '#F48FB1', '#C2185B'],
                        title=f'<b style="color:#C2185B;">{matrix_title}</b>',
                        aspect='auto'
                    )
                    
                    fig_matrix.update_layout(
                        plot_bgcolor='rgba(252, 228, 236, 0.3)',
                        paper_bgcolor='rgba(252, 228, 236, 0.1)',
                        font=dict(color='#880E4F', family='Segoe UI'),
                        xaxis=dict(tickangle=45), height=500
                    )
                    
                    st.plotly_chart(fig_matrix, use_container_width=True)
                    st.info(f"📊 Matrix: {len(heatmap_data.index)} classes × {len(heatmap_data.columns)} terms")
            else:
                # Bar chart for single class
                class_search_data = matrix_data.groupby('search').agg({
                    'Counts': 'sum', 'clicks': 'sum', 'conversions': 'sum'
                }).reset_index()
                
                class_search_data['ctr'] = (class_search_data['clicks'] / class_search_data['Counts'] * 100).round(2)
                class_search_data['cr'] = (class_search_data['conversions'] / class_search_data['Counts'] * 100).round(2)
                class_search_data['classic_cr'] = (class_search_data['conversions'] / class_search_data['clicks'] * 100).fillna(0).round(2)
                class_search_data = class_search_data.sort_values('Counts', ascending=False).head(15)
                
                cr_option = st.radio(
                    "Color by:",
                    options=['CR (Search-based)', 'Classic CR (Click-based)'],
                    index=0, horizontal=True, key="class_cr_radio"
                )
                
                color_col = 'classic_cr' if 'Classic' in cr_option else 'cr'
                
                fig_class_search = px.bar(
                    class_search_data, x='search', y='Counts',
                    title=f'<b style="color:#C2185B;">{matrix_title}</b>',
                    color=color_col,
                    color_continuous_scale=['#FCE4EC', '#F48FB1', '#C2185B'],
                    text='Counts'
                )
                
                fig_class_search.update_traces(
                    texttemplate='%{text}', textposition='outside',
                    text=[format_number(x) for x in class_search_data['Counts']]
                )
                
                fig_class_search.update_layout(
                    plot_bgcolor='rgba(252, 228, 236, 0.3)',
                    paper_bgcolor='rgba(252, 228, 236, 0.1)',
                    font=dict(color='#880E4F', family='Segoe UI'),
                    height=500,
                    xaxis=dict(tickangle=45, showgrid=True, gridcolor='#F8BBD0'),
                    yaxis=dict(showgrid=True, gridcolor='#F8BBD0')
                )
                
                st.plotly_chart(fig_class_search, use_container_width=True)
                
                # Comparison table
                display_comparison = class_search_data[['search', 'Counts', 'ctr', 'cr', 'classic_cr']].copy()
                display_comparison = display_comparison.rename(columns={
                    'search': 'Search Term', 'Counts': 'Volume',
                    'ctr': 'CTR (%)', 'cr': 'CR (%)', 'classic_cr': 'Classic CR (%)'
                })
                display_comparison['Volume'] = display_comparison['Volume'].apply(format_number)
                display_comparison['CTR (%)'] = display_comparison['CTR (%)'].apply(lambda x: f"{x:.1f}%")
                display_comparison['CR (%)'] = display_comparison['CR (%)'].apply(lambda x: f"{x:.1f}%")
                display_comparison['Classic CR (%)'] = display_comparison['Classic CR (%)'].apply(lambda x: f"{x:.1f}%")
                
                display_styled_table(
                    df=display_comparison,
                    title="📋 Search Terms Performance",
                    download_filename=f"class_search_{selected_class.replace(' ', '_')}.csv",
                    scrollable=True, max_height="600px", align="center"
                )
        else:
            st.warning("⚠️ No data for selected filter")
    
    st.markdown("---")
    
    # 🔑 TOP KEYWORDS ANALYSIS (Simplified)
    st.subheader("🔑 Top Keywords by Class")
    
    num_keywords = st.selectbox(
        "🔥 Keywords to analyze:",
        options=[10, 15, 20, 25, 30, 50],
        index=0, key="num_keywords_class"
    )
    
    try:
        rows = []
        for cls_name, grp in class_queries.groupby(class_column):
            keyword_counts = {}
            for idx, row in grp.iterrows():
                keywords_list = row.get('keywords', [])
                query_count = row['Counts']
                
                if isinstance(keywords_list, list):
                    for keyword in keywords_list:
                        keyword_counts[keyword] = keyword_counts.get(keyword, 0) + query_count
                elif pd.notna(row.get('normalized_query')):
                    for keyword in str(row['normalized_query']).lower().split():
                        keyword_counts[keyword] = keyword_counts.get(keyword, 0) + query_count
            
            top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:num_keywords]
            for keyword, count in top_keywords:
                rows.append({'class': cls_name, 'keyword': keyword, 'count': count})
        
        df_ckw = pd.DataFrame(rows)
        
        if not df_ckw.empty:
            display_option = st.radio(
                "Display format:",
                ["Summary Table", "Heatmap"],
                horizontal=True, key="keyword_display_class"
            )
            
            if display_option == "Heatmap":
                pivot_ckw = df_ckw.pivot_table(index='class', columns='keyword', values='count', fill_value=0)
                
                fig_kw_heatmap = px.imshow(
                    pivot_ckw.values,
                    labels=dict(x="Keywords", y="Classes", color="Count"),
                    x=pivot_ckw.columns, y=pivot_ckw.index,
                    color_continuous_scale=['#FCE4EC', '#F48FB1', '#C2185B'],
                    title=f'<b style="color:#C2185B;">💄 Keyword Heatmap (Top {num_keywords})</b>',
                    aspect='auto'
                )
                
                fig_kw_heatmap.update_layout(
                    plot_bgcolor='rgba(252, 228, 236, 0.3)',
                    paper_bgcolor='rgba(252, 228, 236, 0.1)',
                    font=dict(color='#880E4F', family='Segoe UI'),
                    xaxis=dict(tickangle=45), height=600
                )
                
                st.plotly_chart(fig_kw_heatmap, use_container_width=True)
            else:
                st.subheader(f"🔥 Top {num_keywords} Keywords")
                
                summary = []
                total_volume = cls['Counts'].sum()
                
                for cls_name in df_ckw['class'].unique():
                    cls_data = df_ckw[df_ckw['class'] == cls_name].sort_values('count', ascending=False)
                    top_n = cls_data.head(num_keywords)
                    
                    keywords_str = ' | '.join([f"{row['keyword']} ({format_number(row['count'])})" for _, row in top_n.iterrows()])
                    
                    actual_total = cls[cls['class'] == cls_name]['Counts'].iloc[0] if len(cls[cls['class'] == cls_name]) > 0 else cls_data['count'].sum()
                    share_pct = (actual_total / total_volume * 100)
                    
                    summary.append({
                        'Class': cls_name,
                        f'Top {num_keywords} Keywords': keywords_str,
                        'Total Keywords': len(cls_data),
                        'Class Volume': actual_total,
                        'Market Share %': f"{share_pct:.1f}%",
                        'Keyword Volume': cls_data['count'].sum(),
                        'Avg Count': format_number(cls_data['count'].mean()),
                        'Top Keyword': top_n.iloc[0]['keyword'] if len(top_n) > 0 else 'N/A'
                    })
                
                summary = sorted(summary, key=lambda x: x['Class Volume'], reverse=True)
                summary_df = pd.DataFrame(summary)
                summary_df['Class Volume'] = summary_df['Class Volume'].apply(format_number)
                summary_df['Keyword Volume'] = summary_df['Keyword Volume'].apply(format_number)
                
                display_styled_table(df=summary_df, align="center", scrollable=True, max_height="600px")
                
                # Insights
                st.markdown("---")
                st.subheader("📊 Keyword Intelligence")
                
                class_stats = {}
                for cls_name in df_ckw['class'].unique():
                    cls_data = df_ckw[df_ckw['class'] == cls_name]
                    actual_total = cls[cls['class'] == cls_name]['Counts'].iloc[0] if len(cls[cls['class'] == cls_name]) > 0 else cls_data['count'].sum()
                    class_stats[cls_name] = {
                        'total_keywords': len(cls_data),
                        'total_count': actual_total,
                        'share_pct': (actual_total / total_volume * 100)
                    }
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    most_diverse = max(class_stats.items(), key=lambda x: x[1]['total_keywords'])
                    name = most_diverse[0][:15] + "..." if len(most_diverse[0]) > 15 else most_diverse[0]
                    st.markdown(f"""
                    <div class='enhanced-pink-class-metric'>
                        <span class='icon'>🌟</span>
                        <div class='value'>{name}</div>
                        <div class='label'>Most Diverse</div>
                        <div class='sub-label'>{most_diverse[1]['total_keywords']} keywords</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    highest_vol = max(class_stats.items(), key=lambda x: x[1]['total_count'])
                    name = highest_vol[0][:15] + "..." if len(highest_vol[0]) > 15 else highest_vol[0]
                    st.markdown(f"""
                    <div class='enhanced-pink-class-metric'>
                        <span class='icon'>🚀</span>
                        <div class='value'>{name}</div>
                        <div class='label'>Highest Volume</div>
                        <div class='sub-label'>{format_number(highest_vol[1]['total_count'])} searches<br>{highest_vol[1]['share_pct']:.1f}% share</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    most_concentrated = max(class_stats.items(), key=lambda x: x[1]['share_pct'])
                    name = most_concentrated[0][:15] + "..." if len(most_concentrated[0]) > 15 else most_concentrated[0]
                    st.markdown(f"""
                    <div class='enhanced-pink-class-metric'>
                        <span class='icon'>🎯</span>
                        <div class='value'>{name}</div>
                        <div class='label'>Most Concentrated</div>
                        <div class='sub-label'>{most_concentrated[1]['share_pct']:.1f}% share</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Download
            csv_keywords = df_ckw.to_csv(index=False)
            st.download_button(
                label="📥 Download Keywords CSV",
                data=csv_keywords,
                file_name=f"class_keywords_top_{num_keywords}.csv",
                mime="text/csv",
                key="class_keywords_download"
            )
        else:
            st.info("Not enough keyword data")
    
    except Exception as e:
        st.error(f"Error processing keywords: {str(e)}")
        st.info("Not enough keyword data per class")

    

# ----------------- Generic Type Tab (PINK THEME + OPTIMIZED) -----------------
with tab_generic:
    # 🎨 PINK-THEMED HERO HEADER
    st.markdown("""
    <div style="
        text-align: center; 
        padding: 3rem 2rem; 
        background: linear-gradient(135deg, #FCE4EC 0%, #F8BBD0 50%, #F48FB1 100%); 
        border-radius: 20px; 
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(136, 14, 79, 0.15);
        border: 1px solid rgba(233, 30, 99, 0.2);
    ">
        <h1 style="
            color: #880E4F; 
            margin: 0; 
            font-size: 3rem; 
            text-shadow: 2px 2px 8px rgba(136, 14, 79, 0.2);
            font-weight: 700;
            letter-spacing: -1px;
        ">
            💄 Generic Type Performance Analysis ✨
        </h1>
        <p style="
            color: #C2185B; 
            margin: 1rem 0 0 0; 
            font-size: 1.3rem;
            font-weight: 300;
            opacity: 0.9;
        ">
            Deep dive into generic type performance and search trends
        </p>
    </div>
    """, unsafe_allow_html=True)

    try:
        # ✅ DATA VALIDATION
        if generic_type is None or generic_type.empty:
            st.warning("⚠️ No generic type data available.")
            st.info("Please ensure your uploaded file contains a 'generic_type' sheet with data.")
            st.stop()
        
        # 🚀 OPTIMIZED DATA PROCESSING
        @st.cache_data(ttl=1800, show_spinner=False)
        def process_generic_data(data):
            """Optimized data processing with vectorization"""
            gt = data.copy()
            
            required_columns = ['search', 'count', 'Clicks', 'Conversions']
            missing_columns = [col for col in required_columns if col not in gt.columns]
            
            if missing_columns:
                return None, f"Missing required columns: {', '.join(missing_columns)}"
            
            # Vectorized numeric conversion
            numeric_columns = ['count', 'Clicks', 'Conversions']
            for col in numeric_columns:
                gt[col] = pd.to_numeric(gt[col], errors='coerce').fillna(0)
            
            # Efficient cleaning
            gt = gt.dropna(subset=['search'])
            gt = gt[gt['search'].str.strip().astype(bool)]
            
            if gt.empty:
                return None, "No valid data after cleaning"
            
            # Optimized aggregation
            gt_agg = gt.groupby('search', as_index=False).agg({
                'count': 'sum',
                'Clicks': 'sum', 
                'Conversions': 'sum'
            })
            
            # Vectorized calculations
            total_clicks = gt_agg['Clicks'].sum()
            total_conversions = gt_agg['Conversions'].sum()
            
            gt_agg['ctr'] = np.where(gt_agg['count'] > 0, (gt_agg['Clicks'] / gt_agg['count']) * 100, 0)
            gt_agg['classic_cvr'] = np.where(gt_agg['Clicks'] > 0, (gt_agg['Conversions'] / gt_agg['Clicks']) * 100, 0)
            gt_agg['conversion_rate'] = np.where(gt_agg['count'] > 0, (gt_agg['Conversions'] / gt_agg['count']) * 100, 0)
            gt_agg['click_share'] = np.where(total_clicks > 0, (gt_agg['Clicks'] / total_clicks) * 100, 0)
            gt_agg['conversion_share'] = np.where(total_conversions > 0, (gt_agg['Conversions'] / total_conversions) * 100, 0)
            
            gt_agg = gt_agg.sort_values('count', ascending=False).reset_index(drop=True)
            
            return gt_agg, None
        
        with st.spinner("🔄 Processing generic type data..."):
            gt_agg, error = process_generic_data(generic_type)
            
            if error:
                st.error(f"❌ {error}")
                st.stop()
        
        # 📊 PRE-CALCULATE METRICS
        @st.cache_data(ttl=1800, show_spinner=False)
        def calculate_summary_metrics(data):
            """Pre-calculate all summary metrics"""
            total_generic_terms = len(data)
            total_searches = data['count'].sum()
            total_clicks = data['Clicks'].sum()
            total_conversions = data['Conversions'].sum()
            avg_ctr = data['ctr'].mean()
            avg_cr = data['conversion_rate'].mean()
            
            sorted_counts = np.sort(data['count'].values)
            cumsum_counts = np.cumsum(sorted_counts)
            total_count = cumsum_counts[-1]
            n = len(sorted_counts)
            
            gini_coefficient = 1 - 2 * np.sum(cumsum_counts) / (n * total_count) if total_count > 0 else 0
            herfindahl_index = np.sum((data['count'] / total_count) ** 2) if total_count > 0 else 0
            top_5_concentration = data.head(5)['count'].sum() / total_count * 100 if total_count > 0 else 0
            top_10_concentration = data.head(10)['count'].sum() / total_count * 100 if total_count > 0 else 0
            
            return {
                'total_generic_terms': total_generic_terms,
                'total_searches': total_searches,
                'total_clicks': total_clicks,
                'total_conversions': total_conversions,
                'avg_ctr': avg_ctr,
                'avg_cr': avg_cr,
                'gini_coefficient': gini_coefficient,
                'herfindahl_index': herfindahl_index,
                'top_5_concentration': top_5_concentration,
                'top_10_concentration': top_10_concentration,
                'top_generic_term': data.iloc[0]['search'] if len(data) > 0 else 'N/A',
                'top_generic_volume': data.iloc[0]['count'] if len(data) > 0 else 0,
                'top_conversion_generic': data.nlargest(1, 'Conversions')['search'].iloc[0] if len(data) > 0 else 'N/A'
            }
        
        metrics = calculate_summary_metrics(gt_agg)
        
        # 🎨 PINK-THEMED CSS (Load once)
        if 'generic_pink_css_loaded' not in st.session_state:
            st.markdown("""
            <style>
            .pink-generic-metric-card {
                background: linear-gradient(135deg, #FCE4EC 0%, #F8BBD0 100%);
                padding: 20px; border-radius: 12px; text-align: center; color: #880E4F;
                box-shadow: 0 6px 20px rgba(194, 24, 91, 0.25); margin: 8px 0;
                min-height: 140px; display: flex; flex-direction: column; justify-content: center;
                transition: transform 0.15s ease; border-left: 3px solid #E91E63;
            }
            .pink-generic-metric-card:hover { transform: translateY(-1px); }
            .pink-generic-metric-card .icon { font-size: 2.5em; margin-bottom: 8px; color: #C2185B; }
            .pink-generic-metric-card .value { font-size: 1.4em; font-weight: bold; margin-bottom: 6px; color: #880E4F; }
            .pink-generic-metric-card .label { font-size: 1em; opacity: 0.9; font-weight: 600; margin-bottom: 4px; color: #C2185B; }
            .pink-generic-metric-card .sub-label { font-size: 0.9em; opacity: 0.8; color: #AD1457; }
            .pink-performance-badge { padding: 3px 6px; border-radius: 8px; font-size: 0.75em; font-weight: bold; margin-left: 6px; }
            .high-pink-performance { background-color: #E91E63; color: white; }
            .medium-pink-performance { background-color: #F48FB1; color: white; }
            .low-pink-performance { background-color: #F8BBD0; color: #880E4F; }
            .pink-insight-card { background: linear-gradient(135deg, #C2185B 0%, #F06292 100%); padding: 20px; border-radius: 12px; color: white; margin: 12px 0; box-shadow: 0 4px 16px rgba(194, 24, 91, 0.25); }
            </style>
            """, unsafe_allow_html=True)
            st.session_state.generic_pink_css_loaded = True
        
        # 📊 KEY METRICS SECTION
        st.subheader("💖 Generic Type Performance Overview")
        
        # First row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class='pink-generic-metric-card'>
                <span class='icon'>💄</span>
                <div class='value'>{format_number(metrics['total_generic_terms'])}</div>
                <div class='label'>Total Generic Terms</div>
                <div class='sub-label'>Active terms</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='pink-generic-metric-card'>
                <span class='icon'>🔍</span>
                <div class='value'>{format_number(metrics['total_searches'])}</div>
                <div class='label'>Total Searches</div>
                <div class='sub-label'>Across all terms</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            performance_class = "high-pink-performance" if metrics['avg_ctr'] > 5 else "medium-pink-performance" if metrics['avg_ctr'] > 2 else "low-pink-performance"
            performance_text = "High" if metrics['avg_ctr'] > 5 else "Medium" if metrics['avg_ctr'] > 2 else "Low"
            st.markdown(f"""
            <div class='pink-generic-metric-card'>
                <span class='icon'>📈</span>
                <div class='value'>{metrics['avg_ctr']:.1f}% <span class='pink-performance-badge {performance_class}'>{performance_text}</span></div>
                <div class='label'>Average CTR</div>
                <div class='sub-label'>Click-through rate</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            top_generic_display = metrics['top_generic_term'][:12] + "..." if len(metrics['top_generic_term']) > 12 else metrics['top_generic_term']
            market_share = (metrics['top_generic_volume'] / metrics['total_searches'] * 100) if metrics['total_searches'] > 0 else 0
            st.markdown(f"""
            <div class='pink-generic-metric-card'>
                <span class='icon'>👑</span>
                <div class='value'>{top_generic_display}</div>
                <div class='label'>Top Generic Term</div>
                <div class='sub-label'>{market_share:.1f}% market share</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Second row
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            st.markdown(f"""
            <div class='pink-generic-metric-card'>
                <span class='icon'>✨</span>
                <div class='value'>{metrics['avg_cr']:.1f}%</div>
                <div class='label'>Avg Conversion Rate</div>
                <div class='sub-label'>Overall performance</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col6:
            st.markdown(f"""
            <div class='pink-generic-metric-card'>
                <span class='icon'>🖱️</span>
                <div class='value'>{format_number(metrics['total_clicks'])}</div>
                <div class='label'>Total Clicks</div>
                <div class='sub-label'>Across all terms</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col7:
            st.markdown(f"""
            <div class='pink-generic-metric-card'>
                <span class='icon'>✅</span>
                <div class='value'>{format_number(metrics['total_conversions'])}</div>
                <div class='label'>Total Conversions</div>
                <div class='sub-label'>Successful outcomes</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col8:
            top_conversion_display = metrics['top_conversion_generic'][:12] + "..." if len(metrics['top_conversion_generic']) > 12 else metrics['top_conversion_generic']
            st.markdown(f"""
            <div class='pink-generic-metric-card'>
                <span class='icon'>🏆</span>
                <div class='value'>{top_conversion_display}</div>
                <div class='label'>Conversion Leader</div>
                <div class='sub-label'>Most conversions</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 🏆 TOP GENERIC TERMS TABLE
        st.subheader("🏆 Generic Terms Performance")
        
        num_generic_terms = st.slider(
            "Number of generic terms to display:", 
            min_value=10, 
            max_value=50, 
            value=20, 
            step=5,
            key="generic_terms_count_slider"
        )
        
        # Reuse month names
        if 'month_names' not in locals():
            month_names = get_dynamic_month_names(queries)
        
        # Ensure month column
        generic_type_with_month = generic_type.copy()
        if 'month' not in generic_type_with_month.columns and 'start_date' in generic_type_with_month.columns:
            generic_type_with_month['start_date'] = pd.to_datetime(generic_type_with_month['start_date'])
            generic_type_with_month['month'] = generic_type_with_month['start_date'].dt.to_period('M').astype(str)
        
        # 🚀 COMPUTE MONTHLY PERFORMANCE
        @st.cache_data(ttl=1800, show_spinner=False)
        def compute_generic_monthly(_df, _gt, month_dict, num_terms):
            """Build monthly generic terms table"""
            if _df.empty or _gt.empty:
                return pd.DataFrame(), []
            
            top_generics_list = _gt.nlargest(num_terms, 'count')['search'].tolist()
            top_data = _df[_df['search'].isin(top_generics_list)].copy()
            
            unique_months = sorted(top_data['month'].dropna().unique(), key=lambda x: pd.to_datetime(x)) if 'month' in top_data.columns else []
            
            result_data = []
            for generic_term in top_generics_list:
                generic_data = top_data[top_data['search'] == generic_term]
                if generic_data.empty:
                    continue
                
                total_counts = int(generic_data['count'].sum())
                if total_counts == 0:
                    continue
                
                total_clicks = int(generic_data['Clicks'].sum())
                total_conversions = int(generic_data['Conversions'].sum())
                dataset_total = _df['count'].sum()
                share_pct = (total_counts / dataset_total * 100) if dataset_total > 0 else 0
                overall_ctr = (total_clicks / total_counts * 100) if total_counts > 0 else 0
                overall_cr = (total_conversions / total_counts * 100) if total_counts > 0 else 0
                
                row = {
                    'Generic Term': generic_term,
                    'Total Volume': total_counts,
                    'Share %': share_pct,
                    'Overall CTR': overall_ctr,
                    'Overall CR': overall_cr,
                    'Total Clicks': total_clicks,
                    'Total Conversions': total_conversions
                }
                
                for month in unique_months:
                    month_display = month_dict.get(month, month)
                    month_data = generic_data[generic_data['month'] == month]
                    
                    if not month_data.empty:
                        m_counts = int(month_data['count'].sum())
                        m_clicks = int(month_data['Clicks'].sum())
                        m_conversions = int(month_data['Conversions'].sum())
                        m_ctr = (m_clicks / m_counts * 100) if m_counts > 0 else 0
                        m_cr = (m_conversions / m_counts * 100) if m_counts > 0 else 0
                        
                        row[f'{month_display} Vol'] = m_counts
                        row[f'{month_display} CTR'] = m_ctr
                        row[f'{month_display} CR'] = m_cr
                    else:
                        row[f'{month_display} Vol'] = 0
                        row[f'{month_display} CTR'] = 0
                        row[f'{month_display} CR'] = 0
                
                result_data.append(row)
            
            result_df = pd.DataFrame(result_data)
            result_df = result_df.sort_values('Total Volume', ascending=False).reset_index(drop=True)
            return result_df[result_df['Total Volume'] > 0], unique_months
        
        top_generics_monthly, unique_months_gen = compute_generic_monthly(
            generic_type_with_month, gt_agg, month_names, num_generic_terms
        )
        
        if not top_generics_monthly.empty:
            unique_generic_terms_count = generic_type_with_month['search'].nunique()
            
            if st.session_state.get('filters_applied', False):
                st.info(f"🔍 **Filtered**: Top {num_generic_terms} from {unique_generic_terms_count:,} terms")
            else:
                st.info(f"📊 **All Data**: Top {num_generic_terms} from {unique_generic_terms_count:,} terms")
            
            # Column organization
            base_columns = ['Generic Term', 'Total Volume', 'Share %', 'Overall CTR', 'Overall CR', 'Total Clicks', 'Total Conversions']
            sorted_months = sorted(unique_months_gen, key=lambda x: pd.to_datetime(x))
            
            volume_columns = [f'{month_names.get(m, m)} Vol' for m in sorted_months]
            ctr_columns = [f'{month_names.get(m, m)} CTR' for m in sorted_months]
            cr_columns = [f'{month_names.get(m, m)} CR' for m in sorted_months]
            
            ordered_columns = base_columns + volume_columns + ctr_columns + cr_columns
            existing_columns = [col for col in ordered_columns if col in top_generics_monthly.columns]
            top_generics_monthly = top_generics_monthly[existing_columns]
            
            # Format display
            display_generics = top_generics_monthly.copy()
            
            for col in ['Total Volume'] + volume_columns:
                if col in display_generics.columns:
                    display_generics[col] = display_generics[col].apply(lambda x: format_number(int(x)) if pd.notnull(x) else '0')
            
            if 'Total Clicks' in display_generics.columns:
                display_generics['Total Clicks'] = display_generics['Total Clicks'].apply(lambda x: format_number(int(x)))
            if 'Total Conversions' in display_generics.columns:
                display_generics['Total Conversions'] = display_generics['Total Conversions'].apply(lambda x: format_number(int(x)))
            
            # Pink-themed styling
            def highlight_pink_generic_performance(df):
                """Pink-themed performance highlighting"""
                styles = pd.DataFrame('', index=df.index, columns=df.columns)
                
                if len(unique_months_gen) < 2:
                    return styles
                
                sorted_months_local = sorted(unique_months_gen, key=lambda x: pd.to_datetime(x))
                
                for i in range(1, len(sorted_months_local)):
                    current_month = month_names.get(sorted_months_local[i], sorted_months_local[i])
                    prev_month = month_names.get(sorted_months_local[i-1], sorted_months_local[i-1])
                    
                    for metric in ['CTR', 'CR']:
                        current_col = f'{current_month} {metric}'
                        prev_col = f'{prev_month} {metric}'
                        
                        if current_col in df.columns and prev_col in df.columns:
                            for idx in df.index:
                                current_val = df.loc[idx, current_col]
                                prev_val = df.loc[idx, prev_col]
                                
                                if pd.notnull(current_val) and pd.notnull(prev_val) and prev_val > 0:
                                    change_pct = ((current_val - prev_val) / prev_val) * 100
                                    if change_pct > 10:
                                        styles.loc[idx, current_col] = 'background-color: rgba(233, 30, 99, 0.3); color: #880E4F; font-weight: bold;'
                                    elif change_pct < -10:
                                        styles.loc[idx, current_col] = 'background-color: rgba(244, 67, 54, 0.3); color: #B71C1C; font-weight: bold;'
                                    elif abs(change_pct) > 5:
                                        color = 'rgba(233, 30, 99, 0.15)' if change_pct > 0 else 'rgba(244, 67, 54, 0.15)'
                                        styles.loc[idx, current_col] = f'background-color: {color};'
                
                for col in volume_columns:
                    if col in df.columns:
                        styles.loc[:, col] = 'background-color: rgba(194, 24, 91, 0.05);'
                
                return styles
            
            styled_generics = display_generics.style.apply(highlight_pink_generic_performance, axis=None)
            
            styled_generics = styled_generics.set_properties(**{
                'text-align': 'center',
                'vertical-align': 'middle',
                'font-size': '11px',
                'padding': '4px',
                'line-height': '1.1'
            }).set_table_styles([
                {'selector': 'th', 'props': [('background-color', '#FCE4EC'), ('color', '#880E4F'), ('font-weight', 'bold'), ('text-align', 'center'), ('padding', '6px'), ('border', '1px solid #ddd'), ('font-size', '10px')]},
                {'selector': 'td', 'props': [('text-align', 'center'), ('padding', '4px'), ('border', '1px solid #ddd')]},
                {'selector': 'tbody tr:nth-child(even)', 'props': [('background-color', '#FFF0F5')]}
            ])
            
            format_dict = {'Share %': '{:.1f}%', 'Overall CTR': '{:.1f}%', 'Overall CR': '{:.1f}%'}
            for col in ctr_columns + cr_columns:
                if col in display_generics.columns:
                    format_dict[col] = '{:.1f}%'
            
            styled_generics = styled_generics.format(format_dict)
            
            # Display table
            html_content = styled_generics.to_html(index=False, escape=False)
            st.markdown(
                f'<div style="height: 600px; overflow-y: auto; overflow-x: auto; border: 1px solid #ddd; border-radius: 5px;">{html_content}</div>',
                unsafe_allow_html=True
            )
            
            # Legend
            st.markdown("""
            <div style="background: rgba(194, 24, 91, 0.1); padding: 12px; border-radius: 8px; margin: 15px 0;">
                <h4 style="margin: 0 0 8px 0; color: #880E4F;">💄 Performance Guide:</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">
                    <div>📈 <strong style="background-color: rgba(233, 30, 99, 0.3); padding: 2px 6px; border-radius: 4px; color: #880E4F;">Dark Pink</strong> = >10% improvement</div>
                    <div>📈 <strong style="background-color: rgba(233, 30, 99, 0.15); padding: 2px 6px; border-radius: 4px;">Light Pink</strong> = 5-10% improvement</div>
                    <div>📉 <strong style="background-color: rgba(244, 67, 54, 0.3); padding: 2px 6px; border-radius: 4px; color: #B71C1C;">Dark Red</strong> = >10% decline</div>
                    <div>📉 <strong style="background-color: rgba(244, 67, 54, 0.15); padding: 2px 6px; border-radius: 4px;">Light Red</strong> = 5-10% decline</div>
                    <div>💖 <strong style="background-color: rgba(194, 24, 91, 0.05); padding: 2px 6px; border-radius: 4px;">Pink Tint</strong> = Volume columns</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Download
            csv_generics = top_generics_monthly.to_csv(index=False)
            col_download = st.columns([1, 2, 1])
            with col_download[1]:
                filter_suffix = "_filtered" if st.session_state.get('filters_applied', False) else "_all"
                st.download_button(
                    label="📥 Download Generic Terms CSV",
                    data=csv_generics,
                    file_name=f"top_{num_generic_terms}_generic_terms{filter_suffix}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="generic_terms_monthly_download"
                )
        
        st.markdown("---")
        
        # 🎯 INTERACTIVE ANALYSIS
        st.subheader("🎯 Interactive Generic Type Analysis")
        
        analysis_type = st.radio(
            "Choose Analysis Type:",
            ["📊 Top Performers", "🔍 Term Deep Dive", "📈 Performance Comparison", "📊 Distribution Analysis"],
            horizontal=True,
            key="generic_analysis_type"
        )
        
        if analysis_type == "📊 Top Performers":
            with st.spinner('📊 Generating overview...'):
                st.subheader("🏆 Top 20 Generic Terms")
                
                display_count = min(20, len(gt_agg))
                top_20_gt = gt_agg.head(display_count).copy()
                
                # Combined chart
                st.subheader("🚀 Volume vs Performance Matrix")
                
                top_20_gt['conversion_rate_volume'] = (top_20_gt['Conversions'] / top_20_gt['count'] * 100).round(2)
                
                fig_combined = make_subplots(specs=[[{"secondary_y": True}]])
                
                fig_combined.add_trace(
                    go.Bar(
                        name='Search Volume',
                        x=top_20_gt['search'],
                        y=top_20_gt['count'],
                        marker_color='rgba(194, 24, 91, 0.7)',
                        text=[format_number(int(x)) for x in top_20_gt['count']],
                        textposition='outside'
                    ),
                    secondary_y=False
                )
                
                fig_combined.add_trace(
                    go.Scatter(
                        name='CTR %',
                        x=top_20_gt['search'],
                        y=top_20_gt['ctr'],
                        mode='lines+markers',
                        line=dict(color='#FF6B35', width=3),
                        marker=dict(size=8)
                    ),
                    secondary_y=True
                )
                
                fig_combined.add_trace(
                    go.Scatter(
                        name='Conversion Rate %',
                        x=top_20_gt['search'],
                        y=top_20_gt['conversion_rate_volume'],
                        mode='lines+markers',
                        line=dict(color='#9C27B0', width=3, dash='dash'),
                        marker=dict(size=8)
                    ),
                    secondary_y=True
                )
                
                fig_combined.update_layout(
                    title='<b style="color:#C2185B;">💄 Volume vs CTR & Conversion Performance</b>',
                    plot_bgcolor='rgba(252, 228, 236, 0.3)',
                    paper_bgcolor='rgba(252, 228, 236, 0.1)',
                    font=dict(color='#880E4F', family='Segoe UI'),
                    height=600,
                    xaxis=dict(tickangle=45, showgrid=True, gridcolor='#F8BBD0'),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                fig_combined.update_yaxes(title_text="<b>Search Volume</b>", secondary_y=False, showgrid=True, gridcolor='#F8BBD0')
                fig_combined.update_yaxes(title_text="<b>Performance Rate (%)</b>", secondary_y=True, showgrid=False)
                
                st.plotly_chart(fig_combined, use_container_width=True)
                
                # Volume distribution
                st.subheader("📊 Search Volume Distribution")
                
                fig_top = px.bar(
                    top_20_gt, x='search', y='count',
                    title=f'<b style="color:#C2185B;">💖 Top {display_count} Generic Terms</b>',
                    color='count', color_continuous_scale=['#FCE4EC', '#F48FB1', '#C2185B'],
                    text='count'
                )
                
                fig_top.update_traces(
                    texttemplate='%{text}',
                    textposition='outside',
                    text=[format_number(int(x)) for x in top_20_gt['count']]
                )
                
                fig_top.update_layout(
                    plot_bgcolor='rgba(252, 228, 236, 0.3)',
                    paper_bgcolor='rgba(252, 228, 236, 0.1)',
                    font=dict(color='#880E4F', family='Segoe UI'),
                    height=500,
                    xaxis=dict(tickangle=45, showgrid=True, gridcolor='#F8BBD0'),
                    yaxis=dict(showgrid=True, gridcolor='#F8BBD0'),
                    showlegend=False
                )
                
                st.plotly_chart(fig_top, use_container_width=True)
                
                # Metrics comparison
                st.subheader("📊 Performance Metrics Comparison")
                
                fig_metrics = go.Figure()
                fig_metrics.add_trace(go.Bar(
                    name='CTR %', x=top_20_gt['search'], y=top_20_gt['ctr'],
                    marker_color='#E91E63',
                    text=[f'{x:.1f}%' for x in top_20_gt['ctr']],
                    textposition='outside'
                ))
                fig_metrics.add_trace(go.Bar(
                    name='Conversion Rate %', x=top_20_gt['search'], y=top_20_gt['conversion_rate'],
                    marker_color='#F48FB1',
                    text=[f'{x:.1f}%' for x in top_20_gt['conversion_rate']],
                    textposition='outside'
                ))
                
                fig_metrics.update_layout(
                    title='<b style="color:#C2185B;">💄 CTR vs Conversion Rate</b>',
                    barmode='group',
                    plot_bgcolor='rgba(252, 228, 236, 0.3)',
                    paper_bgcolor='rgba(252, 228, 236, 0.1)',
                    font=dict(color='#880E4F', family='Segoe UI'),
                    height=500,
                    xaxis=dict(tickangle=45),
                    yaxis=dict(title='Percentage (%)')
                )
                
                st.plotly_chart(fig_metrics, use_container_width=True)
        
        elif analysis_type == "🔍 Term Deep Dive":
            st.subheader("🔬 Generic Term Deep Dive")
            
            selected_generic = st.selectbox(
                "Select a generic term:",
                options=gt_agg['search'].tolist(),
                index=0
            )
            
            if selected_generic:
                generic_data = gt_agg[gt_agg['search'] == selected_generic].iloc[0]
                generic_rank = gt_agg.index[gt_agg['search'] == selected_generic].tolist()[0] + 1
                
                # Detailed metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    rank_perf = "high-pink-performance" if generic_rank <= 3 else "medium-pink-performance" if generic_rank <= 10 else "low-pink-performance"
                    rank_text = "Top 3" if generic_rank <= 3 else "Top 10" if generic_rank <= 10 else "Lower"
                    st.markdown(f"""
                    <div class='pink-generic-metric-card'>
                        <span class='icon'>🏆</span>
                        <div class='value'>#{generic_rank} <span class='pink-performance-badge {rank_perf}'>{rank_text}</span></div>
                        <div class='label'>Market Rank</div>
                        <div class='sub-label'>Out of {metrics['total_generic_terms']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    market_share = (generic_data['count'] / metrics['total_searches'] * 100)
                    share_perf = "high-pink-performance" if market_share > 5 else "medium-pink-performance" if market_share > 2 else "low-pink-performance"
                    share_text = "High" if market_share > 5 else "Medium" if market_share > 2 else "Low"
                    st.markdown(f"""
                    <div class='pink-generic-metric-card'>
                        <span class='icon'>📊</span>
                        <div class='value'>{market_share:.1f}% <span class='pink-performance-badge {share_perf}'>{share_text}</span></div>
                        <div class='label'>Market Share</div>
                        <div class='sub-label'>Of total volume</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    performance_score = (generic_data['ctr'] + generic_data['conversion_rate']) / 2
                    score_perf = "high-pink-performance" if performance_score > 3 else "medium-pink-performance" if performance_score > 1 else "low-pink-performance"
                    score_text = "High" if performance_score > 3 else "Medium" if performance_score > 1 else "Low"
                    st.markdown(f"""
                    <div class='pink-generic-metric-card'>
                        <span class='icon'>⭐</span>
                        <div class='value'>{performance_score:.1f} <span class='pink-performance-badge {score_perf}'>{score_text}</span></div>
                        <div class='label'>Performance Score</div>
                        <div class='sub-label'>Combined CTR & CR</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    conversion_efficiency = generic_data['conversion_rate'] / generic_data['ctr'] * 100 if generic_data['ctr'] > 0 else 0
                    eff_perf = "high-pink-performance" if conversion_efficiency > 50 else "medium-pink-performance" if conversion_efficiency > 25 else "low-pink-performance"
                    eff_text = "High" if conversion_efficiency > 50 else "Medium" if conversion_efficiency > 25 else "Low"
                    st.markdown(f"""
                    <div class='pink-generic-metric-card'>
                        <span class='icon'>⚡</span>
                        <div class='value'>{conversion_efficiency:.1f}% <span class='pink-performance-badge {eff_perf}'>{eff_text}</span></div>
                        <div class='label'>Conversion Efficiency</div>
                        <div class='sub-label'>CR as % of CTR</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Performance breakdown
                medians = {
                    'count': gt_agg['count'].median(),
                    'Clicks': gt_agg['Clicks'].median(),
                    'Conversions': gt_agg['Conversions'].median(),
                    'ctr': gt_agg['ctr'].median(),
                    'classic_cvr': gt_agg['classic_cvr'].median(),
                    'conversion_rate': gt_agg['conversion_rate'].median(),
                    'click_share': gt_agg['click_share'].median(),
                    'conversion_share': gt_agg['conversion_share'].median()
                }
                
                metrics_data = {
                    'Metric': ['Search Volume', 'Total Clicks', 'Total Conversions', 
                               'CTR', 'Classic CVR', 'Conversion Rate', 'Click Share', 'Conversion Share'],
                    'Value': [
                        format_number(int(generic_data['count'])),
                        format_number(int(generic_data['Clicks'])),
                        format_number(int(generic_data['Conversions'])),
                        f"{generic_data['ctr']:.1f}%",
                        f"{generic_data['classic_cvr']:.1f}%",
                        f"{generic_data['conversion_rate']:.1f}%",
                        f"{generic_data['click_share']:.1f}%",
                        f"{generic_data['conversion_share']:.1f}%"
                    ],
                    'Performance': [
                        'High' if generic_data['count'] > medians['count'] else 'Low',
                        'High' if generic_data['Clicks'] > medians['Clicks'] else 'Low',
                        'High' if generic_data['Conversions'] > medians['Conversions'] else 'Low',
                        'High' if generic_data['ctr'] > medians['ctr'] else 'Low',
                        'High' if generic_data['classic_cvr'] > medians['classic_cvr'] else 'Low',
                        'High' if generic_data['conversion_rate'] > medians['conversion_rate'] else 'Low',
                        'High' if generic_data['click_share'] > medians['click_share'] else 'Low',
                        'High' if generic_data['conversion_share'] > medians['conversion_share'] else 'Low'
                    ]
                }
                
                display_styled_table(
                    df=pd.DataFrame(metrics_data),
                    title="📈 Performance Breakdown",
                    align="center"
                )
                
                # Radar chart
                st.markdown("### 📊 Performance Radar")
                
                max_values = {
                    'count': gt_agg['count'].max(),
                    'ctr': gt_agg['ctr'].max(),
                    'conversion_rate': gt_agg['conversion_rate'].max()
                }
                
                normalized_data = {
                    'Search Volume': generic_data['count'] / max_values['count'] * 100,
                    'CTR': generic_data['ctr'] / max_values['ctr'] * 100 if max_values['ctr'] > 0 else 0,
                    'Conversion Rate': generic_data['conversion_rate'] / max_values['conversion_rate'] * 100 if max_values['conversion_rate'] > 0 else 0,
                    'Click Share': generic_data['click_share'],
                    'Conversion Share': generic_data['conversion_share']
                }
                
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=list(normalized_data.values()),
                    theta=list(normalized_data.keys()),
                    fill='toself',
                    name=selected_generic,
                    line_color='#E91E63',
                    fillcolor='rgba(233, 30, 99, 0.3)'
                ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 100], gridcolor='#F8BBD0'),
                        angularaxis=dict(gridcolor='#F8BBD0')
                    ),
                    title=f'<b style="color:#C2185B;">💄 Performance Radar - {selected_generic}</b>',
                    height=400,
                    plot_bgcolor='rgba(252, 228, 236, 0.3)',
                    paper_bgcolor='rgba(252, 228, 236, 0.1)',
                    font=dict(color='#880E4F', family='Segoe UI')
                )
                
                st.plotly_chart(fig_radar, use_container_width=True)
        
        elif analysis_type == "📈 Performance Comparison":
            st.subheader("⚖️ Generic Terms Comparison")
            
            selected_generics = st.multiselect(
                "Select generic terms to compare (max 10):",
                options=gt_agg['search'].tolist(),
                default=gt_agg['search'].head(5).tolist(),
                max_selections=10
            )
            
            if selected_generics:
                comparison_data = gt_agg[gt_agg['search'].isin(selected_generics)].copy()
                
                # Comparison chart
                fig_comp = go.Figure()
                metrics_list = ['ctr', 'conversion_rate', 'click_share', 'conversion_share']
                metric_names = ['CTR %', 'Conversion Rate %', 'Click Share %', 'Conversion Share %']
                colors = ['#E91E63', '#F48FB1', '#F06292', '#F8BBD0']
                
                for i, (metric, name) in enumerate(zip(metrics_list, metric_names)):
                    fig_comp.add_trace(go.Bar(
                        name=name,
                        x=comparison_data['search'],
                        y=comparison_data[metric],
                        marker_color=colors[i]
                    ))
                
                fig_comp.update_layout(
                    title='<b style="color:#C2185B;">💄 Performance Metrics Comparison</b>',
                    barmode='group',
                    plot_bgcolor='rgba(252, 228, 236, 0.3)',
                    paper_bgcolor='rgba(252, 228, 236, 0.1)',
                    font=dict(color='#880E4F', family='Segoe UI'),
                    height=500,
                    xaxis=dict(tickangle=45),
                    yaxis=dict(title='Percentage (%)')
                )
                
                st.plotly_chart(fig_comp, use_container_width=True)
                
                # Comparison table
                st.markdown("### 📊 Detailed Comparison")
                
                comparison_table = comparison_data[['search', 'count', 'Clicks', 'Conversions', 
                                                    'ctr', 'conversion_rate', 'click_share', 'conversion_share']].copy()
                comparison_table.columns = ['Generic Term', 'Search Volume', 'Clicks', 'Conversions', 
                                            'CTR %', 'Conversion Rate %', 'Click Share %', 'Conversion Share %']
                
                for col in ['Search Volume', 'Clicks', 'Conversions']:
                    comparison_table[col] = comparison_table[col].apply(lambda x: format_number(int(x)))
                for col in ['CTR %', 'Conversion Rate %', 'Click Share %', 'Conversion Share %']:
                    comparison_table[col] = comparison_table[col].apply(lambda x: f"{x:.1f}%")
                
                display_styled_table(
                    df=comparison_table,
                    align="center",
                    scrollable=True,
                    max_height="500px"
                )
                
                # Download
                csv_comparison = comparison_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download Comparison Data",
                    data=csv_comparison,
                    file_name="generic_terms_comparison.csv",
                    mime="text/csv",
                    key="generic_comparison_download"
                )
            else:
                st.info("Please select generic terms to compare.")
        
        elif analysis_type == "📊 Distribution Analysis":
            st.subheader("📊 Market Share & Distribution")
            
            # Market share visualizations
            col_pie, col_treemap = st.columns(2)
            
            with col_pie:
                top_10_market = gt_agg.head(10).copy()
                others_value = gt_agg.iloc[10:]['count'].sum() if len(gt_agg) > 10 else 0
                
                if others_value > 0:
                    others_row = pd.DataFrame({'search': ['Others'], 'count': [others_value]})
                    pie_data = pd.concat([top_10_market[['search', 'count']], others_row], ignore_index=True)
                else:
                    pie_data = top_10_market[['search', 'count']]
                
                fig_pie = px.pie(
                    pie_data, values='count', names='search',
                    title='<b style="color:#C2185B;">💖 Top 10 Market Share</b>',
                    color_discrete_sequence=['#C2185B', '#E91E63', '#F06292', '#F48FB1', '#F8BBD0', '#FCE4EC', '#AD1457', '#D81B60', '#EC407A', '#FF4081', '#F8BBD0']
                )
                
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                fig_pie.update_layout(
                    height=400,
                    plot_bgcolor='rgba(252, 228, 236, 0.3)',
                    paper_bgcolor='rgba(252, 228, 236, 0.1)',
                    font=dict(color='#880E4F', family='Segoe UI')
                )
                
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col_treemap:
                fig_treemap = px.treemap(
                    gt_agg.head(20), path=['search'], values='count',
                    title='<b style="color:#C2185B;">💄 Volume Distribution</b>',
                    color='ctr', color_continuous_scale=['#FCE4EC', '#F48FB1', '#C2185B'],
                    hover_data={'count': ':,', 'ctr': ':.2f'}
                )
                
                fig_treemap.update_layout(
                    height=400,
                    plot_bgcolor='rgba(252, 228, 236, 0.3)',
                    paper_bgcolor='rgba(252, 228, 236, 0.1)',
                    font=dict(color='#880E4F', family='Segoe UI')
                )
                
                st.plotly_chart(fig_treemap, use_container_width=True)
            
            # Distribution metrics
            st.markdown("### 📈 Distribution Analysis")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class='pink-generic-metric-card'>
                    <span class='icon'>📊</span>
                    <div class='value'>{metrics['gini_coefficient']:.3f}</div>
                    <div class='label'>Gini Coefficient</div>
                    <div class='sub-label'>Market concentration</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class='pink-generic-metric-card'>
                    <span class='icon'>📈</span>
                    <div class='value'>{metrics['herfindahl_index']:.4f}</div>
                    <div class='label'>Herfindahl Index</div>
                    <div class='sub-label'>Market dominance</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class='pink-generic-metric-card'>
                    <span class='icon'>🔝</span>
                    <div class='value'>{metrics['top_5_concentration']:.1f}%</div>
                    <div class='label'>Top 5 Share</div>
                    <div class='sub-label'>Market concentration</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class='pink-generic-metric-card'>
                    <span class='icon'>🔟</span>
                    <div class='value'>{metrics['top_10_concentration']:.1f}%</div>
                    <div class='label'>Top 10 Share</div>
                    <div class='sub-label'>Market concentration</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Lorenz Curve
            st.markdown("### 📈 Market Concentration Analysis")
            
            sorted_counts = np.sort(gt_agg['count'].values)
            cumulative_counts = np.cumsum(sorted_counts)
            total_count = cumulative_counts[-1]
            
            lorenz_x = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts) * 100
            lorenz_y = cumulative_counts / total_count * 100
            
            fig_lorenz = go.Figure()
            fig_lorenz.add_trace(go.Scatter(
                x=lorenz_x, y=lorenz_y, mode='lines',
                name='Lorenz Curve',
                line=dict(color='#E91E63', width=3)
            ))
            fig_lorenz.add_trace(go.Scatter(
                x=[0, 100], y=[0, 100], mode='lines',
                name='Line of Equality',
                line=dict(color='#F48FB1', width=2, dash='dash')
            ))
            
            fig_lorenz.update_layout(
                title='<b style="color:#C2185B;">💄 Lorenz Curve - Market Concentration</b>',
                xaxis_title='Cumulative % of Generic Terms',
                yaxis_title='Cumulative % of Search Volume',
                plot_bgcolor='rgba(252, 228, 236, 0.3)',
                paper_bgcolor='rgba(252, 228, 236, 0.1)',
                font=dict(color='#880E4F', family='Segoe UI'),
                height=400,
                showlegend=True,
                xaxis=dict(showgrid=True, gridcolor='#F8BBD0'),
                yaxis=dict(showgrid=True, gridcolor='#F8BBD0')
            )
            
            st.plotly_chart(fig_lorenz, use_container_width=True)
            
            # Insights
            col_ins1, col_ins2 = st.columns(2)
            
            with col_ins1:
                st.markdown("#### 🎯 Market Concentration Insights")
                
                if metrics['gini_coefficient'] > 0.7:
                    st.error("🔴 **Highly Concentrated**: Few terms dominate")
                elif metrics['gini_coefficient'] > 0.5:
                    st.warning("🟡 **Moderately Concentrated**: Some terms have significant share")
                else:
                    st.success("🟢 **Well-Distributed**: Volume is relatively even")
                
                st.markdown(f"- **Gini Coefficient**: {metrics['gini_coefficient']:.3f}")
                st.markdown(f"- **Top 5 Terms**: {metrics['top_5_concentration']:.1f}% of volume")
                st.markdown(f"- **Top 10 Terms**: {metrics['top_10_concentration']:.1f}% of volume")
            
            with col_ins2:
                st.markdown("#### 📊 Performance Distribution")
                
                quartiles = gt_agg['count'].quantile([0.25, 0.5, 0.75])
                q1, q2, q3 = quartiles[0.25], quartiles[0.5], quartiles[0.75]
                
                high_performers = len(gt_agg[gt_agg['count'] >= q3])
                medium_performers = len(gt_agg[(gt_agg['count'] >= q2) & (gt_agg['count'] < q3)])
                low_performers = len(gt_agg[gt_agg['count'] < q2])
                
                st.markdown(f"**📈 High Volume (Top 25%)**: {high_performers} terms")
                st.markdown(f"**📊 Medium Volume (25-75%)**: {medium_performers} terms")
                st.markdown(f"**📉 Low Volume (Bottom 50%)**: {low_performers} terms")
                
                high_avg_ctr = gt_agg[gt_agg['count'] >= q3]['ctr'].mean()
                medium_avg_ctr = gt_agg[(gt_agg['count'] >= q2) & (gt_agg['count'] < q3)]['ctr'].mean()
                low_avg_ctr = gt_agg[gt_agg['count'] < q2]['ctr'].mean()
                
                st.markdown(f"**CTR by Volume:**")
                st.markdown(f"- High: {high_avg_ctr:.1f}%")
                st.markdown(f"- Medium: {medium_avg_ctr:.1f}%")
                st.markdown(f"- Low: {low_avg_ctr:.1f}%")
        
        # 💾 DOWNLOAD SECTION
        st.markdown("---")
        st.subheader("💾 Export & Download Options")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            csv_complete = gt_agg.to_csv(index=False)
            st.download_button(
                label="📊 Complete Analysis",
                data=csv_complete,
                file_name=f"generic_terms_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="complete_generic_download"
            )
        
        with col2:
            top_performers_csv = gt_agg.head(50).to_csv(index=False)
            st.download_button(
                label="🏆 Top 50 Performers",
                data=top_performers_csv,
                file_name=f"top_50_generic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="top_performers_download"
            )
        
        with col3:
            # Summary report
            top_10_list = "\n".join([f"{i+1}. {row['search']}: {format_number(int(row['count']))} searches ({row['ctr']:.1f}% CTR, {row['conversion_rate']:.1f}% CR)" 
                                   for i, (_, row) in enumerate(gt_agg.head(10).iterrows())])
            
            summary = f"""# Generic Terms Analysis Summary
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
- Total Generic Terms: {format_number(metrics['total_generic_terms'])}
- Total Search Volume: {format_number(metrics['total_searches'])}
- Average CTR: {metrics['avg_ctr']:.1f}%
- Average Conversion Rate: {metrics['avg_cr']:.1f}%
- Total Clicks: {format_number(metrics['total_clicks'])}
- Total Conversions: {format_number(metrics['total_conversions'])}

## Top Performing Generic Terms
{top_10_list}

## Market Concentration
- Gini Coefficient: {metrics['gini_coefficient']:.3f}
- Herfindahl Index: {metrics['herfindahl_index']:.4f}
- Top 5 Market Share: {metrics['top_5_concentration']:.1f}%
- Top 10 Market Share: {metrics['top_10_concentration']:.1f}%

## Key Insights
- Top Generic Term: "{metrics['top_generic_term']}" with {format_number(int(metrics['top_generic_volume']))} searches
- Conversion Leader: "{metrics['top_conversion_generic']}"
- Market Concentration: {"High" if metrics['gini_coefficient'] > 0.7 else "Medium" if metrics['gini_coefficient'] > 0.5 else "Low"}

Generated by Generic Terms Analysis Dashboard
"""
            
            st.download_button(
                label="📋 Executive Summary",
                data=summary,
                file_name=f"generic_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                key="summary_download"
            )
        
        with col4:
            high_opportunity = gt_agg[
                (gt_agg['count'] > gt_agg['count'].median()) & 
                (gt_agg['ctr'] < metrics['avg_ctr'])
            ]
            
            if len(high_opportunity) > 0:
                opportunity_csv = high_opportunity.to_csv(index=False)
                st.download_button(
                    label="🎯 High Opportunity Terms",
                    data=opportunity_csv,
                    file_name=f"high_opportunity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="opportunity_download"
                )
            else:
                st.info("No high-opportunity terms")
        
        # 🔍 ADVANCED FILTERING
        st.markdown("---")
        st.subheader("🔍 Advanced Filtering & Custom Analysis")
        
        with st.expander("🎛️ Custom Filter Options", expanded=False):
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            
            with filter_col1:
                st.markdown("**Volume Filters**")
                min_searches = st.number_input(
                    "Minimum Search Volume:",
                    min_value=0,
                    max_value=int(gt_agg['count'].max()),
                    value=0,
                    key="min_searches_filter"
                )
                max_searches = st.number_input(
                    "Maximum Search Volume:",
                    min_value=int(min_searches),
                    max_value=int(gt_agg['count'].max()),
                    value=int(gt_agg['count'].max()),
                    key="max_searches_filter"
                )
            
            with filter_col2:
                st.markdown("**Performance Filters**")
                min_ctr = st.slider(
                    "Minimum CTR (%):",
                    min_value=0.0,
                    max_value=float(gt_agg['ctr'].max()),
                    value=0.0,
                    step=0.1,
                    key="min_ctr_filter"
                )
                min_cr = st.slider(
                    "Minimum Conversion Rate (%):",
                    min_value=0.0,
                    max_value=float(gt_agg['conversion_rate'].max()),
                    value=0.0,
                    step=0.1,
                    key="min_cr_filter"
                )
            
            with filter_col3:
                st.markdown("**Text Filters**")
                search_contains = st.text_input(
                    "Generic term contains:",
                    placeholder="Enter text to search...",
                    key="search_contains_filter"
                )
                exclude_terms = st.text_input(
                    "Exclude terms containing:",
                    placeholder="Enter text to exclude...",
                    key="exclude_terms_filter"
                )
            
            # Apply filters
            @st.cache_data(ttl=1800, show_spinner=False)
            def apply_filters(data, min_s, max_s, min_c, min_conv, contains, exclude):
                filtered = data[
                    (data['count'] >= min_s) & (data['count'] <= max_s) &
                    (data['ctr'] >= min_c) & (data['conversion_rate'] >= min_conv)
                ].copy()
                
                if contains:
                    filtered = filtered[filtered['search'].str.contains(contains, case=False, na=False)]
                if exclude:
                    filtered = filtered[~filtered['search'].str.contains(exclude, case=False, na=False)]
                
                return filtered
            
            filtered_data = apply_filters(
                gt_agg, min_searches, max_searches, min_ctr, min_cr, 
                search_contains, exclude_terms
            )
            
            if len(filtered_data) > 0:
                st.markdown(f"### 📊 Filtered Results: {len(filtered_data)} generic terms")
                
                # Quick stats for filtered data
                filtered_col1, filtered_col2, filtered_col3, filtered_col4 = st.columns(4)
                
                with filtered_col1:
                    st.markdown(f"""
                    <div class='pink-generic-metric-card'>
                        <span class='icon'>📊</span>
                        <div class='value'>{format_number(len(filtered_data))}</div>
                        <div class='label'>Terms Found</div>
                        <div class='sub-label'>Matching filters</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with filtered_col2:
                    total_searches_filtered = filtered_data['count'].sum()
                    st.markdown(f"""
                    <div class='pink-generic-metric-card'>
                        <span class='icon'>🔍</span>
                        <div class='value'>{format_number(int(total_searches_filtered))}</div>
                        <div class='label'>Total Searches</div>
                        <div class='sub-label'>Filtered volume</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with filtered_col3:
                    avg_ctr_filtered = filtered_data['ctr'].mean()
                    ctr_performance = "high-pink-performance" if avg_ctr_filtered > 5 else "medium-pink-performance" if avg_ctr_filtered > 2 else "low-pink-performance"
                    ctr_text = "High" if avg_ctr_filtered > 5 else "Medium" if avg_ctr_filtered > 2 else "Low"
                    st.markdown(f"""
                    <div class='pink-generic-metric-card'>
                        <span class='icon'>📈</span>
                        <div class='value'>{avg_ctr_filtered:.1f}% <span class='pink-performance-badge {ctr_performance}'>{ctr_text}</span></div>
                        <div class='label'>Avg CTR</div>
                        <div class='sub-label'>Filtered average</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with filtered_col4:
                    avg_cr_filtered = filtered_data['conversion_rate'].mean()
                    cr_performance = "high-pink-performance" if avg_cr_filtered > 3 else "medium-pink-performance" if avg_cr_filtered > 1 else "low-pink-performance"
                    cr_text = "High" if avg_cr_filtered > 3 else "Medium" if avg_cr_filtered > 1 else "Low"
                    st.markdown(f"""
                    <div class='pink-generic-metric-card'>
                        <span class='icon'>✨</span>
                        <div class='value'>{avg_cr_filtered:.1f}% <span class='pink-performance-badge {cr_performance}'>{cr_text}</span></div>
                        <div class='label'>Avg CR</div>
                        <div class='sub-label'>Filtered average</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display filtered data table
                display_filtered = filtered_data[['search', 'count', 'Clicks', 'Conversions', 'ctr', 'conversion_rate']].copy()
                display_filtered.columns = ['Generic Term', 'Search Volume', 'Clicks', 'Conversions', 'CTR %', 'Conversion Rate %']
                
                # Format numbers
                for col in ['Search Volume', 'Clicks', 'Conversions']:
                    display_filtered[col] = display_filtered[col].apply(lambda x: format_number(int(x)))
                for col in ['CTR %', 'Conversion Rate %']:
                    display_filtered[col] = display_filtered[col].apply(lambda x: f"{x:.1f}%")
                
                display_styled_table(
                    df=display_filtered,
                    title="🔍 Filtered Generic Terms",
                    align="center",
                    scrollable=True,
                    max_height="600px"
                )
                
                # Download filtered data
                filtered_csv = filtered_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download Filtered Data",
                    data=filtered_csv,
                    file_name=f"filtered_generic_terms_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="filtered_generic_download",
                    use_container_width=True
                )
            else:
                st.warning("⚠️ No generic terms match the selected filters. Try adjusting your criteria.")

    except KeyError as e:
        st.error(f"❌ Missing required column: {str(e)}")
        st.info("Please ensure your data contains: 'search', 'count', 'Clicks', 'Conversions'")
    except ValueError as e:
        st.error(f"❌ Data format error: {str(e)}")
        st.info("Please check that numeric columns contain valid numbers")
    except Exception as e:
        st.error(f"❌ Unexpected error processing generic type data: {str(e)}")
        st.info("Please check your data format and try again.")
        st.markdown("""
        **Expected data format:**
        - Column 'search' with generic term names
        - Column 'count' with search volume data
        - Column 'Clicks' with click data
        - Column 'Conversions' with conversion data
        """)



# ----------------- Time Analysis Tab (PINK THEME + OPTIMIZED) -----------------
with tab_time:

    # 🎨 PINK-THEMED HERO HEADER
    st.markdown("""
    <div style="
        text-align: center; 
        padding: 3rem 2rem; 
        background: linear-gradient(135deg, #FCE4EC 0%, #F8BBD0 50%, #F48FB1 100%); 
        border-radius: 20px; 
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(136, 14, 79, 0.15);
        border: 1px solid rgba(233, 30, 99, 0.2);
    ">
        <h1 style="
            color: #880E4F; 
            margin: 0; 
            font-size: 3rem; 
            text-shadow: 2px 2px 8px rgba(136, 14, 79, 0.2);
            font-weight: 700;
            letter-spacing: -1px;
        ">
            ⏰ Temporal Performance Analysis ✨
        </h1>
        <p style="
            color: #C2185B; 
            margin: 1rem 0 0 0; 
            font-size: 1.3rem;
            font-weight: 300;
            opacity: 0.9;
        ">
            Deep Dive into Performance Metrics and Search Trends Over Time
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 🚀 OPTIMIZED: Lazy CSS loading - Only load once per session
    if 'time_analysis_pink_css_loaded' not in st.session_state:
        st.markdown("""
        <style>
        .time-metric-card {
            background: linear-gradient(135deg, #FCE4EC 0%, #F8BBD0 100%);
            padding: 25px; border-radius: 15px; text-align: center; color: #880E4F;
            box-shadow: 0 8px 32px rgba(194, 24, 91, 0.3); margin: 10px 0;
            min-height: 160px; display: flex; flex-direction: column; justify-content: center;
            transition: transform 0.2s ease; border-left: 4px solid #E91E63;
        }
        .time-metric-card:hover { transform: translateY(-2px); box-shadow: 0 12px 40px rgba(194, 24, 91, 0.4); }
        .time-metric-card .icon { font-size: 3em; margin-bottom: 10px; display: block; color: #C2185B; }
        .time-metric-card .value { font-size: 1.6em; font-weight: bold; margin-bottom: 8px; word-wrap: break-word; overflow-wrap: break-word; line-height: 1.2; color: #880E4F; }
        .time-metric-card .label { font-size: 1.1em; opacity: 0.95; font-weight: 600; margin-bottom: 6px; color: #C2185B; }
        .time-metric-card .sub-label { font-size: 1em; opacity: 0.9; font-weight: 500; line-height: 1.2; color: #AD1457; }
        .time-performance-badge { padding: 4px 8px; border-radius: 12px; font-size: 0.8em; font-weight: bold; margin-left: 8px; }
        .high-time-performance { background-color: #E91E63; color: white; }
        .medium-time-performance { background-color: #F48FB1; color: white; }
        .low-time-performance { background-color: #F8BBD0; color: #880E4F; }
        .time-table-container { background: linear-gradient(135deg, #FCE4EC 0%, #F8BBD0 100%); padding: 20px; border-radius: 15px; border-left: 5px solid #E91E63; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin: 10px 0; transition: transform 0.2s ease; }
        .time-table-container:hover { transform: translateY(-2px); box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15); }
        .time-insight-card { background: linear-gradient(135deg, #C2185B 0%, #F06292 100%); padding: 25px; border-radius: 15px; color: white; margin: 15px 0; box-shadow: 0 6px 20px rgba(194, 24, 91, 0.3); }
        </style>
        """, unsafe_allow_html=True)
        st.session_state.time_analysis_pink_css_loaded = True

    try:
        # 🚀 OPTIMIZED: Fast data validation with early exit
        if queries is None or queries.empty:
            st.warning("⚠️ No time data available.")
            st.info("Please ensure your uploaded file contains valid time data.")
            st.stop()
        
        # 🚀 OPTIMIZED: Batch column validation
        required_columns = ['month', 'Counts', 'clicks', 'conversions']
        missing_columns = [col for col in required_columns if col not in queries.columns]
        
        if missing_columns:
            st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
            st.info("Please ensure your time data contains these columns")
            st.stop()
        
        # 🚀 OPTIMIZED: Vectorized data cleaning
        queries_clean = queries.copy()
        numeric_columns = ['Counts', 'clicks', 'conversions']
        
        # Batch numeric conversion
        for col in numeric_columns:
            queries_clean[col] = pd.to_numeric(queries_clean[col], errors='coerce').fillna(0)
        
        # Fast filtering
        queries_clean = queries_clean.dropna(subset=['month'])
        queries_clean = queries_clean[queries_clean['month'].str.strip() != '']
        
        if queries_clean.empty:
            st.warning("⚠️ No valid time data found after cleaning.")
            st.info("Please check your data for empty or invalid month values.")
            st.stop()
        
        # 🚀 OPTIMIZED: Cached monthly calculations with filter awareness
        time_data_state = {
            'data_shape': queries_clean.shape,
            'data_hash': hash(str(queries_clean['month'].tolist()[:10]) + str(queries_clean['Counts'].sum())),
            'filters_applied': st.session_state.get('filters_applied', False)
        }
        time_cache_key = str(hash(str(time_data_state)))
        
        @st.cache_data(ttl=1800, show_spinner=False)
        def compute_monthly_metrics(df, cache_key):
            """🚀 OPTIMIZED: Vectorized monthly calculations"""
            monthly = df.groupby('month').agg({
                'Counts': 'sum',
                'clicks': 'sum',
                'conversions': 'sum'
            }).reset_index()
            
            # Vectorized calculations
            monthly['ctr'] = np.where(monthly['Counts'] > 0, (monthly['clicks'] / monthly['Counts'] * 100), 0)
            monthly['conversion_rate'] = np.where(monthly['Counts'] > 0, (monthly['conversions'] / monthly['Counts'] * 100), 0)
            monthly['classic_cvr'] = np.where(monthly['clicks'] > 0, (monthly['conversions'] / monthly['clicks'] * 100), 0)
            
            # Share calculations
            total_clicks = monthly['clicks'].sum()
            total_conversions = monthly['conversions'].sum()
            monthly['click_share'] = np.where(total_clicks > 0, (monthly['clicks'] / total_clicks * 100), 0)
            monthly['conversion_share'] = np.where(total_conversions > 0, (monthly['conversions'] / total_conversions * 100), 0)
            
            # Smart date sorting
            try:
                monthly['month_dt'] = pd.to_datetime(monthly['month'], format='%b %Y', errors='coerce')
                monthly = monthly.sort_values('month_dt')
            except:
                monthly = monthly.sort_values('month')
            
            return monthly, total_clicks, total_conversions
        
        monthly, total_clicks, total_conversions = compute_monthly_metrics(queries_clean, time_cache_key)
        
        # 🚀 OPTIMIZED: Fast distribution calculations
        @st.cache_data(ttl=1800, show_spinner=False)
        def compute_distribution_metrics(monthly_df, cache_key):
            """🚀 OPTIMIZED: Fast distribution calculations"""
            if monthly_df.empty:
                return 0, 0
            
            # Gini coefficient calculation
            sorted_counts = monthly_df['Counts'].sort_values().values
            n = len(sorted_counts)
            cumsum = np.cumsum(sorted_counts)
            gini = 1 - 2 * np.sum(cumsum) / (n * sorted_counts.sum())
            
            # Top 3 concentration
            top_3_conc = monthly_df.head(3)['Counts'].sum() / monthly_df['Counts'].sum() * 100
            
            return gini, top_3_conc
        
        gini_coefficient, top_3_concentration = compute_distribution_metrics(monthly, time_cache_key)
        
        # 🚀 OPTIMIZED: Pre-calculated summary metrics
        summary_metrics = {
            'total_months': len(monthly),
            'total_searches': monthly['Counts'].sum(),
            'avg_ctr': monthly['ctr'].mean(),
            'avg_cr': monthly['conversion_rate'].mean(),
            'total_clicks': int(total_clicks),
            'total_conversions': int(total_conversions)
        }
        
        # 📊 KEY METRICS SECTION
        st.subheader("💖 Monthly Performance Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class='time-metric-card'>
                <span class='icon'>📅</span>
                <div class='value'>{summary_metrics['total_months']}</div>
                <div class='label'>Total Months</div>
                <div class='sub-label'>Analyzed periods</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='time-metric-card'>
                <span class='icon'>🔍</span>
                <div class='value'>{format_number(summary_metrics['total_searches'])}</div>
                <div class='label'>Total Searches</div>
                <div class='sub-label'>Across all months</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            monthly_avg_ctr = monthly['ctr'].mean()
            overall_ctr = (monthly['clicks'].sum() / monthly['Counts'].sum() * 100) if monthly['Counts'].sum() > 0 else 0
            
            performance_class = "high-time-performance" if monthly_avg_ctr > 5 else "medium-time-performance" if monthly_avg_ctr > 2 else "low-time-performance"
            st.markdown(f"""
            <div class='time-metric-card'>
                <span class='icon'>📈</span>
                <div class='value'>{monthly_avg_ctr:.1f}% <span class='time-performance-badge {performance_class}'>{"High" if monthly_avg_ctr > 5 else "Medium" if monthly_avg_ctr > 2 else "Low"}</span></div>
                <div class='label'>Average Monthly CTR</div>
                <div class='sub-label'>Overall: {overall_ctr:.1f}% | Monthly Avg: {monthly_avg_ctr:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            monthly_avg_cr = monthly['conversion_rate'].mean()
            overall_cr = (monthly['conversions'].sum() / monthly['Counts'].sum() * 100) if monthly['Counts'].sum() > 0 else 0
            
            performance_class = "high-time-performance" if monthly_avg_cr > 3 else "medium-time-performance" if monthly_avg_cr > 1 else "low-time-performance"
            st.markdown(f"""
            <div class='time-metric-card'>
                <span class='icon'>✨</span>
                <div class='value'>{monthly_avg_cr:.1f}% <span class='time-performance-badge {performance_class}'>{"High" if monthly_avg_cr > 3 else "Medium" if monthly_avg_cr > 1 else "Low"}</span></div>
                <div class='label'>Average Monthly CR</div>
                <div class='sub-label'>Overall: {overall_cr:.1f}% | Monthly Avg: {monthly_avg_cr:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        # 🎯 INTERACTIVE ANALYSIS
        st.markdown("---")
        st.subheader("🎯 Interactive Temporal Analysis")

        analysis_type = st.radio(
            "Choose Analysis Type:",
            ["📊 Trends Overview", "🔍 Detailed Month Analysis", "🏷 Brand Comparison", "📊 Distribution Analysis"],
            horizontal=True,
            key="temporal_analysis_type_radio"
        )

        if analysis_type == "📊 Trends Overview":
            st.subheader("📈 Monthly Trends")
            
            # 🚀 OPTIMIZED: Cached chart creation
            @st.cache_data(ttl=1800, show_spinner=False)
            def create_trends_charts(monthly_df, cache_key):
                """🚀 OPTIMIZED: Pre-built chart configurations"""
                # Line chart for counts
                fig_counts = px.line(
                    monthly_df, x='month', y='Counts',
                    title='<b style="color:#C2185B;">💄 Monthly Search Volume</b>',
                    labels={'Counts': 'Search Volume', 'month': 'Month'},
                    color_discrete_sequence=['#E91E63']
                )
                fig_counts.update_traces(line=dict(width=3))
                fig_counts.update_layout(
                    plot_bgcolor='rgba(252, 228, 236, 0.3)',
                    paper_bgcolor='rgba(252, 228, 236, 0.1)',
                    font=dict(color='#880E4F', family='Segoe UI'),
                    height=400,
                    xaxis=dict(tickangle=45, showgrid=True, gridcolor='#F8BBD0'),
                    yaxis=dict(showgrid=True, gridcolor='#F8BBD0')
                )
                
                # Metrics chart
                fig_metrics = go.Figure()
                fig_metrics.add_trace(go.Scatter(
                    x=monthly_df['month'], y=monthly_df['ctr'],
                    name='CTR %', line=dict(color='#E91E63', width=3)
                ))
                fig_metrics.add_trace(go.Scatter(
                    x=monthly_df['month'], y=monthly_df['conversion_rate'],
                    name='Conversion Rate %', line=dict(color='#F48FB1', width=3)
                ))
                fig_metrics.update_layout(
                    title='<b style="color:#C2185B;">💄 Monthly CTR and Conversion Rate Trends</b>',
                    plot_bgcolor='rgba(252, 228, 236, 0.3)',
                    paper_bgcolor='rgba(252, 228, 236, 0.1)',
                    font=dict(color='#880E4F', family='Segoe UI'),
                    height=400,
                    xaxis=dict(tickangle=45, title='Month'),
                    yaxis=dict(title='Percentage (%)')
                )
                
                return fig_counts, fig_metrics
            
            fig_counts, fig_metrics = create_trends_charts(monthly, time_cache_key)
            st.plotly_chart(fig_counts, use_container_width=True)
            st.plotly_chart(fig_metrics, use_container_width=True)

        elif analysis_type == "🔍 Detailed Month Analysis":
            st.subheader("🔬 Detailed Monthly Performance")
            
            selected_month = st.selectbox(
                "Select a month for detailed analysis:",
                options=monthly['month'].tolist(),
                index=0,
                key="detailed_month_selector"
            )
            
            if selected_month:
                month_data = monthly[monthly['month'] == selected_month].iloc[0]
                month_rank = monthly.reset_index().index[monthly['month'] == selected_month].tolist()[0] + 1
                
                col_detail1, col_detail2, col_detail3, col_detail4 = st.columns(4)
                
                with col_detail1:
                    st.markdown(f"""
                    <div class='time-metric-card'>
                        <span class='icon'>🏆</span>
                        <div class='value'>#{month_rank}</div>
                        <div class='label'>Month Rank</div>
                        <div class='sub-label'>Out of {summary_metrics['total_months']} months</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_detail2:
                    market_share = (month_data['Counts'] / summary_metrics['total_searches'] * 100)
                    st.markdown(f"""
                    <div class='time-metric-card'>
                        <span class='icon'>📊</span>
                        <div class='value'>{format_percentage(market_share)}</div>
                        <div class='label'>Market Share</div>
                        <div class='sub-label'>Of total searches</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_detail3:
                    perf_class = "high-time-performance" if month_data['ctr'] > 5 else "medium-time-performance" if month_data['ctr'] > 2 else "low-time-performance"
                    perf_text = "High" if month_data['ctr'] > 5 else "Medium" if month_data['ctr'] > 2 else "Low"
                    st.markdown(f"""
                    <div class='time-metric-card'>
                        <span class='icon'>📈</span>
                        <div class='value'>{format_percentage(month_data['ctr'])} <span class='time-performance-badge {perf_class}'>{perf_text}</span></div>
                        <div class='label'>CTR</div>
                        <div class='sub-label'>Month performance</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_detail4:
                    perf_class = "high-time-performance" if month_data['conversion_rate'] > 3 else "medium-time-performance" if month_data['conversion_rate'] > 1 else "low-time-performance"
                    perf_text = "High" if month_data['conversion_rate'] > 3 else "Medium" if month_data['conversion_rate'] > 1 else "Low"
                    st.markdown(f"""
                    <div class='time-metric-card'>
                        <span class='icon'>✨</span>
                        <div class='value'>{format_percentage(month_data['conversion_rate'])} <span class='time-performance-badge {perf_class}'>{perf_text}</span></div>
                        <div class='label'>Conversion Rate</div>
                        <div class='sub-label'>Month performance</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Performance breakdown
                st.markdown("### 📊 Performance Breakdown")
                metrics_data = {
                    'Metric': ['Search Volume', 'Total Clicks', 'Total Conversions', 'CTR', 'Conversion Rate', 'Classic CVR', 'Click Share', 'Conversion Share'],
                    'Value': [
                        format_number(month_data['Counts']), 
                        format_number(month_data['clicks']), 
                        format_number(month_data['conversions']),
                        format_percentage(month_data['ctr']), 
                        format_percentage(month_data['conversion_rate']), 
                        format_percentage(month_data['classic_cvr']),
                        format_percentage(month_data['click_share']), 
                        format_percentage(month_data['conversion_share'])
                    ],
                    'Performance': [
                        'High' if month_data['Counts'] > monthly['Counts'].median() else 'Low',
                        'High' if month_data['clicks'] > monthly['clicks'].median() else 'Low',
                        'High' if month_data['conversions'] > monthly['conversions'].median() else 'Low',
                        'High' if month_data['ctr'] > monthly['ctr'].median() else 'Low',
                        'High' if month_data['conversion_rate'] > monthly['conversion_rate'].median() else 'Low',
                        'High' if month_data['classic_cvr'] > monthly['classic_cvr'].median() else 'Low',
                        'High' if month_data['click_share'] > monthly['click_share'].median() else 'Low',
                        'High' if month_data['conversion_share'] > monthly['conversion_share'].median() else 'Low'
                    ]
                }
                metrics_df = pd.DataFrame(metrics_data)
                display_styled_table(
                    df=metrics_df,
                    title="📈 Performance Breakdown",
                    align="center"
                )

        elif analysis_type == "🏷 Brand Comparison":
            st.subheader("🏷 Top Brands Performance by Month")
            
            if 'brand' in queries_clean.columns and queries_clean['brand'].notna().any():
                # 🚀 OPTIMIZED: Cached brand analysis
                @st.cache_data(ttl=1800, show_spinner=False)
                def compute_brand_analysis(df, cache_key):
                    """🚀 OPTIMIZED: Brand analysis with vectorized operations"""
                    brand_counts = df[df['brand'].str.lower() != 'other'].groupby('brand')['Counts'].sum()
                    top_brands = brand_counts.sort_values(ascending=False).head(5).index
                    
                    brand_month = df[df['brand'].isin(top_brands)].groupby(['month', 'brand']).agg({
                        'Counts': 'sum', 'clicks': 'sum', 'conversions': 'sum'
                    }).reset_index()
                    
                    # Vectorized calculations
                    brand_month['ctr'] = np.where(brand_month['Counts'] > 0, (brand_month['clicks'] / brand_month['Counts'] * 100), 0)
                    brand_month['conversion_rate'] = np.where(brand_month['Counts'] > 0, (brand_month['conversions'] / brand_month['Counts'] * 100), 0)
                    
                    try:
                        brand_month['month_dt'] = pd.to_datetime(brand_month['month'], format='%b %Y', errors='coerce')
                        brand_month = brand_month.sort_values('month_dt')
                    except:
                        brand_month = brand_month.sort_values('month')
                    
                    return brand_month
                
                brand_month = compute_brand_analysis(queries_clean, time_cache_key)
                
                # 🚀 OPTIMIZED: Cached chart creation
                @st.cache_data(ttl=1800, show_spinner=False)
                def create_brand_chart(brand_data, cache_key):
                    """🚀 OPTIMIZED: Pre-built brand chart"""
                    fig_brands = px.bar(
                        brand_data, x='month', y='Counts', color='brand',
                        title='<b style="color:#C2185B;">💄 Top 5 Brands by Search Volume per Month</b>',
                        color_discrete_sequence=['#C2185B', '#E91E63', '#F06292', '#F48FB1', '#F8BBD0']
                    )
                    fig_brands.update_layout(
                        plot_bgcolor='rgba(252, 228, 236, 0.3)',
                        paper_bgcolor='rgba(252, 228, 236, 0.1)',
                        font=dict(color='#880E4F', family='Segoe UI'),
                        height=500,
                        xaxis=dict(tickangle=45, title='Month'),
                        yaxis=dict(title='Search Volume')
                    )
                    return fig_brands
                
                fig_brands = create_brand_chart(brand_month, time_cache_key)
                st.plotly_chart(fig_brands, use_container_width=True)
                
                # Display table
                display_brands = brand_month[['month', 'brand', 'Counts', 'clicks', 'conversions', 'ctr', 'conversion_rate']].copy()
                display_brands.columns = ['Month', 'Brand', 'Search Volume', 'Clicks', 'Conversions', 'CTR %', 'Conversion Rate %']
                
                display_brands['Search Volume'] = display_brands['Search Volume'].apply(format_number)
                display_brands['Clicks'] = display_brands['Clicks'].apply(format_number)
                display_brands['Conversions'] = display_brands['Conversions'].apply(format_number)
                display_brands['CTR %'] = display_brands['CTR %'].apply(format_percentage)
                display_brands['Conversion Rate %'] = display_brands['Conversion Rate %'].apply(format_percentage)
                
                display_styled_table(
                    df=display_brands,
                    title="📊 Brand Performance Table",
                    align="center"
                )
                
                # Download
                csv_brands = brand_month.to_csv(index=False)
                st.download_button(
                    label="📥 Download Brand Data CSV",
                    data=csv_brands,
                    file_name=f"brand_monthly_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="brand_monthly_download"
                )
            else:
                st.info("Brand or month data not available for brand-month analysis.")

        elif analysis_type == "📊 Distribution Analysis":
            st.subheader("📊 Monthly Distribution Analysis")
            
            # 🚀 OPTIMIZED: Cached pie chart
            @st.cache_data(ttl=1800, show_spinner=False)
            def create_distribution_chart(monthly_df, cache_key):
                """🚀 OPTIMIZED: Pre-built distribution chart"""
                fig_pie = px.pie(
                    monthly_df, values='Counts', names='month',
                    title='<b style="color:#C2185B;">💄 Monthly Search Volume Distribution</b>',
                    color_discrete_sequence=['#C2185B', '#E91E63', '#F06292', '#F48FB1', '#F8BBD0', '#FCE4EC', '#AD1457', '#D81B60', '#EC407A', '#FF4081', '#F8BBD0']
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                fig_pie.update_layout(
                    height=400,
                    plot_bgcolor='rgba(252, 228, 236, 0.3)',
                    paper_bgcolor='rgba(252, 228, 236, 0.1)',
                    font=dict(color='#880E4F', family='Segoe UI')
                )
                return fig_pie
            
            fig_pie = create_distribution_chart(monthly, time_cache_key)
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Distribution metrics
            col_dist1, col_dist2 = st.columns(2)
            
            with col_dist1:
                st.markdown(f"""
                <div class='time-metric-card'>
                    <span class='icon'>📊</span>
                    <div class='value'>{gini_coefficient:.3f}</div>
                    <div class='label'>Gini Coefficient</div>
                    <div class='sub-label'>Market concentration</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_dist2:
                st.markdown(f"""
                <div class='time-metric-card'>
                    <span class='icon'>🔝</span>
                    <div class='value'>{format_percentage(top_3_concentration)}</div>
                    <div class='label'>Top 3 Months Share</div>
                    <div class='sub-label'>Volume concentration</div>
                </div>
                """, unsafe_allow_html=True)

        # 🔍 ADVANCED FILTERING
        st.markdown("---")
        st.subheader("🔍 Advanced Filtering & Custom Analysis")

        with st.expander("🎛️ Custom Filter Options", expanded=False):
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            
            with filter_col1:
                st.markdown("**Volume Filters**")
                min_searches = st.number_input(
                    "Minimum Search Volume:",
                    min_value=0,
                    max_value=int(monthly['Counts'].max()),
                    value=0,
                    key="min_searches_time_filter"
                )
                max_searches = st.number_input(
                    "Maximum Search Volume:",
                    min_value=int(min_searches),
                    max_value=int(monthly['Counts'].max()),
                    value=int(monthly['Counts'].max()),
                    key="max_searches_time_filter"
                )
            
            with filter_col2:
                st.markdown("**Performance Filters**")
                min_ctr = st.slider(
                    "Minimum CTR (%):",
                    min_value=0.0,
                    max_value=float(monthly['ctr'].max()),
                    value=0.0,
                    step=0.1,
                    key="min_ctr_time_filter"
                )
                min_cr = st.slider(
                    "Minimum Conversion Rate (%):",
                    min_value=0.0,
                    max_value=float(monthly['conversion_rate'].max()),
                    value=0.0,
                    step=0.1,
                    key="min_cr_time_filter"
                )
            
            with filter_col3:
                st.markdown("**Brand Filter**")
                if 'brand' in queries_clean.columns and queries_clean['brand'].notna().any():
                    @st.cache_data(ttl=1800, show_spinner=False)
                    def get_brand_options(df, cache_key):
                        """🚀 OPTIMIZED: Pre-computed brand list"""
                        brand_series = df['brand'].astype(str).replace('nan', '')
                        return [b for b in brand_series.unique().tolist() if b.lower() != 'other' and b]
                    
                    brand_options = get_brand_options(queries_clean, time_cache_key)
                    selected_brands = st.multiselect(
                        "Select brands to include:",
                        options=brand_options,
                        default=brand_options[:min(3, len(brand_options))],
                        key="brand_time_filter"
                    )
                else:
                    selected_brands = []
                    st.info("No brand data available for filtering.")
            
            # Apply filters
            filtered_data = monthly.copy()
            
            filter_mask = (
                (filtered_data['Counts'] >= min_searches) &
                (filtered_data['Counts'] <= max_searches) &
                (filtered_data['ctr'] >= min_ctr) &
                (filtered_data['conversion_rate'] >= min_cr)
            )
            filtered_data = filtered_data[filter_mask]
            
            # Brand filter
            if selected_brands:
                @st.cache_data(ttl=1800, show_spinner=False)
                def apply_brand_filter(df, brands, cache_key):
                    """🚀 OPTIMIZED: Cached brand filtering"""
                    brand_series = df['brand'].astype(str).replace('nan', '')
                    brand_filtered = df[
                        (brand_series.isin(brands)) & 
                        (brand_series.str.lower() != 'other')
                    ].groupby('month').agg({
                        'Counts': 'sum',
                        'clicks': 'sum',
                        'conversions': 'sum'
                    }).reset_index()
                    
                    brand_filtered['ctr'] = np.where(brand_filtered['Counts'] > 0, (brand_filtered['clicks'] / brand_filtered['Counts'] * 100), 0)
                    brand_filtered['conversion_rate'] = np.where(brand_filtered['Counts'] > 0, (brand_filtered['conversions'] / brand_filtered['Counts'] * 100), 0)
                    brand_filtered['classic_cvr'] = np.where(brand_filtered['clicks'] > 0, (brand_filtered['conversions'] / brand_filtered['clicks'] * 100), 0)
                    brand_filtered['click_share'] = np.where(total_clicks > 0, (brand_filtered['clicks'] / total_clicks * 100), 0)
                    brand_filtered['conversion_share'] = np.where(total_conversions > 0, (brand_filtered['conversions'] / total_conversions * 100), 0)
                    
                    return brand_filtered
                
                brand_filtered = apply_brand_filter(queries_clean, selected_brands, time_cache_key + str(selected_brands))
                
                filtered_data = filtered_data.merge(
                    brand_filtered[['month', 'Counts', 'clicks', 'conversions', 'ctr', 'conversion_rate', 'classic_cvr', 'click_share', 'conversion_share']],
                    on='month', how='inner', suffixes=('', '_brand')
                )
                
                for col in ['Counts', 'clicks', 'conversions', 'ctr', 'conversion_rate', 'classic_cvr', 'click_share', 'conversion_share']:
                    filtered_data[col] = filtered_data[f'{col}_brand']
                    filtered_data = filtered_data.drop(columns=f'{col}_brand')
            
            # Display filtered results
            if len(filtered_data) > 0:
                st.markdown(f"### 📊 Filtered Results: {len(filtered_data)} months")
                
                filtered_col1, filtered_col2, filtered_col3, filtered_col4 = st.columns(4)
                
                filtered_metrics = {
                    'count': len(filtered_data),
                    'total_searches': filtered_data['Counts'].sum(),
                    'avg_ctr': filtered_data['ctr'].mean(),
                    'avg_cr': filtered_data['conversion_rate'].mean()
                }
                
                with filtered_col1:
                    st.markdown(f"""
                    <div class='time-metric-card'>
                        <span class='icon'>📅</span>
                        <div class='value'>{filtered_metrics['count']}</div>
                        <div class='label'>Months Found</div>
                        <div class='sub-label'>Matching filters</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with filtered_col2:
                    st.markdown(f"""
                    <div class='time-metric-card'>
                        <span class='icon'>🔍</span>
                        <div class='value'>{format_number(filtered_metrics['total_searches'])}</div>
                        <div class='label'>Total Searches</div>
                        <div class='sub-label'>Filtered volume</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with filtered_col3:
                    ctr_perf = "high-time-performance" if filtered_metrics['avg_ctr'] > 5 else "medium-time-performance" if filtered_metrics['avg_ctr'] > 2 else "low-time-performance"
                    ctr_text = "High" if filtered_metrics['avg_ctr'] > 5 else "Medium" if filtered_metrics['avg_ctr'] > 2 else "Low"
                    st.markdown(f"""
                    <div class='time-metric-card'>
                        <span class='icon'>📈</span>
                        <div class='value'>{format_percentage(filtered_metrics['avg_ctr'])} <span class='time-performance-badge {ctr_perf}'>{ctr_text}</span></div>
                        <div class='label'>Avg CTR</div>
                        <div class='sub-label'>Filtered average</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with filtered_col4:
                    cr_perf = "high-time-performance" if filtered_metrics['avg_cr'] > 3 else "medium-time-performance" if filtered_metrics['avg_cr'] > 1 else "low-time-performance"
                    cr_text = "High" if filtered_metrics['avg_cr'] > 3 else "Medium" if filtered_metrics['avg_cr'] > 1 else "Low"
                    st.markdown(f"""
                    <div class='time-metric-card'>
                        <span class='icon'>✨</span>
                        <div class='value'>{format_percentage(filtered_metrics['avg_cr'])} <span class='time-performance-badge {cr_perf}'>{cr_text}</span></div>
                        <div class='label'>Avg CR</div>
                        <div class='sub-label'>Filtered average</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display table
                display_filtered = filtered_data[['month', 'Counts', 'clicks', 'conversions', 'ctr', 'conversion_rate']].copy()
                display_filtered.columns = ['Month', 'Search Volume', 'Clicks', 'Conversions', 'CTR %', 'Conversion Rate %']
                
                display_filtered['Search Volume'] = display_filtered['Search Volume'].apply(format_number)
                display_filtered['Clicks'] = display_filtered['Clicks'].apply(format_number)
                display_filtered['Conversions'] = display_filtered['Conversions'].apply(format_number)
                display_filtered['CTR %'] = display_filtered['CTR %'].apply(format_percentage)
                display_filtered['Conversion Rate %'] = display_filtered['Conversion Rate %'].apply(format_percentage)
                
                display_styled_table(
                    df=display_filtered,
                    align="center",
                    scrollable=True,
                    max_height="600px"
                )
                
                # Download
                filtered_csv = filtered_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download Filtered Data",
                    data=filtered_csv,
                    file_name=f"filtered_monthly_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="filtered_time_download"
                )
            else:
                st.warning("⚠️ No months match the selected filters. Try adjusting your criteria.")

        # 💾 DOWNLOAD SECTION
        st.markdown("---")
        st.subheader("💾 Advanced Export & Download Options")
        
        col_download1, col_download2 = st.columns(2)
        
        with col_download1:
            csv_complete = monthly.to_csv(index=False)
            st.download_button(
                label="📊 Complete Monthly Analysis CSV",
                data=csv_complete,
                file_name=f"monthly_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="complete_time_download"
            )
        
        with col_download2:
            @st.cache_data(ttl=1800, show_spinner=False)
            def generate_summary_report(monthly_df, summary_stats, gini, top3_conc, cache_key):
                """🚀 OPTIMIZED: Cached report generation"""
                top_months = monthly_df.head(3)
                top_months_text = '\n'.join([
                    f"{row['month']}: {int(row['Counts']):,} searches ({row['ctr']:.1f}% CTR, {row['conversion_rate']:.1f}% CR)" 
                    for _, row in top_months.iterrows()
                ])
                
                return f"""# Monthly Analysis Summary Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
- Total Months Analyzed: {summary_stats['total_months']}
- Total Search Volume: {summary_stats['total_searches']:,}
- Average CTR: {summary_stats['avg_ctr']:.1f}%
- Average Conversion Rate: {summary_stats['avg_cr']:.1f}%
- Total Clicks: {summary_stats['total_clicks']:,}
- Total Conversions: {summary_stats['total_conversions']:,}

## Top Performing Months
{top_months_text}

## Market Concentration
- Gini Coefficient: {gini:.3f}
- Top 3 Months Share: {top3_conc:.1f}%

## Recommendations
- Focus on high-performing months for campaign optimization
- Investigate low-performing months for improvement opportunities

Generated by Temporal Analysis Dashboard
"""
            
            summary_report = generate_summary_report(monthly, summary_metrics, gini_coefficient, top_3_concentration, time_cache_key)
            
            st.download_button(
                label="📋 Executive Summary",
                data=summary_report,
                file_name=f"monthly_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                key="summary_time_download"
            )
    
    except KeyError as e:
        st.error(f"❌ Missing required column: {str(e)}")
        st.info("Please ensure your data contains: 'month', 'Counts', 'clicks', 'conversions'")
    except ValueError as e:
        st.error(f"❌ Data format error: {str(e)}")
        st.info("Please check that numeric columns contain valid numbers")
    except Exception as e:
        st.error(f"❌ Unexpected error processing time data: {str(e)}")
        st.info("Please check your data format and try again.")



# ----------------- Pivot Builder Tab (PINK THEME + OPTIMIZED) -----------------
with tab_pivot:
    st.header("🔄 Pivot Intelligence Hub")
    st.markdown("Deep dive into custom pivots and advanced data insights. 💡")

    # 🚀 OPTIMIZED: Generate cache key for pivot tab
    @st.cache_data(ttl=3600, show_spinner=False)
    def generate_pivot_cache_key(df):
        """Generate unique cache key for pivot data"""
        return hashlib.md5(f"{len(df)}_{df['Counts'].sum()}_{datetime.now().strftime('%Y%m%d')}".encode()).hexdigest()
    
    pivot_cache_key = generate_pivot_cache_key(queries)

    # 🎨 PINK-THEMED HERO HEADER
    st.markdown("""
    <div style="
        text-align: center; 
        padding: 3rem 2rem; 
        background: linear-gradient(135deg, #FCE4EC 0%, #F8BBD0 50%, #F48FB1 100%); 
        border-radius: 20px; 
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(136, 14, 79, 0.15);
        border: 1px solid rgba(233, 30, 99, 0.2);
    ">
        <h1 style="
            color: #880E4F; 
            margin: 0; 
            font-size: 3rem; 
            text-shadow: 2px 2px 8px rgba(136, 14, 79, 0.2);
            font-weight: 700;
            letter-spacing: -1px;
        ">
            🔄 Pivot Intelligence Hub ✨
        </h1>
        <p style="
            color: #C2185B; 
            margin: 1rem 0 0 0; 
            font-size: 1.3rem;
            font-weight: 300;
            opacity: 0.9;
        ">
            Deep Dive into Custom Pivots and Advanced Data Insights
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 🎨 PINK THEME CSS
    if 'pivot_pink_css_loaded' not in st.session_state:
        st.markdown("""
        <style>
        .pivot-metric-card {
            background: linear-gradient(135deg, #FCE4EC 0%, #F8BBD0 100%);
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            color: #880E4F;
            box-shadow: 0 8px 32px rgba(194, 24, 91, 0.3);
            margin: 10px 0;
            min-height: 160px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            transition: transform 0.2s ease;
            border-left: 4px solid #E91E63;
        }
        .pivot-metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 40px rgba(194, 24, 91, 0.4);
        }
        .pivot-metric-card .icon {
            font-size: 3em;
            margin-bottom: 10px;
            display: block;
            color: #C2185B;
        }
        .pivot-metric-card .value {
            font-size: 1.6em;
            font-weight: bold;
            margin-bottom: 8px;
            word-wrap: break-word;
            overflow-wrap: break-word;
            line-height: 1.2;
            color: #880E4F;
        }
        .pivot-metric-card .label {
            font-size: 1.1em;
            opacity: 0.95;
            font-weight: 600;
            margin-bottom: 6px;
            color: #C2185B;
        }
        .pivot-metric-card .sub-label {
            font-size: 1em;
            opacity: 0.9;
            font-weight: 500;
            line-height: 1.2;
            color: #AD1457;
        }
        .pivot-performance-badge {
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: bold;
            margin-left: 8px;
        }
        .high-pivot-performance {
            background-color: #E91E63;
            color: white;
        }
        .medium-pivot-performance {
            background-color: #F48FB1;
            color: white;
        }
        .low-pivot-performance {
            background-color: #F8BBD0;
            color: #880E4F;
        }
        .pivot-table-container {
            background: linear-gradient(135deg, #FCE4EC 0%, #F8BBD0 100%);
            padding: 20px;
            border-radius: 15px;
            border-left: 5px solid #E91E63;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin: 10px 0;
            transition: transform 0.2s ease;
        }
        .pivot-table-container:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        }
        .pivot-insight-card {
            background: linear-gradient(135deg, #C2185B 0%, #F06292 100%);
            padding: 25px;
            border-radius: 15px;
            color: white;
            margin: 15px 0;
            box-shadow: 0 6px 20px rgba(194, 24, 91, 0.3);
        }
        .pivot-insight-card h4 {
            margin-top: 0;
            margin-bottom: 15px;
            font-size: 1.3em;
        }
        .pivot-insight-card p {
            line-height: 1.8;
            margin: 0;
        }
        
        /* Pivot Configuration Styling */
        .stMultiSelect [data-baseweb="tag"] {
            background-color: #E91E63 !important;
            color: white !important;
        }
        .stMultiSelect [data-baseweb="tag"] span[role="button"] {
            color: white !important;
        }
        .stSelectbox > div > div {
            border-color: #E91E63 !important;
        }
        
        /* Preview box styling */
        .pivot-preview-box {
            background: linear-gradient(135deg, #FCE4EC 0%, #F8BBD0 100%);
            padding: 20px;
            border-radius: 12px;
            border-left: 4px solid #E91E63;
            margin: 15px 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .pivot-preview-box h4 {
            color: #C2185B;
            margin-bottom: 12px;
            font-size: 1.1em;
        }
        .pivot-preview-item {
            background: white;
            padding: 10px 15px;
            border-radius: 8px;
            margin: 8px 0;
            border-left: 3px solid #F06292;
            color: #880E4F;
            font-weight: 500;
        }
        .pivot-preview-label {
            color: #C2185B;
            font-weight: 600;
            margin-right: 8px;
        }
        .pivot-preview-value {
            color: #AD1457;
            font-weight: 500;
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #E91E63 0%, #F06292 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 12px 24px !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 6px rgba(233, 30, 99, 0.3) !important;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #C2185B 0%, #E91E63 100%) !important;
            box-shadow: 0 6px 12px rgba(233, 30, 99, 0.4) !important;
            transform: translateY(-2px) !important;
        }
        .stButton > button[kind="secondary"] {
            background: linear-gradient(135deg, #F48FB1 0%, #F8BBD0 100%) !important;
            color: #880E4F !important;
        }
        .stButton > button[kind="secondary"]:hover {
            background: linear-gradient(135deg, #F06292 0%, #F48FB1 100%) !important;
        }
        </style>
        """, unsafe_allow_html=True)
        st.session_state.pivot_pink_css_loaded = True

    try:
        # 🚀 OPTIMIZED: Prebuilt Pivot with Caching
        st.subheader("📋 Prebuilt: Brand × Query (Top 300)")
        
        @st.cache_data(ttl=3600, show_spinner=False)
        def generate_brand_query_pivot(df, cache_key):
            """🚀 OPTIMIZED: Generate brand-query pivot with caching"""
            if 'brand' not in df.columns or 'normalized_query' not in df.columns:
                return None
            
            # Clean and prepare data
            df_clean = df.copy()
            df_clean['brand'] = df_clean['brand'].astype(str).replace('nan', '')
            df_clean['Counts'] = pd.to_numeric(df_clean['Counts'], errors='coerce').fillna(0)
            df_clean['clicks'] = pd.to_numeric(df_clean['clicks'], errors='coerce').fillna(0)
            df_clean['conversions'] = pd.to_numeric(df_clean['conversions'], errors='coerce').fillna(0)
            
            # Filter out 'other' brands
            df_clean = df_clean[df_clean['brand'].str.lower() != 'other']
            
            # Aggregate with vectorized operations
            pv = df_clean.groupby(['brand', 'normalized_query'], as_index=False).agg({
                'Counts': 'sum',
                'clicks': 'sum',
                'conversions': 'sum'
            })
            
            # Vectorized metric calculations
            pv['ctr'] = np.where(pv['Counts'] > 0, (pv['clicks'] / pv['Counts'] * 100), 0)
            pv['conversion_rate'] = np.where(pv['Counts'] > 0, (pv['conversions'] / pv['Counts'] * 100), 0)
            pv['classic_cvr'] = np.where(pv['clicks'] > 0, (pv['conversions'] / pv['clicks'] * 100), 0)
            
            # Sort and get top 300
            pv_top = pv.nlargest(300, 'Counts')
            
            return pv_top
        
        pv_top = generate_brand_query_pivot(queries, pivot_cache_key)
        
        if pv_top is not None and len(pv_top) > 0:
            # 🚀 OPTIMIZED: Pre-calculated metrics
            @st.cache_data(ttl=3600, show_spinner=False)
            def calculate_pivot_metrics(df, cache_key):
                """🚀 OPTIMIZED: Calculate pivot metrics with caching"""
                return {
                    'total_rows': len(df),
                    'total_counts': df['Counts'].sum(),
                    'total_clicks': df['clicks'].sum(),
                    'total_conversions': df['conversions'].sum(),
                    'avg_ctr': df['ctr'].mean(),
                    'avg_cr': df['conversion_rate'].mean(),
                    'avg_classic_cvr': df['classic_cvr'].mean(),
                    'top5_concentration': (df.head(5)['Counts'].sum() / df['Counts'].sum() * 100) if df['Counts'].sum() > 0 else 0
                }
            
            pivot_metrics = calculate_pivot_metrics(pv_top, pivot_cache_key)
            
            # 📊 METRIC CARDS
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class='pivot-metric-card'>
                    <span class='icon'>📋</span>
                    <div class='value'>{pivot_metrics['total_rows']:,}</div>
                    <div class='label'>Total Rows</div>
                    <div class='sub-label'>Top 300 Brand-Query Pairs</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class='pivot-metric-card'>
                    <span class='icon'>🔍</span>
                    <div class='value'>{format_number(int(pivot_metrics['total_counts']))}</div>
                    <div class='label'>Total Searches</div>
                    <div class='sub-label'>Top 300 pairs</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                ctr_performance = "high-pivot-performance" if pivot_metrics['avg_ctr'] > 5 else "medium-pivot-performance" if pivot_metrics['avg_ctr'] > 2 else "low-pivot-performance"
                ctr_label = "High" if pivot_metrics['avg_ctr'] > 5 else "Medium" if pivot_metrics['avg_ctr'] > 2 else "Low"
                st.markdown(f"""
                <div class='pivot-metric-card'>
                    <span class='icon'>📈</span>
                    <div class='value'>{pivot_metrics['avg_ctr']:.1f}% <span class='pivot-performance-badge {ctr_performance}'>{ctr_label}</span></div>
                    <div class='label'>Average CTR</div>
                    <div class='sub-label'>Top 300 pairs</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                cr_performance = "high-pivot-performance" if pivot_metrics['avg_cr'] > 3 else "medium-pivot-performance" if pivot_metrics['avg_cr'] > 1 else "low-pivot-performance"
                cr_label = "High" if pivot_metrics['avg_cr'] > 3 else "Medium" if pivot_metrics['avg_cr'] > 1 else "Low"
                st.markdown(f"""
                <div class='pivot-metric-card'>
                    <span class='icon'>✨</span>
                    <div class='value'>{pivot_metrics['avg_cr']:.1f}% <span class='pivot-performance-badge {cr_performance}'>{cr_label}</span></div>
                    <div class='label'>Avg CR</div>
                    <div class='sub-label'>Top 300 pairs</div>
                </div>
                """, unsafe_allow_html=True)
            
            # 🔍 FILTER & SORT OPTIONS
            with st.expander("🔍 Filter & Sort Options", expanded=False):
                filter_col1, filter_col2 = st.columns(2)
                
                with filter_col1:
                    sort_col = st.selectbox(
                        "Sort By:", 
                        options=['Counts', 'clicks', 'conversions', 'ctr', 'conversion_rate', 'classic_cvr'], 
                        index=0,
                        key="pivot_sort_col"
                    )
                    sort_order = st.radio(
                        "Sort Order:", 
                        options=['Descending', 'Ascending'], 
                        index=0, 
                        horizontal=True,
                        key="pivot_sort_order"
                    )
                
                with filter_col2:
                    min_counts = st.number_input(
                        "Minimum Search Volume:", 
                        min_value=0, 
                        value=0,
                        key="pivot_min_counts"
                    )
                    min_ctr = st.slider(
                        "Minimum CTR (%):",
                        min_value=0.0,
                        max_value=float(pv_top['ctr'].max()),
                        value=0.0,
                        step=0.1,
                        key="pivot_min_ctr"
                    )
                
                # Apply filters
                pv_filtered = pv_top[
                    (pv_top['Counts'] >= min_counts) & 
                    (pv_top['ctr'] >= min_ctr)
                ].sort_values(
                    sort_col, 
                    ascending=(sort_order == 'Ascending')
                ).head(300)
            
            # 📊 DISPLAY PIVOT
            st.markdown(f"### 📊 Showing {len(pv_filtered)} Brand-Query Pairs")
            
            @st.cache_data(ttl=1800, show_spinner=False)
            def format_pivot_display(df, cache_key):
                """🚀 OPTIMIZED: Format pivot for display"""
                display_df = df[['brand', 'normalized_query', 'Counts', 'clicks', 'conversions', 'ctr', 'conversion_rate', 'classic_cvr']].copy()
                display_df.columns = ['Brand', 'Query', 'Search Volume', 'Clicks', 'Conversions', 'CTR %', 'CR %', 'Classic CVR %']
                
                # Vectorized formatting
                display_df['Search Volume'] = display_df['Search Volume'].apply(format_number)
                display_df['Clicks'] = display_df['Clicks'].apply(format_number)
                display_df['Conversions'] = display_df['Conversions'].apply(format_number)
                display_df['CTR %'] = display_df['CTR %'].apply(format_percentage)
                display_df['CR %'] = display_df['CR %'].apply(format_percentage)
                display_df['Classic CVR %'] = display_df['Classic CVR %'].apply(format_percentage)
                
                return display_df
            
            display_pv = format_pivot_display(pv_filtered, pivot_cache_key + str(min_counts) + str(min_ctr))
            
            st.markdown("<div class='pivot-table-container'>", unsafe_allow_html=True)
            if AGGRID_OK:
                gb = GridOptionsBuilder.from_dataframe(display_pv)
                gb.configure_default_column(filterable=True, sortable=True, resizable=True)
                gb.configure_grid_options(enableRangeSelection=True, pagination=True, paginationPageSize=20)
                gb.configure_selection(selection_mode="multiple", use_checkbox=True)
                AgGrid(display_pv, gridOptions=gb.build(), height=500, theme='material', fit_columns_on_grid_load=True)
            else:
                display_styled_table(
                    df=display_pv,
                    align="center",
                    scrollable=True,
                    max_height="500px"
                )
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Download button
            csv_pv = pv_filtered.to_csv(index=False)
            st.download_button(
                label="📥 Download Brand × Query Pivot",
                data=csv_pv,
                file_name=f"brand_query_pivot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="brand_query_pivot_download"
            )
        else:
            st.info("Brand or normalized_query column missing for this pivot.")
            
        # 🔧 CUSTOM PIVOT BUILDER
        st.markdown("---")
        st.subheader("🔧 Custom Pivot Builder")

        # 🚀 OPTIMIZED: Get available columns
        @st.cache_data(ttl=3600, show_spinner=False)
        def get_pivot_columns(df):
            """Get available columns for pivot"""
            return df.columns.tolist()

        columns = get_pivot_columns(queries)

        with st.expander("🎛️ Pivot Configuration", expanded=True):
            col_pivot1, col_pivot2 = st.columns(2)
            
            with col_pivot1:
                idx = st.multiselect(
                    "📊 Rows (Index)",
                    options=columns,
                    default=['normalized_query'] if 'normalized_query' in columns else [],
                    help="Select one or more columns to group as rows.",
                    key="pivot_idx"
                )
                cols = st.multiselect(
                    "📋 Columns",
                    options=[c for c in columns if c not in idx],
                    default=['brand'] if 'brand' in columns else [],
                    help="Select one or more columns to group as columns.",
                    key="pivot_cols"
                )
            
            with col_pivot2:
                val = st.selectbox(
                    "📈 Value (Measure)",
                    options=['Counts', 'clicks', 'conversions', 'ctr', 'conversion_rate'],
                    index=0,
                    help="Select the metric to aggregate (CTR and CR are calculated post-aggregation).",
                    key="pivot_val"
                )
                aggfunc = st.selectbox(
                    "🔢 Aggregation Function",
                    options=['sum', 'mean', 'count', 'max', 'min'],
                    index=0,
                    help="Choose how to aggregate the selected value.",
                    key="pivot_aggfunc"
                )
            
            # Preview pivot structure
            if idx and cols and val:
                st.markdown("""
                <div class='pivot-preview-box'>
                    <h4>👁️ Preview Pivot Structure</h4>
                    <div class='pivot-preview-item'>
                        <span class='pivot-preview-label'>📊 Rows:</span>
                        <span class='pivot-preview-value'>{}</span>
                    </div>
                    <div class='pivot-preview-item'>
                        <span class='pivot-preview-label'>📋 Columns:</span>
                        <span class='pivot-preview-value'>{}</span>
                    </div>
                    <div class='pivot-preview-item'>
                        <span class='pivot-preview-label'>📈 Value:</span>
                        <span class='pivot-preview-value'>{} ({})</span>
                    </div>
                </div>
                """.format(', '.join(idx), ', '.join(cols), val, aggfunc), unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='background: linear-gradient(135deg, #FFF9C4 0%, #FFF59D 100%); 
                            padding: 15px; 
                            border-radius: 10px; 
                            border-left: 4px solid #FBC02D; 
                            margin: 15px 0;'>
                    <span style='color: #F57F17; font-weight: 600;'>⚠️ Please select at least one row, one column, and a value to generate the pivot.</span>
                </div>
                """, unsafe_allow_html=True)

        col_btn1, col_btn2 = st.columns(2)

        with col_btn1:
            generate_pivot = st.button("🔄 Generate Pivot", use_container_width=True, type="primary")

        with col_btn2:
            reset_pivot = st.button("🔃 Reset Selections", use_container_width=True)

        if reset_pivot:
            # Clear session state
            for key in ['pivot_idx', 'pivot_cols', 'pivot_val', 'pivot_aggfunc', 'pivot_sort_col', 'pivot_sort_order', 'pivot_min_counts', 'pivot_min_ctr']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

        if generate_pivot:
            if not idx or not cols or not val:
                st.markdown("""
                <div style='background: linear-gradient(135deg, #FFCDD2 0%, #EF9A9A 100%); 
                            padding: 15px; 
                            border-radius: 10px; 
                            border-left: 4px solid #E53935; 
                            margin: 15px 0;'>
                    <span style='color: #B71C1C; font-weight: 600;'>❌ Please select at least one row, one column, and a value.</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                try:
                    with st.spinner("🔄 Generating custom pivot..."):
                        # 🚀 OPTIMIZED: Generate custom pivot with caching
                        @st.cache_data(ttl=1800, show_spinner=False)
                        def generate_custom_pivot(df, idx_cols, col_cols, val_col, agg_func, cache_key):
                            """🚀 OPTIMIZED: Generate custom pivot with caching"""
                            pivot_data = df.copy()
                            
                            # Calculate derived metrics if needed
                            if val_col in ['ctr', 'conversion_rate']:
                                pivot_data['ctr'] = np.where(
                                    pivot_data['Counts'] > 0, 
                                    (pivot_data['clicks'] / pivot_data['Counts'] * 100), 
                                    0
                                )
                                pivot_data['conversion_rate'] = np.where(
                                    pivot_data['Counts'] > 0, 
                                    (pivot_data['conversions'] / pivot_data['Counts'] * 100), 
                                    0
                                )
                            
                            # Handle brand column if present
                            if 'brand' in pivot_data.columns:
                                pivot_data['brand'] = pivot_data['brand'].astype(str).replace('nan', '')
                                pivot_data = pivot_data[pivot_data['brand'].str.lower() != 'other']
                            
                            # Create pivot table
                            pivot = pd.pivot_table(
                                pivot_data,
                                values=val_col,
                                index=idx_cols,
                                columns=col_cols,
                                aggfunc=agg_func,
                                fill_value=0
                            )
                            
                            return pivot
                        
                        custom_pivot_key = pivot_cache_key + str(idx) + str(cols) + str(val) + str(aggfunc)
                        pivot = generate_custom_pivot(queries, idx, cols, val, aggfunc, custom_pivot_key)
                        
                        # Success message
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #F8BBD0 0%, #F48FB1 100%); 
                                    padding: 15px; 
                                    border-radius: 10px; 
                                    border-left: 4px solid #E91E63; 
                                    margin: 15px 0;'>
                            <span style='color: #880E4F; font-weight: 600;'>✅ Custom pivot generated successfully! Shape: {pivot.shape[0]:,} rows × {pivot.shape[1]:,} columns</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("<div class='pivot-table-container'>", unsafe_allow_html=True)
                        if AGGRID_OK:
                            pivot_display = pivot.reset_index()
                            gb = GridOptionsBuilder.from_dataframe(pivot_display)
                            gb.configure_default_column(filterable=True, sortable=True, resizable=True)
                            gb.configure_grid_options(enableRangeSelection=True, pagination=True, paginationPageSize=20)
                            AgGrid(pivot_display, gridOptions=gb.build(), height=500, theme='material', fit_columns_on_grid_load=True)
                        else:
                            display_styled_table(
                                df=pivot,
                                align="left",
                                scrollable=True,
                                max_height="500px"
                            )
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Download button
                        csv_pivot = pivot.to_csv()
                        st.download_button(
                            label="📥 Download Custom Pivot CSV",
                            data=csv_pivot,
                            file_name=f"custom_pivot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            key="custom_pivot_download"
                        )
                        
                except Exception as e:
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #FFCDD2 0%, #EF9A9A 100%); 
                                padding: 15px; 
                                border-radius: 10px; 
                                border-left: 4px solid #E53935; 
                                margin: 15px 0;'>
                        <span style='color: #B71C1C; font-weight: 600;'>❌ Pivot generation error: {e}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("""
                    <div style='background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%); 
                                padding: 15px; 
                                border-radius: 10px; 
                                border-left: 4px solid #2196F3; 
                                margin: 15px 0;'>
                        <span style='color: #0D47A1; font-weight: 600;'>💡 Ensure selected columns and values are valid and contain data.</span>
                    </div>
                    """, unsafe_allow_html=True)

        # 💡 PIVOT INSIGHTS
        st.markdown("---")
        st.subheader("💡 Pivot Insights & Recommendations")
        
        col_insight1, col_insight2 = st.columns(2)
        
        with col_insight1:
            st.markdown("""
            <div class='pivot-insight-card'>
                <h4>🔍 Key Pivot Insights</h4>
                <p>
                • Analyze brand-query interactions for patterns<br>
                • Identify high-performing combinations<br>
                • Spot seasonal trends in data<br>
                • Uncover conversion opportunities<br>
                • Track performance across dimensions
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_insight2:
            st.markdown("""
            <div class='pivot-insight-card'>
                <h4>📊 Pivot Strategy Recommendations</h4>
                <p>
                • Customize pivots for specific metrics<br>
                • Focus on top brand-query pairs<br>
                • Optimize campaigns based on insights<br>
                • Explore multi-dimensional analysis<br>
                • Export data for deeper analysis
                </p>
            </div>
            """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ Unexpected error in Pivot Builder: {e}")
        st.info("💡 Please check your data format and ensure required columns are present.")
        with st.expander("🔍 Error Details"):
            st.code(str(e))


# ----------------- Insights & Strategic Questions (PINK THEME + OPTIMIZED) -----------------
with tab_insights:
    # 🎨 PINK-THEMED HERO HEADER
    st.markdown("""
    <div style="
        text-align: center; 
        padding: 3rem 2rem; 
        background: linear-gradient(135deg, #FCE4EC 0%, #F8BBD0 50%, #F48FB1 100%); 
        border-radius: 20px; 
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(136, 14, 79, 0.15);
        border: 1px solid rgba(233, 30, 99, 0.2);
    ">
        <h1 style="
            color: #880E4F; 
            margin: 0; 
            font-size: 3rem; 
            text-shadow: 2px 2px 8px rgba(136, 14, 79, 0.2);
            font-weight: 700;
            letter-spacing: -1px;
        ">
            🔍 Strategic Insights Hub ✨
        </h1>
        <p style="
            color: #C2185B; 
            margin: 1rem 0 0 0; 
            font-size: 1.3rem;
            font-weight: 300;
            opacity: 0.9;
        ">
            Data-Driven Decisions Through Advanced Performance Analytics
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 🎨 PINK THEME CSS
    if 'insights_pink_css_loaded' not in st.session_state:
        st.markdown("""
        <style>
        .insight-metric-card {
            background: linear-gradient(135deg, #FCE4EC 0%, #F8BBD0 100%);
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            color: #880E4F;
            box-shadow: 0 4px 16px rgba(194, 24, 91, 0.2);
            margin: 10px 0;
            min-height: 140px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            transition: all 0.3s ease;
            border-left: 4px solid #E91E63;
        }
        .insight-metric-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 24px rgba(194, 24, 91, 0.3);
        }
        .insight-metric-card .icon {
            font-size: 2.5em;
            margin-bottom: 8px;
            color: #C2185B;
        }
        .insight-metric-card .value {
            font-size: 1.8em;
            font-weight: 700;
            margin-bottom: 6px;
            color: #880E4F;
        }
        .insight-metric-card .label {
            font-size: 1em;
            font-weight: 600;
            color: #C2185B;
        }
        .performance-badge {
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.75em;
            font-weight: bold;
            margin-left: 6px;
        }
        .high-performance { background-color: #E91E63; color: white; }
        .medium-performance { background-color: #F48FB1; color: white; }
        .low-performance { background-color: #F8BBD0; color: #880E4F; }
        .insight-box {
            background: linear-gradient(135deg, #FCE4EC 0%, #F8BBD0 100%);
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #E91E63;
            margin-bottom: 12px;
            box-shadow: 0 2px 8px rgba(194, 24, 91, 0.1);
        }
        .insight-box h4 {
            color: #C2185B;
            margin-bottom: 8px;
            font-size: 1.1em;
        }
        .insight-box p {
            color: #AD1457;
            line-height: 1.6;
            margin: 0;
        }
        </style>
        """, unsafe_allow_html=True)
        st.session_state.insights_pink_css_loaded = True

    # 🚀 OPTIMIZED: Preprocess data for insights
    @st.cache_data(ttl=3600, show_spinner=False)
    def preprocess_insights_data(df):
        """🚀 OPTIMIZED: Vectorized data preprocessing"""
        df_clean = df.copy()
        
        # Vectorized numeric conversions
        df_clean['search_volume'] = pd.to_numeric(df_clean['Counts'], errors='coerce').fillna(0)
        df_clean['clicks'] = pd.to_numeric(df_clean['clicks'], errors='coerce').fillna(0)
        df_clean['conversions'] = pd.to_numeric(df_clean['conversions'], errors='coerce').fillna(0)
        
        # Handle CTR
        if 'Click Through Rate' in df_clean.columns:
            df_clean['ctr_calculated'] = pd.to_numeric(df_clean['Click Through Rate'], errors='coerce').fillna(0)
        else:
            df_clean['ctr_calculated'] = np.where(
                df_clean['search_volume'] > 0,
                (df_clean['clicks'] / df_clean['search_volume'] * 100),
                0
            )
        
        # Handle CR
        cr_col = 'Converion Rate' if 'Converion Rate' in df_clean.columns else 'Conversion Rate'
        if cr_col in df_clean.columns:
            df_clean['cr_calculated'] = pd.to_numeric(df_clean[cr_col], errors='coerce').fillna(0)
        else:
            df_clean['cr_calculated'] = np.where(
                df_clean['search_volume'] > 0,
                (df_clean['conversions'] / df_clean['search_volume'] * 100),
                0
            )
        
        # Clean brand column
        if 'Brand' in df_clean.columns:
            df_clean['brand'] = df_clean['Brand'].astype(str).replace(['nan', 'None', ''], 'Other').str.strip()
        else:
            df_clean['brand'] = 'Other'
        
        # Clean category columns
        for col, new_col in [('Category', 'category'), ('Sub Category', 'sub_category'), 
                             ('Department', 'department'), ('Class', 'class')]:
            if col in df_clean.columns:
                df_clean[new_col] = df_clean[col].astype(str).replace(['nan', 'None'], '').str.strip()
            else:
                df_clean[new_col] = ''
        
        # Handle underperforming flag
        if 'underperforming' in df_clean.columns:
            df_clean['underperforming'] = df_clean['underperforming'].astype(str).str.upper().isin(['TRUE', 'T', '1', 'YES'])
        else:
            df_clean['underperforming'] = False
        
        # Handle position
        if 'averageClickPosition' in df_clean.columns:
            df_clean['averageclickposition'] = pd.to_numeric(df_clean['averageClickPosition'], errors='coerce')
        
        # Handle date columns
        for date_col in ['start_date', 'end_date']:
            if date_col in df_clean.columns:
                df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors='coerce')
        
        return df_clean

    # Load preprocessed data
    try:
        df_insights = preprocess_insights_data(queries)
        st.sidebar.success(f"✅ Insights data loaded: {len(df_insights):,} rows")
    except Exception as e:
        st.error(f"⚠️ Data preprocessing error: {e}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()

    st.markdown("---")

    # Helper function for expandable questions
    def q_expand(title, explanation, render_fn, icon="💡"):
        with st.expander(f"{icon} {title}", expanded=False):
            st.markdown(f"<div class='insight-box'><h4>📌 Strategic Value</h4><p>{explanation}</p></div>", unsafe_allow_html=True)
            try:
                render_fn()
            except Exception as e:
                st.error(f"⚠️ Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    # ==================== TOP 11 STRATEGIC QUESTIONS ====================

    # ==================== Q1: Top 20 Search Queries by CTR and CR ====================
    def q1():
        """Top 20 search queries based on both CTR and CR performance"""
        filtered = df_insights[
            (df_insights['Brand'] != 'Other') & 
            (df_insights['Counts'] >= 200)
        ].copy()
        
        if len(filtered) > 0:
            filtered['ctr'] = (filtered['clicks'] / filtered['Counts'] * 100).fillna(0)
            filtered['cr'] = (filtered['conversions'] / filtered['Counts'] * 100).fillna(0)
            
            out = filtered.nlargest(20, ['ctr', 'cr'])[
                ['search', 'Brand', 'Counts', 'clicks', 'conversions', 'ctr', 'cr']
            ].copy()
            
            display_df = out.copy()
            display_df['Counts_fmt'] = display_df['Counts'].apply(lambda x: format_number(int(x)))
            display_df['clicks_fmt'] = display_df['clicks'].apply(lambda x: format_number(int(x)))
            display_df['conversions_fmt'] = display_df['conversions'].apply(lambda x: format_number(int(x)))
            display_df['ctr_fmt'] = display_df['ctr'].apply(format_percentage)
            display_df['cr_fmt'] = display_df['cr'].apply(format_percentage)
            
            display_df = display_df[['search', 'Brand', 'Counts_fmt', 'clicks_fmt', 'conversions_fmt', 
                                    'ctr_fmt', 'cr_fmt']]
            display_df.columns = ['Search Query', 'Brand', 'Search Volume', 'Clicks', 'Conversions', 
                                'CTR', 'CR']
            
            display_styled_table(df=display_df, align="center", scrollable=True, max_height="600px")
            
            st.download_button("📥 Download Data", out.to_csv(index=False), 
                            f"q1_top20_ctr_cr_{datetime.now().strftime('%Y%m%d')}.csv", 
                            "text/csv", key="q1_dl")
            
            fig = px.scatter(out, x='ctr', y='cr', size='Counts', color='cr',
                            hover_data=['search', 'Brand', 'clicks', 'conversions'],
                            title='Top 20 Search Queries: CTR vs CR Performance',
                            color_continuous_scale='PuRd', text='search')
            fig.update_traces(textposition='top center', textfont_size=8)
            fig.update_layout(xaxis_title="CTR (%)", yaxis_title="CR (%)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 No search queries found with volume >= 200")

    q_expand(
        "Q1 — 🏆 Top 20 Search Queries by CTR & CR Performance",
        "Identifies the best-performing search queries based on CTR and CR scores. **Filter: Search Volume >= 200, excludes generic items**. Focus optimization efforts on these high-performing queries.",
        q1, "🏆"
    )

    # ==================== Q2: Bottom 20 Search Queries by CTR and CR ====================
    def q2():
        """Bottom 20 search queries based on both CTR and CR performance"""
        filtered = df_insights[
            (df_insights['Brand'] != 'Other') & 
            (df_insights['Counts'] >= 200)
        ].copy()
        
        if len(filtered) > 0:
            filtered['ctr'] = (filtered['clicks'] / filtered['Counts'] * 100).fillna(0)
            filtered['cr'] = (filtered['conversions'] / filtered['Counts'] * 100).fillna(0)
            
            out = filtered.nsmallest(20, ['ctr', 'cr'])[
                ['search', 'Brand', 'Counts', 'clicks', 'conversions', 'ctr', 'cr']
            ].copy()
            
            display_df = out.copy()
            display_df['Counts_fmt'] = display_df['Counts'].apply(lambda x: format_number(int(x)))
            display_df['clicks_fmt'] = display_df['clicks'].apply(lambda x: format_number(int(x)))
            display_df['conversions_fmt'] = display_df['conversions'].apply(lambda x: format_number(int(x)))
            display_df['ctr_fmt'] = display_df['ctr'].apply(format_percentage)
            display_df['cr_fmt'] = display_df['cr'].apply(format_percentage)
            
            display_df = display_df[['search', 'Brand', 'Counts_fmt', 'clicks_fmt', 'conversions_fmt', 
                                    'ctr_fmt', 'cr_fmt']]
            display_df.columns = ['Search Query', 'Brand', 'Search Volume', 'Clicks', 'Conversions', 
                                'CTR', 'CR']
            
            display_styled_table(df=display_df, align="center", scrollable=True, max_height="600px")
            
            total_volume = out['Counts'].sum()
            st.warning(f"⚠️ **{len(out)} underperforming queries** with {format_number(int(total_volume))} total search volume need immediate attention!")
            
            st.download_button("📥 Download Data", out.to_csv(index=False), 
                            f"q2_bottom20_ctr_cr_{datetime.now().strftime('%Y%m%d')}.csv", 
                            "text/csv", key="q2_dl")
            
            fig = px.scatter(out, x='ctr', y='cr', size='Counts', color='cr',
                            hover_data=['search', 'Brand', 'clicks', 'conversions'],
                            title='Bottom 20 Search Queries: CTR vs CR Performance',
                            color_continuous_scale='Reds_r', text='search')
            fig.update_traces(textposition='top center', textfont_size=8)
            fig.update_layout(xaxis_title="CTR (%)", yaxis_title="CR (%)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 No search queries found with volume >= 200")

    q_expand(
        "Q2 — ⚠️ Bottom 20 Search Queries by CTR & CR Performance",
        "Identifies the worst-performing search queries. **Filter: Search Volume >= 200, excludes generic items**. These queries need immediate optimization: review product relevance, pricing, descriptions, and availability.",
        q2, "⚠️"
    )

    # ==================== Q3: Top 20 Search Queries by CR ====================
    def q3():
        """Top 20 search queries based on Conversion Rate (CR)"""
        filtered = df_insights[
            (df_insights['Brand'] != 'Other') & 
            (df_insights['Counts'] >= 200)
        ].copy()
        
        if len(filtered) > 0:
            filtered['cr'] = (filtered['conversions'] / filtered['Counts'] * 100).fillna(0)
            filtered['ctr'] = (filtered['clicks'] / filtered['Counts'] * 100).fillna(0)
            
            out = filtered.nlargest(20, 'cr')[
                ['search', 'Brand', 'Counts', 'clicks', 'conversions', 'ctr', 'cr']
            ].copy()
            
            display_df = out.copy()
            display_df['Counts_fmt'] = display_df['Counts'].apply(lambda x: format_number(int(x)))
            display_df['clicks_fmt'] = display_df['clicks'].apply(lambda x: format_number(int(x)))
            display_df['conversions_fmt'] = display_df['conversions'].apply(lambda x: format_number(int(x)))
            display_df['ctr_fmt'] = display_df['ctr'].apply(format_percentage)
            display_df['cr_fmt'] = display_df['cr'].apply(format_percentage)
            
            display_df = display_df[['search', 'Brand', 'Counts_fmt', 'clicks_fmt', 'conversions_fmt', 
                                    'ctr_fmt', 'cr_fmt']]
            display_df.columns = ['Search Query', 'Brand', 'Search Volume', 'Clicks', 'Conversions', 
                                'CTR', 'CR']
            
            display_styled_table(df=display_df, align="center", scrollable=True, max_height="600px")
            
            top_query = out.iloc[0]
            st.success(f"🎯 **Top Converting Query:** '{top_query['search']}' ({top_query['Brand']}) with {format_percentage(top_query['cr'])} CR and {format_number(int(top_query['conversions']))} conversions!")
            
            st.download_button("📥 Download Data", out.to_csv(index=False), 
                            f"q3_top20_cr_{datetime.now().strftime('%Y%m%d')}.csv", 
                            "text/csv", key="q3_dl")
            
            fig = px.bar(out, x='search', y='cr', color='cr',
                        title='Top 20 Search Queries by Conversion Rate',
                        color_continuous_scale='PuRd',
                        hover_data=['Brand', 'Counts', 'clicks', 'conversions'])
            fig.update_layout(xaxis_tickangle=-45, xaxis_title="Search Query", yaxis_title="CR (%)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 No search queries found with volume >= 200")

    q_expand(
        "Q3 — 🎯 Top 20 Search Queries by Conversion Rate (CR)",
        "Identifies search queries with the highest conversion rates. **Filter: Search Volume >= 200, excludes generic items**. These queries represent high purchase intent - prioritize them in SEO and PPC campaigns.",
        q3, "🎯"
    )

    # ==================== Q4: Top 20 Search Queries by CTR ====================
    def q4():
        """Top 20 search queries based on Click-Through Rate (CTR)"""
        filtered = df_insights[
            (df_insights['Brand'] != 'Other') & 
            (df_insights['Counts'] >= 200)
        ].copy()
        
        if len(filtered) > 0:
            filtered['ctr'] = (filtered['clicks'] / filtered['Counts'] * 100).fillna(0)
            filtered['cr'] = (filtered['conversions'] / filtered['Counts'] * 100).fillna(0)
            
            out = filtered.nlargest(20, 'ctr')[
                ['search', 'Brand', 'Counts', 'clicks', 'conversions', 'ctr', 'cr']
            ].copy()
            
            display_df = out.copy()
            display_df['Counts_fmt'] = display_df['Counts'].apply(lambda x: format_number(int(x)))
            display_df['clicks_fmt'] = display_df['clicks'].apply(lambda x: format_number(int(x)))
            display_df['conversions_fmt'] = display_df['conversions'].apply(lambda x: format_number(int(x)))
            display_df['ctr_fmt'] = display_df['ctr'].apply(format_percentage)
            display_df['cr_fmt'] = display_df['cr'].apply(format_percentage)
            
            display_df = display_df[['search', 'Brand', 'Counts_fmt', 'clicks_fmt', 'conversions_fmt', 
                                    'ctr_fmt', 'cr_fmt']]
            display_df.columns = ['Search Query', 'Brand', 'Search Volume', 'Clicks', 'Conversions', 
                                'CTR', 'CR']
            
            display_styled_table(df=display_df, align="center", scrollable=True, max_height="600px")
            
            top_query = out.iloc[0]
            st.success(f"👆 **Most Clicked Query:** '{top_query['search']}' ({top_query['Brand']}) with {format_percentage(top_query['ctr'])} CTR and {format_number(int(top_query['clicks']))} clicks!")
            
            st.download_button("📥 Download Data", out.to_csv(index=False), 
                            f"q4_top20_ctr_{datetime.now().strftime('%Y%m%d')}.csv", 
                            "text/csv", key="q4_dl")
            
            fig = px.bar(out, x='search', y='ctr', color='ctr',
                        title='Top 20 Search Queries by Click-Through Rate',
                        color_continuous_scale='RdPu',
                        hover_data=['Brand', 'Counts', 'clicks', 'conversions'])
            fig.update_layout(xaxis_tickangle=-45, xaxis_title="Search Query", yaxis_title="CTR (%)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 No search queries found with volume >= 200")

    q_expand(
        "Q4 — 👆 Top 20 Search Queries by Click-Through Rate (CTR)",
        "Identifies search queries with the highest click-through rates. **Filter: Search Volume >= 200, excludes generic items**. These queries have strong search result appeal - analyze what makes them attractive and replicate across other products.",
        q4, "👆"
    )

    # ==================== Q5: High Search Volume, Low CR ====================
    def q5():
        """High search volume but low conversion rate - optimization opportunities"""
        filtered = df_insights[
            (df_insights['Counts'] >= 200) &
            (df_insights['Brand'] != 'Other')
        ].copy()
        
        if len(filtered) > 0:
            threshold = filtered['Counts'].quantile(0.70)
            median_cr = filtered['cr_calculated'].median()
            
            opportunities = filtered[
                (filtered['Counts'] >= threshold) & 
                (filtered['cr_calculated'] < median_cr)
            ].copy()
            
            if len(opportunities) > 0:
                out = opportunities.nlargest(20, 'Counts')[
                    ['search', 'Brand', 'Counts', 'clicks', 'conversions', 'cr_calculated']
                ].copy()
                
                display_df = out.copy()
                display_df['Counts_fmt'] = display_df['Counts'].apply(lambda x: format_number(int(x)))
                display_df['clicks_fmt'] = display_df['clicks'].apply(lambda x: format_number(int(x)))
                display_df['conversions_fmt'] = display_df['conversions'].apply(lambda x: format_number(int(x)))
                display_df['cr_fmt'] = display_df['cr_calculated'].apply(format_percentage)
                
                display_df = display_df[['search', 'Brand', 'Counts_fmt', 'clicks_fmt', 'conversions_fmt', 'cr_fmt']]
                display_df.columns = ['Search Query', 'Brand', 'Search Volume', 'Clicks', 'Current Conversions', 'Current CR']
                
                display_styled_table(df=display_df, align="center", scrollable=True, max_height="600px")
                
                st.warning(f"💰 **{len(out)} high-traffic queries** with below-median CR need optimization!")
                
                st.download_button("📥 Download Data", out.to_csv(index=False), 
                                  f"q5_conversion_opportunities_{datetime.now().strftime('%Y%m%d')}.csv", 
                                  "text/csv", key="q5_dl")
                
                fig = px.bar(out.head(15), x='search', y='Counts', color='cr_calculated',
                            title='Top 15 Conversion Optimization Opportunities',
                            color_continuous_scale='Reds_r',
                            hover_data=['Brand', 'clicks', 'conversions'])
                fig.update_layout(xaxis_tickangle=-45, xaxis_title="Search Query", yaxis_title="Search Volume")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("📊 No high volume, low CR opportunities found")
        else:
            st.info("📊 No search queries found with volume >= 200")

    q_expand(
        "Q5 — 💰 High Search Volume, Low CR - Conversion Optimization",
        "Identifies high-traffic queries with below-median conversion rates. **Filter: Search Volume >= 200 (top 30%), excludes generic items**. Optimize product pages, pricing, reviews, and checkout flow to capture lost revenue.",
        q5, "💰"
    )

    # ==================== Q6: High CTR, Low CR ====================
    def q6():
        """High CTR but low CR - post-click experience problems"""
        filtered = df_insights[
            (df_insights['Counts'] >= 200) &
            (df_insights['Brand'] != 'Other')
        ].copy()
        
        if len(filtered) > 0:
            high_ctr = filtered['ctr_calculated'].quantile(0.70)
            low_cr = filtered['cr_calculated'].quantile(0.30)
            
            issues = filtered[
                (filtered['ctr_calculated'] >= high_ctr) & 
                (filtered['cr_calculated'] <= low_cr)
            ].copy()
            
            if len(issues) > 0:
                out = issues.nlargest(30, 'Counts')[
                    ['search', 'Brand', 'Counts', 'clicks', 'conversions', 'ctr_calculated', 'cr_calculated']
                ].copy()
                
                display_df = out.copy()
                display_df['Counts_fmt'] = display_df['Counts'].apply(lambda x: format_number(int(x)))
                display_df['clicks_fmt'] = display_df['clicks'].apply(lambda x: format_number(int(x)))
                display_df['conversions_fmt'] = display_df['conversions'].apply(lambda x: format_number(int(x)))
                display_df['ctr_fmt'] = display_df['ctr_calculated'].apply(format_percentage)
                display_df['cr_fmt'] = display_df['cr_calculated'].apply(format_percentage)
                
                display_df = display_df[['search', 'Brand', 'Counts_fmt', 'clicks_fmt', 'conversions_fmt', 
                                        'ctr_fmt', 'cr_fmt']]
                display_df.columns = ['Search Query', 'Brand', 'Search Volume', 'Clicks', 'Conversions', 
                                    'CTR', 'CR']
                
                display_styled_table(df=display_df, align="center", scrollable=True, max_height="600px")
                
                st.warning(f"⚠️ **{len(out)} queries** attract clicks but fail to convert. Post-click experience needs immediate improvement!")
                
                st.download_button("📥 Download Data", out.to_csv(index=False), 
                                  f"q6_experience_issues_{datetime.now().strftime('%Y%m%d')}.csv", 
                                  "text/csv", key="q6_dl")
                
                fig = px.scatter(out.head(20), x='ctr_calculated', y='cr_calculated', 
                                size='Counts', color='ctr_calculated',
                                hover_data=['search', 'Brand', 'clicks', 'conversions'],
                                title='High CTR, Low CR: Post-Click Experience Issues (Top 20)',
                                color_continuous_scale='PuRd', text='search')
                fig.update_traces(textposition='top center', textfont_size=8)
                fig.update_layout(xaxis_title="CTR (%)", yaxis_title="CR (%)")
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("""
                **💡 Common Post-Click Issues to Fix:**
                - ❌ Price mismatch between search result and product page
                - ❌ Out of stock or limited availability
                - ❌ Poor product images or descriptions
                - ❌ Lack of customer reviews or low ratings
                - ❌ Complicated checkout process
                - ❌ Slow page load times
                - ❌ Mobile experience issues
                - ❌ Shipping costs revealed too late
                """)
            else:
                st.info("📊 No high CTR, low CR issues found")
        else:
            st.info("📊 No search queries found with volume >= 200")

    q_expand(
        "Q6 — 🔍 High CTR, Low CR - Post-Click Experience Issues",
        "Search queries attracting clicks but failing to convert. **Filter: CTR >= 70th percentile, CR <= 30th percentile, Search Volume >= 200, excludes generic items**. Shows 30 examples of specific queries with post-click experience problems.",
        q6, "🔍"
    )

    # ==================== Q7: Brand Performance ====================
    def q7():
        """Branded vs Generic search intent comparison"""
        df_temp = df_insights.copy()
        
        if len(df_temp) > 0:
            df_temp['brand_type'] = df_temp['Brand'].apply(
                lambda x: 'Generic' if str(x).lower() == 'other' else 'Branded'
            )
            
            agg = df_temp.groupby('brand_type').agg({
                'Counts': 'sum',
                'clicks': 'sum',
                'conversions': 'sum'
            }).reset_index()
            
            agg['ctr'] = (agg['clicks'] / agg['Counts'] * 100).fillna(0)
            agg['cr'] = (agg['conversions'] / agg['Counts'] * 100).fillna(0)
            total_sv = agg['Counts'].sum()
            agg['search_share'] = (agg['Counts'] / total_sv * 100).fillna(0) if total_sv > 0 else 0
            
            out = agg.sort_values('Counts', ascending=False).copy()
            
            display_df = out.copy()
            display_df['Counts_fmt'] = display_df['Counts'].apply(lambda x: format_number(int(x)))
            display_df['clicks_fmt'] = display_df['clicks'].apply(lambda x: format_number(int(x)))
            display_df['conversions_fmt'] = display_df['conversions'].apply(lambda x: format_number(int(x)))
            display_df['ctr_fmt'] = display_df['ctr'].apply(format_percentage)
            display_df['cr_fmt'] = display_df['cr'].apply(format_percentage)
            display_df['search_share_fmt'] = display_df['search_share'].apply(format_percentage)
            
            display_df = display_df[['brand_type', 'Counts_fmt', 'clicks_fmt', 'conversions_fmt', 
                                    'ctr_fmt', 'cr_fmt', 'search_share_fmt']]
            display_df.columns = ['Brand Type', 'Search Volume', 'Clicks', 'Conversions', 
                                'CTR', 'CR', 'Search Share']
            
            display_styled_table(df=display_df, align="center", scrollable=True, max_height="600px")
            
            st.download_button("📥 Download Data", out.to_csv(index=False), 
                              f"q7_branded_vs_generic_{datetime.now().strftime('%Y%m%d')}.csv", 
                              "text/csv", key="q7_dl")
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Search Volume', x=out['brand_type'], y=out['Counts'],
                text=out['Counts'].apply(lambda x: format_number(int(x))),
                textposition='outside', marker_color='#E91E63'
            ))
            fig.add_trace(go.Bar(
                name='Clicks', x=out['brand_type'], y=out['clicks'],
                text=out['clicks'].apply(lambda x: format_number(int(x))),
                textposition='outside', marker_color='#F48FB1'
            ))
            fig.add_trace(go.Bar(
                name='Conversions', x=out['brand_type'], y=out['conversions'],
                text=out['conversions'].apply(lambda x: format_number(int(x))),
                textposition='outside', marker_color='#F06292'
            ))
            fig.update_layout(title='Branded vs Generic Performance', xaxis_title='Brand Type', 
                            yaxis_title='Count', barmode='group')
            st.plotly_chart(fig, use_container_width=True)
            
            fig2 = go.Figure(data=[go.Pie(
                labels=out['brand_type'], values=out['Counts'],
                textinfo='percent+label+value', textposition='inside',
                marker=dict(colors=['#E91E63', '#F48FB1'])
            )])
            fig2.update_layout(title='Search Volume Distribution: Branded vs Generic')
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("📊 No data found")

    q_expand(
        "Q7 — 🏷️ Brand Performance: Branded vs Generic Search Intent",
        "Compares branded vs generic search behavior. Balance SEO and brand marketing strategies accordingly. Numbers displayed on charts for clarity.",
        q7, "🏷️"
    )

    # ==================== Q8: Seasonal Trends ====================
    def q8():
        """Month-over-month performance trends"""
        if 'start_date' in df_insights.columns:
            filtered = df_insights[df_insights['start_date'].notna()].copy()
            
            if len(filtered) > 0:
                df_temp = filtered.copy()
                df_temp['month'] = pd.to_datetime(df_temp['start_date']).dt.to_period('M').astype(str)
                
                agg = df_temp.groupby('month').agg({
                    'Counts': 'sum',
                    'clicks': 'sum',
                    'conversions': 'sum'
                }).reset_index()
                
                agg['ctr'] = (agg['clicks'] / agg['Counts'] * 100).fillna(0)
                agg['cr'] = (agg['conversions'] / agg['Counts'] * 100).fillna(0)
                
                out = agg.sort_values('month').copy()
                
                display_df = out.copy()
                display_df['Counts_fmt'] = display_df['Counts'].apply(lambda x: format_number(int(x)))
                display_df['clicks_fmt'] = display_df['clicks'].apply(lambda x: format_number(int(x)))
                display_df['conversions_fmt'] = display_df['conversions'].apply(lambda x: format_number(int(x)))
                display_df['ctr_fmt'] = display_df['ctr'].apply(format_percentage)
                display_df['cr_fmt'] = display_df['cr'].apply(format_percentage)
                
                display_df = display_df[['month', 'Counts_fmt', 'clicks_fmt', 'conversions_fmt', 'ctr_fmt', 'cr_fmt']]
                display_df.columns = ['Month', 'Search Volume', 'Clicks', 'Conversions', 'CTR', 'CR']
                
                display_styled_table(df=display_df, align="center", scrollable=True, max_height="600px")
                
                st.download_button("📥 Download Data", out.to_csv(index=False), 
                                  f"q8_seasonal_trends_{datetime.now().strftime('%Y%m%d')}.csv", 
                                  "text/csv", key="q8_dl")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=out['month'], y=out['Counts'], mode='lines+markers', 
                                        name='Search Volume', line=dict(color='#E91E63', width=3)))
                fig.add_trace(go.Scatter(x=out['month'], y=out['clicks'], mode='lines+markers', 
                                        name='Clicks', line=dict(color='#F06292', width=3)))
                fig.add_trace(go.Scatter(x=out['month'], y=out['conversions'], mode='lines+markers', 
                                        name='Conversions', line=dict(color='#F48FB1', width=3)))
                fig.update_layout(title='Month-over-Month Performance Trends', 
                                xaxis_title='Month', yaxis_title='Count', hovermode='x unified')
                st.plotly_chart(fig, use_container_width=True)
                
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=out['month'], y=out['ctr'], mode='lines+markers', 
                                         name='CTR (%)', line=dict(color='#C2185B', width=3)))
                fig2.add_trace(go.Scatter(x=out['month'], y=out['cr'], mode='lines+markers', 
                                         name='CR (%)', line=dict(color='#AD1457', width=3)))
                fig2.update_layout(title='CTR & CR Trends Over Time', 
                                 xaxis_title='Month', yaxis_title='Percentage (%)', hovermode='x unified')
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("📊 No data found with valid dates")
        else:
            st.info("📊 Date column not available")

    q_expand(
        "Q8 — 📅 Seasonal Trends - Month-over-Month Performance",
        "Analyzes performance trends over time. Identify seasonal patterns, peak periods, and plan inventory/marketing campaigns accordingly.",
        q8, "📅"
    )

    # ==================== Q9: Brand Comparison ====================
    def q10():
        """Top brands comparison by key metrics"""
        filtered = df_insights[df_insights['Brand'] != 'Other'].copy()
        
        if len(filtered) > 0:
            agg = filtered.groupby('Brand').agg({
                'search': 'count',
                'Counts': 'sum',
                'clicks': 'sum',
                'conversions': 'sum'
            }).reset_index()
            
            agg.columns = ['Brand', '# Unique Queries', 'Total Search Volume', 'Total Clicks', 'Total Conversions']
            agg['Avg Search Volume'] = (agg['Total Search Volume'] / agg['# Unique Queries']).round(0)
            agg['CTR'] = (agg['Total Clicks'] / agg['Total Search Volume'] * 100).fillna(0)
            agg['CR'] = (agg['Total Conversions'] / agg['Total Search Volume'] * 100).fillna(0)
            
            out = agg.nlargest(20, 'Total Search Volume').copy()
            
            display_df = out.copy()
            display_df['# Unique Queries_fmt'] = display_df['# Unique Queries'].apply(lambda x: format_number(int(x)))
            display_df['Avg Search Volume_fmt'] = display_df['Avg Search Volume'].apply(lambda x: format_number(int(x)))
            display_df['Total Search Volume_fmt'] = display_df['Total Search Volume'].apply(lambda x: format_number(int(x)))
            display_df['Total Clicks_fmt'] = display_df['Total Clicks'].apply(lambda x: format_number(int(x)))
            display_df['Total Conversions_fmt'] = display_df['Total Conversions'].apply(lambda x: format_number(int(x)))
            display_df['CTR_fmt'] = display_df['CTR'].apply(format_percentage)
            display_df['CR_fmt'] = display_df['CR'].apply(format_percentage)
            
            display_df = display_df[['Brand', '# Unique Queries_fmt', 'Avg Search Volume_fmt', 'Total Search Volume_fmt', 
                                    'Total Clicks_fmt', 'Total Conversions_fmt', 'CTR_fmt', 'CR_fmt']]
            display_df.columns = ['Brand', '# Unique Queries', 'Avg Search Volume', 'Total Search Volume', 
                                'Total Clicks', 'Total Conversions', 'CTR', 'CR']
            
            display_styled_table(df=display_df, align="center", scrollable=True, max_height="600px")
            
            st.download_button("📥 Download Data", out.to_csv(index=False), 
                              f"q10_brand_comparison_{datetime.now().strftime('%Y%m%d')}.csv", 
                              "text/csv", key="q10_dl")
            
            fig = px.scatter(out, x='CTR', y='CR', size='Total Search Volume', color='Total Conversions',
                            hover_data=['Brand', '# Unique Queries', 'Total Clicks'],
                            title='Top 20 Brands: CTR vs CR Performance',
                            color_continuous_scale='PuRd', text='Brand')
            fig.update_traces(textposition='top center', textfont_size=9)
            fig.update_layout(xaxis_title="CTR (%)", yaxis_title="CR (%)")
            st.plotly_chart(fig, use_container_width=True)
            
            fig2 = px.bar(out.head(10), x='Brand', y='Total Conversions', color='CR',
                         title='Top 10 Brands by Total Conversions',
                         color_continuous_scale='PuRd')
            fig2.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("📊 No brand data found")

    q_expand(
        "Q9 — 🏅 Brand Comparison - Top Performers",
        "Compares top 20 brands by search volume, engagement, and conversion metrics. **Filter: Excludes generic items, sorted by Search Volume descending**. Identify which brands drive the most value and deserve increased investment.",
        q10, "🏅"
    )


# ----------------- Export / Downloads (PINK THEME) -----------------
with tab_export:
    st.header("⬇ Export & Save Dashboard")
    
    st.markdown("""
    <div style="background: linear-gradient(90deg, #E91E63 0%, #F48FB1 100%); 
                padding: 15px; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="color: white; margin: 0;">📸 Click Any Card to Auto-Print!</h3>
        <p style="color: white; margin: 5px 0 0 0;">Cards will switch tabs and open print dialog automatically!</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab_info = {
        "Overview": {"icon": "📊", "desc": "Key metrics, totals, and summary charts", "tab_index": 0},
        "Search Analysis": {"icon": "🔍", "desc": "Search queries analysis and insights", "tab_index": 1},
        "Brand Analysis": {"icon": "🏷️", "desc": "Brand performance and comparisons", "tab_index": 2},
        "Category Analysis": {"icon": "📂", "desc": "Category breakdown and trends", "tab_index": 3},
        "Subcategory Analysis": {"icon": "📋", "desc": "Detailed subcategory insights", "tab_index": 4},
        "Generic Type": {"icon": "🏷️", "desc": "Generic vs branded analysis", "tab_index": 5},
        "Time Analysis": {"icon": "📈", "desc": "Time-based trends and patterns", "tab_index": 6}
    }
    
    st.subheader("🎯 Click Card to Auto-Screenshot:")
    
    auto_print_js = """
    <script>
    function autoPrintTab(tabName, tabIndex) {
        const sidebar = document.querySelector('[data-testid="stSidebar"]');
        if (sidebar) sidebar.style.display = 'none';
        
        const tabs = document.querySelectorAll('[data-baseweb="tab"]');
        if (tabs[tabIndex]) {
            tabs[tabIndex].click();
            
            setTimeout(() => {
                const header = document.querySelector('[data-testid="stHeader"]');
                const toolbar = document.querySelector('[data-testid="stToolbar"]');
                
                if (header) header.style.display = 'none';
                if (toolbar) toolbar.style.display = 'none';
                
                window.print();
                alert(`✅ Print dialog opened for ${tabName} tab!\\n\\nTip: Choose "Save as PDF" in the print dialog.`);
            }, 2000);
        }
    }
    </script>
    """
    
    st.components.v1.html(auto_print_js, height=0)
    
    cols = st.columns(2)
    for i, (tab_name, info) in enumerate(tab_info.items()):
        with cols[i % 2]:
            if st.button(f"🖨️ Print {tab_name}", key=f"print_{tab_name.lower().replace(' ', '_')}"):
                st.components.v1.html(f"""
                <script>
                    setTimeout(() => {{
                        const sidebar = parent.document.querySelector('[data-testid="stSidebar"]');
                        const header = parent.document.querySelector('[data-testid="stHeader"]');
                        const toolbar = parent.document.querySelector('[data-testid="stToolbar"]');
                        
                        if (sidebar) sidebar.style.display = 'none';
                        if (header) header.style.display = 'none';
                        if (toolbar) toolbar.style.display = 'none';
                        
                        const tabs = parent.document.querySelectorAll('[data-baseweb="tab"]');
                        if (tabs[{info['tab_index']}]) {{
                            tabs[{info['tab_index']}].click();
                            setTimeout(() => {{ parent.window.print(); }}, 1500);
                        }}
                    }}, 100);
                </script>
                """, height=0)
                st.success(f"🎯 Switching to {tab_name} and opening print dialog...")
            
            st.markdown(f"""
            <div style="
                border: 2px solid #E91E63; 
                border-radius: 10px; 
                padding: 15px; 
                margin: 10px 0;
                background: linear-gradient(135deg, #FCE4EC 0%, #F8BBD0 100%);
            ">
                <h4 style="margin: 0 0 8px 0; color: #880E4F;">{info['icon']} {tab_name}</h4>
                <p style="margin: 5px 0; color: #C2185B; font-size: 14px;">{info['desc']}</p>
                <small style="color: #E91E63; font-weight: bold;">✨ One-click auto-print!</small>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("🔧 Alternative: Manual Method")
    
    with st.expander("📋 Manual Screenshot Steps (if auto-print doesn't work)"):
        st.markdown("""
        ### Step-by-Step Manual Process:
        
        1. **Click the tab** you want to screenshot (at the top)
        2. **Wait** for all data to load completely
        3. **Click the 3 dots (⋮)** in the top-right corner
        4. **Select "Print"** from the dropdown menu
        5. **In Print Dialog:**
           - Destination: **Save as PDF**
           - Layout: **Portrait** (or Landscape for wide charts)
           - More settings → **Background graphics: ✅ ON**
        6. **Click "Save"** and choose your location
        
        ✅ **Done!** Perfect PDF saved!
        """)
    
    st.markdown("---")
    st.subheader("📊 Export Raw Data")
    
    if 'queries' in locals() or 'queries' in globals():
        col1, col2, col3 = st.columns(3)
        
        with col1:
            try:
                csv_data = queries.to_csv(index=False)
                st.download_button(
                    label="📄 Download CSV",
                    data=csv_data,
                    file_name=f"beauty_care_queries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help="Download complete queries dataset as CSV file"
                )
            except Exception as e:
                st.error(f"CSV Export Error: {str(e)}")
        
        with col2:
            try:
                from io import BytesIO
                import pandas as pd
                
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    queries.to_excel(writer, sheet_name='Queries', index=False)
                    
                    if brand_summary is not None:
                        brand_summary.to_excel(writer, sheet_name='Brand Summary', index=False)
                    
                    if category_summary is not None:
                        category_summary.to_excel(writer, sheet_name='Category Summary', index=False)
                    
                    if subcategory_summary is not None:
                        subcategory_summary.to_excel(writer, sheet_name='Subcategory Summary', index=False)
                    
                    if generic_type is not None:
                        generic_type.to_excel(writer, sheet_name='Generic Type', index=False)
                    
                    if 'brand' in queries.columns:
                        analysis_summary = queries.groupby('brand').agg({
                            'clicks': 'sum',
                            'conversions': 'sum',
                            'Counts': 'sum',
                            'ctr': 'mean',
                            'cr': 'mean'
                        }).round(3)
                        analysis_summary.to_excel(writer, sheet_name='Analysis Summary')
                
                st.download_button(
                    label="📊 Download Excel",
                    data=buffer.getvalue(),
                    file_name=f"beauty_care_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="Download as Excel with all sheets and summaries"
                )
            except ImportError:
                st.info("📊 Excel export requires openpyxl package")
            except Exception as e:
                st.error(f"Excel Export Error: {str(e)}")
        
        with col3:
            try:
                json_data = queries.to_json(orient='records', date_format='iso')
                st.download_button(
                    label="🔧 Download JSON",
                    data=json_data,
                    file_name=f"beauty_care_queries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    help="Download as JSON for API integration"
                )
            except Exception as e:
                st.error(f"JSON Export Error: {str(e)}")
    
    st.markdown("---")
    st.info("""
    🌐 **Browser Compatibility:**
    - ✅ **Chrome/Edge**: Full auto-print support
    - ✅ **Firefox**: May require manual confirmation
    - ✅ **Safari**: May need manual steps
    
    💡 **Tip**: If auto-print doesn't work, use the manual method above!
    """)
    
    st.markdown("---")
    st.subheader("💡 Pro Tips")
    
    st.markdown("""
    ### 🎯 For Best Screenshot Quality:
    
    - **📱 Use Chrome/Edge** for best auto-print compatibility
    - **🖥️ Full Screen Mode** (F11) before printing
    - **📊 Wait for Charts** to fully load before printing
    - **🎨 Enable Background Graphics** in print settings
    - **📄 Choose A4/Letter size** for standard documents
    - **🔄 Landscape Mode** for wide charts and tables
    
    ### 📊 Data Export Tips:
    
    - **CSV**: Best for Excel analysis and pivot tables
    - **Excel**: Includes all sheets (queries, summaries, analysis)
    - **JSON**: Perfect for API integration and web apps
    
    ### 🚀 Quick Actions:
    
    1. **Screenshot All Tabs**: Click each print button in sequence
    2. **Batch Export**: Use the data export buttons for raw data
    3. **Share Reports**: Combine PDFs into a single presentation
    """)


# ----------------- Footer -----------------
st.markdown(f"""
<div class="footer">
✨ Beauty Care Search Analytics — Noureldeen Mohamed
</div>
""", unsafe_allow_html=True)
