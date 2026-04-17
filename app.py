import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="EduPredict AI | Student Risk Analyzer",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS — DARK PREMIUM THEME
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

:root {
    --bg-primary: #0A0E1A;
    --bg-secondary: #111827;
    --bg-card: #161D2F;
    --bg-card-hover: #1E2740;
    --accent-blue: #3B82F6;
    --accent-cyan: #06B6D4;
    --accent-green: #10B981;
    --accent-red: #EF4444;
    --accent-orange: #F59E0B;
    --accent-purple: #8B5CF6;
    --text-primary: #F1F5F9;
    --text-secondary: #94A3B8;
    --text-muted: #64748B;
    --border: rgba(255,255,255,0.08);
    --glow-blue: rgba(59,130,246,0.3);
    --glow-green: rgba(16,185,129,0.3);
    --glow-red: rgba(239,68,68,0.3);
}

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* Main container */
.main { background-color: var(--bg-primary) !important; }
.block-container { padding: 1.5rem 2rem !important; max-width: 1400px !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1321 0%, #111827 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem !important; }

/* Header */
.hero-header {
    background: linear-gradient(135deg, #0D1321 0%, #1a2540 50%, #0D1321 100%);
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -20%;
    width: 60%;
    height: 200%;
    background: radial-gradient(circle, rgba(59,130,246,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-header::after {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 50%;
    height: 200%;
    background: radial-gradient(circle, rgba(6,182,212,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #F1F5F9 30%, #3B82F6 60%, #06B6D4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.2;
}
.hero-subtitle {
    color: var(--text-secondary);
    font-size: 1.05rem;
    margin-top: 0.6rem;
    font-weight: 400;
}
.hero-badge {
    display: inline-block;
    background: rgba(59,130,246,0.15);
    border: 1px solid rgba(59,130,246,0.4);
    color: #60A5FA;
    padding: 0.3rem 0.9rem;
    border-radius: 50px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 1rem;
    font-family: 'DM Mono', monospace;
}

/* KPI Cards */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.5rem; }
.kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
}
.kpi-card:hover { border-color: rgba(59,130,246,0.3); background: var(--bg-card-hover); }
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    border-radius: 16px 16px 0 0;
}
.kpi-card.blue::before { background: linear-gradient(90deg, #3B82F6, #06B6D4); }
.kpi-card.green::before { background: linear-gradient(90deg, #10B981, #34D399); }
.kpi-card.red::before { background: linear-gradient(90deg, #EF4444, #F87171); }
.kpi-card.purple::before { background: linear-gradient(90deg, #8B5CF6, #A78BFA); }
.kpi-label { font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted); margin-bottom: 0.5rem; }
.kpi-value { font-size: 2.2rem; font-weight: 700; line-height: 1; margin-bottom: 0.3rem; font-family: 'DM Mono', monospace; }
.kpi-value.blue { color: #60A5FA; }
.kpi-value.green { color: #34D399; }
.kpi-value.red { color: #F87171; }
.kpi-value.purple { color: #A78BFA; }
.kpi-sub { font-size: 0.8rem; color: var(--text-secondary); }

/* Section headers */
.section-header {
    font-size: 1.2rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 1.5rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Prediction Result Cards */
.pred-safe {
    background: linear-gradient(135deg, rgba(16,185,129,0.1), rgba(6,182,212,0.05));
    border: 2px solid rgba(16,185,129,0.4);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.pred-safe::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at 50% 0%, rgba(16,185,129,0.12) 0%, transparent 60%);
}
.pred-risk {
    background: linear-gradient(135deg, rgba(239,68,68,0.1), rgba(245,158,11,0.05));
    border: 2px solid rgba(239,68,68,0.4);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.pred-risk::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at 50% 0%, rgba(239,68,68,0.12) 0%, transparent 60%);
}
.pred-title { font-size: 1.8rem; font-weight: 700; margin: 0.5rem 0; }
.pred-icon { font-size: 3rem; }
.pred-pct { font-size: 3.5rem; font-weight: 700; font-family: 'DM Mono', monospace; margin: 0.5rem 0; }
.pred-sub { color: var(--text-secondary); font-size: 0.9rem; }

/* Input styling */
.stSlider > div > div { background: var(--bg-card) !important; }
[data-testid="stSlider"] { padding: 0.5rem 0; }
.stNumberInput input, .stTextInput input, .stSelectbox > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-secondary) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(59,130,246,0.2) !important;
    color: #60A5FA !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #3B82F6, #06B6D4) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.75rem 2.5rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 0 20px rgba(59,130,246,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(59,130,246,0.5) !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1rem 1.2rem !important;
}

/* Dataframe */
[data-testid="stDataFrame"] { border-radius: 12px !important; overflow: hidden; }
.dataframe { background: var(--bg-card) !important; }

/* Alert boxes */
.risk-alert {
    background: rgba(239,68,68,0.1);
    border-left: 4px solid #EF4444;
    border-radius: 8px;
    padding: 1rem 1.5rem;
    margin: 0.5rem 0;
    color: #FCA5A5;
}
.safe-alert {
    background: rgba(16,185,129,0.1);
    border-left: 4px solid #10B981;
    border-radius: 8px;
    padding: 1rem 1.5rem;
    margin: 0.5rem 0;
    color: #6EE7B7;
}
.info-alert {
    background: rgba(59,130,246,0.1);
    border-left: 4px solid #3B82F6;
    border-radius: 8px;
    padding: 1rem 1.5rem;
    margin: 0.5rem 0;
    color: #93C5FD;
}
.warning-alert {
    background: rgba(245,158,11,0.1);
    border-left: 4px solid #F59E0B;
    border-radius: 8px;
    padding: 1rem 1.5rem;
    margin: 0.5rem 0;
    color: #FCD34D;
}

/* Gauge container */
.gauge-container {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1rem;
}

/* Sidebar */
.sidebar-section {
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
}
.sidebar-title {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    margin-bottom: 0.8rem;
}

/* Hide streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Plotly charts */
.js-plotly-plot .plotly { background: transparent !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-secondary); }
::-webkit-scrollbar-thumb { background: rgba(59,130,246,0.4); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(59,130,246,0.7); }

/* Status indicator */
.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #10B981;
    box-shadow: 0 0 8px rgba(16,185,129,0.8);
    margin-right: 6px;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* Grade badges */
.grade-badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 50px;
    font-size: 0.78rem;
    font-weight: 700;
    font-family: 'DM Mono', monospace;
}
.grade-S { background: rgba(16,185,129,0.2); color: #34D399; border: 1px solid rgba(16,185,129,0.4); }
.grade-A { background: rgba(59,130,246,0.2); color: #60A5FA; border: 1px solid rgba(59,130,246,0.4); }
.grade-B { background: rgba(139,92,246,0.2); color: #A78BFA; border: 1px solid rgba(139,92,246,0.4); }
.grade-F { background: rgba(239,68,68,0.2); color: #F87171; border: 1px solid rgba(239,68,68,0.4); }

/* Feature bar */
.feature-bar {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin: 0.4rem 0;
}
.feature-label { font-size: 0.8rem; color: var(--text-secondary); width: 180px; flex-shrink: 0; }
.feature-bar-bg { flex: 1; background: rgba(255,255,255,0.05); border-radius: 4px; height: 6px; overflow: hidden; }
.feature-bar-fill { height: 100%; border-radius: 4px; background: linear-gradient(90deg, #3B82F6, #06B6D4); }
.feature-pct { font-size: 0.8rem; color: var(--text-muted); font-family: 'DM Mono', monospace; width: 40px; text-align: right; }

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD DATA & MODEL
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_excel('R23_R24_structured.xlsx')
    df['Internals'] = pd.to_numeric(df['Internals'], errors='coerce').fillna(0)
    
    student_df = df.groupby('Htno').agg(
        avg_internal=('Internals', 'mean'),
        min_internal=('Internals', 'min'),
        max_internal=('Internals', 'max'),
        std_internal=('Internals', 'std'),
        num_subjects=('Internals', 'count'),
        zero_internals=('Internals', lambda x: (x == 0).sum()),
        low_internals=('Internals', lambda x: (x < 15).sum()),
        high_internals=('Internals', lambda x: (x >= 20).sum()),
        num_failed=('Grade', lambda x: (x == 'F').sum()),
        num_absent=('Grade', lambda x: (x == 'ABSENT').sum()),
        grades_list=('Grade', list),
        subjects_list=('Subject', list),
        internals_list=('Internals', list),
    ).reset_index()
    
    student_df['std_internal'] = student_df['std_internal'].fillna(0)
    student_df['fail_ratio'] = student_df['num_failed'] / student_df['num_subjects']
    student_df['zero_ratio'] = student_df['zero_internals'] / student_df['num_subjects']
    student_df['avg_internal'] = student_df['avg_internal'].round(2)
    student_df['AtRisk'] = (student_df['fail_ratio'] > 0.3).astype(int)
    
    return df, student_df

@st.cache_resource
def load_model():
    try:
        model = joblib.load('model.pkl')
        features = joblib.load('features.pkl')
        return model, features
    except:
        return None, None

@st.cache_data
def load_analytics():
    try:
        with open('analytics.json') as f:
            return json.load(f)
    except:
        return {}

df, student_df = load_data()
model, features = load_model()
analytics = load_analytics()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 1.5rem 0;">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🎓</div>
        <div style="font-size: 1.1rem; font-weight: 700; color: #F1F5F9;">EduPredict AI</div>
        <div style="font-size: 0.75rem; color: #64748B; margin-top: 0.2rem;">
            <span class='status-dot'></span>System Online
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    nav = st.radio(
        "Navigation",
        ["🏠 Dashboard", "🔮 Risk Predictor", "🔍 Student Lookup", "📊 Analytics", "📖 About"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    st.markdown("""
    <div class="sidebar-section">
        <div class="sidebar-title">📦 Dataset Info</div>
        <div style="font-size: 0.83rem; color: #94A3B8; line-height: 1.8;">
            <div>📁 R23 & R24 Batches</div>
            <div>👥 4,388 Students</div>
            <div>📚 249 Subjects</div>
            <div>📋 12,236 Records</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sidebar-section">
        <div class="sidebar-title">🤖 Model Info</div>
        <div style="font-size: 0.83rem; color: #94A3B8; line-height: 1.8;">
            <div>Algorithm: XGBoost</div>
            <div>Accuracy: 69.8%</div>
            <div>AUC-ROC: 76.5%</div>
            <div>Type: Binary Classifier</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE: DASHBOARD
# ─────────────────────────────────────────────
if nav == "🏠 Dashboard":
    
    # Hero Header
    st.markdown("""
    <div class="hero-header">
        <div class="hero-badge">🤖 AI-Powered Analytics Platform</div>
        <h1 class="hero-title">Student Risk Intelligence</h1>
        <p class="hero-subtitle">Identify at-risk students early. Predict academic outcomes. Enable proactive intervention.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # KPI Cards
    pass_r = analytics.get('pass_rate', 73.8)
    fail_r = analytics.get('fail_rate', 20.7)
    total_s = analytics.get('total_students', 4388)
    total_r = analytics.get('total_records', 12236)
    
    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card blue">
            <div class="kpi-label">Total Students</div>
            <div class="kpi-value blue">{total_s:,}</div>
            <div class="kpi-sub">R23 + R24 Batches</div>
        </div>
        <div class="kpi-card green">
            <div class="kpi-label">Overall Pass Rate</div>
            <div class="kpi-value green">{pass_r}%</div>
            <div class="kpi-sub">Across all subjects</div>
        </div>
        <div class="kpi-card red">
            <div class="kpi-label">Failure Rate</div>
            <div class="kpi-value red">{fail_r}%</div>
            <div class="kpi-sub">Needs intervention</div>
        </div>
        <div class="kpi-card purple">
            <div class="kpi-label">Total Records</div>
            <div class="kpi-value purple">{total_r:,}</div>
            <div class="kpi-sub">249 subjects tracked</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Charts row
    col1, col2 = st.columns([1.2, 0.8])
    
    with col1:
        st.markdown('<div class="section-header">📊 Grade Distribution</div>', unsafe_allow_html=True)
        
        grade_counts = analytics.get('grade_counts', {})
        grade_order = ['S', 'A', 'A+', 'B', 'C', 'D', 'E', 'F', 'ABSENT', 'COMPLE']
        grades = [g for g in grade_order if g in grade_counts]
        counts = [grade_counts[g] for g in grades]
        colors_map = {
            'S': '#10B981', 'A': '#34D399', 'A+': '#6EE7B7',
            'B': '#3B82F6', 'C': '#60A5FA',
            'D': '#F59E0B', 'E': '#FCD34D',
            'F': '#EF4444', 'ABSENT': '#6B7280', 'COMPLE': '#8B5CF6'
        }
        colors = [colors_map.get(g, '#64748B') for g in grades]
        
        fig = go.Figure(go.Bar(
            x=grades, y=counts,
            marker=dict(color=colors, line=dict(width=0)),
            hovertemplate='<b>Grade %{x}</b><br>Count: %{y:,}<extra></extra>',
            text=counts,
            textposition='outside',
            textfont=dict(color='#94A3B8', size=11),
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Space Grotesk', color='#94A3B8'),
            margin=dict(l=10, r=10, t=20, b=10),
            height=300,
            xaxis=dict(showgrid=False, showline=False, color='#64748B'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.04)', color='#64748B', showline=False),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    with col2:
        st.markdown('<div class="section-header">🥧 Pass vs Fail Split</div>', unsafe_allow_html=True)
        
        gc = analytics.get('grade_counts', {})
        excellent = gc.get('S',0) + gc.get('A',0) + gc.get('A+',0)
        good = gc.get('B',0) + gc.get('C',0)
        below = gc.get('D',0) + gc.get('E',0)
        fail = gc.get('F',0)
        absent = gc.get('ABSENT',0)
        other_cat = gc.get('COMPLE',0)
        
        fig2 = go.Figure(go.Pie(
            labels=['Excellent (S/A)', 'Good (B/C)', 'Below Avg (D/E)', 'Failed (F)', 'Absent', 'Other'],
            values=[excellent, good, below, fail, absent, other_cat],
            hole=0.55,
            marker=dict(colors=['#10B981','#3B82F6','#F59E0B','#EF4444','#6B7280','#8B5CF6'],
                        line=dict(color='rgba(0,0,0,0.3)', width=2)),
            textinfo='percent',
            textfont=dict(size=11, family='Space Grotesk'),
            hovertemplate='<b>%{label}</b><br>%{value:,} records (%{percent})<extra></extra>',
        ))
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Space Grotesk', color='#94A3B8'),
            margin=dict(l=10, r=10, t=20, b=10),
            height=300,
            legend=dict(font=dict(size=10), bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,0,0,0)'),
            annotations=[dict(text=f'<b>{pass_r}%</b><br>Pass', x=0.5, y=0.5, font=dict(size=14, color='#F1F5F9'), showarrow=False)]
        )
        st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
    
    # Internal marks distribution
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown('<div class="section-header">📈 Internal Marks Distribution</div>', unsafe_allow_html=True)
        
        internal_vals = df['Internals'].clip(0, 30)
        fig3 = go.Figure(go.Histogram(
            x=internal_vals,
            nbinsx=30,
            marker=dict(
                color='rgba(59,130,246,0.7)',
                line=dict(width=0)
            ),
            hovertemplate='Internal Score: %{x}<br>Count: %{y}<extra></extra>'
        ))
        fig3.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Space Grotesk', color='#94A3B8'),
            margin=dict(l=10, r=10, t=20, b=10),
            height=250,
            xaxis=dict(showgrid=False, showline=False, color='#64748B', title='Internal Marks'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.04)', color='#64748B', showline=False),
            showlegend=False,
        )
        st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})
    
    with col4:
        st.markdown('<div class="section-header">⚠️ Top Subjects by Failure Rate</div>', unsafe_allow_html=True)
        
        top_fail = analytics.get('top_fail_subjects', [])[:8]
        if top_fail:
            subjects_short = [s['Subject'][:25]+'...' if len(s['Subject'])>25 else s['Subject'] for s in top_fail]
            fail_rates = [s['fail_rate'] for s in top_fail]
            
            fig4 = go.Figure(go.Bar(
                y=subjects_short,
                x=fail_rates,
                orientation='h',
                marker=dict(
                    color=fail_rates,
                    colorscale=[[0,'rgba(239,68,68,0.3)'],[1,'rgba(239,68,68,0.9)']],
                    line=dict(width=0)
                ),
                hovertemplate='<b>%{y}</b><br>Fail Rate: %{x:.1f}%<extra></extra>',
                text=[f'{r}%' for r in fail_rates],
                textposition='outside',
                textfont=dict(color='#94A3B8', size=10),
            ))
            fig4.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Space Grotesk', color='#94A3B8', size=10),
                margin=dict(l=5, r=40, t=20, b=10),
                height=250,
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.04)', color='#64748B', showline=False),
                yaxis=dict(showgrid=False, color='#94A3B8', showline=False),
                showlegend=False,
            )
            st.plotly_chart(fig4, use_container_width=True, config={'displayModeBar': False})
    
    # Batch comparison
    st.markdown('<div class="section-header">🏫 Batch Comparison: R23 vs R24</div>', unsafe_allow_html=True)
    
    r23 = df[df['Subcode'].str.startswith('23', na=False)]
    r24 = df[df['Subcode'].str.startswith('24', na=False)]
    
    def get_grade_pcts(batch_df):
        gc = batch_df['Grade'].value_counts()
        total = len(batch_df)
        pass_grades = ['S','A','A+','B','C','D','E']
        return {
            'pass': round(batch_df['Grade'].isin(pass_grades).mean()*100, 1),
            'fail': round((batch_df['Grade']=='F').mean()*100, 1),
            'absent': round((batch_df['Grade']=='ABSENT').mean()*100, 1),
            'excellent': round(batch_df['Grade'].isin(['S','A','A+']).mean()*100, 1),
        }
    
    r23_stats = get_grade_pcts(r23)
    r24_stats = get_grade_pcts(r24)
    
    categories = ['Pass Rate', 'Excellence Rate', 'Fail Rate', 'Absent Rate']
    r23_vals = [r23_stats['pass'], r23_stats['excellent'], r23_stats['fail'], r23_stats['absent']]
    r24_vals = [r24_stats['pass'], r24_stats['excellent'], r24_stats['fail'], r24_stats['absent']]
    
    fig5 = go.Figure()
    fig5.add_trace(go.Bar(name='R23 Batch', x=categories, y=r23_vals,
                          marker_color='rgba(59,130,246,0.7)',
                          hovertemplate='R23 %{x}: %{y}%<extra></extra>',
                          text=[f'{v}%' for v in r23_vals], textposition='outside',
                          textfont=dict(color='#60A5FA', size=11)))
    fig5.add_trace(go.Bar(name='R24 Batch', x=categories, y=r24_vals,
                          marker_color='rgba(6,182,212,0.7)',
                          hovertemplate='R24 %{x}: %{y}%<extra></extra>',
                          text=[f'{v}%' for v in r24_vals], textposition='outside',
                          textfont=dict(color='#67E8F9', size=11)))
    fig5.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Space Grotesk', color='#94A3B8'),
        margin=dict(l=10, r=10, t=20, b=10),
        height=280,
        barmode='group',
        xaxis=dict(showgrid=False, showline=False, color='#64748B'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.04)', color='#64748B', showline=False),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=12)),
        bargap=0.2, bargroupgap=0.05,
    )
    st.plotly_chart(fig5, use_container_width=True, config={'displayModeBar': False})

# ─────────────────────────────────────────────
# PAGE: RISK PREDICTOR
# ─────────────────────────────────────────────
elif nav == "🔮 Risk Predictor":
    
    st.markdown("""
    <div class="hero-header">
        <div class="hero-badge">🔮 XGBoost Predictor</div>
        <h1 class="hero-title">Student Risk Assessment</h1>
        <p class="hero-subtitle">Enter student academic profile to predict risk of academic failure</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_form, col_result = st.columns([1, 1], gap="large")
    
    with col_form:
        st.markdown('<div class="section-header">📝 Student Academic Profile</div>', unsafe_allow_html=True)
        
        with st.container():
            avg_internal = st.slider(
                "Average Internal Marks",
                min_value=0.0, max_value=30.0, value=18.0, step=0.5,
                help="Average of all internal exam marks (0–30)"
            )
            
            col_a, col_b = st.columns(2)
            with col_a:
                min_internal = st.number_input("Minimum Internal", 0, 30, 10,
                                               help="Lowest internal mark across subjects")
            with col_b:
                max_internal = st.number_input("Maximum Internal", 0, 30, 26,
                                               help="Highest internal mark across subjects")
            
            std_internal = st.slider(
                "Marks Variability (Std Dev)",
                min_value=0.0, max_value=15.0, value=4.0, step=0.5,
                help="How much marks vary across subjects. Higher = more inconsistent"
            )
            
            col_c, col_d = st.columns(2)
            with col_c:
                num_subjects = st.number_input("Number of Subjects", 1, 20, 8)
            with col_d:
                zero_internals = st.number_input("Zero-Mark Subjects", 0, 20, 0,
                                                  help="Subjects where student scored 0 in internals")
            
            col_e, col_f = st.columns(2)
            with col_e:
                low_internals = st.number_input("Low-Score Subjects (<15)", 0, 20, 1)
            with col_f:
                high_internals = st.number_input("High-Score Subjects (≥20)", 0, 20, 4)
            
            zero_ratio = zero_internals / max(num_subjects, 1)
            
            st.markdown(f"""
            <div class="info-alert">
                📊 Computed: Zero-mark ratio = <b>{zero_ratio:.1%}</b> &nbsp;|&nbsp; 
                Coverage: {num_subjects} subjects
            </div>
            """, unsafe_allow_html=True)
        
        predict_btn = st.button("🔮 Generate Risk Assessment", use_container_width=True)
    
    with col_result:
        st.markdown('<div class="section-header">🎯 Prediction Result</div>', unsafe_allow_html=True)
        
        if predict_btn and model is not None:
            input_data = pd.DataFrame([{
                'avg_internal': avg_internal,
                'min_internal': min_internal,
                'max_internal': max_internal,
                'std_internal': std_internal,
                'num_subjects': num_subjects,
                'zero_internals': zero_internals,
                'low_internals': low_internals,
                'high_internals': high_internals,
                'zero_ratio': zero_ratio,
            }])
            
            prob = model.predict_proba(input_data[features])[0]
            risk_prob = prob[1]
            safe_prob = prob[0]
            prediction = model.predict(input_data[features])[0]
            
            if prediction == 0:
                risk_level = "LOW RISK" if risk_prob < 0.3 else "MODERATE RISK"
                color_class = "pred-safe"
                icon = "✅"
                color = "#10B981"
                pct_display = f"{safe_prob*100:.1f}%"
                pct_label = "Likelihood of Passing"
            else:
                risk_level = "HIGH RISK" if risk_prob > 0.6 else "ELEVATED RISK"
                color_class = "pred-risk"
                icon = "⚠️"
                color = "#EF4444"
                pct_display = f"{risk_prob*100:.1f}%"
                pct_label = "Risk of Failure"
            
            st.markdown(f"""
            <div class="{color_class}">
                <div class="pred-icon">{icon}</div>
                <div class="pred-title" style="color:{color}">{risk_level}</div>
                <div class="pred-pct" style="color:{color}">{pct_display}</div>
                <div class="pred-sub">{pct_label}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Probability gauge
            st.markdown("")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Risk Score", 'font': {'size': 16, 'color': '#94A3B8', 'family': 'Space Grotesk'}},
                number={'suffix': '%', 'font': {'size': 32, 'color': '#F1F5F9', 'family': 'DM Mono'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#64748B', 'tickfont': {'color': '#64748B'}},
                    'bar': {'color': '#EF4444' if risk_prob > 0.5 else '#10B981', 'thickness': 0.3},
                    'bgcolor': 'rgba(255,255,255,0.02)',
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(16,185,129,0.12)'},
                        {'range': [30, 60], 'color': 'rgba(245,158,11,0.12)'},
                        {'range': [60, 100], 'color': 'rgba(239,68,68,0.12)'}
                    ],
                    'threshold': {'line': {'color': 'white', 'width': 2}, 'value': 50}
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=220,
                margin=dict(l=20, r=20, t=30, b=10),
                font=dict(family='Space Grotesk', color='#94A3B8')
            )
            st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})
            
            # Recommendations
            st.markdown('<div class="section-header">💡 Recommendations</div>', unsafe_allow_html=True)
            
            if zero_internals > 0:
                st.markdown(f'<div class="risk-alert">🔴 <b>Critical:</b> {zero_internals} zero-mark subject(s). Immediate intervention needed — check attendance and engagement.</div>', unsafe_allow_html=True)
            if low_internals > num_subjects * 0.4:
                st.markdown(f'<div class="warning-alert">🟡 <b>Warning:</b> {low_internals} subjects below 15 marks. Academic support sessions recommended.</div>', unsafe_allow_html=True)
            if avg_internal < 15:
                st.markdown('<div class="warning-alert">🟡 <b>Warning:</b> Average internal marks below 15. Core concept revision required across subjects.</div>', unsafe_allow_html=True)
            if std_internal > 8:
                st.markdown('<div class="info-alert">🔵 <b>Note:</b> High variability in marks. Student may have subject-specific weaknesses — targeted tutoring advised.</div>', unsafe_allow_html=True)
            if prediction == 0:
                st.markdown('<div class="safe-alert">✅ <b>Good standing:</b> Student appears on track. Continue current performance and aim for excellence.</div>', unsafe_allow_html=True)
        
        elif predict_btn and model is None:
            st.error("Model not loaded. Please ensure model.pkl is available.")
        else:
            st.markdown("""
            <div style="background: rgba(255,255,255,0.02); border: 1px dashed rgba(255,255,255,0.1); 
                        border-radius: 16px; padding: 3rem; text-align: center; color: #64748B;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🔮</div>
                <div style="font-size: 1rem; font-weight: 500;">Fill in the student profile</div>
                <div style="font-size: 0.85rem; margin-top: 0.5rem;">and click <b style="color:#3B82F6">Generate Risk Assessment</b></div>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE: STUDENT LOOKUP
# ─────────────────────────────────────────────
elif nav == "🔍 Student Lookup":
    
    st.markdown("""
    <div class="hero-header">
        <div class="hero-badge">🔍 Individual Analysis</div>
        <h1 class="hero-title">Student Profile Lookup</h1>
        <p class="hero-subtitle">Search any student by Hall Ticket Number for detailed academic analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_search, col_quick = st.columns([2, 1])
    
    with col_search:
        htno_input = st.text_input("🔍 Enter Hall Ticket Number", 
                                   placeholder="e.g. 319, 23MC1A0501...",
                                   help="Enter the student's Hall Ticket Number")
    with col_quick:
        st.markdown("<br>", unsafe_allow_html=True)
        sample_htnos = student_df['Htno'].head(5).tolist()
        quick_select = st.selectbox("Or pick a sample:", ["Select..."] + [str(h) for h in sample_htnos])
        if quick_select != "Select...":
            htno_input = quick_select
    
    if htno_input:
        # Try to find student
        mask = student_df['Htno'].astype(str).str.strip() == str(htno_input).strip()
        
        if not mask.any():
            st.markdown(f"""
            <div class="risk-alert">
                ❌ Student with Hall Ticket <b>{htno_input}</b> not found in the database.
            </div>
            """, unsafe_allow_html=True)
        else:
            student = student_df[mask].iloc[0]
            raw_records = df[df['Htno'].astype(str).str.strip() == str(htno_input).strip()]
            
            # Header
            risk_tag = "⚠️ AT RISK" if student['AtRisk'] == 1 else "✅ ON TRACK"
            risk_color = "#EF4444" if student['AtRisk'] == 1 else "#10B981"
            
            st.markdown(f"""
            <div style="background: var(--bg-card); border: 1px solid var(--border); border-radius: 16px; 
                        padding: 1.5rem 2rem; margin: 1rem 0; display: flex; align-items: center; 
                        justify-content: space-between; flex-wrap: wrap; gap: 1rem;">
                <div>
                    <div style="font-size: 0.75rem; color: #64748B; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.3rem;">Student</div>
                    <div style="font-size: 1.8rem; font-weight: 700; color: #F1F5F9; font-family: 'DM Mono', monospace;">HTN: {htno_input}</div>
                    <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 0.3rem;">
                        {len(raw_records)} subjects enrolled
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 1.2rem; font-weight: 700; color: {risk_color}; 
                                background: rgba(255,255,255,0.05); padding: 0.5rem 1.5rem; 
                                border-radius: 50px; border: 1px solid {risk_color}40;">
                        {risk_tag}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Stats row
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Avg Internal", f"{student['avg_internal']:.1f}/30")
            c2.metric("Subjects Failed", int(student['num_failed']), 
                      delta=None if student['num_failed']==0 else f"{int(student['num_failed'])} needs attention",
                      delta_color="inverse")
            c3.metric("Fail Rate", f"{student['fail_ratio']*100:.0f}%")
            c4.metric("Zero Marks", int(student['zero_internals']))
            
            tab1, tab2, tab3 = st.tabs(["📋 Subject Records", "📊 Performance Chart", "🤖 ML Prediction"])
            
            with tab1:
                display_df = raw_records[['Subject', 'Internals', 'Grade']].copy()
                display_df['Subject'] = display_df['Subject'].str[:45]
                display_df['Status'] = display_df['Grade'].apply(
                    lambda g: '✅ Pass' if g in ['S','A','A+','B','C','D','E','COMPLE'] 
                             else ('❌ Fail' if g == 'F' else '⚠️ Absent')
                )
                st.dataframe(display_df, use_container_width=True, hide_index=True,
                             height=350)
            
            with tab2:
                internal_list = raw_records['Internals'].tolist()
                subjects_short = [str(s)[:20] for s in raw_records['Subject'].tolist()]
                grade_list = raw_records['Grade'].tolist()
                
                bar_colors = []
                for g in grade_list:
                    if g in ['S','A','A+']: bar_colors.append('#10B981')
                    elif g in ['B','C','D','E']: bar_colors.append('#3B82F6')
                    elif g == 'F': bar_colors.append('#EF4444')
                    else: bar_colors.append('#6B7280')
                
                fig_s = go.Figure()
                fig_s.add_trace(go.Bar(
                    y=subjects_short, x=internal_list,
                    orientation='h',
                    marker=dict(color=bar_colors, line=dict(width=0)),
                    hovertemplate='<b>%{y}</b><br>Internals: %{x}<extra></extra>',
                    text=grade_list,
                    textposition='outside',
                    textfont=dict(size=10, color='#94A3B8')
                ))
                fig_s.add_vline(x=15, line_dash='dash', line_color='rgba(245,158,11,0.6)', 
                                annotation_text='Min. required', 
                                annotation_font=dict(color='#F59E0B', size=10))
                fig_s.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family='Space Grotesk', color='#94A3B8', size=10),
                    margin=dict(l=5, r=60, t=20, b=10),
                    height=max(300, len(subjects_short)*25),
                    xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.04)', 
                               color='#64748B', range=[0, 32], title='Internal Marks'),
                    yaxis=dict(showgrid=False, color='#94A3B8', showline=False),
                    showlegend=False,
                )
                st.plotly_chart(fig_s, use_container_width=True, config={'displayModeBar': False})
            
            with tab3:
                if model is not None:
                    input_data = pd.DataFrame([{
                        'avg_internal': student['avg_internal'],
                        'min_internal': student['min_internal'],
                        'max_internal': student['max_internal'],
                        'std_internal': student['std_internal'],
                        'num_subjects': student['num_subjects'],
                        'zero_internals': student['zero_internals'],
                        'low_internals': student['low_internals'],
                        'high_internals': student['high_internals'],
                        'zero_ratio': student['zero_ratio'],
                    }])
                    
                    prob = model.predict_proba(input_data[features])[0]
                    risk_prob = prob[1]
                    
                    col_g, col_i = st.columns(2)
                    
                    with col_g:
                        fig_g2 = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=risk_prob * 100,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': f"Risk Score for HTN {htno_input}", 
                                   'font': {'size': 13, 'color': '#94A3B8'}},
                            number={'suffix': '%', 'font': {'size': 28, 'color': '#F1F5F9', 'family': 'DM Mono'}},
                            gauge={
                                'axis': {'range': [0, 100], 'tickcolor': '#64748B'},
                                'bar': {'color': '#EF4444' if risk_prob > 0.5 else '#10B981', 'thickness': 0.25},
                                'bgcolor': 'rgba(0,0,0,0)',
                                'borderwidth': 0,
                                'steps': [
                                    {'range': [0, 30], 'color': 'rgba(16,185,129,0.1)'},
                                    {'range': [30, 60], 'color': 'rgba(245,158,11,0.1)'},
                                    {'range': [60, 100], 'color': 'rgba(239,68,68,0.1)'}
                                ],
                            }
                        ))
                        fig_g2.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            height=220, margin=dict(l=20, r=20, t=40, b=10),
                            font=dict(family='Space Grotesk', color='#94A3B8')
                        )
                        st.plotly_chart(fig_g2, use_container_width=True, config={'displayModeBar': False})
                    
                    with col_i:
                        st.markdown("**📊 Key Metrics**")
                        metrics_data = {
                            'Avg Internal': f"{student['avg_internal']:.1f}",
                            'Min Internal': f"{int(student['min_internal'])}",
                            'Zero Marks': f"{int(student['zero_internals'])}",
                            'Low Subjects': f"{int(student['low_internals'])}",
                            'High Subjects': f"{int(student['high_internals'])}",
                            'Variability': f"{student['std_internal']:.2f}",
                        }
                        for k, v in metrics_data.items():
                            st.markdown(f"""
                            <div style="display:flex; justify-content:space-between; padding:0.35rem 0; 
                                        border-bottom: 1px solid rgba(255,255,255,0.04); font-size:0.85rem;">
                                <span style="color:#94A3B8">{k}</span>
                                <span style="color:#F1F5F9; font-family:'DM Mono',monospace; font-weight:600">{v}</span>
                            </div>
                            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE: ANALYTICS
# ─────────────────────────────────────────────
elif nav == "📊 Analytics":
    
    st.markdown("""
    <div class="hero-header">
        <div class="hero-badge">📊 Deep Analytics</div>
        <h1 class="hero-title">Academic Performance Analytics</h1>
        <p class="hero-subtitle">Comprehensive analysis of student performance across batches, departments and subjects</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab_a, tab_b, tab_c = st.tabs(["📈 Internal Marks Analysis", "🏆 Grade Patterns", "🔬 At-Risk Distribution"])
    
    with tab_a:
        col1, col2 = st.columns(2)
        
        with col1:
            # Box plot: internals vs grade
            grade_order_box = ['S', 'A', 'B', 'C', 'D', 'E', 'F']
            colors_box = {'S':'#10B981','A':'#34D399','B':'#3B82F6','C':'#60A5FA',
                          'D':'#F59E0B','E':'#FCD34D','F':'#EF4444'}
            
            fig_box = go.Figure()
            for g in grade_order_box:
                sub = df[df['Grade'] == g]['Internals']
                if len(sub) > 0:
                    fig_box.add_trace(go.Box(
                        y=sub, name=g, 
                        marker_color=colors_box.get(g,'#64748B'),
                        line_color=colors_box.get(g,'#64748B'),
                        fillcolor=f"rgba({','.join(str(int(int(colors_box.get(g,'#64748B').lstrip('#')[i:i+2],16)) for i in (0,2,4))},0.15)",
                        boxmean=True,
                        hovertemplate=f'Grade {g}<br>Marks: %{{y}}<extra></extra>'
                    ))
            fig_box.update_layout(
                title=dict(text='Internal Marks by Grade', font=dict(color='#F1F5F9', size=14), x=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Space Grotesk', color='#94A3B8'),
                margin=dict(l=10, r=10, t=40, b=10),
                height=350,
                xaxis=dict(showgrid=False, color='#64748B'),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.04)', color='#64748B'),
                showlegend=False,
            )
            st.plotly_chart(fig_box, use_container_width=True, config={'displayModeBar': False})
        
        with col2:
            # Scatter: avg_internal vs fail_ratio per student
            sample = student_df.sample(min(500, len(student_df)), random_state=42)
            
            fig_scat = go.Figure(go.Scatter(
                x=sample['avg_internal'],
                y=sample['fail_ratio'] * 100,
                mode='markers',
                marker=dict(
                    size=6,
                    color=sample['fail_ratio'],
                    colorscale=[[0,'rgba(16,185,129,0.7)'],[0.5,'rgba(245,158,11,0.7)'],[1,'rgba(239,68,68,0.7)']],
                    opacity=0.7,
                    line=dict(width=0)
                ),
                hovertemplate='Avg Internal: %{x:.1f}<br>Fail Rate: %{y:.1f}%<extra></extra>'
            ))
            fig_scat.update_layout(
                title=dict(text='Avg Internal vs Fail Rate (per student)', font=dict(color='#F1F5F9', size=14), x=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Space Grotesk', color='#94A3B8'),
                margin=dict(l=10, r=10, t=40, b=10),
                height=350,
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.04)', color='#64748B', title='Avg Internal Marks'),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.04)', color='#64748B', title='Fail Rate (%)'),
            )
            st.plotly_chart(fig_scat, use_container_width=True, config={'displayModeBar': False})
    
    with tab_b:
        # Grade heatmap - grade distribution per batch
        r23 = df[df['Subcode'].str.startswith('23', na=False)]
        r24 = df[df['Subcode'].str.startswith('24', na=False)]
        
        grade_order2 = ['S','A','A+','B','C','D','E','F','ABSENT']
        r23_pct = [(r23['Grade']==g).mean()*100 for g in grade_order2]
        r24_pct = [(r24['Grade']==g).mean()*100 for g in grade_order2]
        
        fig_heat = go.Figure()
        fig_heat.add_trace(go.Scatterpolar(
            r=r23_pct, theta=grade_order2, fill='toself',
            name='R23 Batch',
            line=dict(color='#3B82F6', width=2),
            fillcolor='rgba(59,130,246,0.15)',
            hovertemplate='Grade %{theta}: %{r:.1f}%<extra>R23</extra>'
        ))
        fig_heat.add_trace(go.Scatterpolar(
            r=r24_pct, theta=grade_order2, fill='toself',
            name='R24 Batch',
            line=dict(color='#06B6D4', width=2),
            fillcolor='rgba(6,182,212,0.15)',
            hovertemplate='Grade %{theta}: %{r:.1f}%<extra>R24</extra>'
        ))
        fig_heat.update_layout(
            title=dict(text='Grade Distribution Radar: R23 vs R24', font=dict(color='#F1F5F9', size=14), x=0),
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 35], color='#64748B', gridcolor='rgba(255,255,255,0.08)'),
                angularaxis=dict(color='#94A3B8', gridcolor='rgba(255,255,255,0.08)'),
                bgcolor='rgba(0,0,0,0)'
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Space Grotesk', color='#94A3B8'),
            margin=dict(l=30, r=30, t=50, b=30),
            height=400,
            showlegend=True,
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=12)),
        )
        st.plotly_chart(fig_heat, use_container_width=True, config={'displayModeBar': False})
    
    with tab_c:
        # Risk distribution among students
        col_r1, col_r2 = st.columns(2)
        
        with col_r1:
            risk_counts = student_df['AtRisk'].value_counts()
            fig_risk = go.Figure(go.Pie(
                labels=['Safe Students', 'At-Risk Students'],
                values=[risk_counts.get(0, 0), risk_counts.get(1, 0)],
                hole=0.6,
                marker=dict(colors=['#10B981','#EF4444'],
                            line=dict(color='rgba(0,0,0,0.4)', width=2)),
                textinfo='percent+value',
                textfont=dict(size=12),
                hovertemplate='<b>%{label}</b><br>%{value:,} students (%{percent})<extra></extra>',
            ))
            fig_risk.update_layout(
                title=dict(text='Student Risk Distribution', font=dict(color='#F1F5F9', size=14), x=0),
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Space Grotesk', color='#94A3B8'),
                margin=dict(l=10, r=10, t=40, b=10),
                height=350,
                annotations=[dict(text='<b>4,388</b><br>total', x=0.5, y=0.5, 
                                  font=dict(size=14, color='#F1F5F9'), showarrow=False)]
            )
            st.plotly_chart(fig_risk, use_container_width=True, config={'displayModeBar': False})
        
        with col_r2:
            # Fail count distribution
            fail_dist = student_df['num_failed'].value_counts().sort_index().head(12)
            fig_fail = go.Figure(go.Bar(
                x=fail_dist.index,
                y=fail_dist.values,
                marker=dict(
                    color=fail_dist.index,
                    colorscale=[[0,'rgba(16,185,129,0.7)'],[0.5,'rgba(245,158,11,0.7)'],[1,'rgba(239,68,68,0.7)']],
                    line=dict(width=0)
                ),
                hovertemplate='<b>%{x} failed subjects</b><br>%{y:,} students<extra></extra>',
                text=fail_dist.values,
                textposition='outside',
                textfont=dict(color='#94A3B8', size=10),
            ))
            fig_fail.update_layout(
                title=dict(text='How Many Subjects Did Students Fail?', font=dict(color='#F1F5F9', size=14), x=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Space Grotesk', color='#94A3B8'),
                margin=dict(l=10, r=10, t=40, b=10),
                height=350,
                xaxis=dict(showgrid=False, color='#64748B', title='Number of Failed Subjects', tickmode='linear'),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.04)', color='#64748B', title='Number of Students'),
            )
            st.plotly_chart(fig_fail, use_container_width=True, config={'displayModeBar': False})
        
        # Top at-risk students
        st.markdown('<div class="section-header">🆘 Students Requiring Immediate Intervention</div>', unsafe_allow_html=True)
        
        high_risk = student_df.nlargest(15, 'fail_ratio')[
            ['Htno', 'avg_internal', 'min_internal', 'num_failed', 'num_absent', 'fail_ratio', 'zero_internals']
        ].copy()
        high_risk['fail_rate_%'] = (high_risk['fail_ratio'] * 100).round(1)
        high_risk['avg_internal'] = high_risk['avg_internal'].round(1)
        high_risk['Risk Level'] = high_risk['fail_ratio'].apply(
            lambda x: '🔴 Critical' if x > 0.7 else ('🟠 High' if x > 0.5 else '🟡 Elevated')
        )
        display_hr = high_risk[['Htno','avg_internal','min_internal','num_failed','zero_internals','fail_rate_%','Risk Level']].copy()
        display_hr.columns = ['Hall Ticket', 'Avg Internal', 'Min Internal', 'Subjects Failed', 'Zero Marks', 'Fail Rate %', 'Risk Level']
        st.dataframe(display_hr, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────
# PAGE: ABOUT
# ─────────────────────────────────────────────
elif nav == "📖 About":
    
    st.markdown("""
    <div class="hero-header">
        <div class="hero-badge">📖 Project Documentation</div>
        <h1 class="hero-title">About EduPredict AI</h1>
        <p class="hero-subtitle">Smart Academic Early Warning System — Built with XGBoost + Streamlit</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: var(--bg-card); border: 1px solid var(--border); border-radius: 16px; padding: 2rem;">
            <div style="font-size: 1.1rem; font-weight: 700; color: #F1F5F9; margin-bottom: 1.2rem; 
                        padding-bottom: 0.7rem; border-bottom: 1px solid var(--border);">🎯 Project Goal</div>
            <p style="color: #94A3B8; font-size: 0.9rem; line-height: 1.8;">
                EduPredict AI is a machine learning-powered academic risk prediction system 
                designed to identify students at risk of failure <b style="color:#60A5FA">before</b> final exams, 
                enabling timely intervention by faculty and counselors.
            </p>
            <p style="color: #94A3B8; font-size: 0.9rem; line-height: 1.8; margin-top: 1rem;">
                Built using real institutional data from <b style="color:#60A5FA">4,388 students</b> across 
                <b style="color:#60A5FA">249 subjects</b> (R23 & R24 batches), covering CE, EEE, ME, ECE, CSE, 
                Data Science, and AI/ML departments.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: var(--bg-card); border: 1px solid var(--border); border-radius: 16px; padding: 2rem;">
            <div style="font-size: 1.1rem; font-weight: 700; color: #F1F5F9; margin-bottom: 1.2rem; 
                        padding-bottom: 0.7rem; border-bottom: 1px solid var(--border);">🔬 ML Pipeline</div>
            <div style="display: flex; flex-direction: column; gap: 0.8rem;">
                <div style="display:flex; align-items:center; gap:1rem;">
                    <div style="background:rgba(59,130,246,0.2); border-radius:8px; padding:0.5rem 0.8rem; font-size:0.85rem; font-weight:700; color:#60A5FA; min-width:130px; text-align:center;">Data Loading</div>
                    <div style="color:#94A3B8; font-size:0.85rem;">Excel → Pandas preprocessing</div>
                </div>
                <div style="display:flex; align-items:center; gap:1rem;">
                    <div style="background:rgba(6,182,212,0.2); border-radius:8px; padding:0.5rem 0.8rem; font-size:0.85rem; font-weight:700; color:#67E8F9; min-width:130px; text-align:center;">Feature Eng.</div>
                    <div style="color:#94A3B8; font-size:0.85rem;">9 aggregated student features</div>
                </div>
                <div style="display:flex; align-items:center; gap:1rem;">
                    <div style="background:rgba(16,185,129,0.2); border-radius:8px; padding:0.5rem 0.8rem; font-size:0.85rem; font-weight:700; color:#34D399; min-width:130px; text-align:center;">XGBoost</div>
                    <div style="color:#94A3B8; font-size:0.85rem;">150 estimators, depth=5, lr=0.1</div>
                </div>
                <div style="display:flex; align-items:center; gap:1rem;">
                    <div style="background:rgba(245,158,11,0.2); border-radius:8px; padding:0.5rem 0.8rem; font-size:0.85rem; font-weight:700; color:#FCD34D; min-width:130px; text-align:center;">Evaluation</div>
                    <div style="color:#94A3B8; font-size:0.85rem;">69.8% accuracy, 76.5% AUC-ROC</div>
                </div>
                <div style="display:flex; align-items:center; gap:1rem;">
                    <div style="background:rgba(139,92,246,0.2); border-radius:8px; padding:0.5rem 0.8rem; font-size:0.85rem; font-weight:700; color:#A78BFA; min-width:130px; text-align:center;">Deployment</div>
                    <div style="color:#94A3B8; font-size:0.85rem;">Streamlit Cloud (free tier)</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: var(--bg-card); border: 1px solid var(--border); border-radius: 16px; padding: 2rem;">
            <div style="font-size: 1.1rem; font-weight: 700; color: #F1F5F9; margin-bottom: 1.2rem; 
                        padding-bottom: 0.7rem; border-bottom: 1px solid var(--border);">📦 Features Used in Model</div>
            <div style="font-size: 0.85rem; color: #94A3B8; line-height: 1.6;">
        """, unsafe_allow_html=True)
        
        feature_info = [
            ("avg_internal", "Average internal marks across all subjects", 100),
            ("min_internal", "Minimum internal mark — catches worst subject", 90),
            ("std_internal", "Standard deviation — measures inconsistency", 72),
            ("zero_internals", "Count of zero-mark subjects (biggest risk signal)", 85),
            ("low_internals", "Subjects with marks < 15 (below threshold)", 78),
            ("high_internals", "Subjects with marks ≥ 20 (consistency indicator)", 65),
            ("num_subjects", "Total subjects enrolled", 55),
            ("max_internal", "Peak performance indicator", 48),
            ("zero_ratio", "Proportion of zero-mark subjects", 82),
        ]
        
        for feat, desc, imp in feature_info:
            st.markdown(f"""
            <div class="feature-bar">
                <div class="feature-label">{feat}</div>
                <div class="feature-bar-bg"><div class="feature-bar-fill" style="width:{imp}%"></div></div>
                <div class="feature-pct">{imp}%</div>
            </div>
            <div style="font-size:0.75rem; color:#64748B; margin-left:180px; margin-top:-0.2rem; margin-bottom:0.5rem;">{desc}</div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div></div>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: var(--bg-card); border: 1px solid var(--border); border-radius: 16px; padding: 2rem;">
            <div style="font-size: 1.1rem; font-weight: 700; color: #F1F5F9; margin-bottom: 1.2rem; 
                        padding-bottom: 0.7rem; border-bottom: 1px solid var(--border);">🚀 Tech Stack</div>
            <div style="display: flex; flex-wrap: wrap; gap: 0.6rem;">
                <span style="background:rgba(59,130,246,0.15); border:1px solid rgba(59,130,246,0.4); color:#60A5FA; padding:0.3rem 0.8rem; border-radius:50px; font-size:0.8rem; font-weight:600;">Python 3.11</span>
                <span style="background:rgba(16,185,129,0.15); border:1px solid rgba(16,185,129,0.4); color:#34D399; padding:0.3rem 0.8rem; border-radius:50px; font-size:0.8rem; font-weight:600;">Streamlit</span>
                <span style="background:rgba(245,158,11,0.15); border:1px solid rgba(245,158,11,0.4); color:#FCD34D; padding:0.3rem 0.8rem; border-radius:50px; font-size:0.8rem; font-weight:600;">XGBoost</span>
                <span style="background:rgba(139,92,246,0.15); border:1px solid rgba(139,92,246,0.4); color:#A78BFA; padding:0.3rem 0.8rem; border-radius:50px; font-size:0.8rem; font-weight:600;">Plotly</span>
                <span style="background:rgba(6,182,212,0.15); border:1px solid rgba(6,182,212,0.4); color:#67E8F9; padding:0.3rem 0.8rem; border-radius:50px; font-size:0.8rem; font-weight:600;">Pandas</span>
                <span style="background:rgba(239,68,68,0.15); border:1px solid rgba(239,68,68,0.4); color:#F87171; padding:0.3rem 0.8rem; border-radius:50px; font-size:0.8rem; font-weight:600;">Scikit-learn</span>
                <span style="background:rgba(251,146,60,0.15); border:1px solid rgba(251,146,60,0.4); color:#FCA5A5; padding:0.3rem 0.8rem; border-radius:50px; font-size:0.8rem; font-weight:600;">NumPy</span>
                <span style="background:rgba(16,185,129,0.15); border:1px solid rgba(16,185,129,0.4); color:#6EE7B7; padding:0.3rem 0.8rem; border-radius:50px; font-size:0.8rem; font-weight:600;">Joblib</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Deploy instructions
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">🚀 How to Deploy on Streamlit Cloud (FREE)</div>', unsafe_allow_html=True)
    
    steps = [
        ("1️⃣ Create GitHub Repo", "Go to github.com → New repository → Name it `student-predictor` → Add these files:\n`app.py`, `model.pkl`, `features.pkl`, `analytics.json`, `requirements.txt`, `R23_R24_structured.xlsx`"),
        ("2️⃣ Add requirements.txt", "Create a file with: `streamlit`, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `plotly`, `openpyxl`, `joblib`"),
        ("3️⃣ Sign in to Streamlit Cloud", "Visit share.streamlit.io → Sign in with your GitHub account"),
        ("4️⃣ Deploy App", "Click 'New App' → Select your repo → Set main file as `app.py` → Click Deploy!"),
        ("5️⃣ Share Your App", "Your app gets a free URL like: `https://yourname-student-predictor.streamlit.app` — share it on LinkedIn!"),
    ]
    
    for title, content in steps:
        st.markdown(f"""
        <div style="background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px; 
                    padding: 1.2rem 1.5rem; margin-bottom: 0.7rem; display: flex; gap: 1rem; align-items: flex-start;">
            <div style="font-size: 1.2rem; flex-shrink: 0; margin-top: 0.1rem;">{title.split(' ')[0]}</div>
            <div>
                <div style="font-weight: 600; color: #F1F5F9; margin-bottom: 0.3rem;">{' '.join(title.split(' ')[1:])}</div>
                <div style="color: #94A3B8; font-size: 0.87rem; line-height: 1.6;">{content}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

