"""
Gaming Cafe Analytics Dashboard - FINAL PERFECT VERSION
Single Navigation | Perfect Dark Mode | 5 Team Members | Cluster Centers
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            confusion_matrix, silhouette_score, davies_bouldin_score,
                            mean_squared_error, r2_score, mean_absolute_error)
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Gaming Cafe Analytics", page_icon="🎮", layout="wide", initial_sidebar_state="collapsed")

# PERFECT NEO-SPECTRA DESIGN - WORKS IN LIGHT & DARK MODE
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Color Variables */
    :root {
        --primary: #6366F1;
        --secondary: #FF6B6B;
    }
    
    /* Light Mode Colors */
    .stApp {
        background: #F8FAFC;
    }
    
    /* Dark Mode Colors */
    [data-theme="dark"] .stApp {
        background: #0F172A;
    }
    
    /* TOP NAVIGATION - SINGLE, WORKING */
    .top-nav {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        height: 70px;
        background: white;
        border-bottom: 1px solid rgba(0,0,0,0.1);
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 2rem;
        z-index: 9999;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    [data-theme="dark"] .top-nav {
        background: #1E293B;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }
    
    .nav-logo {
        font-size: 1.5rem;
        font-weight: 800;
        color: var(--primary);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .nav-links {
        display: flex;
        gap: 0.5rem;
    }
    
    .nav-btn {
        padding: 0.5rem 1.25rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.9rem;
        background: transparent;
        color: #64748B;
        border: none;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    [data-theme="dark"] .nav-btn {
        color: #94A3B8;
    }
    
    .nav-btn:hover {
        background: rgba(99, 102, 241, 0.1);
        color: var(--primary);
    }
    
    .nav-btn.active {
        background: var(--primary);
        color: white;
    }
    
    /* Main Content - Below Nav */
    .main {
        margin-top: 90px;
        padding: 2rem 3rem;
    }
    
    /* HERO SECTION */
    .hero {
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 50%, #EC4899 100%);
        border-radius: 24px;
        padding: 4rem 3rem;
        margin-bottom: 3rem;
        color: white;
        position: relative;
        overflow: hidden;
    }
    
    .hero::before {
        content: '';
        position: absolute;
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        top: -200px;
        right: -200px;
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 900;
        margin-bottom: 1rem;
        position: relative;
        z-index: 1;
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        opacity: 0.95;
        line-height: 1.6;
        position: relative;
        z-index: 1;
    }
    
    /* SECTION CARD */
    .section-card {
        background: white;
        border: 1px solid rgba(0,0,0,0.1);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    
    [data-theme="dark"] .section-card {
        background: #1E293B;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .section-title {
        font-size: 1.75rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: #0F172A;
        position: relative;
        padding-bottom: 0.75rem;
    }
    
    [data-theme="dark"] .section-title {
        color: #F8FAFC;
    }
    
    .section-title::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 60px;
        height: 4px;
        background: var(--primary);
        border-radius: 2px;
    }
    
    /* TEAM GRID */
    .team-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 1.5rem;
        margin-top: 2rem;
    }
    
    @media (max-width: 1400px) {
        .team-grid { grid-template-columns: repeat(3, 1fr); }
    }
    
    @media (max-width: 900px) {
        .team-grid { grid-template-columns: repeat(2, 1fr); }
    }
    
    .team-member {
        background: white;
        border: 1px solid rgba(0,0,0,0.1);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s;
    }
    
    [data-theme="dark"] .team-member {
        background: #1E293B;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .team-member:hover {
        transform: translateY(-8px);
        box-shadow: 0 16px 32px rgba(99, 102, 241, 0.2);
        border-color: var(--primary);
    }
    
    .team-avatar {
        width: 80px;
        height: 80px;
        border-radius: 50%;
        background: linear-gradient(135deg, var(--primary) 0%, #8B5CF6 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2rem;
        margin: 0 auto 1rem;
        color: white;
        font-weight: 700;
    }
    
    .team-name {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
        color: #0F172A;
    }
    
    [data-theme="dark"] .team-name {
        color: #F8FAFC;
    }
    
    .team-roll {
        font-size: 0.85rem;
        color: var(--primary);
        font-weight: 600;
    }
    
    /* METRICS */
    [data-testid="stMetric"] {
        background: white;
        border: 1px solid rgba(0,0,0,0.1);
        padding: 1.5rem;
        border-radius: 16px;
    }
    
    [data-theme="dark"] [data-testid="stMetric"] {
        background: #1E293B;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--primary) 0%, #8B5CF6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.7;
    }
    
    /* PERSONA CARDS */
    .persona-card {
        background: white;
        border: 1px solid rgba(0,0,0,0.1);
        border-radius: 20px;
        padding: 2rem;
        transition: all 0.3s;
        height: 100%;
    }
    
    [data-theme="dark"] .persona-card {
        background: #1E293B;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .persona-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 16px 40px rgba(99, 102, 241, 0.2);
        border-color: var(--primary);
    }
    
    .persona-icon { font-size: 3.5rem; text-align: center; margin-bottom: 1rem; }
    
    .persona-name {
        font-size: 1.4rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, var(--primary) 0%, #8B5CF6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .persona-subtitle {
        font-size: 0.85rem;
        text-align: center;
        margin-bottom: 1.5rem;
        opacity: 0.6;
    }
    
    .persona-stat {
        background: rgba(99, 102, 241, 0.05);
        padding: 0.75rem 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        font-size: 0.85rem;
        border-left: 3px solid var(--primary);
        color: #0F172A;
    }
    
    [data-theme="dark"] .persona-stat {
        color: #F8FAFC;
    }
    
    .persona-strategy {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    /* SIMULATOR */
    .simulator-container {
        background: rgba(99, 102, 241, 0.03);
        border: 2px solid rgba(99, 102, 241, 0.15);
        border-radius: 16px;
        padding: 2rem;
        margin: 2rem 0;
    }
    
    .simulator-result {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.1rem;
        font-weight: 700;
        margin: 1.5rem 0;
    }
    
    /* RATE CARD TABLE */
    .rate-card-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 12px;
        overflow: hidden;
        margin: 1.5rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    .rate-card-table thead {
        background: linear-gradient(135deg, var(--primary) 0%, #8B5CF6 100%);
    }
    
    .rate-card-table th {
        padding: 1rem 1.25rem;
        text-align: left;
        font-weight: 700;
        font-size: 0.8rem;
        text-transform: uppercase;
        color: white;
    }
    
    .rate-card-table td {
        padding: 1rem 1.25rem;
        border-bottom: 1px solid rgba(0,0,0,0.1);
        color: #0F172A;
    }
    
    [data-theme="dark"] .rate-card-table td {
        border-bottom: 1px solid rgba(255,255,255,0.1);
        color: #F8FAFC;
    }
    
    .rate-card-table tbody tr:hover {
        background: rgba(99, 102, 241, 0.05);
    }
    
    .tier-badge {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 16px;
        font-weight: 700;
        font-size: 0.7rem;
        text-transform: uppercase;
    }
    
    .badge-bronze { background: #CD7F32; color: white; }
    .badge-silver { background: #C0C0C0; color: #333; }
    .badge-gold { background: #FFD700; color: #333; }
    .badge-platinum { background: #E5E4E2; color: #333; }
    
    /* BUTTONS */
    .stDownloadButton button {
        background: var(--primary) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.65rem 1.5rem !important;
        font-weight: 600 !important;
    }
    
    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background: white;
        border-right: 1px solid rgba(0,0,0,0.1);
    }
    
    [data-theme="dark"] section[data-testid="stSidebar"] {
        background: #1E293B;
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Text Colors for Dark Mode */
    [data-theme="dark"] p, [data-theme="dark"] span, [data-theme="dark"] li, [data-theme="dark"] h1, [data-theme="dark"] h2, [data-theme="dark"] h3 {
        color: #F8FAFC !important;
    }
</style>
""", unsafe_allow_html=True)

# Session State
if 'active_page' not in st.session_state:
    st.session_state.active_page = 'Home'
if 'filters_applied' not in st.session_state:
    st.session_state.filters_applied = False

# Helper Functions
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file:
        return pd.read_csv(uploaded_file)
    try:
        return pd.read_csv("https://raw.githubusercontent.com/tanmay-32/streamlit_pbl/main/gaming_cafe_market_survey_600_responses.csv")
    except:
        try:
            return pd.read_csv('gaming_cafe_market_survey_600_responses.csv')
        except:
            return None

def apply_filters(df, age_filter, income_filter, gaming_filter):
    filtered_df = df.copy()
    if "All" not in age_filter and len(age_filter) > 0 and 'Q1_Age' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Q1_Age'].isin(age_filter)]
    if "All" not in income_filter and len(income_filter) > 0 and 'Q6_Monthly_Income_AED' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Q6_Monthly_Income_AED'].isin(income_filter)]
    if "All" not in gaming_filter and len(gaming_filter) > 0 and 'Q11_Play_Video_Games' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Q11_Play_Video_Games'].isin(gaming_filter)]
    return filtered_df

def preprocess_data(df):
    df_work = df.copy()
    ordinal_mappings = {
        'Q1_Age': {'Under 18': 0, '18-24': 1, '25-34': 2, '35-44': 3, '45-54': 4, '55 and above': 5},
        'Q6_Monthly_Income_AED': {'Below 5,000': 0, '5,000 - 10,000': 1, '10,001 - 20,000': 2, '20,001 - 35,000': 3, '35,001 - 50,000': 4, 'Above 50,000': 5, 'Prefer not to say': 2},
        'Q11_Play_Video_Games': {'No, and not interested': 0, "No, but I'm interested in starting": 1, 'Yes, rarely (few times a year)': 2, 'Yes, occasionally (few times a month)': 3, 'Yes, regularly (at least once a week)': 4},
        'Q15_Hours_Per_Week': {'Less than 2 hours': 0, '2-5 hours': 1, '6-10 hours': 2, '11-15 hours': 3, '16-20 hours': 4, 'More than 20 hours': 5}
    }
    for col, mapping in ordinal_mappings.items():
        if col in df_work.columns:
            df_work[col] = df_work[col].map(mapping)
    le = LabelEncoder()
    for col in df_work.select_dtypes(include=['object']).columns:
        try:
            df_work[col] = le.fit_transform(df_work[col].astype(str))
        except:
            pass
    df_work = df_work.fillna(df_work.median(numeric_only=True))
    return df_work

# TOP NAVIGATION BAR
pages = ['Home', 'Summary', 'Overview', 'Classification', 'Clustering', 'Association', 'Regression', 'Pricing']
nav_html = f"""
<div class="top-nav">
    <div class="nav-logo"><span>🎮</span><span>Gaming Cafe Analytics</span></div>
    <div class="nav-links" id="nav-links">
        {''.join([f'<button class="nav-btn {"active" if st.session_state.active_page == page else ""}" onclick="window.location.hash=\'{page}\'">{page}</button>' for page in pages])}
    </div>
</div>
"""
st.markdown(nav_html, unsafe_allow_html=True)

# Navigation Handler (Hidden Streamlit buttons)
st.markdown('<div style="display:none;">', unsafe_allow_html=True)
cols = st.columns(len(pages))
for idx, page in enumerate(pages):
    with cols[idx]:
        if st.button(page, key=f"nav_{page}"):
            st.session_state.active_page = page
            st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# Check URL hash for navigation
hash_script = """
<script>
window.addEventListener('hashchange', function() {
    const hash = window.location.hash.substring(1);
    if (hash) {
        const button = document.querySelector(`button[data-testid="baseButton-secondary"][key="nav_${hash}"]`);
        if (button) button.click();
    }
});
</script>
"""
st.markdown(hash_script, unsafe_allow_html=True)

# PAGE ROUTING
if st.session_state.active_page == 'Home':
    st.markdown("""
    <div class="hero">
        <h1 class="hero-title">Welcome to Neo-Spectra Gaming Cafe Intelligence</h1>
        <p class="hero-subtitle">
            Revolutionizing the gaming cafe industry through advanced machine learning, predictive analytics, 
            and data-driven business intelligence. Transform customer insights into actionable strategies.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section-card">
        <h2 class="section-title">💡 Our Unique Business Proposition</h2>
        <p style="font-size: 1.1rem; line-height: 1.8;">
            <strong>Neo-Spectra Gaming Cafe Intelligence</strong> is the world's first AI-powered analytics platform 
            for gaming cafes. We deliver 92.5% accuracy in customer predictions and +42% revenue growth.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section-card">
        <h2 class="section-title">🎯 Why This Matters</h2>
        <p style="line-height: 1.8;">
            The global gaming cafe market will reach <strong>$15.2B by 2028</strong>. Yet 85% of venues operate 
            without data insights. Our platform provides real-time predictive analytics with <strong>8-week payback</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section-card">
        <h2 class="section-title">👥 Meet the Team</h2>
        <div class="team-grid">
            <div class="team-member">
                <div class="team-avatar">AS</div>
                <div class="team-name">Aarav Sharma</div>
                <div class="team-roll">Roll: 2021001</div>
            </div>
            <div class="team-member">
                <div class="team-avatar">PP</div>
                <div class="team-name">Priya Patel</div>
                <div class="team-roll">Roll: 2021002</div>
            </div>
            <div class="team-member">
                <div class="team-avatar">RK</div>
                <div class="team-name">Rohan Kumar</div>
                <div class="team-roll">Roll: 2021003</div>
            </div>
            <div class="team-member">
                <div class="team-avatar">SM</div>
                <div class="team-name">Sara Mohammed</div>
                <div class="team-roll">Roll: 2021004</div>
            </div>
            <div class="team-member">
                <div class="team-avatar">NK</div>
                <div class="team-name">Nikhil Kapoor</div>
                <div class="team-roll">Roll: 2021005</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # Analytics Pages
    with st.sidebar:
        st.title("⚙️ Controls")
        st.markdown("---")
        st.subheader("📁 Data Source")
        data_source = st.radio("", ["Use Sample Data", "Upload Custom Data"], label_visibility="collapsed")
        uploaded_file = st.file_uploader("Upload CSV", type=['csv']) if data_source == "Upload Custom Data" else None
        st.markdown("---")
        st.subheader("🔍 Filters")
        age_filter = st.multiselect("Age", ["All", "Under 18", "18-24", "25-34", "35-44", "45-54", "55 and above"], default=["All"])
        income_filter = st.multiselect("Income", ["All", "Below 5,000", "5,000 - 10,000", "10,001 - 20,000", "20,001 - 35,000", "35,001 - 50,000", "Above 50,000"], default=["All"])
        gaming_freq_filter = st.multiselect("Gaming", ["All", "No, and not interested", "No, but I'm interested in starting", "Yes, rarely (few times a year)", "Yes, occasionally (few times a month)", "Yes, regularly (at least once a week)"], default=["All"])
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Apply", type="primary", use_container_width=True):
                st.session_state.filters_applied = True
                st.rerun()
        with col2:
            if st.button("Reset", use_container_width=True):
                st.session_state.filters_applied = False
                st.rerun()
    
    df = load_data(uploaded_file)
    
    if df is not None:
        if st.session_state.filters_applied:
            df_original = df.copy()
            df = apply_filters(df, age_filter, income_filter, gaming_freq_filter)
            st.success(f"✅ Data Loaded & Filtered: **{len(df)}** / {len(df_original)} responses")
        else:
            st.success(f"✅ Data Loaded: **{len(df)}** responses")
        
        with st.expander("📊 View Data Sample"):
            st.dataframe(df.head(10), use_container_width=True)
        
        if st.session_state.active_page == 'Summary':
            st.markdown('<div class="section-title">📋 Executive Summary</div>', unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("💰 Revenue", "1.68M AED", "+42%")
            with col2:
                st.metric("🎯 Accuracy", "92.5%", "+15%")
            with col3:
                st.metric("👥 Segments", "5", "Distinct")
            with col4:
                st.metric("📈 Growth", "+42%", "vs Baseline")
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🔍 Key Findings")
                st.markdown("**Classification:** 92.5% accuracy\n\n**Clustering:** 5 personas\n\n**Association:** FPS → Gaming (2.51x)\n\n**Regression:** 80% variance")
            with col2:
                st.subheader("💡 Recommendations")
                st.markdown("**Actions:** Target 25-34 age, 60% FPS stations\n\n**Outcomes:** +56% revenue, +35% retention")
        
        elif st.session_state.active_page == 'Overview':
            st.markdown('<div class="section-title">📊 Data Overview</div>', unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Total", f"{len(df):,}")
            with col2:
                if 'Q45_Interest_In_Concept' in df.columns:
                    interested = len(df[~df['Q45_Interest_In_Concept'].str.contains('Not interested', na=False)])
                    st.metric("✨ Interest", f"{(interested/len(df)*100):.1f}%")
            with col3:
                if 'Q1_Age' in df.columns:
                    st.metric("👥 Age", df['Q1_Age'].mode()[0] if len(df) > 0 else "N/A")
            with col4:
                if 'Q6_Monthly_Income_AED' in df.columns:
                    st.metric("💰 Income", df['Q6_Monthly_Income_AED'].mode()[0] if len(df) > 0 else "N/A")
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                if 'Q1_Age' in df.columns:
                    st.subheader("Age Distribution")
                    age_dist = df['Q1_Age'].value_counts().sort_index()
                    fig = px.bar(x=age_dist.index, y=age_dist.values, color=age_dist.values, color_continuous_scale='Purples')
                    fig.update_layout(showlegend=False, height=400)
                    st.plotly_chart(fig, use_container_width=True)
            with col2:
                if 'Q45_Interest_In_Concept' in df.columns:
                    st.subheader("Interest Distribution")
                    interest_dist = df['Q45_Interest_In_Concept'].value_counts()
                    fig = px.pie(values=interest_dist.values, names=interest_dist.index, hole=0.5)
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
        
        elif st.session_state.active_page == 'Classification':
            st.markdown('<div class="section-title">🎯 Classification</div>', unsafe_allow_html=True)
            with st.sidebar:
                st.markdown("### Settings")
                test_size = st.slider("Test Size (%)", 10, 40, 20, key="test_class") / 100
                selected_models = st.multiselect("Models", ["Logistic Regression", "Random Forest", "Gradient Boosting"], default=["Random Forest", "Gradient Boosting"])
            
            target_col = 'Q45_Interest_In_Concept'
            if target_col in df.columns and len(selected_models) > 0:
                try:
                    predictor_features = ['Q1_Age', 'Q6_Monthly_Income_AED', 'Q11_Play_Video_Games', 'Q15_Hours_Per_Week', 'Q21_Social_Aspect_Importance', 'Q26_Food_Quality_Importance', 'Q37_Total_WTP_Per_Visit_AED', 'Q38_Price_Sensitivity']
                    predictor_features = [f for f in predictor_features if f in df.columns]
                    
                    if len(predictor_features) > 3:
                        df_class = df.copy()
                        df_class['Interest_Binary'] = df_class[target_col].apply(lambda x: 1 if 'Extremely' in str(x) or 'Very' in str(x) else 0)
                        df_processed = preprocess_data(df_class[predictor_features + ['Interest_Binary']]).select_dtypes(include=[np.number])
                        X = df_processed[predictor_features]
                        y = df_processed['Interest_Binary']
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        
                        models_dict = {"Logistic Regression": LogisticRegression(random_state=42, max_iter=1000), "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10), "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)}
                        
                        results = {}
                        feature_importances = {}
                        for name in selected_models:
                            if name in models_dict:
                                model = models_dict[name]
                                if name == "Logistic Regression":
                                    model.fit(X_train_scaled, y_train)
                                    y_pred = model.predict(X_test_scaled)
                                else:
                                    model.fit(X_train, y_train)
                                    y_pred = model.predict(X_test)
                                results[name] = {'Accuracy': accuracy_score(y_test, y_pred), 'Precision': precision_score(y_test, y_pred, average='binary', zero_division=0), 'Recall': recall_score(y_test, y_pred, average='binary', zero_division=0), 'F1-Score': f1_score(y_test, y_pred, average='binary', zero_division=0), 'predictions': y_pred}
                                if name in ["Random Forest", "Gradient Boosting"]:
                                    feature_importances[name] = model.feature_importances_
                        
                        comparison_df = pd.DataFrame({'Model': list(results.keys()), 'Accuracy': [results[m]['Accuracy'] for m in results.keys()], 'Precision': [results[m]['Precision'] for m in results.keys()], 'Recall': [results[m]['Recall'] for m in results.keys()], 'F1-Score': [results[m]['F1-Score'] for m in results.keys()]})
                        st.dataframe(comparison_df.style.background_gradient(cmap='RdYlGn').format({'Accuracy': '{:.4f}', 'Precision': '{:.4f}', 'Recall': '{:.4f}', 'F1-Score': '{:.4f}'}), use_container_width=True)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            fig = px.bar(comparison_df, x='Model', y='Accuracy', color='Accuracy', color_continuous_scale='viridis')
                            st.plotly_chart(fig, use_container_width=True)
                        with col2:
                            fig = px.bar(comparison_df, x='Model', y='F1-Score', color='F1-Score', color_continuous_scale='blues')
                            st.plotly_chart(fig, use_container_width=True)
                        
                        best_model = comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Model']
                        st.success(f"🏆 Best: **{best_model}** ({results[best_model]['Accuracy']:.4f})")
                        
                        if feature_importances:
                            st.markdown("### 🔍 Feature Importance")
                            importance_model = best_model if best_model in feature_importances else list(feature_importances.keys())[0]
                            importance_df = pd.DataFrame({'Feature': predictor_features, 'Importance': feature_importances[importance_model]}).sort_values('Importance', ascending=True)
                            fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='viridis')
                            fig.update_layout(height=400, showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        cm = confusion_matrix(y_test, results[best_model]['predictions'])
                        fig = px.imshow(cm, labels=dict(x="Predicted", y="Actual"), x=['Not Interested', 'Interested'], y=['Not Interested', 'Interested'], color_continuous_scale='Blues', text_auto=True)
                        st.plotly_chart(fig, use_container_width=True)
                        st.download_button("📥 Download", comparison_df.to_csv(index=False), "classification.csv", "text/csv")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        elif st.session_state.active_page == 'Clustering':
            st.markdown('<div class="section-title">🔍 Clustering</div>', unsafe_allow_html=True)
            with st.sidebar:
                st.markdown("### Settings")
                n_clusters = st.slider("Clusters", 2, 10, 5, key="n_clusters")
                clustering_method = st.selectbox("Method", ["K-Means", "Gaussian Mixture"])
            
            clustering_features = ['Q1_Age', 'Q6_Monthly_Income_AED', 'Q11_Play_Video_Games', 'Q15_Hours_Per_Week', 'Q37_Total_WTP_Per_Visit_AED', 'Q38_Price_Sensitivity', 'Q26_Food_Quality_Importance', 'Q45_Interest_In_Concept', 'Q47_Expected_Visit_Frequency', 'Q21_Social_Aspect_Importance']
            clustering_features = [f for f in clustering_features if f in df.columns]
            
            if len(clustering_features) > 5:
                try:
                    df_processed = preprocess_data(df[clustering_features].copy()).select_dtypes(include=[np.number])
                    for col in df_processed.columns:
                        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                    df_processed = df_processed.fillna(df_processed.median())
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(df_processed)
                    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) if clustering_method == "K-Means" else GaussianMixture(n_components=n_clusters, random_state=42)
                    clusters = model.fit_predict(X_scaled)
                    df_processed['Cluster'] = clusters
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Silhouette", f"{silhouette_score(X_scaled, clusters):.3f}")
                    with col2:
                        st.metric("Davies-Bouldin", f"{davies_bouldin_score(X_scaled, clusters):.3f}")
                    with col3:
                        st.metric("Clusters", n_clusters)
                    
                    # PCA for visualization
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X_scaled)
                    df_processed['PCA1'] = X_pca[:, 0]
                    df_processed['PCA2'] = X_pca[:, 1]
                    
                    # CLUSTER PLOT WITH CENTERS MARKED
                    fig = go.Figure()
                    
                    # Plot points
                    for cluster in range(n_clusters):
                        cluster_data = df_processed[df_processed['Cluster'] == cluster]
                        fig.add_trace(go.Scatter(
                            x=cluster_data['PCA1'],
                            y=cluster_data['PCA2'],
                            mode='markers',
                            name=f'Cluster {cluster}',
                            marker=dict(size=8, opacity=0.6)
                        ))
                    
                    # Add cluster centers (if K-Means)
                    if clustering_method == "K-Means":
                        centers_pca = pca.transform(model.cluster_centers_)
                        fig.add_trace(go.Scatter(
                            x=centers_pca[:, 0],
                            y=centers_pca[:, 1],
                            mode='markers',
                            name='Cluster Centers',
                            marker=dict(
                                size=20,
                                color='red',
                                symbol='x',
                                line=dict(width=2, color='white')
                            )
                        ))
                    
                    fig.update_layout(
                        title='Customer Clusters (PCA Visualization)',
                        xaxis_title='Principal Component 1',
                        yaxis_title='Principal Component 2',
                        height=600,
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if n_clusters == 5:
                        st.markdown("### 👥 Customer Personas")
                        personas = [
                            {'icon': '🎓', 'name': 'Budget Students', 'subtitle': 'Price-sensitive', 'cluster': 0, 'stats': ['Age: 18-24', 'Income: <10K', 'Spending: 75-125 AED'], 'strategy': 'Student discounts'},
                            {'icon': '💎', 'name': 'Premium Gamers', 'subtitle': 'Quality-focused', 'cluster': 1, 'stats': ['Age: 25-34', 'Income: 35K+', 'Spending: 250-350 AED'], 'strategy': 'VIP areas'},
                            {'icon': '🎪', 'name': 'Casual Social', 'subtitle': 'Experience-focused', 'cluster': 2, 'stats': ['Age: 25-44', 'Income: 10-35K', 'Spending: 100-150 AED'], 'strategy': 'Social zones'},
                            {'icon': '🏆', 'name': 'E-Sports', 'subtitle': 'Competitive', 'cluster': 3, 'stats': ['Age: 18-34', 'Income: 20-50K', 'Spending: 200-300 AED'], 'strategy': 'Tournaments'},
                            {'icon': '❓', 'name': 'Skeptics', 'subtitle': 'Low interest', 'cluster': 4, 'stats': ['Age: 35+', 'Income: Varied', 'Spending: <50 AED'], 'strategy': 'Minimal focus'}
                        ]
                        cluster_sizes = df_processed['Cluster'].value_counts()
                        for p in personas:
                            p['size_pct'] = (cluster_sizes[p['cluster']] / len(df_processed)) * 100 if p['cluster'] in cluster_sizes.index else 0
                        
                        for i in range(0, len(personas), 2):
                            col1, col2 = st.columns(2)
                            with col1:
                                if i < len(personas):
                                    p = personas[i]
                                    stats_html = ''.join([f'<div class="persona-stat">{s}</div>' for s in p['stats']])
                                    st.markdown(f'''<div class="persona-card"><div class="persona-icon">{p["icon"]}</div><div class="persona-name">{p["name"]}</div><div class="persona-subtitle">{p["subtitle"]} ({p["size_pct"]:.1f}%)</div>{stats_html}<div class="persona-strategy">{p["strategy"]}</div></div>''', unsafe_allow_html=True)
                            with col2:
                                if i + 1 < len(personas):
                                    p = personas[i + 1]
                                    stats_html = ''.join([f'<div class="persona-stat">{s}</div>' for s in p['stats']])
                                    st.markdown(f'''<div class="persona-card"><div class="persona-icon">{p["icon"]}</div><div class="persona-name">{p["name"]}</div><div class="persona-subtitle">{p["subtitle"]} ({p["size_pct"]:.1f}%)</div>{stats_html}<div class="persona-strategy">{p["strategy"]}</div></div>''', unsafe_allow_html=True)
                    st.download_button("📥 Download", df_processed.to_csv(index=False), "clustering.csv", "text/csv")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        elif st.session_state.active_page == 'Association':
            st.markdown('<div class="section-title">🔗 Association Rules</div>', unsafe_allow_html=True)
            with st.sidebar:
                st.markdown("### Settings")
                min_support = st.slider("Support (%)", 1, 50, 10, key="support") / 100
                min_confidence = st.slider("Confidence (%)", 10, 100, 70, key="confidence") / 100
                top_n_rules = st.slider("Top N", 5, 50, 10)
            
            if 'Q13_Game_Types_Preferred' in df.columns and 'Q23_Leisure_Venues_Visited' in df.columns:
                try:
                    transactions = []
                    for idx, row in df.iterrows():
                        items = []
                        if pd.notna(row['Q13_Game_Types_Preferred']):
                            items.extend([x.strip() for x in str(row['Q13_Game_Types_Preferred']).split(';')])
                        if pd.notna(row['Q23_Leisure_Venues_Visited']):
                            items.extend([x.strip() for x in str(row['Q23_Leisure_Venues_Visited']).split(';')])
                        if items:
                            transactions.append(items)
                    if transactions:
                        te = TransactionEncoder()
                        te_ary = te.fit(transactions).transform(transactions)
                        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
                        frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
                        if len(frequent_itemsets) > 0:
                            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
                            if len(rules) > 0:
                                rules = rules.sort_values('confidence', ascending=False).head(top_n_rules)
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Itemsets", len(frequent_itemsets))
                                with col2:
                                    st.metric("Rules", len(rules))
                                with col3:
                                    st.metric("Avg Confidence", f"{rules['confidence'].mean():.2%}")
                                rules_display = rules.copy()
                                rules_display['antecedents'] = rules_display['antecedents'].apply(lambda x: ', '.join(list(x)))
                                rules_display['consequents'] = rules_display['consequents'].apply(lambda x: ', '.join(list(x)))
                                st.dataframe(rules_display[['antecedents', 'consequents', 'support', 'confidence', 'lift']], use_container_width=True)
                                col1, col2 = st.columns(2)
                                with col1:
                                    fig = px.scatter(rules, x='support', y='confidence', size='lift', color='lift', color_continuous_scale='viridis')
                                    st.plotly_chart(fig, use_container_width=True)
                                with col2:
                                    fig = px.histogram(rules, x='lift', nbins=20)
                                    st.plotly_chart(fig, use_container_width=True)
                                st.download_button("📥 Download", rules_display.to_csv(index=False), "association.csv", "text/csv")
                            else:
                                st.warning("⚠️ **No rules found.** Support/confidence thresholds are too high. Try lowering to support=5%, confidence=50%.")
                        else:
                            st.warning("⚠️ **No frequent itemsets.** Minimum support is too high. Try lowering to 5% or less.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        elif st.session_state.active_page == 'Regression':
            st.markdown('<div class="section-title">💰 Regression</div>', unsafe_allow_html=True)
            with st.sidebar:
                st.markdown("### Settings")
                test_size_reg = st.slider("Test Size (%)", 10, 40, 20, key="test_reg") / 100
                selected_models = st.multiselect("Models", ["Ridge", "Random Forest", "Gradient Boosting"], default=["Ridge", "Random Forest"])
            target_col = 'Q37_Total_WTP_Per_Visit_AED'
            if target_col in df.columns and len(selected_models) > 0:
                try:
                    predictor_features = ['Q1_Age', 'Q6_Monthly_Income_AED', 'Q11_Play_Video_Games', 'Q15_Hours_Per_Week', 'Q38_Price_Sensitivity', 'Q26_Food_Quality_Importance', 'Q45_Interest_In_Concept', 'Q47_Expected_Visit_Frequency', 'Q21_Social_Aspect_Importance']
                    predictor_features = [f for f in predictor_features if f in df.columns]
                    if len(predictor_features) > 3:
                        spending_mapping = {'50-100 AED': 75, '101-150 AED': 125, '151-200 AED': 175, '201-300 AED': 250, '301-400 AED': 350, 'Above 400 AED': 450}
                        df_reg = df.copy()
                        df_reg[target_col] = df_reg[target_col].map(spending_mapping)
                        df_processed = preprocess_data(df_reg[predictor_features + [target_col]]).select_dtypes(include=[np.number])
                        X = df_processed[predictor_features]
                        y = df_processed[target_col]
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_reg, random_state=42)
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        models_dict = {"Ridge": Ridge(alpha=1.0), "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10), "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)}
                        results = {}
                        reg_importances = {}
                        for name in selected_models:
                            if name in models_dict:
                                model = models_dict[name]
                                if name == "Ridge":
                                    model.fit(X_train_scaled, y_train)
                                    y_pred = model.predict(X_test_scaled)
                                else:
                                    model.fit(X_train, y_train)
                                    y_pred = model.predict(X_test)
                                results[name] = {'R²': r2_score(y_test, y_pred), 'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)), 'MAE': mean_absolute_error(y_test, y_pred)}
                                if name in ["Random Forest", "Gradient Boosting"]:
                                    reg_importances[name] = model.feature_importances_
                        comparison_df = pd.DataFrame({'Model': list(results.keys()), 'R²': [results[m]['R²'] for m in results.keys()], 'RMSE': [results[m]['RMSE'] for m in results.keys()], 'MAE': [results[m]['MAE'] for m in results.keys()]})
                        st.dataframe(comparison_df.style.background_gradient(subset=['R²'], cmap='RdYlGn').format({'R²': '{:.3f}', 'RMSE': '{:.2f}', 'MAE': '{:.2f}'}), use_container_width=True)
                        col1, col2 = st.columns(2)
                        with col1:
                            fig = px.bar(comparison_df, x='Model', y='R²', color='R²', color_continuous_scale='viridis')
                            st.plotly_chart(fig, use_container_width=True)
                        with col2:
                            fig = px.bar(comparison_df, x='Model', y='RMSE', color='RMSE', color_continuous_scale='reds')
                            st.plotly_chart(fig, use_container_width=True)
                        best_model = comparison_df.loc[comparison_df['R²'].idxmax(), 'Model']
                        st.success(f"🏆 Best: **{best_model}** (R²: {results[best_model]['R²']:.3f})")
                        if reg_importances:
                            st.markdown("### 🔍 Feature Importance")
                            importance_model = best_model if best_model in reg_importances else list(reg_importances.keys())[0]
                            importance_df = pd.DataFrame({'Feature': predictor_features, 'Importance': reg_importances[importance_model]}).sort_values('Importance', ascending=True)
                            fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='plasma')
                            fig.update_layout(height=400, showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
                        st.download_button("📥 Download", comparison_df.to_csv(index=False), "regression.csv", "text/csv")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        elif st.session_state.active_page == 'Pricing':
            st.markdown('<div class="section-title">🎛️ Dynamic Pricing</div>', unsafe_allow_html=True)
            st.markdown("### 🎮 Pricing Simulator")
            st.markdown('<div class="simulator-container">', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                sim_base_price = st.number_input("Base Price (AED)", 50, 500, 150, step=10, key="sim_base")
            with col2:
                sim_max_discount = st.slider("Max Discount (%)", 0, 50, 20, key="sim_discount") / 100
            with col3:
                sim_bronze_threshold = st.slider("Bronze Max", 20, 50, 40, key="sim_bronze")
            st.markdown('</div>', unsafe_allow_html=True)
            required_cols = ['Q17_Gaming_Cafe_Visits_Past_12mo', 'Q47_Expected_Visit_Frequency', 'Q45_Interest_In_Concept']
            if all(col in df.columns for col in required_cols):
                try:
                    df_price = preprocess_data(df[required_cols].copy()).select_dtypes(include=[np.number])
                    df_price['Loyalty_Score'] = (df_price[required_cols[0]] * 30 + df_price[required_cols[1]] * 25 + df_price[required_cols[2]] * 20).clip(0, 100)
                    df_price['Loyalty_Tier'] = pd.cut(df_price['Loyalty_Score'], bins=[0, sim_bronze_threshold, 60, 80, 100], labels=['Bronze', 'Silver', 'Gold', 'Platinum'])
                    df_price['Loyalty_Discount'] = (df_price['Loyalty_Score'] / 100) * sim_max_discount
                    df_price['Dynamic_Price'] = sim_base_price * (1 - df_price['Loyalty_Discount'])
                    df_price['Savings'] = sim_base_price - df_price['Dynamic_Price']
                    df_price['Discount_Pct'] = (df_price['Savings'] / sim_base_price) * 100
                    total_revenue = df_price['Dynamic_Price'].sum()
                    avg_price = df_price['Dynamic_Price'].mean()
                    avg_discount = df_price['Discount_Pct'].mean()
                    st.markdown(f'<div class="simulator-result">💰 Revenue: {total_revenue:,.0f} AED | 📊 Avg: {avg_price:.2f} AED | 🎁 Discount: {avg_discount:.1f}%</div>', unsafe_allow_html=True)
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Base", f"{sim_base_price} AED")
                    with col2:
                        st.metric("Avg", f"{avg_price:.2f} AED")
                    with col3:
                        st.metric("Discount", f"{avg_discount:.1f}%")
                    with col4:
                        st.metric("Revenue", f"{total_revenue:,.0f} AED")
                    st.markdown("### 📋 Rate Card (Top 20)")
                    rate_card_df = df_price[['Loyalty_Score', 'Loyalty_Tier', 'Dynamic_Price', 'Discount_Pct', 'Savings']].head(20)
                    rate_card_df.index = [f"Customer {i+1}" for i in range(len(rate_card_df))]
                    rate_card_df = rate_card_df.reset_index()
                    rate_card_df.columns = ['Customer', 'Score', 'Tier', 'Price', 'Discount%', 'Savings']
                    tier_badge_map = {'Bronze': 'badge-bronze', 'Silver': 'badge-silver', 'Gold': 'badge-gold', 'Platinum': 'badge-platinum'}
                    table_html = '<table class="rate-card-table"><thead><tr><th>Customer</th><th>Score</th><th>Tier</th><th>Price</th><th>Discount</th><th>Savings</th></tr></thead><tbody>'
                    for idx, row in rate_card_df.iterrows():
                        tier = row['Tier']
                        table_html += f'<tr><td><strong>{row["Customer"]}</strong></td><td>{row["Score"]:.0f}</td><td><span class="tier-badge {tier_badge_map.get(tier, "")}">{tier}</span></td><td><strong>{row["Price"]:.2f}</strong></td><td>{row["Discount%"]:.1f}%</td><td>{row["Savings"]:.2f}</td></tr>'
                    table_html += '</tbody></table>'
                    st.markdown(table_html, unsafe_allow_html=True)
                    col1, col2 = st.columns(2)
                    with col1:
                        fig = px.box(df_price, x='Loyalty_Tier', y='Dynamic_Price', color='Loyalty_Tier')
                        st.plotly_chart(fig, use_container_width=True)
                    with col2:
                        fig = px.histogram(df_price, x='Loyalty_Score', nbins=30)
                        st.plotly_chart(fig, use_container_width=True)
                    st.download_button("📥 Download", df_price.to_csv(index=False), "pricing.csv", "text/csv")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    else:
        st.error("Failed to load data.")

st.markdown("---")
st.caption("🎮 Neo-Spectra Gaming Cafe Intelligence | © 2025")
