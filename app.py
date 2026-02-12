import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu
import warnings
from auth import check_authentication, logout, init_session_state
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Health AI Recommender",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize authentication
init_session_state()

# Check authentication - redirect to login if not authenticated
if not check_authentication():
    st.switch_page("pages/0_🔐_Login.py")

# Custom CSS (keep your existing CSS)
st.markdown("""
<style>
    /* Your existing CSS here */
</style>
""", unsafe_allow_html=True)

# Initialize session state (keep your existing code)
# In app.py, update the user_data initialization
if 'user_data' not in st.session_state:
    st.session_state.user_data = {
        'age': st.session_state.user.get('age', 25),
        'gender': st.session_state.user.get('gender', 'Male'),
        'medical_history': st.session_state.user.get('medical_history', []),
        'medicine_preferences': [],
        'symptoms': [],
        'height_cm': st.session_state.user.get('height_cm'),
        'weight_kg': st.session_state.user.get('weight_kg'),
        'bmi': st.session_state.user.get('bmi'),
        'bmi_category': st.session_state.user.get('bmi_category')
    }

if 'drug_data' not in st.session_state:
    try:
        st.session_state.drug_data = pd.read_csv('Data/Drug.csv')
    except:
        st.session_state.drug_data = pd.DataFrame()

# Update sidebar to include user profile
with st.sidebar:
    # User profile section
    if st.session_state.user:
        st.markdown(f"""
        <div style="background-color: #3A3A3A; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="
                    width: 40px;
                    height: 40px;
                    border-radius: 50%;
                    background: linear-gradient(135deg, #FF6B35, #FF8B35);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: white;
                    font-weight: bold;
                    font-size: 1.2rem;
                ">
                    {st.session_state.user['full_name'][0] if st.session_state.user.get('full_name') else 'U'}
                </div>
                <div>
                    <div style="color: white; font-weight: bold;">
                        {st.session_state.user.get('full_name', 'User')}
                    </div>
                    <div style="color: #AAAAAA; font-size: 0.8rem;">
                        @{st.session_state.user['username']}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Logout button
    if st.button("🚪 Logout", use_container_width=True):
        logout()
        st.rerun()
    
# Navigation sidebar
with st.sidebar:
    st.markdown("<h1 style='color: #FF6B35; text-align: center;'>🏥 Health AI</h1>", unsafe_allow_html=True)
    
    selected = option_menu(
        menu_title="Navigation",
        options=["Dashboard", "Disease Prediction", "Medicine Recommendation", "Analytics", "Admin"],
        icons=["house", "activity", "capsule", "graph-up", "gear"],
        menu_icon="menu-app",
        default_index=0,
        styles={
            "container": {"padding": "5!important", "background-color": "#1A1A1A"},
            "icon": {"color": "#FF6B35", "font-size": "20px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#3A3A3A",
                "color": "white"
            },
            "nav-link-selected": {"background-color": "#FF6B35", "color": "#1A1A1A"},
        }
    )
    
    # User profile section
    st.markdown("---")
    st.markdown("<h3 style='color: #FF6B35;'>👤 User Profile</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.user_data['age'] = st.number_input(
            "Age", 
            min_value=1, 
            max_value=100, 
            value=st.session_state.user_data['age'],
            key="age_input"
        )
    
    with col2:
        st.session_state.user_data['gender'] = st.selectbox(
            "Gender",
            ["Male", "Female", "Other"],
            index=0 if st.session_state.user_data['gender'] == "Male" else 1,
            key="gender_input"
        )
    
    st.markdown("---")
    
    # Quick stats
    st.markdown("<h3 style='color: #FF6B35;'>📊 Quick Stats</h3>", unsafe_allow_html=True)
    
    if not st.session_state.drug_data.empty:
        total_drugs = len(st.session_state.drug_data)
        total_diseases = st.session_state.drug_data['Disease'].nunique()
        avg_age = st.session_state.drug_data['Age'].mean()
        
        st.metric("Total Drugs", f"{total_drugs:,}")
        st.metric("Diseases Covered", total_diseases)
        st.metric("Average Age", f"{avg_age:.1f}")

# Main content based on navigation
if selected == "Dashboard":
    st.markdown("<h1 class='main-header'>🏥 Health AI Dashboard</h1>", unsafe_allow_html=True)
    
    # Welcome message
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div class='card'>
            <h3 style='color: #FF6B35;'>Welcome Back!</h3>
            <p>Age: {st.session_state.user_data['age']} | Gender: {st.session_state.user_data['gender']}</p>
            <p>Get personalized healthcare recommendations based on your profile and symptoms.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick actions
    st.markdown("<h3 class='sub-header'>🚀 Quick Actions</h3>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🩺 Predict Disease", use_container_width=True):
            st.switch_page("pages/2_Disease_Prediction.py")
    
    with col2:
        if st.button("💊 Get Medicine", use_container_width=True):
            st.switch_page("pages/3_Medicine_Recommendation.py")
    
    with col3:
        if st.button("📈 View Analytics", use_container_width=True):
            st.switch_page("pages/4_Analytics.py")
    
    with col4:
        if st.button("⚙️ Admin Panel", use_container_width=True):
            st.switch_page("pages/5_Admin.py")
    
    # Stats overview
    st.markdown("<h3 class='sub-header'>📊 System Overview</h3>", unsafe_allow_html=True)
    
    if not st.session_state.drug_data.empty:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class='metric-card'>
                <div class='metric-value'>2,500+</div>
                <div class='metric-label'>Medicines in Database</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='metric-card'>
                <div class='metric-value'>3</div>
                <div class='metric-label'>Diseases Covered</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class='metric-card'>
                <div class='metric-value'>95%</div>
                <div class='metric-label'>Prediction Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class='metric-card'>
                <div class='metric-value'>24/7</div>
                <div class='metric-label'>Availability</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Disease distribution chart
        st.markdown("<h3 class='sub-header'>📈 Disease Distribution</h3>", unsafe_allow_html=True)
        
        disease_counts = st.session_state.drug_data['Disease'].value_counts()
        
        fig = go.Figure(data=[
            go.Pie(
                labels=disease_counts.index,
                values=disease_counts.values,
                hole=.3,
                marker=dict(colors=['#FF6B35', '#FF8B35', '#FFA935'])
            )
        ])
        
        fig.update_layout(
            plot_bgcolor='#1A1A1A',
            paper_bgcolor='#1A1A1A',
            font_color='white',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent recommendations preview
        st.markdown("<h3 class='sub-header'>💡 Recent Insights</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class='card'>
                <h4>🔬 Most Common Prescriptions</h4>
                <ul>
                    <li>Acne: Benzoyl Peroxide based gels</li>
                    <li>Allergy: Levocetirizine tablets</li>
                    <li>Diabetes: Metformin formulations</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='card'>
                <h4>📋 System Recommendations</h4>
                <ul>
                    <li>Update your medical history regularly</li>
                    <li>Consult doctor for personalized advice</li>
                    <li>Report any adverse effects immediately</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

elif selected == "Disease Prediction":
    st.switch_page("pages/2_🤒_Disease_Prediction.py")
elif selected == "Medicine Recommendation":
    st.switch_page("pages/3_💊_Medicine_Recommendation.py")
elif selected == "Analytics":
    st.switch_page("pages/4_📊_Analytics.py")
elif selected == "Admin":
    st.switch_page("pages/5_⚙️_Admin.py")