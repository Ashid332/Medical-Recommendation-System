import streamlit as st

def apply_theme():
    """Inject a global CSS theme for the app."""
    st.markdown("""
    <style>
    :root{
        --primary: #2B8CFF; /* Soft blue */
        --accent: #08C6A3;  /* Teal accent */
        --bg: #071021;      /* Dark background */
        --surface: #0b1b2b; /* Card surface */
        --text: #E6EEF6;    /* Light text */
        --muted: #9FB3C8;
    }

    .stApp {
        background-color: var(--bg) !important;
        color: var(--text) !important;
    }

    .main-header, .sub-header, h1, h2, h3, h4, h5 {
        color: var(--primary) !important;
    }

    .card, .prediction-card, .medicine-card, .login-container, .login-card {
        background-color: var(--surface) !important;
        color: var(--text) !important;
        border-left: 5px solid var(--primary) !important;
    }

    .stButton>button {
        background: linear-gradient(90deg, var(--primary), var(--accent)) !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: 600 !important;
    }

    .stButton>button:hover {
        filter: brightness(1.05) !important;
        transform: translateY(-1px) !important;
    }

    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>div {
        background-color: rgba(255,255,255,0.03) !important;
        color: var(--text) !important;
        border: 1px solid rgba(255,255,255,0.04) !important;
    }

    .metric-card { background-color: var(--surface) !important; }

    a { color: var(--primary) !important; }
    </style>
    """, unsafe_allow_html=True)
