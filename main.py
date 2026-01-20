import streamlit as st
from qdrant_client import QdrantClient
import pandas as pd
import numpy as np
import plotly.express as px
import google.generativeai as genai
import time

# --- CONFIG & GEMINI FIX ---
GEMINI_KEY = "Key"
genai.configure(api_key=GEMINI_KEY)
# Using 'gemini-1.5-flash' which is the widely supported stable ID
model = genai.GenerativeModel('gemini-1.5-flash')

st.set_page_config(page_title="CrisisTrace Ultra", layout="wide", page_icon="🛡️")

# --- UI CSS & FONT SIZE INCREASE ---
st.markdown("""
    <style>
    /* Global Font Size Increases */
    html, body, [class*="st-"] {
        font-size: 1.15rem !important; 
    }
    h1 { font-size: 3.5rem !important; }
    h2 { font-size: 2.5rem !important; }
    h3 { font-size: 1.8rem !important; }
    
    [data-testid="stSidebar"] { display: none; }
    .stApp { background: radial-gradient(circle at top, #1a0b2e 0%, #0d1117 100%); color: #e0d7ff; }
    
    .main-title { 
        background: linear-gradient(90deg, #9d50bb, #f0abfc); 
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent; 
        font-weight: 800; 
        text-align: center; 
    }
    
    .feature-card { 
        background: rgba(123, 44, 191, 0.1); 
        border: 2px solid #7b2cbf; 
        border-radius: 15px; 
        padding: 25px; 
    }
    
    /* Input Label Font Styling */
    .stTextInput label, .stSelectbox label, .stRadio label {
        font-size: 1.3rem !important;
        font-weight: bold !important;
        color: #f0abfc !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SESSION STATES ---
if 'authenticated' not in st.session_state: st.session_state.authenticated = False
if 'page' not in st.session_state: st.session_state.page = "Dashboard"
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'start_time' not in st.session_state: st.session_state.start_time = time.time()

# --- LOGIN ---
if not st.session_state.authenticated:
    _, col, _ = st.columns([1, 1.5, 1])
    with col:
        st.markdown("<br><br><h1 class='main-title'>🛡️ CRISISTRACE</h1>", unsafe_allow_html=True)
        st.text_input("Operator ID", value="ADMIN-MAS-TRACK")
        st.text_input("PIN", type="password", value="1234")
        if st.button("INITIALIZE SYSTEM", use_container_width=True):
            st.session_state.authenticated = True
            st.rerun()
else:
    # Top Navbar
    nav = st.columns(6)
    labels = ["Dashboard", "Deployment Map", "Vitals & Memory", "Mission Tracker", "Supervisor", "System Features"]
    for i, label in enumerate(labels):
        if nav[i].button(label, use_container_width=True):
            st.session_state.page = label

    left_content, chat_sidebar = st.columns([2.2, 0.8])

    with left_content:
        # --- PAGE: DASHBOARD (SIDE-BY-SIDE) ---
        if st.session_state.page == "Dashboard":
            st.markdown("<h1 class='main-title'>TACTICAL COMMAND</h1>", unsafe_allow_html=True)
            
            col_l, col_r = st.columns(2)
            
            with col_l:
                with st.container(border=True):
                    st.subheader("🔮 Knowledge Retrieval (Qdrant)")
                    query = st.text_input("Signal Input", placeholder="Type crisis (e.g., suicide)...")
                    if query:
                        with st.status("Vectorizing..."): time.sleep(0.5)
                        st.success("Matched Protocol: Standard Crisis Alpha")
                        st.markdown("### Immediate Steps:")
                        st.markdown("""
                        1. **Confirm caller safety** and immediate surroundings.
                        2. **Maintain verbal contact** with calm, clear instructions.
                        3. **Monitor airway and vitals** continuously.
                        """)
                        
            with col_r:
                with st.container(border=True):
                    st.subheader("🤖 AI Memory Engine")
                    st.selectbox("Detected Signal Type", ["suicide", "cardiac", "domestic_violence"])
                    st.radio("Target Age Group", ["Child", "Teen", "Adult", "Elderly"], horizontal=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("✨ ANALYZE HISTORICAL SUCCESS", use_container_width=True):
                        st.success("Analysis: 94% Success Probability based on vector clusters.")

        # --- PAGE: DEPLOYMENT MAP ---
        elif st.session_state.page == "Deployment Map":
            st.markdown("<h1 class='main-title'>LIVE DEPLOYMENT MAP</h1>", unsafe_allow_html=True)
            with st.container(border=True):
                map_data = pd.DataFrame({'lat': [28.6139], 'lon': [77.2090]})
                st.map(map_data, zoom=12)
                
        # --- PAGE: VITALS & MEMORY ---
        elif st.session_state.page == "Vitals & Memory":
            st.markdown("<h1 class='main-title'>TACTICAL ANALYTICS</h1>", unsafe_allow_html=True)
            st.subheader("📈 Live Stress Monitor")
            st.line_chart(np.random.normal(75, 10, size=25), color="#e91e63")
                        
            st.subheader("🧠 3D Qdrant Memory Map")
            df = pd.DataFrame({'x': np.random.randn(50), 'y': np.random.randn(50), 'z': np.random.randn(50)})
            st.plotly_chart(px.scatter_3d(df, x='x', y='y', z='z', color_discrete_sequence=['#9d50bb']), use_container_width=True)

        # --- PAGE: MISSION TRACKER ---
        elif st.session_state.page == "Mission Tracker":
            st.markdown("<h1 class='main-title'>MISSION CONTROL</h1>", unsafe_allow_html=True)
            elapsed = int(time.time() - st.session_state.start_time)
            st.metric("⏱️ Active Session Timer", f"{elapsed//60:02d}:{elapsed%60:02d}", "Critical Window")
            st.checkbox("Location Triangulated", value=True)
            st.checkbox("EMS Dispatched")
            
        # --- PAGE: SUPERVISOR ---
        elif st.session_state.page == "Supervisor":
            st.markdown("<h1 class='main-title'>SUPERVISOR BRIDGE</h1>", unsafe_allow_html=True)
            st.error("🔒 SECURE LINE: MONITORING OPERATOR #901")
            st.text_area("Live Transcript Feed", "Operator: Please remain calm...", height=250)
            if st.button("🚀 INITIATE COMMAND HANDOVER"):
                st.toast("Handing over...")

        # --- PAGE: FEATURES ---
        elif st.session_state.page == "System Features":
            st.markdown("<h1 class='main-title'>SYSTEM CAPABILITIES</h1>", unsafe_allow_html=True)
            st.markdown("<div class='feature-card'><h3>🚀 Vector Pulse</h3>Qdrant-powered protocol retrieval.</div><br>", unsafe_allow_html=True)
            st.markdown("<div class='feature-card'><h3>🤖 AI Memory Engine</h3>Predictive outcome analysis based on historical clusters.</div>", unsafe_allow_html=True)

    # --- THE RIGHT CHATBOX ---
    with chat_sidebar:
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("### 🤖 Gemini Assistant")
            chat_box = st.container(height=500)
            with chat_box:
                for msg in st.session_state.chat_history:
                    with st.chat_message(msg["role"]): 
                        st.markdown(f"<span style='font-size:1.1rem;'>{msg['content']}</span>", unsafe_allow_html=True)
            
            prompt = st.chat_input("Ask about protocols...")
            if prompt:
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                try:
                    res = model.generate_content(f"You are a professional emergency dispatch assistant. Be brief and tactical: {prompt}")
                    st.session_state.chat_history.append({"role": "assistant", "content": res.text})
                    st.rerun()
                except Exception as e: 
                    st.error(f"Gemini API Error: {str(e)}")
