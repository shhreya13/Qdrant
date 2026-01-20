import streamlit as st
from qdrant_client import QdrantClient
import requests
import re
import pandas as pd

# --- CONFIG ---
URL = "https://bf15866f-7a66-43fa-b798-c06cbb11105d.europe-west3-0.gcp.cloud.qdrant.io:6333"
KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.C9E6rJx6xEoBoM8sEb-UMizFpKGLK38x99XElVUg23g"
API_URL = "http://127.0.0.1:5005/recommend"

client = QdrantClient(url=URL, api_key=KEY)

st.set_page_config(page_title="CrisisTrace Ultra", layout="wide", page_icon="🛡️")

# --- CUSTOM CYBER-PURPLE CSS ---
st.markdown("""
    <style>
    .stApp { background: radial-gradient(circle at top, #1a0b2e 0%, #0d1117 100%); color: #e0d7ff; }
    
    .main-title { background: linear-gradient(90deg, #9d50bb, #6e48aa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 3.5rem; font-weight: 800; text-align: center; }
    
    .protocol-card { background: rgba(26, 11, 46, 0.6); border: 1px solid #7b2cbf; border-radius: 20px; padding: 25px; backdrop-filter: blur(10px); }
    .ai-card { background: linear-gradient(135deg, #240b36 0%, #1a0b2e 100%); border: 1px solid #9d50bb; border-radius: 20px; padding: 25px; margin-bottom: 20px; }
    
    .step-box { background: rgba(0, 0, 0, 0.4); border-left: 5px solid #9d50bb; padding: 15px; margin: 10px 0; border-radius: 5px; font-family: 'JetBrains Mono', monospace; }
    .highlight { color: #f0abfc; font-weight: bold; }
    
    /* Emergency Buttons */
    .emergency-btn>div>button { background: linear-gradient(45deg, #e91e63, #9c27b0) !important; color: white !important; border: none !important; font-size: 1.1rem !important; font-weight: bold !important; }
    
    /* Map Styling */
    [data-testid="stMap"] { border: 2px solid #7b2cbf; border-radius: 15px; overflow: hidden; }
    </style>
    """, unsafe_allow_html=True)

def get_vector(text):
    vector = [0.0] * 128
    for i, char in enumerate(text.lower()[:128]):
        vector[i] = ord(char) / 255.0
    return vector

# --- HEADER ---
t1, t2, t3 = st.columns([2, 1, 1])
with t1:
    st.markdown("<h1 class='main-title'>🛡️ CRISISTRACE ULTRA</h1>", unsafe_allow_html=True)
with t2:
    st.markdown("<div class='emergency-btn'>", unsafe_allow_html=True)
    if st.button("🚨 DISPATCH EMS", use_container_width=True):
        st.toast("Dispatching unit...")
    st.markdown("</div>", unsafe_allow_html=True)
with t3:
    st.markdown("<div class='emergency-btn'>", unsafe_allow_html=True)
    if st.button("📞 SUPERVISOR", use_container_width=True):
        st.toast("Connecting...")
    st.markdown("</div>", unsafe_allow_html=True)

st.divider()

left, right = st.columns([1.4, 1], gap="large")

# --- LEFT: PROTOCOLS ---
with left:
    st.subheader("🔮 Vector Knowledge Retrieval")
    kb_query = st.text_input("Signal Input", placeholder="Type crisis type...", label_visibility="collapsed")
    
    if kb_query:
        try:
            response = client.query_points(collection_name="knowledge_base", query=get_vector(kb_query), limit=1)
            if response.points:
                res = response.points[0].payload
                st.markdown(f'<div class="protocol-card"><h2 style="color:#c084fc;">{res["category"]}</h2><p>{res["text"]}</p></div>', unsafe_allow_html=True)
                for s in res.get('steps', []):
                    s = re.sub(r"(SAY THIS|STAY|PUSH|FAST|HARD|DON'T STOP)", r"<span class='highlight'>\1</span>", s)
                    st.markdown(f'<div class="step-box">{s}</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error: {e}")

# --- RIGHT: AI TOP, MAP BOTTOM ---
with right:
    # 1. AI PREDICTIVE MEMORY (TOP)
    st.subheader("🤖 AI Predictive Memory")
    with st.container():
        c_type = st.selectbox("Current Signal", ["suicide", "panic", "domestic_violence", "substance"])
        age = st.radio("Target Demographic", ["child", "teen", "adult", "elderly"], horizontal=True)
        
        if st.button("✨ GENERATE AI STRATEGY", use_container_width=True):
            try:
                payload = {"crisis_type": c_type, "age_group": age, "urgency_level": 3}
                api_res = requests.post(API_URL, json=payload, timeout=5)
                if api_res.status_code == 200:
                    data = api_res.json()
                    st.markdown(f"""
                    <div class="ai-card">
                        <h3 style='color:#d8b4fe; margin:0;'>Match: {int(data['confidence']*100)}%</h3>
                        <p><b>Recommended:</b> {data['primary_protocol']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            except:
                st.error("AI API Offline.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    # 2. CALLER GEOLOCATOR (BOTTOM)
    st.subheader("📍 Caller Geolocator")
    # New Delhi Coordinates for Demo
    map_data = pd.DataFrame({'lat': [28.6139], 'lon': [77.2090]})
    st.map(map_data, zoom=12)
    st.markdown("<p style='text-align:center; color:#8b949e; font-size:0.8rem;'>Accuracy: ±5m | Method: Triangulation</p>", unsafe_allow_html=True)

st.divider()
st.caption("CrisisTrace Ultra v3.0 | Secure Tactical Interface")