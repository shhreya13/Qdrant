# 🛡️ CrisisTrace Ultra
**Tactical Intelligence & Emergency Dispatch Portal**

CrisisTrace Ultra is a professional, high-performance emergency response dashboard designed for tactical crisis management. It integrates real-time vector-based protocol retrieval, AI-driven predictive memory, and coordinated mission tracking into a single, cohesive interface.

---

## 🚀 Core Features

### 🔮 Knowledge Retrieval (Qdrant)
* **Semantic Vector Search:** Matches emergency signals (e.g., "cardiac", "suicide") to specific crisis protocols using high-dimensional embeddings.
* **Instant Protocols:** Displays matched protocols with immediate "Golden Hour" steps to stabilize situations.


### 🤖 AI Memory Engine
* **Historical Success Probability:** Uses **Gemini 1.5 Flash** to analyze historical vector clusters and predict success rates for chosen response paths.
* **Demographic Context:** Refines strategies based on age group and situational variables to ensure precision care.

### 📍 Deployment Map
* **Geospatial Intelligence:** Dedicated full-page map for tracking caller signals and identifying the nearest trauma centers.
* **Cellular Triangulation:** Real-time location visualization for field units and caller coordinates.


### ⏱️ Mission Control
* **Tactical Operations Stack:** Includes specialized management modules for:
    * **🚑 Unit Tracking:** Monitoring ambulance dispatch and ETA.
    * **🏥 Hospital Prep:** Ensuring trauma centers and bed availability are secured.
    * **👥 Civilian Safety:** Managing bystander instructions and perimeter security.
* **Action Timer:** Persistent session clock to maintain operational tempo.


### 🔒 Supervisor Bridge
* **Command Handover:** A secure protocol that allows operators to transfer control to a supervisor station instantly.
* **Live Notification System:** Visually confirms once the supervisor has taken over the active signal.

---

## 🛠️ Technical Architecture
* **Frontend:** Streamlit (Python)
* **AI Engine:** Google Gemini 1.5 Flash
* **Layout:** Custom CSS Flexbox with wide-mode optimization
* **Vector Ops:** Qdrant Client (Mock Integration)
* **Charts:** Plotly Express & Streamlit Native Metrics

---

## 🚦 Getting Started

### 1. Installation
Ensure you have Python 3.9+ installed, then run:
```bash
pip install streamlit google-generativeai pandas numpy plotly qdrant-client
```
###2. API Configuration
Open the application file and ensure your Google AI API Key is correctly set:
```bash
GEMINI_KEY = "YOUR_API_KEY_HERE"
```
3. Launching the System
```bash
streamlit run app.py
```
### 5. Access Credentials
```bash
Operator ID: ADMIN-MAS-TRACK
Security PIN: 1234
```
📜 System Disclaimer
This application is a Tactical Simulation Environment. It is designed for workflow visualization and training purposes. For live emergency field use, this interface must be integrated with certified Emergency Medical Dispatch (EMD) backend systems.

© 2026 CrisisTrace Systems. Tactical. Precise. Reliable.
