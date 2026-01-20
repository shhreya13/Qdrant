from flask import Flask, request, jsonify
from flask_cors import CORS
import sys

# Try to import your engine
try:
    from memory_engine import CrisisAssistantSystem
except ImportError:
    print("ERROR: Could not find memory_engine.py in this folder!")
    sys.exit(1)

app = Flask(__name__)
CORS(app)

# --- INITIALIZE DATABASE ---
QDRANT_URL = "https://bf15866f-7a66-43fa-b798-c06cbb11105d.europe-west3-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.C9E6rJx6xEoBoM8sEb-UMizFpKGLK38x99XElVUg23g"

print("📡 Connecting to Qdrant...")
system = CrisisAssistantSystem(QDRANT_URL, QDRANT_API_KEY)
system.initialize(recreate_db=False)

# --- ROUTES ---

@app.route('/', methods=['GET'])
def index():
    return "<h1>Crisis API is Running</h1><p>Try /health or POST to /recommend</p>"

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "online",
        "database": "connected"
    }), 200

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    if not data:
        return jsonify({"error": "No JSON data"}), 400
    try:
        results = system.get_recommendations(
            crisis_type=data.get('crisis_type'),
            age_group=data.get('age_group'),
            triggers=data.get('triggers', []),
            urgency_level=data.get('urgency_level', 2)
        )
        return jsonify(results), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    # We change the port to 5005 to avoid the "ghost" on 5000
    print("🚀 SERVER STARTING ON PORT 5005")
    app.run(host='127.0.0.1', port=5005, debug=False)