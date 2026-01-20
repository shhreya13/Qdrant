import json
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# --- CONFIG ---
URL = "https://bf15866f-7a66-43fa-b798-c06cbb11105d.europe-west3-0.gcp.cloud.qdrant.io:6333"
KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.C9E6rJx6xEoBoM8sEb-UMizFpKGLK38x99XElVUg23g"

client = QdrantClient(url=URL, api_key=KEY)

def get_vector(text):
    vector = [0.0] * 128
    for i, char in enumerate(text.lower()[:128]):
        vector[i] = ord(char) / 255.0
    return vector

def ingest():
    try:
        client.delete_collection("knowledge_base")
    except:
        pass
        
    client.create_collection(
        collection_name="knowledge_base",
        vectors_config=VectorParams(size=128, distance=Distance.COSINE),
    )
    
    with open('data.json', 'r') as f:
        data = json.load(f)

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=get_vector(f"{item.get('category')} {item.get('text')}"),
            payload=item
        ) for item in data
    ]

    client.upsert(collection_name="knowledge_base", points=points)
    print("🚀 Protocol Sync Complete.")

if __name__ == "__main__":
    ingest()