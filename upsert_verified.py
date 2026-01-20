# ============================
# UPSERT VERIFIED SCIENTIFIC FACTS
# ============================

import json, uuid
from sentence_transformers import SentenceTransformer
from qdrant_client.http.models import PointStruct
from qdrant_client import QdrantClient
from config import *

# Load curated verified dataset
with open("data/verified_facts.json", "r") as f:
    data = json.load(f)

texts = [item["text"] for item in data]

model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = model.encode(texts)

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

points=[]
for vec, item in zip(vectors, data):
    points.append(PointStruct(
        id=str(uuid.uuid4()),
        vector=vec.tolist(),
        payload={
            "text": item["text"],
            "status": "verified",
            "source": item["source"],
            "analysis_link": item["analysis_link"]
        }
    ))

client.upsert(collection_name=COLLECTION_NAME, points=points)

print("[OK] Verified facts uploaded.")
