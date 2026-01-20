# ============================
# UPSERT POLITIFACT FACT-CHECK DATA
# ============================

import json
import pandas as pd
from qdrant_client.http.models import PointStruct
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from config import *
import uuid

# Load dataset (user downloads from provided link)
df = pd.read_json("politifact_factcheck_data.jsonl", lines=True)

# Clean + Label mapping
def map_status(label):
    label = label.lower()
    if "false" in label or "pants" in label:
        return "debunked"
    return "verified"

df["status"] = df["verdict"].apply(map_status)
df["text"] = df["statement"]

# Initialize model
model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = model.encode(df["text"].tolist())

# Initialize Qdrant
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Upload in batches
points = []
for vec, row in zip(vectors, df.itertuples()):
    points.append(PointStruct(
        id=str(uuid.uuid4()),
        vector=vec.tolist(),
        payload={
            "text": row.text,
            "status": row.status,
            "analysis_link": row.factcheck_analysis_link
        }
    ))

client.upsert(collection_name=COLLECTION_NAME, points=points)

print("[OK] PolitiFact upsert completed.")
