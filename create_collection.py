# ============================
# CREATE QDRANT COLLECTION
# ============================

from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from config import QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME, VECTOR_SIZE

# Connect to Qdrant Cloud / Local
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Create new collection (run once)
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
)

print(f"[OK] Collection '{COLLECTION_NAME}' created successfully!")
