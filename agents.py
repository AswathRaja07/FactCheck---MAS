# ============================
# MULTI-AGENT IMPLEMENTATION
# ============================

class ClaimNormalizationAgent:
    """Preprocesses input claim (lowercase, strip punctuation, trim spaces)"""
    def run(self, claim: str) -> str:
        return claim.strip().lower().replace("?", "").replace(".", "")


class EmbeddingAgent:
    """Encodes normalized text into semantic embeddings"""
    def __init__(self, model):
        self.model = model
    
    def run(self, text: str):
        return self.model.encode(text).tolist()


class EvidenceRetrievalAgent:
    """Queries Qdrant to retrieve top-K similar fact-check evidence"""
    def __init__(self, client, collection_name):
        self.client = client
        self.collection_name = collection_name
    
    def run(self, vector, top_k=3):
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=top_k
        )
        return results.points


class VerdictAgent:
    """Computes final verdict using threshold + ratio logic"""
    def run(self, points, threshold=0.60):
        status_counts = {"verified": 0, "debunked": 0}
        filtered = [p for p in points if p.score >= threshold]

        # Case-1: No strong evidence → Unknown
        if not filtered:
            return "unknown", status_counts, filtered
        
        # Count verified/debunked
        for p in filtered:
            status_counts[p.payload["status"]] += 1
        
        total = sum(status_counts.values())

        # Case-2: No evidence after filtering
        if total == 0:
            return "unknown", status_counts, filtered
        
        # Case-3: Ratio-based final decision
        if status_counts["debunked"]/total >= 0.7:
            return "debunked", status_counts, filtered
        elif status_counts["verified"]/total >= 0.7:
            return "verified", status_counts, filtered
        else:
            return "unknown", status_counts, filtered
