import streamlit as st
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

from agents import (
    ClaimNormalizationAgent,
    EmbeddingAgent,
    EvidenceRetrievalAgent,
    VerdictAgent
)
from config import QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME


# ============================
# LOAD MODELS & CLIENT
# ============================

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_client():
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

model = load_model()
client = load_client()


# ============================
# AGENT INITIALIZATION
# ============================

normalize_agent = ClaimNormalizationAgent()
embed_agent = EmbeddingAgent(model)
retrieve_agent = EvidenceRetrievalAgent(client, COLLECTION_NAME)
verdict_agent = VerdictAgent()


# ============================
# STREAMLIT UI
# ============================

st.set_page_config(
    page_title="MAS Fact Check System",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Multi-Agent Fact Verification System (MAS + Qdrant)")
st.write("Enter a claim below and the system will analyze it using semantic evidence retrieval & reasoning.")

claim = st.text_input("Enter a claim:", placeholder="e.g., COVID vaccines cause infertility")


# ============================
# MAIN EXECUTION BLOCK
# ============================

if st.button("Verify Claim"):

    if claim.strip() == "":
        st.warning("⚠ Please enter a claim to verify.")
    else:
        with st.spinner("Analyzing via multi-agent pipeline..."):

            # STEP 1 — Normalization
            normalized = normalize_agent.run(claim)

            # STEP 2 — Embedding
            vector = embed_agent.run(normalized)

            # STEP 3 — Evidence Retrieval
            evidence_points = retrieve_agent.run(vector)

            if len(evidence_points) == 0:
                st.error("❌ No relevant evidence found in the vector database.")
            else:
                # STEP 4 — Verdict Computation
                verdict, confidence, filtered = verdict_agent.run(evidence_points)

                # ============================
                # SHOW VERDICT + CONFIDENCE
                # ============================

                verdict_map = {
                    "verified": "✔ VERIFIED (Supported by Evidence)",
                    "debunked": "❌ DEBUNKED (Contradicted by Evidence)",
                    "unknown": "❓ UNKNOWN (Insufficient Evidence)"
                }

                st.subheader("Final Verdict")
                st.markdown(f"### {verdict_map[verdict]}")

                st.subheader("Confidence Scores")
                st.write(confidence)

                # ============================
                # PLOT BAR CHART
                # ============================
                st.subheader("Confidence Distribution")

                labels = list(confidence.keys())
                values = list(confidence.values())

                fig, ax = plt.subplots(figsize=(4, 3))
                ax.bar(labels, values)
                ax.set_ylabel("Count")
                ax.set_title("Evidence Confidence Counts")
                st.pyplot(fig)

                # ============================
                # DISPLAY EVIDENCE
                # ============================
                st.subheader("Top Evidence Retrieved")

                for p in evidence_points:
                    st.write(f"**Claim:** {p.payload['text']}")
                    st.write(f"- Status: `{p.payload['status']}`")
                    if "analysis_link" in p.payload:
                        st.write(f"- Source: {p.payload['analysis_link']}")
                    st.write(f"- Similarity: `{p.score:.3f}`")
                    st.markdown("---")
