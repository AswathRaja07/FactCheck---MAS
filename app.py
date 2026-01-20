import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import SearchRequest
import numpy as np
import matplotlib.pyplot as plt

# =========================
# APP HEADER
# =========================
st.set_page_config(page_title="FactCheckAI", layout="centered")

st.markdown("""
<h1 style='text-align:center; color:#4da6ff;'>🛡️ FactCheckAI</h1>
<p style='text-align:center;'>AI-powered fact verification using real fact-check evidence</p>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL (CPU SAFE)
# =========================
@st.cache_resource
def load_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

model = load_model()

# =========================
# CONNECT TO QDRANT
# =========================
QDRANT_URL = "https://19f65b30-a9ee-4dfe-a464-7ab559058c66.us-east4-0.gcp.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.4CzqMS8FHJfFp0vdTRdpP2ewarxdKfdLw5MFFYvggDM"

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

COLLECTION = "verified_facts_text"
TOP_K = 3


# =========================
# RETRIEVAL FUNCTION
# =========================
def retrieve(claim):
    vec = model.encode(claim).tolist()
    
    results = client.search(
        collection_name=COLLECTION,
        query_vector=vec,
        limit=TOP_K
    )
    
    return results


# =========================
# VERDICT LOGIC
# =========================
def get_verdict(results):
    deb = sum(1 for r in results if r.payload.get("status") == "debunked")
    ver = sum(1 for r in results if r.payload.get("status") == "verified")

    if ver > deb:
        return "verified", ver, deb
    else:
        return "debunked", ver, deb


# =========================
# UI INPUT
# =========================
claim = st.text_input("Enter a factual claim to verify:")

if st.button("Verify"):
    if claim.strip() == "":
        st.warning("Please enter a claim.")
    else:
        with st.spinner("Checking evidence..."):
            results = retrieve(claim)

        if len(results) == 0:
            st.error("No matching evidence found.")
        else:
            verdict, ver_count, deb_count = get_verdict(results)

            # =========================
            # SHOW VERDICT
            # =========================
            color = "green" if verdict == "verified" else "red"
            st.markdown(f"""
            <h2>Verdict: <span style='color:{color}; text-transform:capitalize;'>{verdict}</span></h2>
            """, unsafe_allow_html=True)

            # =========================
            # PIE CHART (SMALL)
            # =========================
            labels = ['Verified', 'Debunked']
            values = [ver_count, deb_count]
            colors = ['#4CAF50', '#E74C3C']  # green/red

            fig, ax = plt.subplots(figsize=(3, 3))
            ax.pie(values, labels=labels, colors=colors, autopct='%1.0f%%')
            st.pyplot(fig)

            # =========================
            # SHOW EVIDENCE
            # =========================
            st.markdown("### Top Evidence")
            for r in results:
                status = r.payload.get("status", "unknown")
                link = r.payload.get("analysis_link", None)
                score = round(1 - r.score, 3)

                color = "#4CAF50" if status == "verified" else "#E74C3C"

                st.markdown(f"**Status:** <span style='color:{color};'>{status}</span>", unsafe_allow_html=True)
                st.markdown(f"**Similarity Score:** `{score}`")

                if link:
                    st.markdown(f"🔗 **Proof:** [View Fact Check]({link})")
                
                st.markdown("---")
