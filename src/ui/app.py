import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://api:8000")

st.set_page_config(page_title="Edge-RAG", layout="wide")

if "history" not in st.session_state:
    st.session_state["history"] = []

with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_file is not None:
        if st.button("Index Document"):
            with st.spinner("Indexing..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                res = requests.post(f"{API_URL}/upload", files=files)
                if res.status_code == 200:
                    st.success("Successfully indexed!")
                else:
                    st.error("Indexing failed")

    st.divider()
    top_k = st.slider("Top K Chunks", min_value=3, max_value=10, value=5)
    use_reranker = st.checkbox("Enable Reranker", value=False)
    rag_technique = st.selectbox("RAG Technique", ["Baseline", "Parent-Child", "Structural"])

st.title("Edge-RAG Assistant")

for msg in st.session_state["history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("Response Details"):
                tab1, tab2, tab3 = st.tabs(["Sources", "Retrieved Chunks", "RAGAS Metrics"])
                with tab1:
                    st.write(msg["sources"])
                with tab2:
                    st.write("Chunk texts and scores...")
                with tab3:
                    st.progress(0.8, text="Context Relevance (Example)")
                    st.progress(0.9, text="Faithfulness (Example)")

prompt = st.chat_input("Ask a question about the document...")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            payload = {
                "question": prompt,
                "top_k": top_k,
                "use_reranker": use_reranker,
                "technique": rag_technique
            }
            try:
                response = requests.post(f"{API_URL}/ask", json=payload).json()
                answer = response.get("answer", "Sorry, an error occurred.")
                sources = response.get("sources", [])

                st.markdown(answer)

                st.session_state["history"].append({"role": "user", "content": prompt})
                st.session_state["history"].append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
                st.rerun()
            except Exception as e:
                st.error(f"API connection error: {e}")