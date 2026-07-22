import streamlit as st
import os

# Prevent Windows FAISS and HuggingFace terminal errors
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import tempfile
from pathlib import Path
# pyrefly: ignore [missing-import]
from src.search import RAGSearch

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="RAG bot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# CUSTOM STYLES (Lightweight, mainly for padding/spacing)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Adjust main padding */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
    }
    /* Style headers */
    h1 {
        font-weight: 700 !important;
        background: -webkit-linear-gradient(45deg, #4F8BF9, #9b51e0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px !important;
    }
    .subtitle {
        color: #A0AEC0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    /* Style the source badges */
    .source-badge {
        background-color: #2D3748;
        color: #E2E8F0;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-right: 5px;
        display: inline-block;
        border: 1px solid #4A5568;
    }
    /* Style the chat input container */
    div[data-testid="stChatInput"] {
        padding-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SYSTEM INITIALIZATION
# -----------------------------------------------------------------------------
os.makedirs("data", exist_ok=True)

@st.cache_resource(show_spinner=False)
def load_rag_system():
    from dotenv import load_dotenv
    load_dotenv()
    return RAGSearch()

try:
    rag = load_rag_system()
except Exception as e:
    st.error("⚠️ System Offline: Failed to initialize AI core.")
    st.exception(e)
    st.stop()

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_history" not in st.session_state:
    st.session_state.rag_history = []

# -----------------------------------------------------------------------------
# SIDEBAR / ADMIN CONSOLE
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    st.caption("RAG Enterprise Configuration")
    
    st.divider()
    
    # Metrics
    stats = rag.vector_store.stats()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Vectors", f"{stats.get('total_vectors', 0):,}")
    with col2:
        st.metric("Unique Files", len(stats.get("unique_sources", [])))
        
    st.divider()
    
    # Document Upload Area
    st.markdown("### 📥 Knowledge Ingestion")
    with st.expander("Upload New Documents", expanded=False):
        uploaded_files = st.file_uploader(
            "Supported: PDF, TXT, CSV, JSON, DOCX", 
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        
        if st.button("Process & Index", use_container_width=True) and uploaded_files:
            with st.spinner("Indexing into Vector Database..."):
                # pyrefly: ignore [missing-import]
                from src.data_loader import load_all_documents
                with tempfile.TemporaryDirectory() as temp_dir:
                    for f in uploaded_files:
                        temp_path = os.path.join(temp_dir, f.name)
                        with open(temp_path, "wb") as out_f:
                            out_f.write(f.read())
                    
                    new_docs = load_all_documents(temp_dir)
                    if new_docs:
                        rag.vector_store.add_documents(new_docs)
                        st.success(f"Indexed {len(new_docs)} chunks!")
                        st.rerun()
                    else:
                        st.warning("No readable content found.")

    st.divider()
    
    indexed_sources = sorted(stats.get("unique_sources", []))
    if indexed_sources:
        st.markdown("###  Indexed Documents")
        sources_to_delete = st.multiselect(
            "Select documents to delete",
            indexed_sources,
            label_visibility="collapsed"
        )
        if st.button("🗑️ Delete Selected Documents", use_container_width=True, disabled=not sources_to_delete):
            rag.vector_store.remove_sources(sources_to_delete)
            st.session_state.messages = []
            st.session_state.rag_history = []
            st.rerun()

    # Reset
    if st.button("🗑️ Clear Session Memory", use_container_width=True):
        st.session_state.messages = []
        st.session_state.rag_history = []
        st.rerun()

# -----------------------------------------------------------------------------
# MAIN INTERFACE
# -----------------------------------------------------------------------------
st.markdown("<h1>RAG Intelligence</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">Secure, citation-backed answers powered by your enterprise knowledge base.</p>', unsafe_allow_html=True)

# Empty state
if not st.session_state.messages:
    if stats.get("total_vectors", 0) == 0:
        st.info("No documents found. Please upload a document to start.")
    else:
        st.info("👋 Welcome! Ask a question to start exploring your documents.")

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧠" if msg["role"] == "assistant" else "👤"):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"] != "unknown":
            sources_html = "".join([f'<span class="source-badge">📄 {src.strip()}</span>' for src in msg["sources"].split(",")])
            st.markdown(f"<div style='margin-top: 10px;'>{sources_html}</div>", unsafe_allow_html=True)

# Input
if prompt := st.chat_input("Query the knowledge base..."):
    # Render user prompt immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Process and render response
    with st.chat_message("assistant", avatar="🧠"):
        with st.spinner("Synthesizing answer..."):
            try:
                if rag.vector_store.stats().get("total_vectors", 0) == 0:
                    response = "No documents found. Please upload a document before asking a question."
                    sources = "unknown"
                    st.markdown(response)
                    st.session_state.rag_history.append({"role": "user", "content": prompt})
                    st.session_state.rag_history.append({"role": "assistant", "content": response})
                    st.session_state.messages.append({"role": "assistant", "content": response, "sources": sources})
                    st.stop()

                # Core RAG Logic
                st.session_state.rag_history = rag._summarize_history(st.session_state.rag_history)
                standalone_query = rag._rewrite_query(prompt, st.session_state.rag_history)
                results = rag._multi_query_retrieve(standalone_query, top_k=5)
                results = rag._filter_results(results)
                
                if not results:
                    response = "I couldn't find any highly relevant information regarding this in the current knowledge base."
                    sources = "unknown"
                    st.markdown(response)
                else:
                    texts = [r["metadata"].get("text", "") for r in results]
                    context = "\n\n".join(texts)
                    sources = rag._format_sources(results)
                    
                    response = rag.chat_with_context(prompt, context, st.session_state.rag_history)
                    
                    st.markdown(response)
                    
                    # Render fancy source badges
                    sources_html = "".join([f'<span class="source-badge">📄 {src.strip()}</span>' for src in sources.split(",")])
                    st.markdown(f"<div style='margin-top: 10px;'>{sources_html}</div>", unsafe_allow_html=True)
                
                # Commit to memory
                st.session_state.rag_history.append({"role": "user", "content": prompt})
                st.session_state.rag_history.append({"role": "assistant", "content": response})
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "sources": sources
                })
                
            except Exception as e:
                st.error(f"Execution Error: {e}")
