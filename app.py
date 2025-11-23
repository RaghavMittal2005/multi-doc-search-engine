import streamlit as st
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from preprocessor import DocumentPreprocessor
from embedder import EmbeddingGenerator
from cache_manager import CacheManager
from search_engine import VectorSearchEngine

st.set_page_config(
    page_title="Embedding Search Engine",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource
def initialize_search_engine():
    """Initialize all components (cached)"""
    preprocessor = DocumentPreprocessor()
    embedder = EmbeddingGenerator()
    cache_manager = CacheManager()
    search_engine = VectorSearchEngine(embedding_dim=embedder.embedding_dim)
    
    # Load documents
    documents = preprocessor.load_documents()
    
    if not documents:
        return None, None, None, None, []
    
    # Generate or load embeddings
    doc_ids = []
    embeddings_list = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, doc in enumerate(documents):
        cached_embedding = cache_manager.get_embedding(doc['doc_id'], doc['hash'])
        
        if cached_embedding is not None:
            status_text.text(f"Loading cached: {doc['doc_id']}")
            embeddings_list.append(cached_embedding)
        else:
            status_text.text(f"Generating embedding: {doc['doc_id']}")
            embedding = embedder.generate_embedding(doc['text'])
            cache_manager.save_embedding(doc['doc_id'], embedding, doc['hash'])
            embeddings_list.append(embedding)
        
        doc_ids.append(doc['doc_id'])
        progress_bar.progress((i + 1) / len(documents))
    
    progress_bar.empty()
    status_text.empty()
    
    # Build search index
    embeddings_array = np.array(embeddings_list)
    search_engine.add_documents(doc_ids, embeddings_array, documents)
    
    return preprocessor, embedder, cache_manager, search_engine, documents

# Header
st.title("🔍 Multi-document Embedding Search Engine")
st.markdown("Semantic search powered by sentence-transformers and FAISS")

# Initialize
with st.spinner("Initializing search engine..."):
    preprocessor, embedder, cache_manager, search_engine, documents = initialize_search_engine()

if not documents:
    st.error("⚠️ No documents found! Please add .txt files to `data/docs/` directory.")
    st.info("💡 Run `python src/utils.py` to download the 20 newsgroups dataset.")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Number of results", min_value=1, max_value=20, value=5)
    
    st.header("📊 Statistics")
    st.metric("Total Documents", len(documents))
    st.metric("Embedding Dimension", embedder.embedding_dim)
    st.metric("Index Type", "FAISS IndexFlatIP")
    
    st.header("🗄️ Cache Info")
    st.info(f"Database: `{cache_manager.db_path}`")
    
    if st.button("🗑️ Clear Cache"):
        cache_manager.clear_cache()
        st.success("Cache cleared! Please restart the app.")

# Search interface
st.header("Search Documents")
query = st.text_input(
    "Enter your search query:",
    placeholder="e.g., quantum physics, machine learning, space exploration...",
    key="search_query"
)

if query:
    with st.spinner("Searching..."):
        # Generate query embedding
        query_embedding = embedder.generate_embedding(query)
        
        # Search
        results = search_engine.search(query_embedding, query, top_k=top_k)
    
    st.success(f"Found {len(results)} results for: **{query}**")
    
    # Display results
    for i, result in enumerate(results, 1):
        with st.expander(f"#{i} - {result['filename']} (Score: {result['score']:.3f})", expanded=(i==1)):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Preview")
                st.write(result['preview'])
                
                st.subheader("Document Info")
                st.write(f"**Document ID:** `{result['doc_id']}`")
                st.write(f"**Length:** {result['explanation']['doc_length']} characters")
            
            with col2:
                st.subheader("🎯 Ranking Explanation")
                
                exp = result['explanation']
                
                st.metric(
                    "Semantic Similarity", 
                    f"{exp['semantic_similarity']:.3f}",
                    help="Cosine similarity between query and document embeddings"
                )
                
                st.metric(
                    "Keyword Overlap", 
                    f"{exp['keyword_overlap_ratio']:.1%}",
                    help="Percentage of query keywords found in document"
                )
                
                if exp['overlapping_keywords']:
                    st.write("**Overlapping Keywords:**")
                    st.write(", ".join(f"`{kw}`" for kw in exp['overlapping_keywords'][:10]))
                
                st.metric(
                    "Length Normalization",
                    f"{exp['length_normalization']:.3f}",
                    help="Adjustment factor for document length"
                )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built with Sentence-Transformers, FAISS, and Streamlit</p>
        <p>Cache: SQLite | Index: FAISS IndexFlatIP | Model: all-MiniLM-L6-v2</p>
    </div>
    """,
    unsafe_allow_html=True
)