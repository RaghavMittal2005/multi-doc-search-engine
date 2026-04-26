# Multi-document Embedding Search Engine

A production-ready semantic search engine with FAISS indexing, SQLite caching, and FastAPI backend.

## 🎯 Features

✅ **Efficient Embedding Generation** - Uses sentence-transformers (all-MiniLM-L6-v2)  
✅ **Smart Caching** - SQLite-based cache with SHA256 hash invalidation  
✅ **Vector Search** - FAISS IndexFlatIP for fast similarity search  
✅ **REST API** - FastAPI with automatic documentation  
✅ **Ranking Explanation** - Transparent scoring with keyword overlap analysis  
✅ **Streamlit UI** - Interactive web interface (bonus)  
✅ **Modular Architecture** - Clean separation of concerns

## 📁 Project Structure

```
embedding-search-engine/
├── src/
│   ├── __init__.py
│   ├── preprocessor.py      # Document loading & cleaning
│   ├── embedder.py           # Embedding generation
│   ├── cache_manager.py      # SQLite caching system
│   ├── search_engine.py      # FAISS vector search
│   ├── api.py                # FastAPI endpoints
│   └── utils.py              # Dataset download utilities
├── data/
│   └── docs/                 # Text documents (.txt files)
|── tests/
|   └── test_1.py/                 # Test file
├── cache/
│   └── embeddings.db         # SQLite cache database
├── app.py                    # Streamlit UI (bonus)
├── requirements.txt
├── README.md
├── .gitignore
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
conda create -p venv python=3.10.19 #environment
```

### 2. Download Dataset

```bash
python src/utils.py
```

This downloads 200 documents from the 20 Newsgroups dataset.

### 3. Start the API Server

```bash
python src/api.py
```

API will be available at: `http://localhost:8000`

### 4. Or Run Streamlit UI (Bonus)

```bash
streamlit run app.py
```

## 🔧 How It Works

### Caching System

The cache system ensures embeddings are never recomputed unnecessarily:

1. **Hash Generation**: Each document gets a SHA256 hash of its cleaned content
2. **Cache Lookup**: Before embedding, check SQLite cache for matching doc_id + hash
3. **Cache Hit**: If hash matches, reuse cached embedding
4. **Cache Miss**: If hash differs or doc is new, generate new embedding and cache it
5. **Persistence**: SQLite ensures cache survives between runs

**Cache Schema:**
```sql
CREATE TABLE embeddings (
    doc_id TEXT PRIMARY KEY,
    embedding BLOB NOT NULL,
    hash TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
```

### Embedding Pipeline

```
Document → Clean Text → Compute Hash → Check Cache
    ↓                                       ↓
    ↓ (cache miss)                    (cache hit)
    ↓                                       ↓
Generate Embedding ← ─ ─ ─ ─ ─ ─ ─ → Return Cached
    ↓
Save to Cache
    ↓
Add to FAISS Index
```

### Search Process

1. **Query Embedding**: Convert search query to vector
2. **FAISS Search**: Find top-k similar documents using inner product
3. **Ranking Explanation**: Calculate:
   - Semantic similarity score (cosine similarity)
   - Keyword overlap ratio
   - Overlapping keywords list
   - Document length normalization
4. **Return Results**: JSON response with scores and explanations

## 📡 API Endpoints

### POST /search

Search for documents matching a query.

**Request:**
```json
{
  "query": "quantum physics basics",
  "top_k": 5
}
```

**Response:**
```json
{
  "results": [
    {
      "doc_id": "doc_014",
      "filename": "doc_014_sci.physics.txt",
      "score": 0.88,
      "preview": "Quantum theory is concerned with...",
      "explanation": {
        "semantic_similarity": 0.88,
        "keyword_overlap_ratio": 0.6,
        "overlapping_keywords": ["quantum", "physics", "theory"],
        "length_normalization": 0.95,
        "doc_length": 523
      }
    }
  ],
  "query": "quantum physics basics",
  "total_docs": 200
}
```

### GET /stats

Get search engine statistics.

### GET /docs

Interactive API documentation (Swagger UI).

## 🧪 Testing

```bash
# Test with curl
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning", "top_k": 3}'

# Or use Python
import requests

response = requests.post(
    "http://localhost:8000/search",
    json={"query": "artificial intelligence", "top_k": 5}
)
print(response.json())
```

## 🎨 Design Choices

### Why FAISS IndexFlatIP?

- **Inner Product**: With normalized embeddings, inner product = cosine similarity
- **Exact Search**: No approximation, perfect for 100-200 documents
- **Fast**: Optimized C++ implementation with SIMD instructions
- **Scalable**: Can easily upgrade to approximate indexes (IVF, HNSW) for larger datasets

### Why SQLite for Caching?

- **Zero Configuration**: No external database server needed
- **File-Based**: Single file, easy to backup and version control
- **ACID Compliant**: Ensures data integrity
- **Fast**: Efficient for 100-200 documents
- **Portable**: Works on any platform

### Why sentence-transformers?

- **High Quality**: State-of-the-art embeddings for semantic search
- **Efficient**: Small model (all-MiniLM-L6-v2) with 384 dimensions
- **Fast**: Optimized for inference
- **Pre-trained**: No training required

### Modular Architecture

Each component is independent and testable:

- **preprocessor.py**: Document I/O and cleaning
- **embedder.py**: Embedding generation (can swap models)
- **cache_manager.py**: Caching logic (can swap to Redis)
- **search_engine.py**: Vector search (can swap to other indexes)
- **api.py**: REST API (can add more endpoints)

## 🔬 Ranking Explanation

Each search result includes transparent ranking metrics:

1. **Semantic Similarity** (0-1): Cosine similarity between query and document embeddings
2. **Keyword Overlap Ratio** (0-1): Percentage of query keywords found in document
3. **Overlapping Keywords**: List of matching keywords for interpretability
4. **Length Normalization** (0-1): Adjustment factor favoring moderate-length documents

## 🛠️ Advanced Usage

### Using Your Own Documents

```bash
# Place .txt files in data/docs/
cp /path/to/your/documents/*.txt data/docs/

# Restart API (will auto-generate embeddings)
python src/api.py
```

### Clearing Cache

```python
from cache_manager import CacheManager

cache = CacheManager()
cache.clear_cache()
```

### Saving/Loading FAISS Index

```python
from search_engine import VectorSearchEngine

engine = VectorSearchEngine()
# ... add documents ...
engine.save_index("cache/faiss_index.bin")

# Later
engine.load_index("cache/faiss_index.bin")
```

## 📊 Performance

- **Embedding Generation**: ~50-100 docs/second (CPU)
- **Search Latency**: <10ms for 200 documents
- **Cache Hit Rate**: 100% for unchanged documents
- **Memory Usage**: ~50MB for 200 documents + model

## 🚧 Future Enhancements

- [ ] Batch embedding with multiprocessing
- [ ] Query expansion using WordNet
- [ ] Persistent FAISS index
- [ ] Filtering by metadata
- [ ] Hybrid search (semantic + keyword)
- [ ] Document clustering
- [ ] Evaluation metrics (MRR, NDCG)

## 📝 Requirements

```
sentence-transformers==2.2.2
faiss-cpu==1.7.4
fastapi==0.104.1
uvicorn==0.24.0
scikit-learn==1.3.2
numpy==1.24.3
pandas==2.1.3
beautifulsoup4==4.12.2
python-multipart==0.0.6
streamlit==1.28.2
pydantic==2.5.0
```
