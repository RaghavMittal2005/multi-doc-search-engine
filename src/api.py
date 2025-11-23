from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from preprocessor import DocumentPreprocessor
from embedder import EmbeddingGenerator
from cache_manager import CacheManager
from search_engine import VectorSearchEngine

app = FastAPI(title="Embedding Search Engine API")

# Global instances
preprocessor = None
embedder = None
cache_manager = None
search_engine = None
documents = []

class SearchQuery(BaseModel):
    query: str
    top_k: int = 5

class SearchResponse(BaseModel):
    results: List[dict]
    query: str
    total_docs: int

@app.on_event("startup")
async def startup_event():
    """Initialize all components on startup"""
    global preprocessor, embedder, cache_manager, search_engine, documents
    
    print("Initializing search engine...")
    
    # Initialize components
    preprocessor = DocumentPreprocessor()
    embedder = EmbeddingGenerator()
    cache_manager = CacheManager()
    search_engine = VectorSearchEngine(embedding_dim=embedder.embedding_dim)
    
    # Load documents
    documents = preprocessor.load_documents()
    
    if not documents:
        print("No documents found. Please add .txt files to data/docs/")
        return
    
    # Generate or load embeddings
    doc_ids = []
    embeddings_list = []
    
    for doc in documents:
        # Try to get from cache
        cached_embedding = cache_manager.get_embedding(doc['doc_id'], doc['hash'])
        
        if cached_embedding is not None:
            print(f"Using cached embedding for {doc['doc_id']}")
            embeddings_list.append(cached_embedding)
        else:
            print(f"Generating embedding for {doc['doc_id']}")
            embedding = embedder.generate_embedding(doc['text'])
            cache_manager.save_embedding(doc['doc_id'], embedding, doc['hash'])
            embeddings_list.append(embedding)
        
        doc_ids.append(doc['doc_id'])
    
    # Build search index
    import numpy as np
    embeddings_array = np.array(embeddings_list)
    search_engine.add_documents(doc_ids, embeddings_array, documents)
    
    print(f"Search engine ready with {len(documents)} documents")

@app.get("/")
async def root():
    return {
        "message": "Embedding Search Engine API",
        "endpoints": {
            "/search": "POST - Search documents",
            "/stats": "GET - Get statistics",
            "/docs": "GET - API documentation"
        }
    }

@app.post("/search", response_model=SearchResponse)
async def search(query: SearchQuery):
    """Search for documents matching the query"""
    if not documents:
        raise HTTPException(status_code=503, detail="No documents loaded")
    
    # Generate query embedding
    query_embedding = embedder.generate_embedding(query.query)
    
    # Search
    results = search_engine.search(
        query_embedding, 
        query.query, 
        top_k=query.top_k
    )
    
    return SearchResponse(
        results=results,
        query=query.query,
        total_docs=len(documents)
    )

@app.get("/stats")
async def get_stats():
    """Get search engine statistics"""
    return {
        "total_documents": len(documents),
        "embedding_dimension": embedder.embedding_dim if embedder else 0,
        "cache_database": str(cache_manager.db_path) if cache_manager else None
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)