"""
Test script for the embedding search engine
Run this to verify all components work correctly
"""

import sys
from pathlib import Path
import numpy as np
import time

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))
from preprocessor import DocumentPreprocessor
from embedder import EmbeddingGenerator
from cache_manager import CacheManager
from search_engine import VectorSearchEngine

def print_section(title):
    """Print a section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60 + "\n")

def test_preprocessor():
    """Test document preprocessing"""
    print_section("Testing Document Preprocessor")
    
    preprocessor = DocumentPreprocessor()
    
    # Test text cleaning
    raw_text = "<html>  HELLO World  </html>"
    cleaned = preprocessor.clean_text(raw_text)
    print(f"Raw text: {raw_text}")
    print(f"Cleaned: {cleaned}")
    assert cleaned == "hello world", "Text cleaning failed"
    print("✅ Text cleaning works\n")
    
    # Test hash generation
    hash1 = preprocessor.compute_hash("test text")
    hash2 = preprocessor.compute_hash("test text")
    hash3 = preprocessor.compute_hash("different text")
    assert hash1 == hash2, "Hash should be consistent"
    assert hash1 != hash3, "Different text should have different hash"
    print(f"Hash 1: {hash1[:16]}...")
    print(f"Hash 2: {hash2[:16]}...")
    print(f"Hash 3: {hash3[:16]}...")
    print("✅ Hash generation works\n")
    
    # Test document loading
    documents = preprocessor.load_documents()
    print(f"Loaded {len(documents)} documents")
    if documents:
        print(f"Sample document: {documents[0]['doc_id']}")
        print(f"  Length: {documents[0]['doc_length']}")
        print(f"  Preview: {documents[0]['text'][:100]}...")
    print("✅ Document loading works\n")
    
    return documents

def test_embedder():
    """Test embedding generation"""
    print_section("Testing Embedding Generator")
    
    embedder = EmbeddingGenerator()
    print(f"Model loaded: sentence-transformers/all-MiniLM-L6-v2")
    print(f"Embedding dimension: {embedder.embedding_dim}")
    
    # Test single embedding
    text = "artificial intelligence and machine learning"
    embedding = embedder.generate_embedding(text)
    print(f"\nText: {text}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")
    assert embedding.shape[0] == embedder.embedding_dim, "Wrong embedding dimension"
    assert abs(np.linalg.norm(embedding) - 1.0) < 0.01, "Embedding should be normalized"
    print("✅ Single embedding works\n")
    
    # Test batch embedding
    texts = ["quantum physics", "computer science", "biology"]
    embeddings = embedder.generate_embeddings(texts)
    print(f"Batch texts: {texts}")
    print(f"Embeddings shape: {embeddings.shape}")
    print("✅ Batch embedding works\n")
    
    return embedder

def test_cache_manager():
    """Test cache management"""
    print_section("Testing Cache Manager")
    
    cache = CacheManager(cache_dir="cache_test")
    print(f"Cache database: {cache.db_path}")
    
    # Test saving and retrieving
    doc_id = "test_doc_001"
    doc_hash = "test_hash_123"
    embedding = np.random.rand(384).astype(np.float32)
    
    print(f"\nSaving embedding for: {doc_id}")
    cache.save_embedding(doc_id, embedding, doc_hash)
    print("✅ Embedding saved\n")
    
    # Test cache hit
    print("Testing cache hit (same hash)...")
    retrieved = cache.get_embedding(doc_id, doc_hash)
    assert retrieved is not None, "Cache should return embedding"
    assert np.allclose(retrieved, embedding), "Retrieved embedding should match"
    print(f"Retrieved embedding shape: {retrieved.shape}")
    print("✅ Cache hit works\n")
    
    # Test cache miss
    print("Testing cache miss (different hash)...")
    retrieved = cache.get_embedding(doc_id, "different_hash")
    assert retrieved is None, "Cache should return None for different hash"
    print("✅ Cache miss works\n")
    
    # Clean up
    cache.clear_cache()
    print("✅ Cache cleared\n")
    
    return cache

def test_search_engine(documents, embedder):
    """Test vector search"""
    print_section("Testing Vector Search Engine")
    
    if not documents:
        print("⚠️  No documents available for search testing")
        return None
    
    # Use only first 20 documents for quick testing
    test_docs = documents[:min(20, len(documents))]
    
    search_engine = VectorSearchEngine(embedding_dim=embedder.embedding_dim)
    
    # Generate embeddings
    print(f"Generating embeddings for {len(test_docs)} documents...")
    start_time = time.time()
    texts = [doc['text'] for doc in test_docs]
    embeddings = embedder.generate_embeddings(texts)
    doc_ids = [doc['doc_id'] for doc in test_docs]
    elapsed = time.time() - start_time
    print(f"Generated in {elapsed:.2f}s ({len(test_docs)/elapsed:.1f} docs/sec)")
    
    # Add to index
    search_engine.add_documents(doc_ids, embeddings, test_docs)
    print("✅ Documents added to index\n")
    
    # Test search
    test_queries = [
        "artificial intelligence",
        "space exploration",
        "medical research"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        query_embedding = embedder.generate_embedding(query)
        
        start_time = time.time()
        results = search_engine.search(query_embedding, query, top_k=3)
        search_time = (time.time() - start_time) * 1000
        
        print(f"Search time: {search_time:.2f}ms")
        print(f"Found {len(results)} results:\n")
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['doc_id']} (score: {result['score']:.3f})")
            print(f"     Keywords: {', '.join(result['explanation']['overlapping_keywords'][:5])}")
            print(f"     Overlap: {result['explanation']['keyword_overlap_ratio']:.1%}")
            print(f"     Preview: {result['preview'][:80]}...")
            print()
    
    print("✅ Search works\n")
    return search_engine

def test_full_pipeline():
    """Test the complete pipeline with caching"""
    print_section("Testing Full Pipeline with Caching")
    
    preprocessor = DocumentPreprocessor()
    embedder = EmbeddingGenerator()
    cache = CacheManager()
    search_engine = VectorSearchEngine(embedding_dim=embedder.embedding_dim)
    
    documents = preprocessor.load_documents()
    
    if not documents:
        print("⚠️  No documents found. Run 'python src/utils.py' first")
        return
    
    # Use only first 10 documents for quick testing
    test_docs = documents[:min(10, len(documents))]
    
    print(f"Processing {len(test_docs)} documents...\n")
    
    # First pass - generate embeddings
    print("First pass (generating embeddings):")
    start_time = time.time()
    cache_hits = 0
    cache_misses = 0
    
    doc_ids = []
    embeddings_list = []
    
    for doc in test_docs:
        cached_embedding = cache.get_embedding(doc['doc_id'], doc['hash'])
        
        if cached_embedding is not None:
            cache_hits += 1
            embeddings_list.append(cached_embedding)
        else:
            cache_misses += 1
            embedding = embedder.generate_embedding(doc['text'])
            cache.save_embedding(doc['doc_id'], embedding, doc['hash'])
            embeddings_list.append(embedding)
        
        doc_ids.append(doc['doc_id'])
    
    first_pass_time = time.time() - start_time
    print(f"  Cache hits: {cache_hits}")
    print(f"  Cache misses: {cache_misses}")
    print(f"  Time: {first_pass_time:.2f}s\n")
    
    # Second pass - should be all cache hits
    print("Second pass (using cache):")
    start_time = time.time()
    cache_hits = 0
    cache_misses = 0
    
    for doc in test_docs:
        cached_embedding = cache.get_embedding(doc['doc_id'], doc['hash'])
        
        if cached_embedding is not None:
            cache_hits += 1
        else:
            cache_misses += 1
    
    second_pass_time = time.time() - start_time
    print(f"  Cache hits: {cache_hits}")
    print(f"  Cache misses: {cache_misses}")
    print(f"  Time: {second_pass_time:.2f}s")
    print(f"  Speedup: {first_pass_time/second_pass_time:.1f}x\n")
    
    assert cache_hits == len(test_docs), "All documents should be cached"
    print("✅ Caching works perfectly!\n")
    
    # Build index and test search
    embeddings_array = np.array(embeddings_list)
    search_engine.add_documents(doc_ids, embeddings_array, test_docs)
    
    query = "technology and computers"
    print(f"Testing search with query: '{query}'")
    query_embedding = embedder.generate_embedding(query)
    results = search_engine.search(query_embedding, query, top_k=3)
    
    print(f"Found {len(results)} results")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['doc_id']} - Score: {result['score']:.3f}")
    
    print("\n✅ Full pipeline works!\n")

def main():
    """Run all tests"""
    print("\n" + "🔬 EMBEDDING SEARCH ENGINE TEST SUITE ".center(60, "="))
    print("Testing all components...\n")
    
    try:
        # Test individual components
        documents = test_preprocessor()
        embedder = test_embedder()
        cache = test_cache_manager()
        
        if documents:
            search_engine = test_search_engine(documents, embedder)
        
        # Test full pipeline
        test_full_pipeline()
        
        print_section("All Tests Passed! ✅")
        print("The embedding search engine is working correctly.")
        print("\nNext steps:")
        print("  1. Run 'python src/api.py' to start the API server")
        print("  2. Run 'streamlit run app.py' for the web UI")
        print("  3. Test API: curl -X POST http://localhost:8000/search \\")
        print("               -H 'Content-Type: application/json' \\")
        print("               -d '{\"query\": \"your query\", \"top_k\": 5}'")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()