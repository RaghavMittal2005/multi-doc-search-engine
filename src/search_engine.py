import faiss
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter
import re

class VectorSearchEngine:
    """FAISS-based vector search engine with ranking explanation"""
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product for normalized vectors
        self.doc_ids = []
        self.documents = {}
    
    def add_documents(self, doc_ids: List[str], embeddings: np.ndarray, documents: List[Dict]):
        """Add documents to the search index"""
        self.doc_ids = doc_ids
        self.documents = {doc['doc_id']: doc for doc in documents}
        
        # Ensure embeddings are normalized
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        
        self.index.add(embeddings.astype('float32'))
        print(f"Added {len(doc_ids)} documents to search index")
    
    def _extract_keywords(self, text: str) -> set:
        """Extract keywords from text"""
        words = re.findall(r'\b\w+\b', text.lower())
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'are', 'was', 'were'}
        return set(w for w in words if len(w) > 3 and w not in stop_words)
    
    def _compute_overlap_ratio(self, query: str, doc_text: str) -> Tuple[float, List[str]]:
        """Compute keyword overlap between query and document"""
        query_keywords = self._extract_keywords(query)
        doc_keywords = self._extract_keywords(doc_text)
        
        if not query_keywords:
            return 0.0, []
        
        overlapping = query_keywords.intersection(doc_keywords)
        ratio = len(overlapping) / len(query_keywords)
        
        return ratio, list(overlapping)
    
    def search(self, query_embedding: np.ndarray, query_text: str, top_k: int = 5) -> List[Dict]:
        """Search for similar documents"""
        # Normalize query embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Search using FAISS
        scores, indices = self.index.search(query_embedding, min(top_k, len(self.doc_ids)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            
            doc_id = self.doc_ids[idx]
            doc = self.documents[doc_id]
            
            # Compute ranking explanation
            overlap_ratio, overlapping_keywords = self._compute_overlap_ratio(
                query_text, doc['text']
            )
            
            # Document length normalization
            length_norm = min(1.0, 500 / max(doc['doc_length'], 1))
            
            # Preview text (first 200 chars)
            preview = doc['text'][:200] + "..." if len(doc['text']) > 200 else doc['text']
            
            results.append({
                'doc_id': doc_id,
                'filename': doc['filename'],
                'score': float(score),
                'preview': preview,
                'explanation': {
                    'semantic_similarity': float(score),
                    'keyword_overlap_ratio': overlap_ratio,
                    'overlapping_keywords': overlapping_keywords,
                    'length_normalization': length_norm,
                    'doc_length': doc['doc_length']
                }
            })
        
        return results
    
    def save_index(self, path: str = "cache/faiss_index.bin"):
        """Save FAISS index to disk"""
        faiss.write_index(self.index, path)
        print(f"Index saved to {path}")
    
    def load_index(self, path: str = "cache/faiss_index.bin"):
        """Load FAISS index from disk"""
        self.index = faiss.read_index(path)
        print(f"Index loaded from {path}")