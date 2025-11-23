import sqlite3
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List

class CacheManager:
    """Manage embedding cache using SQLite"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "embeddings.db"
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                doc_id TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                hash TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
        print(f"Cache database initialized at {self.db_path}")
    
    def get_embedding(self, doc_id: str, doc_hash: str) -> Optional[np.ndarray]:
        """Retrieve embedding from cache if hash matches"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT embedding, hash FROM embeddings WHERE doc_id = ?",
            (doc_id,)
        )
        result = cursor.fetchone()
        conn.close()
        
        if result and result[1] == doc_hash:
            # Hash matches, return cached embedding
            embedding = np.frombuffer(result[0], dtype=np.float32)
            return embedding
        
        return None
    
    def save_embedding(self, doc_id: str, embedding: np.ndarray, doc_hash: str):
        """Save embedding to cache"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        embedding_blob = embedding.astype(np.float32).tobytes()
        timestamp = datetime.now().isoformat()
        
        cursor.execute("""
            INSERT OR REPLACE INTO embeddings (doc_id, embedding, hash, updated_at)
            VALUES (?, ?, ?, ?)
        """, (doc_id, embedding_blob, doc_hash, timestamp))
        
        conn.commit()
        conn.close()
    
    def get_all_embeddings(self) -> Dict[str, np.ndarray]:
        """Retrieve all embeddings from cache"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT doc_id, embedding FROM embeddings")
        results = cursor.fetchall()
        conn.close()
        
        embeddings = {}
        for doc_id, embedding_blob in results:
            embeddings[doc_id] = np.frombuffer(embedding_blob, dtype=np.float32)
        
        return embeddings
    
    def clear_cache(self):
        """Clear all cached embeddings"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM embeddings")
        conn.commit()
        conn.close()
        print("Cache cleared")