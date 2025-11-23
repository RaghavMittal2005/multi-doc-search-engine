import os
import re
import hashlib
from pathlib import Path
from typing import List, Dict
from bs4 import BeautifulSoup

class DocumentPreprocessor:
    """Handles document loading and preprocessing"""
    
    def __init__(self, data_dir: str = "data/docs"):
        self.data_dir = Path(data_dir)
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove HTML tags
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text()
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-z0-9\s.,!?-]', '', text)
        
        return text.strip()
    
    def compute_hash(self, text: str) -> str:
        """Compute SHA256 hash of text for cache invalidation"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def load_documents(self) -> List[Dict]:
        """Load all documents from data directory"""
        documents = []
        
        if not self.data_dir.exists():
            print(f"Creating directory: {self.data_dir}")
            self.data_dir.mkdir(parents=True, exist_ok=True)
            return documents
        
        for file_path in self.data_dir.glob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    raw_text = f.read()
                
                cleaned_text = self.clean_text(raw_text)
                
                doc = {
                    'doc_id': file_path.stem,
                    'filename': file_path.name,
                    'text': cleaned_text,
                    'raw_text': raw_text,
                    'doc_length': len(cleaned_text),
                    'hash': self.compute_hash(cleaned_text)
                }
                documents.append(doc)
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        print(f"Loaded {len(documents)} documents")
        return documents