#!/usr/bin/env python3
"""
🔗 Hybrid Embedder - Fixed version that returns (doc, embedding) tuples
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Union, Tuple
import hashlib
import json
from datetime import datetime

from ..core.embedding_generator import EmbeddingGenerator

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

class HybridEmbedder:
    """
    Hybrid embedding system that intelligently routes between:
    - Local embeddings (free, fast, good for development)
    - Cloud embeddings (better quality, costs money)
    
    Returns (document, embedding) tuples as expected by vector store.
    """
    
    def __init__(self, environment: str = None, force_openai: bool = True):
        """Initialize hybrid embedder"""
        self.environment = environment or os.getenv('LLMFY_ENV', 'production')  # Default to production for OpenAI
        self.force_openai = force_openai
        
        # Initialize base embedder (has OpenAI)
        self.cloud_embedder = EmbeddingGenerator()
        
        # Initialize local embedder if available (but we'll prefer OpenAI)
        self.local_embedder = None
        if SENTENCE_TRANSFORMERS_AVAILABLE and not force_openai:
            try:
                self.local_embedder = SentenceTransformer('all-MiniLM-L6-v2')
                print("✅ Local embedder initialized (all-MiniLM-L6-v2)")
            except Exception as e:
                print(f"⚠️  Could not initialize local embedder: {e}")
        
        # Cost tracking
        self.cost_tracker = {
            'local_embeddings': 0,
            'cloud_embeddings': 0,
            'cached_embeddings': 0,
            'estimated_cost': 0.0
        }
        
        # Embedding cache
        self.cache_dir = Path("data/embedding_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = self._load_cache()
        
    def process_documents(self, documents: List[Any]) -> List[Tuple[Any, List[float]]]:
        """
        Process documents with hybrid embedding strategy
        
        Returns: List of (document, embedding) tuples
        """
        doc_embeddings = []
        
        for doc in documents:
            # Determine embedding strategy
            use_local = self._should_use_local(doc)
            
            # Check cache first
            cache_key = self._get_cache_key(doc.page_content)
            if cache_key in self.cache:
                # Use cached embedding
                embedding = self.cache[cache_key]['embedding']
                self.cost_tracker['cached_embeddings'] += 1
                
                # Add embedding metadata
                doc.metadata['embedding_type'] = self.cache[cache_key]['type']
                doc.metadata['embedding_cached'] = True
                
            else:
                # Generate new embedding
                if use_local and self.local_embedder:
                    # Use local embedding
                    embedding = self._generate_local_embedding(doc.page_content)
                    
                    # Track and cache
                    self.cost_tracker['local_embeddings'] += 1
                    self._cache_embedding(cache_key, embedding, 'local')
                    
                    # Add metadata
                    doc.metadata['embedding_type'] = 'local'
                    doc.metadata['embedding_model'] = 'all-MiniLM-L6-v2'
                    doc.metadata['embedding_dim'] = 384
                    
                else:
                    # Use cloud embedding
                    embedding = self._generate_cloud_embedding(doc.page_content)
                    
                    # Track cost
                    self.cost_tracker['cloud_embeddings'] += 1
                    tokens = self._estimate_tokens(doc.page_content)
                    cost = (tokens / 1_000_000) * 0.10  # $0.10 per 1M tokens
                    self.cost_tracker['estimated_cost'] += cost
                    
                    # Cache
                    self._cache_embedding(cache_key, embedding, 'cloud')
                    
                    # Add metadata
                    doc.metadata['embedding_type'] = 'cloud'
                    doc.metadata['embedding_model'] = 'text-embedding-ada-002'
                    doc.metadata['embedding_dim'] = 1536
                    doc.metadata['embedding_cost'] = cost
                
                doc.metadata['embedding_cached'] = False
            
            # Add embedding timestamp
            doc.metadata['embedding_generated_at'] = datetime.utcnow().isoformat()
            
            # Append (document, embedding) tuple
            doc_embeddings.append((doc, embedding))
        
        # Save cache and print summary
        self._save_cache()
        self._print_cost_summary()
        
        return doc_embeddings
    
    def embed_documents(self, texts: List[str]) -> List[Tuple[Any, Any]]:
        """
        Embed raw text strings (compatibility method for ChunkOptimizer)
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of (document, embedding) tuples
        """
        # Convert texts to Document objects
        from langchain.schema import Document
        documents = [Document(page_content=text) for text in texts]
        
        # Process using existing method
        return self.process_documents(documents)
    
    def _should_use_local(self, document: Any) -> bool:
        """Determine whether to use local embeddings"""
        # Force OpenAI if requested
        if self.force_openai:
            return False
        
        # Use OpenAI if available, regardless of environment
        if hasattr(self.cloud_embedder, 'client') and self.cloud_embedder.client:
            return False
        
        # Fallback to local in development if OpenAI not available
        if self.environment == 'development':
            return True
        
        # Check document metadata for hints
        metadata = document.metadata
        
        # Use local for draft or temporary content
        if metadata.get('status') in ['draft', 'temporary', 'test']:
            return True
        
        # Use local for frequently changing content
        if metadata.get('update_frequency') == 'high':
            return True
        
        # Use local if quality score is below threshold
        quality_score = metadata.get('final_quality_score', 0)
        if quality_score < 9.7:  # Only use cloud for exceptional content
            return True
        
        # Use cloud for production high-quality content
        return False
    
    def _generate_local_embedding(self, text: str) -> List[float]:
        """Generate embedding using local model"""
        if not self.local_embedder:
            raise ValueError("Local embedder not available")
        
        # Generate embedding
        embedding = self.local_embedder.encode(text, normalize_embeddings=True)
        
        return embedding.tolist()
    
    def _generate_cloud_embedding(self, text: str) -> List[float]:
        """Generate embedding using cloud API"""
        # Use the base embedding generator
        embedding = self.cloud_embedder.generate_embedding(text)
        
        return embedding
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.sha256(text.encode()).hexdigest()
    
    def _cache_embedding(self, key: str, embedding: List[float], embedding_type: str):
        """Cache an embedding"""
        self.cache[key] = {
            'embedding': embedding,
            'type': embedding_type,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _load_cache(self) -> Dict:
        """Load embedding cache"""
        cache_file = self.cache_dir / "hybrid_embeddings_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache(self):
        """Save embedding cache"""
        cache_file = self.cache_dir / "hybrid_embeddings_cache.json"
        with open(cache_file, 'w') as f:
            json.dump(self.cache, f)
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)"""
        # Rough estimate: 1 token ≈ 4 characters
        return len(text) // 4
    
    def _print_cost_summary(self):
        """Print cost summary"""
        print(f"\n💰 Embedding Generation Summary:")
        print(f"   • Local embeddings: {self.cost_tracker['local_embeddings']}")
        print(f"   • Cloud embeddings: {self.cost_tracker['cloud_embeddings']}")
        print(f"   • Cached embeddings: {self.cost_tracker['cached_embeddings']}")
        print(f"   • Estimated cost: ${self.cost_tracker['estimated_cost']:.4f}")
        
        total = sum([
            self.cost_tracker['local_embeddings'],
            self.cost_tracker['cloud_embeddings'],
            self.cost_tracker['cached_embeddings']
        ])
        
        if total > 0:
            cache_rate = (self.cost_tracker['cached_embeddings'] / total) * 100
            local_rate = (self.cost_tracker['local_embeddings'] / total) * 100
            print(f"   • Cache hit rate: {cache_rate:.1f}%")
            print(f"   • Local usage rate: {local_rate:.1f}%")
