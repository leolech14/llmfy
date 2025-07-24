#!/usr/bin/env python3
"""
🔗 Hybrid Embedder - Intelligent local/cloud embedding generation

Optimizes costs by using local embeddings for development and 
selectively using OpenAI for production.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Union
import hashlib
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.embedding_generator import EmbeddingGenerator

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
    
    Cost optimization strategies:
    1. Use local for development environment
    2. Use local for frequently changing content
    3. Use cloud for high-value, stable content
    4. Cache all embeddings to avoid regeneration
    """
    
    def __init__(self, environment: str = None):
        """Initialize hybrid embedder"""
        self.environment = environment or os.getenv('LLMFY_ENV', 'development')
        
        # Initialize base embedder (has OpenAI)
        self.cloud_embedder = EmbeddingGenerator()
        
        # Initialize local embedder if available
        self.local_embedder = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
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
        
    def process_documents(self, documents: List[Any]) -> List[Any]:
        """Process documents with hybrid embedding strategy"""
        embedded_documents = []
        
        for doc in documents:
            # Determine embedding strategy
            use_local = self._should_use_local(doc)
            
            # Check cache first
            cache_key = self._get_cache_key(doc.page_content)
            if cache_key in self.cache:
                # Use cached embedding
                doc.embedding = self.cache[cache_key]['embedding']
                self.cost_tracker['cached_embeddings'] += 1
                
                # Add embedding metadata
                doc.metadata['embedding_type'] = self.cache[cache_key]['type']
                doc.metadata['embedding_cached'] = True
                
            else:
                # Generate new embedding
                if use_local and self.local_embedder:
                    # Use local embedding
                    embedding = self._generate_local_embedding(doc.page_content)
                    doc.embedding = embedding
                    
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
                    doc.embedding = embedding
                    
                    # Track cost
                    self.cost_tracker['cloud_embeddings'] += 1
                    tokens = self._estimate_tokens(doc.page_content)
                    cost = self._calculate_embedding_cost(tokens)
                    self.cost_tracker['estimated_cost'] += cost
                    
                    # Cache embedding
                    self._cache_embedding(cache_key, embedding, 'cloud')
                    
                    # Add metadata
                    doc.metadata['embedding_type'] = 'cloud'
                    doc.metadata['embedding_model'] = 'text-embedding-ada-002'
                    doc.metadata['embedding_dim'] = 1536
                    doc.metadata['embedding_cost'] = cost
                
                doc.metadata['embedding_cached'] = False
            
            # Add embedding timestamp
            doc.metadata['embedding_generated_at'] = datetime.utcnow().isoformat()
            
            embedded_documents.append(doc)
        
        # Print cost summary
        self._print_cost_summary()
        
        return embedded_documents
    
    def _should_use_local(self, document: Any) -> bool:
        """Determine whether to use local embeddings"""
        # Always use local in development
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
        # Use SHA256 hash of text
        return hashlib.sha256(text.encode()).hexdigest()
    
    def _cache_embedding(self, key: str, embedding: List[float], embedding_type: str):
        """Cache embedding for future use"""
        self.cache[key] = {
            'embedding': embedding,
            'type': embedding_type,
            'cached_at': datetime.utcnow().isoformat()
        }
        
        # Save cache periodically
        if len(self.cache) % 100 == 0:
            self._save_cache()
    
    def _load_cache(self) -> Dict:
        """Load embedding cache from disk"""
        cache_file = self.cache_dir / "embedding_cache.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load embedding cache: {e}")
                return {}
        
        return {}
    
    def _save_cache(self):
        """Save embedding cache to disk"""
        cache_file = self.cache_dir / "embedding_cache.json"
        
        try:
            # Don't save actual embeddings in JSON (too large)
            # In production, use a proper embedding database
            cache_metadata = {}
            for key, value in self.cache.items():
                cache_metadata[key] = {
                    'type': value['type'],
                    'cached_at': value.get('cached_at'),
                    'embedding_size': len(value['embedding'])
                }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_metadata, f, indent=2)
        
        except Exception as e:
            print(f"Warning: Could not save embedding cache: {e}")
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        # Rough estimation: 1 token ≈ 4 characters
        return len(text) // 4
    
    def _calculate_embedding_cost(self, tokens: int) -> float:
        """Calculate embedding cost in USD"""
        # OpenAI text-embedding-ada-002 pricing
        # $0.0001 per 1K tokens
        cost_per_1k_tokens = 0.0001
        return (tokens / 1000) * cost_per_1k_tokens
    
    def _print_cost_summary(self):
        """Print cost optimization summary"""
        total = (self.cost_tracker['local_embeddings'] + 
                self.cost_tracker['cloud_embeddings'] + 
                self.cost_tracker['cached_embeddings'])
        
        if total > 0:
            print("\n💰 Embedding Cost Summary:")
            print(f"  Local embeddings: {self.cost_tracker['local_embeddings']} (free)")
            print(f"  Cloud embeddings: {self.cost_tracker['cloud_embeddings']} (${self.cost_tracker['estimated_cost']:.4f})")
            print(f"  Cached embeddings: {self.cost_tracker['cached_embeddings']} (free)")
            
            # Calculate savings
            potential_cost = total * 0.0001  # If all were cloud
            actual_cost = self.cost_tracker['estimated_cost']
            savings = potential_cost - actual_cost
            savings_percent = (savings / potential_cost * 100) if potential_cost > 0 else 0
            
            print(f"\n  Potential cost: ${potential_cost:.4f}")
            print(f"  Actual cost: ${actual_cost:.4f}")
            print(f"  Savings: ${savings:.4f} ({savings_percent:.1f}%)")
    
    def get_cost_report(self) -> Dict:
        """Get detailed cost report"""
        return {
            'embeddings_generated': {
                'local': self.cost_tracker['local_embeddings'],
                'cloud': self.cost_tracker['cloud_embeddings'],
                'cached': self.cost_tracker['cached_embeddings']
            },
            'costs': {
                'actual': self.cost_tracker['estimated_cost'],
                'potential': (self.cost_tracker['local_embeddings'] + 
                            self.cost_tracker['cloud_embeddings']) * 0.0001,
                'savings': ((self.cost_tracker['local_embeddings'] + 
                           self.cost_tracker['cloud_embeddings']) * 0.0001 - 
                          self.cost_tracker['estimated_cost'])
            },
            'cache_hit_rate': (self.cost_tracker['cached_embeddings'] / 
                             (self.cost_tracker['local_embeddings'] + 
                              self.cost_tracker['cloud_embeddings'] + 
                              self.cost_tracker['cached_embeddings'])
                             if (self.cost_tracker['local_embeddings'] + 
                                 self.cost_tracker['cloud_embeddings'] + 
                                 self.cost_tracker['cached_embeddings']) > 0 else 0)
        }