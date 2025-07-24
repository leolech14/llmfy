from typing import List, Dict, Any, Optional, Tuple
import time
import pickle
from pathlib import Path
import numpy as np

from openai import OpenAI
from langchain.schema import Document
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

from .config import Config

console = Console()

class EmbeddingGenerator:
    def __init__(self, model: Optional[str] = None, use_cache: bool = True):
        self.config = Config()
        self.model = model or self.config.EMBEDDING_MODEL
        self.use_cache = use_cache
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.config.OPENAI_API_KEY)
        
        # Initialize cache
        self.cache_dir = Path(self.config.EMBEDDINGS_CACHE_PATH)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict[str, List[float]]:
        """Load embeddings cache from disk"""
        cache_file = self.cache_dir / "embeddings_cache.pkl"
        if cache_file.exists() and self.use_cache:
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                console.print(f"[yellow]Warning: Could not load cache: {e}[/yellow]")
        return {}
    
    def _save_cache(self):
        """Save embeddings cache to disk"""
        if self.use_cache:
            cache_file = self.cache_dir / "embeddings_cache.pkl"
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(self.cache, f)
            except Exception as e:
                console.print(f"[yellow]Warning: Could not save cache: {e}[/yellow]")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return f"{self.model}:{hash(text)}"
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        # Check cache first
        cache_key = self._get_cache_key(text)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            embedding = response.data[0].embedding
            
            # Cache the result
            self.cache[cache_key] = embedding
            
            return embedding
            
        except Exception as e:
            console.print(f"[red]Error generating embedding: {e}[/red]")
            raise
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Generate embeddings for multiple texts in batches"""
        embeddings = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("Generating embeddings...", total=len(texts))
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = []
                
                # Check cache for each text in batch
                texts_to_embed = []
                cached_indices = []
                
                for j, text in enumerate(batch):
                    cache_key = self._get_cache_key(text)
                    if cache_key in self.cache:
                        batch_embeddings.append(self.cache[cache_key])
                        cached_indices.append(j)
                    else:
                        texts_to_embed.append(text)
                
                # Generate embeddings for non-cached texts
                if texts_to_embed:
                    try:
                        response = self.client.embeddings.create(
                            model=self.model,
                            input=texts_to_embed
                        )
                        
                        # Insert new embeddings in correct positions
                        new_embeddings = [e.embedding for e in response.data]
                        
                        # Merge cached and new embeddings
                        final_batch_embeddings = []
                        new_idx = 0
                        
                        for j in range(len(batch)):
                            if j in cached_indices:
                                # Use cached embedding
                                idx = cached_indices.index(j)
                                final_batch_embeddings.append(batch_embeddings[idx])
                            else:
                                # Use new embedding
                                embedding = new_embeddings[new_idx]
                                final_batch_embeddings.append(embedding)
                                
                                # Cache it
                                cache_key = self._get_cache_key(texts_to_embed[new_idx])
                                self.cache[cache_key] = embedding
                                
                                new_idx += 1
                        
                        batch_embeddings = final_batch_embeddings
                        
                        # Rate limiting
                        time.sleep(0.1)
                        
                    except Exception as e:
                        console.print(f"[red]Error in batch {i//batch_size + 1}: {e}[/red]")
                        raise
                else:
                    # All embeddings were cached
                    final_batch_embeddings = batch_embeddings
                
                embeddings.extend(batch_embeddings)
                progress.update(task, advance=len(batch))
        
        # Save cache after processing
        self._save_cache()
        
        return embeddings
    
    def process_documents(self, documents: List[Document]) -> List[Tuple[Document, List[float]]]:
        """Generate embeddings for a list of documents"""
        console.print(f"[blue]Generating embeddings for {len(documents)} documents...[/blue]")
        
        # Extract texts
        texts = [doc.page_content for doc in documents]
        
        # Generate embeddings
        embeddings = self.generate_embeddings_batch(texts, batch_size=self.config.BATCH_SIZE)
        
        # Combine documents with their embeddings
        doc_embeddings = list(zip(documents, embeddings))
        
        console.print(f"[green]Successfully generated {len(embeddings)} embeddings[/green]")
        
        # Calculate and display statistics
        self._display_statistics(embeddings)
        
        return doc_embeddings
    
    def _display_statistics(self, embeddings: List[List[float]]):
        """Display statistics about the embeddings"""
        if not embeddings:
            return
        
        embeddings_array = np.array(embeddings)
        
        stats = {
            "Total embeddings": len(embeddings),
            "Embedding dimension": len(embeddings[0]),
            "Cached embeddings": len([k for k in self.cache.keys() if k.startswith(self.model)]),
            "Mean norm": float(np.mean(np.linalg.norm(embeddings_array, axis=1))),
            "Std norm": float(np.std(np.linalg.norm(embeddings_array, axis=1)))
        }
        
        console.print("\n[bold]Embedding Statistics:[/bold]")
        for key, value in stats.items():
            if isinstance(value, float):
                console.print(f"  {key}: {value:.4f}")
            else:
                console.print(f"  {key}: {value}")
    
    def estimate_cost(self, num_tokens: int) -> float:
        """Estimate the cost of generating embeddings"""
        # OpenAI pricing for text-embedding-ada-002: $0.10 per 1M tokens
        cost_per_million = 0.10
        estimated_cost = (num_tokens / 1_000_000) * cost_per_million
        return estimated_cost