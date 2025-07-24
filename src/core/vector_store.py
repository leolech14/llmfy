from typing import List, Dict, Any, Optional, Tuple
import uuid
from datetime import datetime
import json

import chromadb
from chromadb.config import Settings
from pinecone import Pinecone, ServerlessSpec
from langchain.schema import Document
from rich.console import Console
from rich.table import Table
from rich.progress import track

from .config import Config

console = Console()

class VectorStore:
    def __init__(self, storage_mode: Optional[str] = None):
        self.config = Config()
        self.storage_mode = storage_mode or self.config.STORAGE_MODE
        
        # Initialize stores based on mode
        self.chroma_client = None
        self.chroma_collection = None
        self.pinecone_index = None
        
        if self.storage_mode in ['local', 'hybrid']:
            self._init_chromadb()
        
        if self.storage_mode in ['pinecone', 'hybrid']:
            self._init_pinecone()
    
    def _init_chromadb(self):
        """Initialize ChromaDB client and collection"""
        console.print("[blue]Initializing ChromaDB...[/blue]")
        
        self.chroma_client = chromadb.PersistentClient(
            path=self.config.CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get collection - use different collections for different embedding dimensions
        # This avoids dimension mismatch errors
        try:
            # Try to get existing collection
            self.chroma_collection = self.chroma_client.get_collection("knowledge_base")
        except:
            # Create new collection if it doesn't exist
            self.chroma_collection = self.chroma_client.create_collection(
                name="knowledge_base",
                metadata={"description": "Personal knowledge base embeddings"}
            )
        
        # Also create a hybrid collection for mixed embeddings
        try:
            self.hybrid_collection = self.chroma_client.get_or_create_collection(
                name="knowledge_base_hybrid",
                metadata={"description": "Hybrid embeddings (384/1536 dims)"}
            )
        except:
            self.hybrid_collection = None
        
        console.print(f"[green]ChromaDB initialized with {self.chroma_collection.count()} documents[/green]")
    
    def _init_pinecone(self):
        """Initialize Pinecone client and index"""
        console.print("[blue]Initializing Pinecone...[/blue]")
        
        # Initialize Pinecone
        pc = Pinecone(api_key=self.config.PINECONE_API_KEY)
        
        # Create index if it doesn't exist
        index_name = self.config.PINECONE_INDEX_NAME
        
        try:
            # Check if index exists
            existing_indexes = pc.list_indexes()
            index_exists = any(idx.name == index_name for idx in existing_indexes)
            
            if not index_exists:
                console.print(f"[yellow]Creating new Pinecone index: {index_name}[/yellow]")
                pc.create_index(
                    name=index_name,
                    dimension=1536,  # OpenAI ada-002 dimension
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region=self.config.PINECONE_ENVIRONMENT
                    )
                )
                console.print(f"[green]Pinecone index created: {index_name}[/green]")
            
            self.pinecone_index = pc.Index(index_name)
            
            # Get index stats
            stats = self.pinecone_index.describe_index_stats()
            total_vectors = stats.get('total_vector_count', 0)
            console.print(f"[green]Pinecone initialized with {total_vectors} vectors[/green]")
            
        except Exception as e:
            console.print(f"[red]Error initializing Pinecone: {e}[/red]")
            if self.storage_mode == 'pinecone':
                raise
            else:
                console.print("[yellow]Continuing with local storage only[/yellow]")
                self.storage_mode = 'local'
    
    def add_documents(self, doc_embeddings: List[Tuple[Document, List[float]]]) -> int:
        """Add documents with embeddings to the vector store(s)"""
        console.print(f"[blue]Adding {len(doc_embeddings)} documents to {self.storage_mode} storage...[/blue]")
        
        success_count = 0
        
        if self.storage_mode in ['local', 'hybrid']:
            success_count = self._add_to_chromadb(doc_embeddings)
        
        if self.storage_mode in ['pinecone', 'hybrid']:
            success_count = max(success_count, self._add_to_pinecone(doc_embeddings))
        
        console.print(f"[green]Successfully added {success_count} documents[/green]")
        return success_count
    
    def _add_to_chromadb(self, doc_embeddings: List[Tuple[Document, List[float]]]) -> int:
        """Add documents to ChromaDB"""
        ids = []
        embeddings = []
        metadatas = []
        documents = []
        
        for doc, embedding in doc_embeddings:
            # Generate unique ID
            doc_id = str(uuid.uuid4())
            
            # Prepare metadata (ChromaDB requires JSON-serializable metadata)
            metadata = {
                k: v for k, v in doc.metadata.items() 
                if isinstance(v, (str, int, float, bool))
            }
            metadata['added_at'] = datetime.now().isoformat()
            
            ids.append(doc_id)
            embeddings.append(embedding)
            metadatas.append(metadata)
            documents.append(doc.page_content)
        
        try:
            # Check embedding dimension and use appropriate collection
            if embeddings and len(embeddings[0]) == 384:
                # Use main collection for 384-dim embeddings
                self.chroma_collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=documents
                )
            elif self.hybrid_collection and embeddings and len(embeddings[0]) == 1536:
                # Use hybrid collection for 1536-dim embeddings
                self.hybrid_collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=documents
                )
            else:
                # Try to add to whichever collection accepts it
                try:
                    self.chroma_collection.add(
                        ids=ids,
                        embeddings=embeddings,
                        metadatas=metadatas,
                        documents=documents
                    )
                except:
                    if self.hybrid_collection:
                        self.hybrid_collection.add(
                            ids=ids,
                            embeddings=embeddings,
                            metadatas=metadatas,
                            documents=documents
                        )
                    else:
                        raise
            
            return len(ids)
        except Exception as e:
            console.print(f"[red]Error adding to ChromaDB: {e}[/red]")
            console.print(f"[yellow]Embedding dimension: {len(embeddings[0]) if embeddings else 'unknown'}[/yellow]")
            return 0
    
    def _add_to_pinecone(self, doc_embeddings: List[Tuple[Document, List[float]]]) -> int:
        """Add documents to Pinecone"""
        vectors = []
        
        for doc, embedding in doc_embeddings:
            # Generate unique ID
            doc_id = str(uuid.uuid4())
            
            # Prepare metadata (Pinecone has size limits)
            metadata = {
                'text': doc.page_content[:1000],  # Truncate long text
                'source': doc.metadata.get('source', ''),
                'filename': doc.metadata.get('filename', ''),
                'file_type': doc.metadata.get('file_type', ''),
                'chunk_index': doc.metadata.get('chunk_index', 0),
                'added_at': datetime.now().isoformat()
            }
            
            # Add other metadata if within size limits
            for key, value in doc.metadata.items():
                if key not in metadata and isinstance(value, (str, int, float, bool)):
                    metadata[key] = value
            
            vectors.append({
                'id': doc_id,
                'values': embedding,
                'metadata': metadata
            })
        
        try:
            # Upsert in batches
            batch_size = 100
            success_count = 0
            
            for i in track(range(0, len(vectors), batch_size), description="Uploading to Pinecone..."):
                batch = vectors[i:i + batch_size]
                self.pinecone_index.upsert(vectors=batch)
                success_count += len(batch)
            
            return success_count
        except Exception as e:
            console.print(f"[red]Error adding to Pinecone: {e}[/red]")
            return 0
    
    def search(self, query_embedding: List[float], top_k: int = 5, 
              filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        results = []
        
        if self.storage_mode in ['local', 'hybrid']:
            chroma_results = self._search_chromadb(query_embedding, top_k, filter_dict)
            results.extend(chroma_results)
        
        if self.storage_mode in ['pinecone', 'hybrid']:
            pinecone_results = self._search_pinecone(query_embedding, top_k, filter_dict)
            results.extend(pinecone_results)
        
        # Deduplicate and sort by score if using hybrid mode
        if self.storage_mode == 'hybrid':
            seen = set()
            unique_results = []
            for result in results:
                content_hash = hash(result['content'][:100])
                if content_hash not in seen:
                    seen.add(content_hash)
                    unique_results.append(result)
            
            results = sorted(unique_results, key=lambda x: x['score'], reverse=True)[:top_k]
        
        return results
    
    def _search_chromadb(self, query_embedding: List[float], top_k: int, 
                        filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search in ChromaDB"""
        try:
            where_clause = filter_dict if filter_dict else None
            
            results = self.chroma_collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause,
                include=['metadatas', 'documents', 'distances']
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'score': 1 - results['distances'][0][i],  # Convert distance to similarity
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'source': 'chromadb'
                })
            
            return formatted_results
            
        except Exception as e:
            console.print(f"[red]Error searching ChromaDB: {e}[/red]")
            return []
    
    def _search_pinecone(self, query_embedding: List[float], top_k: int, 
                        filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search in Pinecone"""
        try:
            results = self.pinecone_index.query(
                vector=query_embedding,
                top_k=top_k,
                filter=filter_dict,
                include_metadata=True
            )
            
            # Format results
            formatted_results = []
            for match in results['matches']:
                formatted_results.append({
                    'id': match['id'],
                    'score': match['score'],
                    'content': match['metadata'].get('text', ''),
                    'metadata': match['metadata'],
                    'source': 'pinecone'
                })
            
            return formatted_results
            
        except Exception as e:
            console.print(f"[red]Error searching Pinecone: {e}[/red]")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the vector stores"""
        stats = {
            'storage_mode': self.storage_mode,
            'chromadb': None,
            'pinecone': None
        }
        
        if self.chroma_collection:
            stats['chromadb'] = {
                'total_documents': self.chroma_collection.count(),
                'collection_name': 'knowledge_base'
            }
        
        if self.pinecone_index:
            index_stats = self.pinecone_index.describe_index_stats()
            stats['pinecone'] = {
                'total_vectors': index_stats.get('total_vector_count', 0),
                'index_name': self.config.PINECONE_INDEX_NAME,
                'dimension': index_stats.get('dimension', 1536)
            }
        
        return stats
    
    def display_statistics(self):
        """Display statistics in a formatted table"""
        stats = self.get_statistics()
        
        table = Table(title="Vector Store Statistics")
        table.add_column("Storage", style="cyan")
        table.add_column("Documents/Vectors", style="green")
        table.add_column("Details", style="yellow")
        
        if stats['chromadb']:
            table.add_row(
                "ChromaDB",
                str(stats['chromadb']['total_documents']),
                f"Collection: {stats['chromadb']['collection_name']}"
            )
        
        if stats['pinecone']:
            table.add_row(
                "Pinecone",
                str(stats['pinecone']['total_vectors']),
                f"Index: {stats['pinecone']['index_name']} (dim: {stats['pinecone']['dimension']})"
            )
        
        console.print(table)
