#!/usr/bin/env python3
"""
🔍 Unified Search - Search across all collections regardless of embedding dimension
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
import chromadb
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import numpy as np
import json
import re

console = Console()

class UnifiedSearch:
    """Unified search that handles multiple collections with different embeddings"""
    
    def __init__(self, db_path: str = "data/vector_db/chroma_db"):
        """Initialize unified search"""
        self.db_path = Path(db_path)
        
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found at {db_path}")
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        
        # Get all collections
        self.collections = self._get_all_collections()
        
        # Initialize embedders for different dimensions
        self._init_embedders()
        
        # Load semantic links if available
        self.semantic_links = self._load_semantic_links()
    
    def _get_all_collections(self) -> Dict[str, Any]:
        """Get all available collections"""
        collections = {}
        
        # Try to get main collection (384-dim)
        try:
            main = self.client.get_collection("knowledge_base")
            collections['main'] = {
                'collection': main,
                'count': main.count(),
                'embedding_dim': 384,
                'embedding_type': 'local'
            }
        except:
            pass
        
        # Try to get hybrid collection (1536-dim)
        try:
            hybrid = self.client.get_collection("knowledge_base_hybrid")
            collections['hybrid'] = {
                'collection': hybrid,
                'count': hybrid.count(),
                'embedding_dim': 1536,
                'embedding_type': 'openai'
            }
        except:
            pass
        
        # Display summary
        total_chunks = sum(c['count'] for c in collections.values())
        console.print(f"[green]✅ Connected to {len(collections)} collections with {total_chunks} total chunks[/green]")
        
        for name, info in collections.items():
            console.print(f"[dim]   {name}: {info['count']} chunks ({info['embedding_dim']}-dim {info['embedding_type']})[/dim]")
        
        return collections
    
    def _init_embedders(self):
        """Initialize embedders for different dimensions"""
        self.embedders = {}
        
        # Local embedder (384-dim)
        try:
            from sentence_transformers import SentenceTransformer
            self.embedders[384] = SentenceTransformer('all-MiniLM-L6-v2')
            console.print("[dim]   ✓ Local embedder ready (384-dim)[/dim]")
        except:
            console.print("[yellow]   ⚠ Local embedder not available[/yellow]")
        
        # OpenAI embedder (1536-dim)
        try:
            import openai
            import os
            from dotenv import load_dotenv
            load_dotenv()
            
            if os.getenv('OPENAI_API_KEY'):
                self.embedders[1536] = 'openai'  # Flag for OpenAI
                console.print("[dim]   ✓ OpenAI embedder ready (1536-dim)[/dim]")
            else:
                console.print("[yellow]   ⚠ OpenAI API key not found[/yellow]")
        except:
            console.print("[yellow]   ⚠ OpenAI embedder not available[/yellow]")
    
    def _generate_embedding(self, text: str, dimension: int) -> Optional[List[float]]:
        """Generate embedding for specific dimension"""
        if dimension not in self.embedders:
            return None
        
        if dimension == 384:
            # Local embedding
            return self.embedders[384].encode(text).tolist()
        
        elif dimension == 1536 and self.embedders.get(1536) == 'openai':
            # OpenAI embedding
            import openai
            try:
                response = openai.embeddings.create(
                    model="text-embedding-ada-002",
                    input=text
                )
                return response.data[0].embedding
            except Exception as e:
                console.print(f"[red]OpenAI embedding error: {e}[/red]")
                return None
        
        return None
    
    def search(self, 
               query: str, 
               n_results: int = 10,
               collections_to_search: Optional[List[str]] = None,
               use_hybrid: bool = True) -> List[Dict[str, Any]]:
        """
        Search across all collections with hybrid keyword + semantic matching
        
        Args:
            query: Search query text
            n_results: Number of results to return
            collections_to_search: Specific collections to search (None = all)
            use_hybrid: Use hybrid search (keyword + semantic)
        
        Returns:
            List of search results with content and metadata
        """
        console.print(f"\n[cyan]🔍 Searching for: '{query}'[/cyan]")
        
        all_results = []
        
        # Extract potential keywords/exact phrases for hybrid search
        keywords = self._extract_keywords(query) if use_hybrid else []
        if keywords:
            console.print(f"[dim]   Keywords: {', '.join(keywords)}[/dim]")
        
        # Search each collection
        for name, info in self.collections.items():
            if collections_to_search and name not in collections_to_search:
                continue
            
            # Generate embedding for this collection's dimension
            embedding = self._generate_embedding(query, info['embedding_dim'])
            
            if not embedding:
                console.print(f"[yellow]   ⚠ Skipping {name} collection (no embedder)[/yellow]")
                continue
            
            try:
                # Semantic search
                results = info['collection'].query(
                    query_embeddings=[embedding],
                    n_results=n_results * 2,  # Get more for filtering
                    include=["documents", "metadatas", "distances"]
                )
                
                # Get IDs separately if needed
                ids = results.get('ids', [[]])[0] if 'ids' in results else [f"chunk_{i}" for i in range(len(results['documents'][0]))]
                
                # Process results
                for i, doc in enumerate(results['documents'][0]):
                    if doc:  # Skip empty results
                        result = {
                            'id': ids[i] if i < len(ids) else f"chunk_{i}",
                            'content': doc,
                            'metadata': results['metadatas'][0][i],
                            'distance': results['distances'][0][i],
                            'similarity_score': 1 - results['distances'][0][i],
                            'collection': name,
                            'keyword_matches': 0,
                            'hybrid_score': 0
                        }
                        
                        # Add keyword matching score if hybrid search
                        if use_hybrid and keywords:
                            keyword_score = self._calculate_keyword_score(doc, keywords)
                            result['keyword_matches'] = keyword_score
                            # Hybrid score: 70% semantic, 30% keyword
                            result['hybrid_score'] = (0.7 * result['similarity_score']) + (0.3 * keyword_score)
                        else:
                            result['hybrid_score'] = result['similarity_score']
                        
                        all_results.append(result)
                
                console.print(f"[dim]   ✓ Found {len(results['documents'][0])} results in {name}[/dim]")
                
            except Exception as e:
                console.print(f"[red]   ✗ Error searching {name}: {e}[/red]")
        
        # Sort by hybrid score
        all_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        # Expand with semantic links
        if self.semantic_links:
            all_results = self._expand_with_links(all_results, n_results * 2)
        
        # Take top results
        top_results = all_results[:n_results]
        
        # Add ranks
        for i, result in enumerate(top_results):
            result['rank'] = i + 1
        
        # Display results
        self._display_results(top_results)
        
        return top_results
    
    def _display_results(self, results: List[Dict[str, Any]]):
        """Display search results in a formatted table"""
        if not results:
            console.print("[yellow]No results found[/yellow]")
            return
        
        # Create results table
        table = Table(
            title=f"\n🔍 Search Results ({len(results)} matches)",
            show_header=True,
            header_style="bold cyan"
        )
        
        table.add_column("Rank", style="cyan", width=6)
        table.add_column("Content", style="white", width=70)
        table.add_column("Score", style="green", width=8)
        table.add_column("Source", style="blue", width=25)
        table.add_column("Collection", style="yellow", width=10)
        
        for result in results:
            # Truncate content for display
            content = result['content']
            if len(content) > 150:
                content = content[:150] + "..."
            
            # Format metadata
            metadata = result['metadata']
            source = metadata.get('filename', 'Unknown')[:25]
            chunk_info = f"Chunk {metadata.get('chunk_index', '?')}/{metadata.get('chunk_total', '?')}"
            
            table.add_row(
                str(result['rank']),
                content,
                f"{result['similarity_score']:.3f}",
                f"{source}\n{chunk_info}",
                result['collection']
            )
        
        console.print(table)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about all collections"""
        stats = {
            'total_collections': len(self.collections),
            'total_chunks': sum(c['count'] for c in self.collections.values()),
            'collections': {}
        }
        
        for name, info in self.collections.items():
            stats['collections'][name] = {
                'chunks': info['count'],
                'embedding_dim': info['embedding_dim'],
                'embedding_type': info['embedding_type']
            }
        
        # Display stats
        console.print(Panel.fit(
            f"[bold cyan]📊 Knowledge Base Statistics[/bold cyan]\n\n"
            f"Total Collections: {stats['total_collections']}\n"
            f"Total Chunks: {stats['total_chunks']}\n\n"
            f"Collections:\n" + 
            "\n".join(f"  • {name}: {info['chunks']} chunks ({info['embedding_dim']}-dim)"
                     for name, info in stats['collections'].items()),
            border_style="cyan"
        ))
        
        return stats
    
    def _load_semantic_links(self) -> Dict[str, List[Dict]]:
        """Load semantic links from file"""
        links_file = self.db_path / "semantic_links.json"
        if links_file.exists():
            try:
                with open(links_file, 'r') as f:
                    data = json.load(f)
                    # Create bidirectional lookup
                    links_by_id = {}
                    for link in data.get('links', []):
                        # Forward links
                        if link['source_id'] not in links_by_id:
                            links_by_id[link['source_id']] = []
                        links_by_id[link['source_id']].append(link)
                        
                        # Backward links (for bidirectional traversal)
                        if link['target_id'] not in links_by_id:
                            links_by_id[link['target_id']] = []
                        links_by_id[link['target_id']].append({
                            'source_id': link['target_id'],
                            'target_id': link['source_id'],
                            'type': f"reverse_{link['type']}",
                            'strength': link['strength'],
                            'description': f"Reverse: {link['description']}"
                        })
                    
                    console.print(f"[dim]   ✓ Loaded {len(data.get('links', []))} semantic links[/dim]")
                    return links_by_id
            except Exception as e:
                console.print(f"[yellow]   ⚠ Could not load semantic links: {e}[/yellow]")
        return {}
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords and exact phrases from query"""
        keywords = []
        
        # Extract quoted phrases
        quoted = re.findall(r'"([^"]+)"', query)
        keywords.extend(quoted)
        
        # Extract numbers and technical terms
        numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', query)
        keywords.extend(numbers)
        
        # Extract technical terms (camelCase, snake_case, hyphenated)
        technical = re.findall(r'\b(?:[a-z]+_[a-z_]+|[a-zA-Z]+(?:[A-Z][a-z]*)+|[a-z]+-[a-z-]+)\b', query)
        keywords.extend(technical)
        
        # Extract specific important keywords
        important_terms = ['failure', 'reduction', 'optimal', 'chunk size', 'tokens', 'embedding', 'dimension']
        for term in important_terms:
            if term.lower() in query.lower():
                keywords.append(term)
        
        return list(set(keywords))  # Remove duplicates
    
    def _calculate_keyword_score(self, content: str, keywords: List[str]) -> float:
        """Calculate keyword matching score"""
        content_lower = content.lower()
        matches = 0
        
        for keyword in keywords:
            # Exact match gets full point
            if keyword.lower() in content_lower:
                matches += 1
            # Partial match gets half point
            elif any(word in content_lower for word in keyword.lower().split()):
                matches += 0.5
        
        # Normalize to 0-1 range
        return min(matches / max(len(keywords), 1), 1.0)
    
    def _expand_with_links(self, results: List[Dict], max_results: int) -> List[Dict]:
        """Expand search results using semantic links"""
        if not self.semantic_links:
            return results
        
        # Track seen IDs to avoid duplicates
        seen_ids = {r['id'] for r in results}
        expanded_results = results.copy()
        
        # Get linked chunks for top results
        for result in results[:5]:  # Only expand top 5 to avoid explosion
            chunk_id = result['id']
            
            if chunk_id in self.semantic_links:
                links = self.semantic_links[chunk_id]
                
                # Sort by strength and take top links
                strong_links = sorted(links, key=lambda x: x['strength'], reverse=True)[:3]
                
                for link in strong_links:
                    target_id = link['target_id']
                    
                    if target_id not in seen_ids:
                        # Fetch the linked chunk
                        linked_chunk = self._fetch_chunk_by_id(target_id)
                        
                        if linked_chunk:
                            # Add with adjusted score
                            linked_result = linked_chunk.copy()
                            linked_result['hybrid_score'] = result['hybrid_score'] * link['strength'] * 0.8
                            linked_result['via_link'] = {
                                'from': chunk_id,
                                'type': link['type'],
                                'strength': link['strength']
                            }
                            expanded_results.append(linked_result)
                            seen_ids.add(target_id)
        
        # Re-sort by score
        expanded_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        return expanded_results[:max_results]
    
    def _fetch_chunk_by_id(self, chunk_id: str) -> Optional[Dict]:
        """Fetch a specific chunk by ID from any collection"""
        for name, info in self.collections.items():
            try:
                results = info['collection'].get(
                    ids=[chunk_id],
                    include=["documents", "metadatas"]
                )
                
                if results['documents']:
                    return {
                        'id': chunk_id,
                        'content': results['documents'][0],
                        'metadata': results['metadatas'][0],
                        'collection': name,
                        'similarity_score': 0,
                        'keyword_matches': 0
                    }
            except:
                continue
        return None


def main():
    """CLI interface for unified search"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Search across all llmfy collections")
    parser.add_argument('query', nargs='*', help='Search query')
    parser.add_argument('--results', '-n', type=int, default=10,
                       help='Number of results (default: 10)')
    parser.add_argument('--collection', '-c', choices=['main', 'hybrid', 'all'],
                       default='all', help='Which collection to search')
    parser.add_argument('--stats', '-s', action='store_true',
                       help='Show statistics')
    
    args = parser.parse_args()
    
    # Initialize search
    try:
        search = UnifiedSearch()
    except Exception as e:
        console.print(f"[red]Failed to initialize search: {e}[/red]")
        return
    
    # Handle different modes
    if args.stats:
        search.get_stats()
    elif args.query:
        query = ' '.join(args.query)
        collections = None if args.collection == 'all' else [args.collection]
        search.search(query, n_results=args.results, collections_to_search=collections)
    else:
        console.print("[yellow]Please provide a search query or use --stats[/yellow]")


if __name__ == "__main__":
    main()