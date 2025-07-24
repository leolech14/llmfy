#!/usr/bin/env python3
"""
🔍 Knowledge Search - Search through the vector database
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()

class KnowledgeSearch:
    """Search interface for the llmfy knowledge base"""
    
    def __init__(self, db_path: str = "data/vector_db/chroma_db"):
        """Initialize search with ChromaDB"""
        self.db_path = Path(db_path)
        
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found at {db_path}")
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection("knowledge_base")
            console.print(f"[green]✅ Connected to knowledge base with {self.collection.count()} chunks[/green]")
        except Exception as e:
            console.print(f"[red]Error: Could not access knowledge base: {e}[/red]")
            raise
    
    def search(self, 
               query: str, 
               n_results: int = 10,
               filter_metadata: Optional[Dict] = None,
               show_scores: bool = True) -> List[Dict[str, Any]]:
        """
        Search the knowledge base
        
        Args:
            query: Search query text
            n_results: Number of results to return
            filter_metadata: Optional metadata filters
            show_scores: Whether to display similarity scores
        
        Returns:
            List of search results with content and metadata
        """
        console.print(f"\n[cyan]🔍 Searching for: '{query}'[/cyan]")
        
        try:
            # Perform similarity search
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filter_metadata,
                include=["documents", "metadatas", "distances"]
            )
            
            if not results['documents'][0]:
                console.print("[yellow]No results found[/yellow]")
                return []
            
            # Format results
            formatted_results = []
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i]
                
                # Calculate similarity score (1 - distance for cosine similarity)
                similarity_score = 1 - distance
                
                result = {
                    'content': doc,
                    'metadata': metadata,
                    'similarity_score': similarity_score,
                    'rank': i + 1
                }
                formatted_results.append(result)
            
            # Display results
            self._display_results(formatted_results, show_scores)
            
            return formatted_results
            
        except Exception as e:
            console.print(f"[red]Search error: {e}[/red]")
            return []
    
    def search_by_filename(self, filename_pattern: str, n_results: int = 50) -> List[Dict[str, Any]]:
        """Search for chunks from specific files"""
        console.print(f"\n[cyan]📄 Searching for files matching: '{filename_pattern}'[/cyan]")
        
        # Get all chunks and filter by filename
        try:
            # Get sample to check metadata structure
            sample = self.collection.get(limit=1, include=["metadatas"])
            if not sample['metadatas']:
                console.print("[yellow]No documents in knowledge base[/yellow]")
                return []
            
            # Search all documents (up to 1000)
            all_results = self.collection.get(
                limit=1000,
                include=["documents", "metadatas"]
            )
            
            # Filter by filename pattern
            matching_results = []
            for i, metadata in enumerate(all_results['metadatas']):
                filename = metadata.get('filename', '')
                if filename_pattern.lower() in filename.lower():
                    matching_results.append({
                        'content': all_results['documents'][i],
                        'metadata': metadata,
                        'filename': filename
                    })
            
            if not matching_results:
                console.print(f"[yellow]No files matching '{filename_pattern}' found[/yellow]")
                return []
            
            # Group by filename
            files = {}
            for result in matching_results:
                fname = result['filename']
                if fname not in files:
                    files[fname] = []
                files[fname].append(result)
            
            # Display summary
            console.print(f"\n[green]Found {len(matching_results)} chunks from {len(files)} files:[/green]")
            for fname, chunks in files.items():
                console.print(f"  • {fname}: {len(chunks)} chunks")
            
            return matching_results[:n_results]
            
        except Exception as e:
            console.print(f"[red]Search error: {e}[/red]")
            return []
    
    def _display_results(self, results: List[Dict[str, Any]], show_scores: bool = True):
        """Display search results in a formatted table"""
        if not results:
            return
        
        # Create results table
        table = Table(
            title=f"\n🔍 Search Results ({len(results)} matches)",
            show_header=True,
            header_style="bold cyan"
        )
        
        table.add_column("Rank", style="cyan", width=6)
        table.add_column("Content", style="white", width=80)
        if show_scores:
            table.add_column("Score", style="green", width=8)
        table.add_column("Source", style="blue", width=30)
        
        for result in results:
            # Truncate content for display
            content = result['content']
            if len(content) > 200:
                content = content[:200] + "..."
            
            # Format metadata
            metadata = result['metadata']
            source = metadata.get('filename', 'Unknown')
            chunk_info = f"Chunk {metadata.get('chunk_index', '?')}/{metadata.get('chunk_total', '?')}"
            
            # Build row
            row = [
                str(result['rank']),
                content,
            ]
            
            if show_scores:
                row.append(f"{result['similarity_score']:.3f}")
            
            row.append(f"{source}\n{chunk_info}")
            
            table.add_row(*row)
        
        console.print(table)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        try:
            count = self.collection.count()
            
            # Get sample metadata to analyze
            sample = self.collection.get(
                limit=min(100, count),
                include=["metadatas"]
            )
            
            # Analyze files
            files = set()
            total_quality = 0
            quality_count = 0
            
            for metadata in sample['metadatas']:
                if 'filename' in metadata:
                    files.add(metadata['filename'])
                if 'final_quality_score' in metadata:
                    total_quality += metadata['final_quality_score']
                    quality_count += 1
            
            stats = {
                'total_chunks': count,
                'unique_files': len(files),
                'average_quality': total_quality / quality_count if quality_count > 0 else 0,
                'sample_files': list(files)[:5]  # First 5 files
            }
            
            # Display stats
            console.print(Panel.fit(
                f"[bold cyan]📊 Knowledge Base Statistics[/bold cyan]\n\n"
                f"Total Chunks: {stats['total_chunks']}\n"
                f"Unique Files: {stats['unique_files']}\n"
                f"Average Quality: {stats['average_quality']:.1f}/10\n\n"
                f"Sample Files:\n" + 
                "\n".join(f"  • {f}" for f in stats['sample_files']),
                border_style="cyan"
            ))
            
            return stats
            
        except Exception as e:
            console.print(f"[red]Error getting stats: {e}[/red]")
            return {}


def main():
    """CLI interface for knowledge search"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Search the llmfy knowledge base")
    parser.add_argument('query', nargs='*', help='Search query')
    parser.add_argument('--results', '-n', type=int, default=10,
                       help='Number of results to return (default: 10)')
    parser.add_argument('--file', '-f', help='Search for specific filename')
    parser.add_argument('--stats', '-s', action='store_true',
                       help='Show knowledge base statistics')
    parser.add_argument('--no-scores', action='store_true',
                       help='Hide similarity scores')
    
    args = parser.parse_args()
    
    # Initialize search
    try:
        search = KnowledgeSearch()
    except Exception as e:
        console.print(f"[red]Failed to initialize search: {e}[/red]")
        return
    
    # Handle different modes
    if args.stats:
        search.get_stats()
    elif args.file:
        search.search_by_filename(args.file, n_results=args.results)
    elif args.query:
        query = ' '.join(args.query)
        search.search(
            query, 
            n_results=args.results,
            show_scores=not args.no_scores
        )
    else:
        console.print("[yellow]Please provide a search query or use --stats[/yellow]")
        console.print("Usage: llmfy search \"your query here\"")


if __name__ == "__main__":
    main()