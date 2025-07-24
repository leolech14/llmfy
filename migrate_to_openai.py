#!/usr/bin/env python3
"""
🔄 Migrate existing embeddings to OpenAI (1536-dim)

This script re-processes all documents in the main collection (384-dim)
and migrates them to use OpenAI embeddings in the hybrid collection.
"""

import chromadb
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from src.core.embedding_generator import EmbeddingGenerator
from src.embeddings.hybrid_embedder import HybridEmbedder

console = Console()

def migrate_embeddings():
    """Migrate all 384-dim embeddings to 1536-dim OpenAI embeddings"""
    
    console.print("[bold cyan]🔄 Starting OpenAI Embedding Migration[/bold cyan]\n")
    
    # Initialize ChromaDB
    db_path = "data/vector_db/chroma_db"
    client = chromadb.PersistentClient(path=db_path)
    
    # Get collections
    try:
        main_collection = client.get_collection("knowledge_base")
        main_count = main_collection.count()
        console.print(f"[green]✓ Found main collection with {main_count} chunks (384-dim)[/green]")
    except:
        console.print("[red]✗ Main collection not found[/red]")
        return
    
    try:
        hybrid_collection = client.get_collection("knowledge_base_hybrid")
        hybrid_count = hybrid_collection.count()
        console.print(f"[green]✓ Found hybrid collection with {hybrid_count} chunks (1536-dim)[/green]")
    except:
        console.print("[yellow]Creating hybrid collection...[/yellow]")
        hybrid_collection = client.create_collection("knowledge_base_hybrid")
        hybrid_count = 0
    
    # Get all documents from main collection
    console.print("\n[blue]Fetching documents from main collection...[/blue]")
    
    batch_size = 100
    migrated = 0
    
    # Initialize OpenAI embedder
    embedder = EmbeddingGenerator()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        
        task = progress.add_task("Migrating embeddings...", total=main_count)
        
        # Process in batches
        offset = 0
        while offset < main_count:
            # Get batch
            results = main_collection.get(
                limit=batch_size,
                offset=offset,
                include=["documents", "metadatas"]
            )
            
            if not results['documents']:
                break
            
            # Generate new OpenAI embeddings
            new_embeddings = []
            for doc_text in results['documents']:
                try:
                    embedding = embedder.generate_embedding(doc_text)
                    new_embeddings.append(embedding)
                except Exception as e:
                    console.print(f"\n[red]Error generating embedding: {e}[/red]")
                    continue
            
            # Add to hybrid collection
            if new_embeddings:
                try:
                    # Update metadata to indicate OpenAI embeddings
                    metadatas = results['metadatas']
                    for metadata in metadatas:
                        metadata['embedding_type'] = 'openai'
                        metadata['embedding_model'] = 'text-embedding-ada-002'
                        metadata['embedding_dim'] = 1536
                        metadata['migrated_from_384'] = True
                    
                    hybrid_collection.add(
                        ids=results['ids'][:len(new_embeddings)],
                        embeddings=new_embeddings,
                        metadatas=metadatas[:len(new_embeddings)],
                        documents=results['documents'][:len(new_embeddings)]
                    )
                    
                    migrated += len(new_embeddings)
                    progress.update(task, advance=len(new_embeddings))
                    
                except Exception as e:
                    console.print(f"\n[red]Error adding to hybrid collection: {e}[/red]")
            
            offset += batch_size
    
    # Summary
    console.print(f"\n[bold green]✅ Migration Complete![/bold green]")
    console.print(f"  • Migrated: {migrated} chunks")
    console.print(f"  • Original: {main_count} chunks (384-dim)")
    console.print(f"  • Hybrid total: {hybrid_collection.count()} chunks (1536-dim)")
    
    # Cost estimate
    tokens_processed = migrated * 250  # Estimate 250 tokens per chunk
    cost = (tokens_processed / 1_000_000) * 0.10  # $0.10 per 1M tokens
    console.print(f"\n💰 Estimated cost: ${cost:.4f}")
    
    # Recommendation
    console.print("\n[yellow]📌 Recommendation:[/yellow]")
    console.print("  1. Update search to use only the hybrid collection")
    console.print("  2. Consider removing the main collection after verification")
    console.print("  3. Set all new processing to use OpenAI embeddings")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate embeddings to OpenAI")
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be migrated without doing it')
    
    args = parser.parse_args()
    
    if args.dry_run:
        console.print("[yellow]DRY RUN MODE - No changes will be made[/yellow]\n")
    
    # Check for API key
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    if not os.getenv('OPENAI_API_KEY'):
        console.print("[red]Error: OPENAI_API_KEY not found in environment[/red]")
        console.print("Please set your OpenAI API key in .env file")
        sys.exit(1)
    
    migrate_embeddings()