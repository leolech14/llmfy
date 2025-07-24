#!/usr/bin/env python3

import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import argparse
from datetime import datetime
import json

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from .config import Config
from .document_loader import DocumentLoader
from .text_processor import TextProcessor
from .embedding_generator import EmbeddingGenerator
from .vector_store import VectorStore
from .data_assessor import DataAssessor
from .processing_planner import ProcessingPlanner

console = Console()

class KnowledgeBasePipeline:
    def __init__(self, storage_mode: Optional[str] = None):
        self.config = Config()
        self.config.create_directories()
        
        self.loader = DocumentLoader()
        self.processor = TextProcessor()
        self.embedder = EmbeddingGenerator()
        self.vector_store = VectorStore(storage_mode)
        self.assessor = DataAssessor()
        self.planner = ProcessingPlanner()
        
        self.stats = {
            'documents_loaded': 0,
            'chunks_created': 0,
            'embeddings_generated': 0,
            'vectors_stored': 0,
            'errors': []
        }
    
    def process_inbox(self, move_processed: bool = True):
        """Process all documents in the inbox directory"""
        console.print(Panel.fit("📥 Processing Inbox Documents", style="bold blue"))
        
        # Load documents from inbox
        documents = self.loader.load_from_inbox()
        self.stats['documents_loaded'] = len(documents)
        
        if not documents:
            console.print("[yellow]No new documents to process in inbox[/yellow]")
            return
        
        # Process documents into chunks
        chunks = self.processor.process_documents(documents)
        self.stats['chunks_created'] = len(chunks)
        
        if not chunks:
            console.print("[yellow]No chunks created from documents[/yellow]")
            return
        
        # Generate embeddings
        doc_embeddings = self.embedder.process_documents(chunks)
        self.stats['embeddings_generated'] = len(doc_embeddings)
        
        # Store in vector database(s)
        stored_count = self.vector_store.add_documents(doc_embeddings)
        self.stats['vectors_stored'] = stored_count
        
        # Move processed files
        if move_processed:
            processed_files = set()
            for doc in documents:
                source_path = Path(doc.metadata.get('source', ''))
                if source_path.exists() and str(source_path).startswith(str(self.config.INBOX_DIR)):
                    processed_files.add(source_path)
            
            for file_path in processed_files:
                self.loader.move_to_processed(file_path)
        
        self._display_results()
    
    def process_directory(self, directory_path: str, recursive: bool = True):
        """Process documents from a specific directory"""
        console.print(Panel.fit(f"📁 Processing Directory: {directory_path}", style="bold blue"))
        
        path = Path(directory_path)
        if not path.exists():
            console.print(f"[red]Directory not found: {directory_path}[/red]")
            return
        
        # Load documents
        documents = self.loader.load_directory(path, recursive)
        self.stats['documents_loaded'] = len(documents)
        
        if not documents:
            console.print("[yellow]No documents found in directory[/yellow]")
            return
        
        # Continue with processing pipeline
        self._process_documents(documents)
        self._display_results()
    
    def process_github_repo(self, repo_url: str, branch: str = "main", 
                           file_patterns: Optional[List[str]] = None):
        """Process documents from a GitHub repository"""
        console.print(Panel.fit(f"🐙 Processing GitHub Repo: {repo_url}", style="bold blue"))
        
        # Load documents from repo
        documents = self.loader.load_github_repo(repo_url, branch, file_patterns)
        self.stats['documents_loaded'] = len(documents)
        
        if not documents:
            console.print("[yellow]No documents found in repository[/yellow]")
            return
        
        # Continue with processing pipeline
        self._process_documents(documents)
        self._display_results()
    
    def assess_and_plan(self, source_path: Optional[str] = None, 
                       repo_url: Optional[str] = None, save_plan: bool = True) -> Dict[str, Any]:
        """Assess data and create processing plan before actual processing"""
        console.print(Panel.fit("🔍 Assess and Plan Mode", style="bold cyan"))
        
        # Load documents for assessment
        if repo_url:
            documents = self.loader.load_github_repo(repo_url)
        elif source_path:
            path = Path(source_path)
            if path.is_dir():
                documents = self.loader.load_directory(path)
            else:
                documents = [self.loader.load_file(path)]
                documents = [d for sublist in documents for d in sublist]  # Flatten
        else:
            # Default to inbox
            documents = self.loader.load_directory(self.config.INBOX_DIR)
        
        if not documents:
            console.print("[red]No documents found for assessment[/red]")
            return {}
        
        # Run assessment
        assessment = self.assessor.assess_documents(documents)
        self.assessor.display_assessment(assessment)
        
        # Create processing plan
        plan = self.planner.create_processing_plan(documents, assessment)
        self.planner.display_plan_summary(plan)
        
        # Save plan and assessment
        if save_plan:
            assessment_path = self.config.DATA_DIR / f"assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(assessment_path, 'w') as f:
                json.dump(assessment, f, indent=2)
            
            plan_path = self.planner.save_plan(plan)
            
            # Generate and save report
            report = self.assessor.generate_assessment_report(assessment)
            report_path = self.config.DATA_DIR / f"assessment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            with open(report_path, 'w') as f:
                f.write(report)
            
            console.print(f"\n[green]✓ Assessment saved to: {assessment_path}[/green]")
            console.print(f"[green]✓ Report saved to: {report_path}[/green]")
        
        return {
            'assessment': assessment,
            'plan': plan,
            'documents': documents
        }
    
    def process_with_plan(self, plan_path: Optional[str] = None):
        """Process documents using a saved processing plan"""
        if not plan_path:
            # Find most recent plan
            plan_files = list(self.config.DATA_DIR.glob("processing_plan_*.json"))
            if not plan_files:
                console.print("[red]No processing plan found. Run assess_and_plan first.[/red]")
                return
            plan_path = max(plan_files, key=lambda p: p.stat().st_mtime)
        
        # Load plan
        with open(plan_path, 'r') as f:
            plan = json.load(f)
        
        console.print(f"[blue]Loading processing plan from: {plan_path}[/blue]")
        
        # Execute plan
        # This would integrate with the planner's execute_plan method
        console.print("[yellow]Plan-based processing not fully implemented yet[/yellow]")
        console.print("Falling back to standard processing...")
        
        # For now, process normally
        self.process_inbox()
    
    def _process_documents(self, documents):
        """Internal method to process documents through the pipeline"""
        # Process documents into chunks
        chunks = self.processor.process_documents(documents)
        self.stats['chunks_created'] = len(chunks)
        
        if not chunks:
            console.print("[yellow]No chunks created from documents[/yellow]")
            return
        
        # Generate embeddings
        doc_embeddings = self.embedder.process_documents(chunks)
        self.stats['embeddings_generated'] = len(doc_embeddings)
        
        # Store in vector database(s)
        stored_count = self.vector_store.add_documents(doc_embeddings)
        self.stats['vectors_stored'] = stored_count
    
    def _display_results(self):
        """Display processing results"""
        table = Table(title="Processing Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="green")
        
        table.add_row("Documents Loaded", str(self.stats['documents_loaded']))
        table.add_row("Chunks Created", str(self.stats['chunks_created']))
        table.add_row("Embeddings Generated", str(self.stats['embeddings_generated']))
        table.add_row("Vectors Stored", str(self.stats['vectors_stored']))
        
        if self.stats['errors']:
            table.add_row("Errors", str(len(self.stats['errors'])), style="red")
        
        console.print("\n")
        console.print(table)
        
        # Display vector store statistics
        console.print("\n")
        self.vector_store.display_statistics()
        
        # Estimate costs
        if self.stats['embeddings_generated'] > 0:
            # Rough estimate: average 100 tokens per chunk
            estimated_tokens = self.stats['chunks_created'] * 100
            estimated_cost = self.embedder.estimate_cost(estimated_tokens)
            console.print(f"\n💰 Estimated embedding cost: ${estimated_cost:.4f}")
    
    def search(self, query: str, top_k: int = 5):
        """Search the knowledge base"""
        console.print(Panel.fit(f"🔍 Searching: {query}", style="bold blue"))
        
        # Generate query embedding
        query_embedding = self.embedder.generate_embedding(query)
        
        # Search vector stores
        results = self.vector_store.search(query_embedding, top_k)
        
        # Display results
        if results:
            console.print(f"\n[green]Found {len(results)} results:[/green]\n")
            
            for i, result in enumerate(results, 1):
                console.print(f"[bold]{i}. Score: {result['score']:.4f}[/bold]")
                console.print(f"   Source: {result['metadata'].get('filename', 'Unknown')}")
                console.print(f"   Type: {result['metadata'].get('file_type', 'Unknown')}")
                console.print(f"   Content: {result['content'][:200]}...")
                console.print()
        else:
            console.print("[yellow]No results found[/yellow]")

def main():
    parser = argparse.ArgumentParser(description="Knowledge Base Pipeline")
    parser.add_argument('command', choices=['inbox', 'directory', 'github', 'search', 'stats', 'assess'],
                       help='Command to execute')
    parser.add_argument('--path', help='Path to directory (for directory command)')
    parser.add_argument('--url', help='GitHub repository URL (for github command)')
    parser.add_argument('--branch', default='main', help='Git branch (default: main)')
    parser.add_argument('--patterns', nargs='+', help='File patterns to include')
    parser.add_argument('--query', help='Search query (for search command)')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results (default: 5)')
    parser.add_argument('--storage', choices=['local', 'pinecone', 'hybrid'],
                       help='Override storage mode')
    parser.add_argument('--no-move', action='store_true', 
                       help="Don't move processed files from inbox")
    parser.add_argument('--no-save', action='store_true',
                       help="Don't save assessment and plan (for assess command)")
    parser.add_argument('--plan', help='Path to processing plan (for process-with-plan)')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = KnowledgeBasePipeline(storage_mode=args.storage)
    
    # Execute command
    if args.command == 'inbox':
        pipeline.process_inbox(move_processed=not args.no_move)
    
    elif args.command == 'directory':
        if not args.path:
            console.print("[red]Error: --path required for directory command[/red]")
            sys.exit(1)
        pipeline.process_directory(args.path)
    
    elif args.command == 'github':
        if not args.url:
            console.print("[red]Error: --url required for github command[/red]")
            sys.exit(1)
        pipeline.process_github_repo(args.url, args.branch, args.patterns)
    
    elif args.command == 'search':
        if not args.query:
            console.print("[red]Error: --query required for search command[/red]")
            sys.exit(1)
        pipeline.search(args.query, args.top_k)
    
    elif args.command == 'stats':
        pipeline.vector_store.display_statistics()
    
    elif args.command == 'assess':
        # Assess and plan for data processing
        if args.url:
            pipeline.assess_and_plan(repo_url=args.url, save_plan=not args.no_save)
        elif args.path:
            pipeline.assess_and_plan(source_path=args.path, save_plan=not args.no_save)
        else:
            # Default to inbox
            pipeline.assess_and_plan(save_plan=not args.no_save)

if __name__ == "__main__":
    main()