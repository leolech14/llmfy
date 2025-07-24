#!/usr/bin/env python3
"""
🏗️ llmfy - LLM-ify Your Documents

The one command to transform your documents into LLM-ready knowledge!
"""

import sys
import os
import argparse
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

console = Console()

def print_banner():
    console.print(Panel.fit(
        "[bold cyan]🏗️ llmfy - LLM-ify Your Documents[/bold cyan]\n"
        "[white]Quality-First Document Processing[/white]\n"
        "[dim]Every chunk must score 9.5/10 or higher[/dim]",
        border_style="cyan"
    ))

def main():
    parser = argparse.ArgumentParser(
        description="llmfy - Transform documents into LLM-ready knowledge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  llmfy ingest /path/to/document.md      # Add document to system
  llmfy process                           # Process all inbox documents
  llmfy validate                          # Check quality of processed docs
  llmfy search "query terms"              # Search the knowledge base
  llmfy status                            # Show system status
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Add document to llmfy')
    ingest_parser.add_argument('document', help='Path to document')
    ingest_parser.add_argument('--process', '-p', action='store_true',
                              help='Process immediately after ingesting')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process inbox documents')
    process_parser.add_argument('--file', '-f', help='Process specific file')
    process_parser.add_argument('--threshold', '-t', type=float, default=9.5,
                               help='Quality threshold (default: 9.5)')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate document quality')
    validate_parser.add_argument('--path', '-p', default='data/processed',
                                help='Path to validate')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search knowledge base')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--limit', '-l', type=int, default=5,
                              help='Number of results')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show system status')
    
    args = parser.parse_args()
    
    if not args.command:
        print_banner()
        parser.print_help()
        return
    
    # Execute commands
    if args.command == 'ingest':
        print_banner()
        # Import and run ingestion
        from llmfy_ingest import ingest_document
        success = ingest_document(args.document, move=True)
        
        if success and args.process:
            console.print("\n[cyan]🔄 Processing document...[/cyan]\n")
            os.system(f"{sys.executable} -m src.core.llmfy_pipeline")
    
    elif args.command == 'process':
        print_banner()
        console.print("\n[cyan]🔄 Processing documents...[/cyan]\n")
        if args.file:
            cmd = f"{sys.executable} -m src.core.llmfy_pipeline --input {args.file}"
        else:
            cmd = f"{sys.executable} -m src.core.llmfy_pipeline"
        os.system(cmd)
    
    elif args.command == 'validate':
        print_banner()
        console.print("\n[cyan]🧪 Validating quality...[/cyan]\n")
        cmd = f"{sys.executable} llmfy_validator.py {args.path}"
        os.system(cmd)
    
    elif args.command == 'search':
        print_banner()
        console.print(f"\n[cyan]🔍 Searching for: {args.query}[/cyan]\n")
        console.print("[yellow]Search functionality coming soon![/yellow]")
        console.print("For now, documents are in data/processed/")
    
    elif args.command == 'status':
        print_banner()
        llmfy_root = Path(__file__).parent
        
        # Count documents
        inbox_count = len(list((llmfy_root / "data/inbox").glob("*.md")))
        processed_count = len(list((llmfy_root / "data/processed").glob("*.json")))
        
        console.print("\n[bold]System Status:[/bold]")
        console.print(f"  📥 Inbox: {inbox_count} documents")
        console.print(f"  ✅ Processed: {processed_count} documents")
        console.print(f"  🎯 Quality Threshold: 9.5/10")
        console.print(f"  🌍 Environment: Development (free embeddings)")
        console.print(f"  💾 Storage: ChromaDB (local)")
        
        # Check if virtual environment is active
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            console.print(f"  🐍 Virtual Environment: Active")
        else:
            console.print(f"  ⚠️  Virtual Environment: Not active")
            console.print("\n[yellow]Tip: Activate with: source venv/bin/activate[/yellow]")

if __name__ == "__main__":
    main()
