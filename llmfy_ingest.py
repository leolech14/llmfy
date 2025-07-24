#!/usr/bin/env python3
"""
🚀 llmfy Smart Universal Ingestion Tool

Intelligently transform ANY document into LLM-ready chunks!
"""

import sys
import os
import argparse
from pathlib import Path
import shutil
import json
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.prompt import Confirm

from src.core.smart_ingestion import SmartIngestionPlanner

console = Console()

def ingest_document(source_path: str, move: bool = True, smart: bool = True, auto_process: bool = False):
    """Smart ingestion with analysis and planning"""
    
    source = Path(source_path)
    if not source.exists():
        console.print(f"[red]❌ Error: File not found: {source_path}[/red]")
        return False
    
    # Get llmfy inbox
    llmfy_root = Path(__file__).parent
    inbox = llmfy_root / "data" / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    
    # Clean filename for safety
    safe_name = source.name.replace(':', '_').replace('/', '_')
    dest = inbox / safe_name
    
    if smart:
        # Smart analysis and planning
        planner = SmartIngestionPlanner()
        profile, plan = planner.analyze_document(source)
        
        # Save the plan
        plan_file = planner.save_plan(plan, source)
        console.print(f"\n[green]✅ Ingestion plan saved: {plan_file.name}[/green]")
        
        # Ask for confirmation
        console.print("\n[bold]Ready to ingest with this plan?[/bold]")
        
        if auto_process or Confirm.ask("Proceed with ingestion?"):
            # Copy or move file
            if move:
                console.print("\n[yellow]Moving file to inbox...[/yellow]")
                shutil.move(str(source), str(dest))
            else:
                console.print("\n[yellow]Copying file to inbox...[/yellow]")
                shutil.copy2(str(source), str(dest))
            
            console.print(f"[green]✅ File ready for processing: {dest.name}[/green]")
            
            # If auto-process, start processing with the plan
            if auto_process:
                console.print("\n[bold cyan]🔄 Starting smart processing...[/bold cyan]\n")
                
                # Create custom pipeline command with plan parameters
                cmd = f"{sys.executable} -m src.core.llmfy_pipeline"
                cmd += f" --input {dest}"
                cmd += f" --chunk-size {plan.chunk_size}"
                cmd += f" --overlap {plan.overlap}"
                cmd += f" --strategy {plan.chunking_strategy}"
                
                # Add enhancement flags
                if plan.enhancement_strategies:
                    cmd += f" --enhance {','.join(plan.enhancement_strategies)}"
                
                os.system(cmd)
            else:
                # Show processing command
                console.print("\n[bold]To process with smart plan:[/bold]")
                console.print(f"[cyan]python -m src.core.llmfy_pipeline --input {dest} --use-plan {plan_file}[/cyan]")
        else:
            console.print("[yellow]Ingestion cancelled[/yellow]")
            return False
    else:
        # Simple ingestion without analysis
        console.print(Panel.fit(
            f"[bold cyan]📥 Nexus Document Ingestion[/bold cyan]\n\n"
            f"Source: {source.name}\n"
            f"Size: {source.stat().st_size:,} bytes\n"
            f"Destination: {dest.name}",
            border_style="cyan"
        ))
        
        # Copy or move file
        if move:
            console.print("\n[yellow]Moving file to inbox...[/yellow]")
            shutil.move(str(source), str(dest))
        else:
            console.print("\n[yellow]Copying file to inbox...[/yellow]")
            shutil.copy2(str(source), str(dest))
        
        console.print(f"[green]✅ File ready for processing: {dest.name}[/green]")
        
        # Show next steps
        console.print("\n[bold]Next Steps:[/bold]")
        console.print("\n1. Process with quality pipeline:")
        console.print(f"   [cyan]python -m src.core.llmfy_pipeline[/cyan]")
        console.print("\n2. Or process just this file:")
        console.print(f"   [cyan]python -m src.core.llmfy_pipeline --input {dest}[/cyan]")
        console.print("\n3. Check quality after processing:")
        console.print(f"   [cyan]python llmfy_validator.py data/processed/[/cyan]")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="🚀 Smart Universal Ingestion for llmfy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Smart ingestion with analysis (recommended)
  nexus_ingest document.md
  
  # Smart ingestion with auto-processing
  nexus_ingest document.md --process
  
  # Simple ingestion without analysis
  nexus_ingest document.md --no-smart
  
  # Copy file instead of moving
  nexus_ingest document.md --copy
"""
    )
    parser.add_argument(
        'document',
        help='Path to document to ingest'
    )
    parser.add_argument(
        '--copy', '-c',
        action='store_true',
        help='Copy instead of move'
    )
    parser.add_argument(
        '--process', '-p',
        action='store_true',
        help='Immediately process after ingestion'
    )
    parser.add_argument(
        '--no-smart', '-n',
        action='store_true',
        help='Skip smart analysis (simple ingestion)'
    )
    
    args = parser.parse_args()
    
    # Ingest the document
    success = ingest_document(
        args.document, 
        move=not args.copy,
        smart=not args.no_smart,
        auto_process=args.process
    )

if __name__ == "__main__":
    main()
