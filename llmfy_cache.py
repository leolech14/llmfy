#!/usr/bin/env python3
"""
🧠 Nexus Cache Management Tool

Manage the processing cache for the Nexus AI Library.
"""

import sys
import json
from pathlib import Path
import argparse
from rich.console import Console
from rich.table import Table
from rich.prompt import Confirm

console = Console()

def clear_cache():
    """Clear all processing cache"""
    cache_file = Path("data/.processed/metadata.json")
    
    if not cache_file.exists():
        console.print("[yellow]No cache file found[/yellow]")
        return
    
    if Confirm.ask("Are you sure you want to clear the processing cache?"):
        cache_file.unlink()
        console.print("[green]✅ Cache cleared successfully[/green]")
    else:
        console.print("[yellow]Cache clear cancelled[/yellow]")

def list_cache():
    """List all cached files"""
    cache_file = Path("data/.processed/metadata.json")
    
    if not cache_file.exists():
        console.print("[yellow]No cache file found[/yellow]")
        return
    
    with open(cache_file, 'r') as f:
        cache_data = json.load(f)
    
    table = Table(title="Cached Files")
    table.add_column("File", style="cyan")
    table.add_column("Hash", style="dim")
    table.add_column("Processed At", style="green")
    
    for file_path, metadata in cache_data.items():
        table.add_row(
            Path(file_path).name,
            metadata.get('hash', 'Unknown')[:8] + "...",
            metadata.get('processed_at', 'Unknown')
        )
    
    console.print(table)
    console.print(f"\nTotal cached files: {len(cache_data)}")

def remove_from_cache(file_pattern: str):
    """Remove specific files from cache"""
    cache_file = Path("data/.processed/metadata.json")
    
    if not cache_file.exists():
        console.print("[yellow]No cache file found[/yellow]")
        return
    
    with open(cache_file, 'r') as f:
        cache_data = json.load(f)
    
    # Find matching files
    matches = []
    for file_path in cache_data:
        if file_pattern in file_path or Path(file_path).name == file_pattern:
            matches.append(file_path)
    
    if not matches:
        console.print(f"[yellow]No files matching '{file_pattern}' found in cache[/yellow]")
        return
    
    console.print(f"Found {len(matches)} matching file(s):")
    for match in matches:
        console.print(f"  - {match}")
    
    if Confirm.ask("Remove these files from cache?"):
        for match in matches:
            del cache_data[match]
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        console.print(f"[green]✅ Removed {len(matches)} file(s) from cache[/green]")
    else:
        console.print("[yellow]Removal cancelled[/yellow]")

def main():
    parser = argparse.ArgumentParser(
        description="Nexus Cache Management Tool"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear all cache')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List cached files')
    
    # Remove command
    remove_parser = subparsers.add_parser('remove', help='Remove specific files from cache')
    remove_parser.add_argument('pattern', help='File name or pattern to remove')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'clear':
        clear_cache()
    elif args.command == 'list':
        list_cache()
    elif args.command == 'remove':
        remove_from_cache(args.pattern)

if __name__ == "__main__":
    main()
