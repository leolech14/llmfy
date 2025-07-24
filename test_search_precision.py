#!/usr/bin/env python3
"""Test search precision for specific technical details"""

from src.search.unified_search import UnifiedSearch
from rich.console import Console
from rich.table import Table

console = Console()

# Test queries
test_queries = [
    {
        'query': '67% failure reduction',
        'expected': 'Anthropic context independence 67% retrieval failure reduction'
    },
    {
        'query': '250 tokens optimal chunk size',
        'expected': 'Research recommended chunk size of 250 tokens'
    },
    {
        'query': 'MCP servers shared between Claude Desktop and Claude Code',
        'expected': 'MCP configuration shared between Claude Desktop and Code'
    },
    {
        'query': 'Pinecone serverless',
        'expected': 'Pinecone serverless embedding system'
    },
    {
        'query': 'text-embedding-ada-002',
        'expected': 'OpenAI text-embedding-ada-002 model'
    }
]

# Initialize search
search = UnifiedSearch()

# Create results table
results_table = Table(
    title="Search Precision Test Results",
    show_header=True,
    header_style="bold cyan"
)
results_table.add_column("Query", style="cyan", width=40)
results_table.add_column("Found?", style="green", width=8)
results_table.add_column("Top Score", style="yellow", width=10)
results_table.add_column("Top Result Preview", style="white", width=60)

# Run searches
for test in test_queries:
    console.print(f"\n[cyan]Testing: {test['query']}[/cyan]")
    results = search.search(test['query'], n_results=5)
    
    found = False
    top_score = 0
    top_content = ""
    
    if results:
        top_result = results[0]
        top_score = top_result['similarity_score']
        top_content = top_result['content'][:100] + "..."
        
        # Check if expected content appears in any top results
        for result in results[:3]:
            if any(keyword.lower() in result['content'].lower() 
                   for keyword in test['expected'].split()):
                found = True
                break
    
    results_table.add_row(
        test['query'],
        "✓" if found else "✗",
        f"{top_score:.3f}",
        top_content
    )

console.print("\n")
console.print(results_table)

# Summary statistics
console.print("\n[bold cyan]Summary:[/bold cyan]")
console.print(f"• Total documents indexed: {search.get_stats()['total_chunks']}")
console.print(f"• Collections: {', '.join(search.collections.keys())}")
console.print(f"• Embedding models: local (384-dim), OpenAI (1536-dim)")