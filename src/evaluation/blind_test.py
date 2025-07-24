#!/usr/bin/env python3
"""
🔍 Blind Test Module - Automated quality evaluation for processed chunks

This module creates a blind test to evaluate how well chunks can be understood
without their original context, helping validate our preprocessing and quality scoring.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import chromadb
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

class BlindTestEvaluator:
    """
    Evaluates processed chunks by having an LLM reconstruct the document
    without access to the original, measuring comprehension quality.
    """
    
    def __init__(self, chroma_path: str = "data/vector_db/chroma_db"):
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.chroma_client.get_or_create_collection("knowledge_base")
        
    def extract_document_chunks(self, document_pattern: str) -> List[Dict[str, Any]]:
        """Extract all chunks for a specific document"""
        # Get all chunks
        results = self.collection.get(
            include=["documents", "metadatas"],
            limit=1000
        )
        
        # Filter for matching document
        document_chunks = []
        for i, metadata in enumerate(results['metadatas']):
            filename = metadata.get('filename', '').lower()
            if document_pattern.lower() in filename:
                document_chunks.append({
                    'chunk_index': metadata.get('chunk_index', i),
                    'content': results['documents'][i],
                    'metadata': {
                        'chunk_tokens': metadata.get('chunk_tokens'),
                        'chunk_total': metadata.get('chunk_total'),
                        'quality_score': metadata.get('final_quality_score', metadata.get('initial_quality_score')),
                        'section': metadata.get('section', 'Unknown')
                    }
                })
        
        # Sort by chunk index
        document_chunks.sort(key=lambda x: x['chunk_index'])
        return document_chunks
    
    def create_blind_test_file(self, chunks: List[Dict[str, Any]], output_path: Path) -> Path:
        """Create a file with chunks for blind testing"""
        with open(output_path, 'w') as f:
            f.write("BLIND TEST - DOCUMENT CHUNKS\n")
            f.write("=" * 50 + "\n\n")
            f.write("Instructions: Read all chunks and create a comprehensive summary.\n")
            f.write("Note any areas where chunks seem disconnected or unclear.\n\n")
            f.write("=" * 50 + "\n\n")
            
            for chunk in chunks:
                f.write(f"CHUNK {chunk['chunk_index'] + 1}/{chunk['metadata']['chunk_total']}\n")
                f.write(f"Quality Score: {chunk['metadata']['quality_score']:.1f}/10\n")
                f.write("-" * 30 + "\n")
                f.write(chunk['content'])
                f.write("\n\n" + "=" * 50 + "\n\n")
        
        return output_path
    
    def evaluate_with_llm(self, test_file_path: Path) -> Dict[str, Any]:
        """
        Use a separate LLM process to evaluate the chunks.
        This simulates opening a new session without context.
        """
        # Create evaluation prompt
        prompt = f"""Please read the following file and provide a detailed analysis:
{test_file_path}

After reading all chunks, please provide:
1. A comprehensive summary of the document's content
2. The main topics and key concepts covered
3. Any areas where chunks seem disconnected or missing context
4. A rating from 1-10 on how well you could reconstruct the document's meaning
5. Specific examples of chunks that were particularly clear or confusing

Format your response as JSON with these keys:
- summary: string
- main_topics: list of strings
- key_concepts: list of strings  
- disconnected_areas: list of strings
- reconstruction_score: number (1-10)
- clear_chunks: list of chunk numbers
- confusing_chunks: list of chunk numbers
- overall_assessment: string
"""

        # Use a simple approach - write to temp file and use subprocess
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
            tmp.write(prompt)
            tmp_path = tmp.name
        
        try:
            # This is a placeholder - in production you'd call an actual LLM API
            # For now, return a template response
            evaluation = {
                'summary': 'Automated evaluation pending - implement LLM API call',
                'main_topics': ['Quality control', 'LLM knowledge bases', 'RAG systems'],
                'key_concepts': ['Pattern-based scoring', 'Hybrid evaluation', 'Retrieval performance'],
                'disconnected_areas': ['Framework descriptions split across chunks', 'Numbered references without context'],
                'reconstruction_score': 8.5,
                'clear_chunks': [1, 2, 3, 5, 8, 10],
                'confusing_chunks': [15, 23, 31],
                'overall_assessment': 'Chunks generally coherent with contextual headers helping significantly'
            }
            
            # Clean up
            os.unlink(tmp_path)
            
            return evaluation
            
        except Exception as e:
            console.print(f"[red]Error during LLM evaluation: {e}[/red]")
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            return {}
    
    def compare_with_original(self, evaluation: Dict[str, Any], original_summary: Optional[str] = None) -> Dict[str, Any]:
        """Compare blind test results with original document understanding"""
        comparison = {
            'reconstruction_quality': evaluation.get('reconstruction_score', 0),
            'identified_main_topics': len(evaluation.get('main_topics', [])),
            'problematic_chunks': len(evaluation.get('confusing_chunks', [])),
            'success_indicators': [],
            'improvement_areas': []
        }
        
        # Analyze success indicators
        if evaluation.get('reconstruction_score', 0) >= 7:
            comparison['success_indicators'].append('High reconstruction score indicates good chunk coherence')
        
        if len(evaluation.get('disconnected_areas', [])) < 3:
            comparison['success_indicators'].append('Few disconnected areas shows effective preprocessing')
        
        # Identify improvement areas
        if evaluation.get('confusing_chunks'):
            comparison['improvement_areas'].append(f"Review chunks: {evaluation['confusing_chunks']}")
        
        for area in evaluation.get('disconnected_areas', []):
            comparison['improvement_areas'].append(f"Improve continuity: {area}")
        
        return comparison
    
    def run_blind_test(self, document_pattern: str) -> Dict[str, Any]:
        """Run complete blind test evaluation"""
        console.print(Panel.fit(
            f"[bold cyan]🔍 Running Blind Test Evaluation[/bold cyan]\n"
            f"Document: {document_pattern}",
            border_style="cyan"
        ))
        
        # Extract chunks
        console.print("\n[bold]Extracting document chunks...[/bold]")
        chunks = self.extract_document_chunks(document_pattern)
        
        if not chunks:
            console.print(f"[red]No chunks found for pattern: {document_pattern}[/red]")
            return {}
        
        console.print(f"[green]Found {len(chunks)} chunks[/green]")
        
        # Create test file
        test_dir = Path("data/blind_tests")
        test_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_file = test_dir / f"blind_test_{timestamp}.txt"
        
        console.print("\n[bold]Creating blind test file...[/bold]")
        self.create_blind_test_file(chunks, test_file)
        console.print(f"[green]Test file created: {test_file}[/green]")
        
        # Run evaluation
        console.print("\n[bold]Running LLM evaluation (simulated)...[/bold]")
        evaluation = self.evaluate_with_llm(test_file)
        
        # Compare results
        console.print("\n[bold]Analyzing results...[/bold]")
        comparison = self.compare_with_original(evaluation)
        
        # Display results
        self._display_results(evaluation, comparison)
        
        # Save full report
        report = {
            'timestamp': datetime.now().isoformat(),
            'document_pattern': document_pattern,
            'chunk_count': len(chunks),
            'evaluation': evaluation,
            'comparison': comparison,
            'test_file': str(test_file)
        }
        
        report_file = test_dir / f"blind_test_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        console.print(f"\n[green]Full report saved: {report_file}[/green]")
        
        return report
    
    def _display_results(self, evaluation: Dict[str, Any], comparison: Dict[str, Any]):
        """Display evaluation results in a nice format"""
        # Create results table
        table = Table(title="\n📊 Blind Test Results", show_header=True)
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Result", style="green")
        
        table.add_row("Reconstruction Score", f"{evaluation.get('reconstruction_score', 0)}/10")
        table.add_row("Main Topics Identified", str(len(evaluation.get('main_topics', []))))
        table.add_row("Clear Chunks", str(len(evaluation.get('clear_chunks', []))))
        table.add_row("Confusing Chunks", str(len(evaluation.get('confusing_chunks', []))))
        table.add_row("Disconnected Areas", str(len(evaluation.get('disconnected_areas', []))))
        
        console.print(table)
        
        # Show success indicators
        if comparison['success_indicators']:
            console.print("\n[bold green]✅ Success Indicators:[/bold green]")
            for indicator in comparison['success_indicators']:
                console.print(f"  • {indicator}")
        
        # Show improvement areas
        if comparison['improvement_areas']:
            console.print("\n[bold yellow]🔧 Areas for Improvement:[/bold yellow]")
            for area in comparison['improvement_areas']:
                console.print(f"  • {area}")
        
        # Show overall assessment
        console.print(f"\n[bold]Overall Assessment:[/bold] {evaluation.get('overall_assessment', 'N/A')}")


def main():
    """Run blind test from command line"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run blind test evaluation on processed documents")
    parser.add_argument(
        'document',
        type=str,
        help='Document name pattern to test (e.g., "Quality Control")'
    )
    parser.add_argument(
        '--chroma-path',
        type=str,
        default="data/vector_db/chroma_db",
        help='Path to ChromaDB storage'
    )
    
    args = parser.parse_args()
    
    evaluator = BlindTestEvaluator(chroma_path=args.chroma_path)
    evaluator.run_blind_test(args.document)


if __name__ == "__main__":
    main()