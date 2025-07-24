#!/usr/bin/env python3
"""
🏗️ llmfy Pipeline - Quality-First Document Processing

This is the main pipeline that transforms documents into LLM-ready chunks.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import json

from .pipeline import KnowledgeBasePipeline
from .document_loader import DocumentLoader
from .text_processor_v2 import TextProcessorV2SlidingWindow, ChunkingConfig
from ..quality.quality_scorer_v2 import ImprovedQualityAnalyzer as QualityAnalyzer
from ..quality.quality_enhancer import QualityEnhancer
from ..embeddings.hybrid_embedder import HybridEmbedder
from .chunk_optimizer import ChunkOptimizer

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()

class LlmfyPipeline(KnowledgeBasePipeline):
    """
    Enhanced pipeline with mandatory quality assessment and enhancement.
    
    Key Features:
    - Every chunk must score 7.0/10 or higher (retrieval-oriented scoring)
    - Automatic quality enhancement for low-scoring chunks
    - Quality metadata attached to all chunks
    - Detailed quality reporting
    """
    
    def __init__(self, storage_mode: Optional[str] = None, quality_threshold: float = 7.0):
        """Initialize llmfy Pipeline with quality features"""
        super().__init__(storage_mode)
        
        # Use improved text processor
        # Use sliding window semantic chunking with increased overlap for 10/10 quality
        self.processor = TextProcessorV2SlidingWindow(
            config=ChunkingConfig(
                chunk_size=250,
                chunk_overlap=100,  # Increased from 50 for better continuity
                min_chunk_size=100
            ),
            use_semantic_chunking=True
        )
        
        # Add chunk optimizer for post-processing
        self.chunk_optimizer = ChunkOptimizer(quality_threshold=9.0)
        
        self.quality_threshold = quality_threshold
        self.quality_analyzer = QualityAnalyzer()
        self.quality_enhancer = QualityEnhancer(threshold=quality_threshold)
        self.hybrid_embedder = HybridEmbedder()
        
        # Quality statistics
        self.quality_stats = {
            'total_chunks': 0,
            'passed_first_try': 0,
            'enhanced': 0,
            'failed': 0,
            'average_initial_score': 0,
            'average_final_score': 0,
            'enhancement_improvements': []
        }
        
    def process_documents(self, documents: List[Any]) -> List[Any]:
        """Process documents with quality enforcement"""
        console.print(Panel.fit(
            f"[bold cyan]🏗️ llmfy Quality-First Processing[/bold cyan]\n"
            f"Quality Threshold: {self.quality_threshold}/10",
            border_style="cyan"
        ))
        
        # First, use base chunking
        console.print("\n[bold]🔪 Creating initial chunks...[/bold]")
        initial_chunks = self.processor.process_documents(documents)
        
        # Quality assessment and enhancement
        console.print(f"\n[bold]🔍 Assessing quality of {len(initial_chunks)} chunks...[/bold]")
        
        high_quality_chunks = []
        quality_reports = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            
            task = progress.add_task(
                "Processing chunks...", 
                total=len(initial_chunks)
            )
            
            for i, chunk in enumerate(initial_chunks):
                # Analyze quality
                quality_result = self.quality_analyzer.analyze(chunk.page_content)
                initial_score = quality_result['overall_score']
                
                # Update statistics
                self.quality_stats['total_chunks'] += 1
                
                # Create quality metadata
                quality_metadata = {
                    'initial_quality_score': initial_score,
                    'quality_dimensions': quality_result['dimension_scores'],
                    'quality_assessed_at': datetime.now(timezone.utc).isoformat(),
                    'llmfy_version': '1.0'
                }
                
                # Check if enhancement needed
                if initial_score >= self.quality_threshold:
                    # Chunk passes quality threshold
                    self.quality_stats['passed_first_try'] += 1
                    quality_metadata['quality_status'] = 'passed'
                    quality_metadata['final_quality_score'] = initial_score
                    
                else:
                    # Try to enhance
                    progress.update(task, description=f"Enhancing chunk {i+1}...")
                    
                    enhanced_result = self.quality_enhancer.enhance_chunk(
                        chunk.page_content,
                        quality_result,
                        chunk.metadata
                    )
                    
                    if enhanced_result['success']:
                        # Re-assess enhanced chunk
                        final_quality = self.quality_analyzer.analyze(enhanced_result['enhanced_text'])
                        final_score = final_quality['overall_score']
                        
                        if final_score >= self.quality_threshold:
                            # Enhancement successful
                            chunk.page_content = enhanced_result['enhanced_text']
                            self.quality_stats['enhanced'] += 1
                            self.quality_stats['enhancement_improvements'].append(
                                final_score - initial_score
                            )
                            
                            quality_metadata['quality_status'] = 'enhanced'
                            quality_metadata['final_quality_score'] = final_score
                            quality_metadata['enhancement_delta'] = final_score - initial_score
                            quality_metadata['enhancements_applied'] = enhanced_result['enhancements']
                        else:
                            # Enhancement failed to meet threshold
                            self.quality_stats['failed'] += 1
                            quality_metadata['quality_status'] = 'failed'
                            quality_metadata['final_quality_score'] = final_score
                            
                            # Log for manual review
                            quality_reports.append({
                                'chunk_id': chunk.metadata.get('chunk_id', f'chunk_{i}'),
                                'content_preview': chunk.page_content[:200] + '...',
                                'initial_score': initial_score,
                                'final_score': final_score,
                                'quality_issues': quality_result['suggestions']
                            })
                            
                            # Skip this chunk - doesn't meet standards
                            progress.update(task, advance=1)
                            continue
                    else:
                        # Enhancement failed
                        self.quality_stats['failed'] += 1
                        quality_metadata['quality_status'] = 'failed'
                        quality_reports.append({
                            'chunk_id': chunk.metadata.get('chunk_id', f'chunk_{i}'),
                            'content_preview': chunk.page_content[:200] + '...',
                            'initial_score': initial_score,
                            'enhancement_error': enhanced_result.get('error', 'Unknown error')
                        })
                        progress.update(task, advance=1)
                        continue
                
                # Add quality metadata to chunk
                chunk.metadata.update(quality_metadata)
                
                # Add to high-quality chunks
                high_quality_chunks.append(chunk)
                
                progress.update(task, advance=1)
        
        # Apply chunk optimization for 10/10 quality
        if high_quality_chunks and hasattr(self, 'chunk_optimizer'):
            console.print("\n[bold]🔧 Optimizing chunks for perfect continuity...[/bold]")
            optimized_chunks, optimization_report = self.chunk_optimizer.optimize_chunks(
                [{'text': c.page_content, 'metadata': c.metadata} for c in high_quality_chunks]
            )
            
            # Convert back to Document objects
            high_quality_chunks = []
            for opt_chunk in optimized_chunks:
                chunk = initial_chunks[0].__class__(  # Use same Document class
                    page_content=opt_chunk['text'],
                    metadata=opt_chunk.get('metadata', {})
                )
                high_quality_chunks.append(chunk)
            
            # Display optimization results
            if optimization_report:
                console.print(f"\n[green]✨ Optimization improved average score from {optimization_report['summary']['average_score_before']:.1f} to {optimization_report['summary']['average_score_after']:.1f}[/green]")
                console.print(f"[green]   Merged {optimization_report['summary']['chunks_merged']} chunks for better continuity[/green]")
                console.print(f"[green]   Perfect chunks (9.5+): {optimization_report['summary']['perfect_chunks']}[/green]")
        
        # Calculate final statistics
        self._calculate_final_stats(initial_chunks, high_quality_chunks)
        
        # Display results
        self._display_quality_results()
        
        # Save quality report if there were failures
        if quality_reports:
            self._save_quality_report(quality_reports)
        
        # Continue with embedding generation for high-quality chunks
        console.print(f"\n[bold]🔗 Generating embeddings for {len(high_quality_chunks)} quality chunks...[/bold]")
        
        # Use hybrid embedder
        doc_embeddings = self.hybrid_embedder.process_documents(high_quality_chunks)
        
        # Store in vector database
        console.print("\n[bold]💾 Storing in vector database...[/bold]")
        stored_count = self.vector_store.add_documents(doc_embeddings)
        
        console.print(f"\n[green]✅ Successfully processed {stored_count} high-quality chunks![/green]")
        
        # Offer to run blind test
        if stored_count > 0:
            self._offer_blind_test(documents[0].metadata.get('filename', 'Unknown') if documents else 'Unknown')
        
        return high_quality_chunks
    
    def _calculate_final_stats(self, initial_chunks, final_chunks):
        """Calculate final quality statistics"""
        if self.quality_stats['total_chunks'] > 0:
            # Get all scores
            initial_scores = []
            final_scores = []
            
            for chunk in final_chunks:
                initial_scores.append(chunk.metadata.get('initial_quality_score', 0))
                final_scores.append(chunk.metadata.get('final_quality_score', 0))
            
            if initial_scores:
                self.quality_stats['average_initial_score'] = sum(initial_scores) / len(initial_scores)
            if final_scores:
                self.quality_stats['average_final_score'] = sum(final_scores) / len(final_scores)
    
    def _display_quality_results(self):
        """Display quality processing results"""
        # Create summary table
        table = Table(title="\n📊 Quality Processing Summary", show_header=True)
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        table.add_column("Percentage", style="yellow")
        
        total = self.quality_stats['total_chunks']
        if total > 0:
            table.add_row(
                "Total Chunks", 
                str(total),
                "100%"
            )
            table.add_row(
                "Passed First Try", 
                str(self.quality_stats['passed_first_try']),
                f"{(self.quality_stats['passed_first_try']/total)*100:.1f}%"
            )
            table.add_row(
                "Enhanced Successfully", 
                str(self.quality_stats['enhanced']),
                f"{(self.quality_stats['enhanced']/total)*100:.1f}%"
            )
            table.add_row(
                "Failed Quality Check", 
                str(self.quality_stats['failed']),
                f"{(self.quality_stats['failed']/total)*100:.1f}%"
            )
            table.add_row(
                "Average Initial Score", 
                f"{self.quality_stats['average_initial_score']:.2f}/10",
                "-"
            )
            table.add_row(
                "Average Final Score", 
                f"{self.quality_stats['average_final_score']:.2f}/10",
                "-"
            )
            
            if self.quality_stats['enhancement_improvements']:
                avg_improvement = sum(self.quality_stats['enhancement_improvements']) / len(self.quality_stats['enhancement_improvements'])
                table.add_row(
                    "Average Enhancement Improvement", 
                    f"+{avg_improvement:.2f}",
                    "-"
                )
        
        console.print(table)
    
    def _save_quality_report(self, reports: List[Dict]):
        """Save quality report for failed chunks"""
        report_dir = Path("data/quality_reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = report_dir / f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'quality_threshold': self.quality_threshold,
            'failed_chunks': len(reports),
            'chunks': reports
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        console.print(f"\n[yellow]📄 Quality report saved: {report_file}[/yellow]")
        console.print("[dim]Review failed chunks for manual improvement[/dim]")
    
    def _offer_blind_test(self, filename: str):
        """Offer to run blind test evaluation"""
        from rich.prompt import Confirm
        
        if Confirm.ask("\n[cyan]Would you like to run a blind test evaluation?[/cyan]"):
            try:
                from ..evaluation.blind_test import BlindTestEvaluator
                console.print("\n[bold]🔍 Starting blind test evaluation...[/bold]")
                
                evaluator = BlindTestEvaluator()
                # Extract base name for search
                base_name = Path(filename).stem
                evaluator.run_blind_test(base_name)
                
            except Exception as e:
                console.print(f"[red]Error running blind test: {e}[/red]")


def main():
    """Command-line interface for llmfy Pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="llmfy - Transform Documents into LLM-Ready Knowledge"
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Input file or directory'
    )
    parser.add_argument(
        '--quality-threshold', '-q',
        type=float,
        default=7.0,
        help='Minimum quality score (default: 7.0)'
    )
    parser.add_argument(
        '--storage', '-s',
        choices=['local', 'cloud', 'hybrid'],
        default='local',
        help='Storage mode (default: local)'
    )
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Force reprocessing of already processed files'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = LlmfyPipeline(
        storage_mode=args.storage,
        quality_threshold=args.quality_threshold
    )
    
    # Process input
    if args.input:
        input_path = Path(args.input)
        if input_path.is_file():
            documents = pipeline.loader.load_file(input_path, force=args.force)
        elif input_path.is_dir():
            documents = pipeline.loader.load_directory(input_path, force=args.force)
        else:
            console.print(f"[red]Error: Invalid input path: {args.input}[/red]")
            return
        
        # Process documents
        pipeline.process_documents(documents)
    else:
        # Process inbox
        console.print("[blue]Processing inbox directory...[/blue]")
        pipeline.process_inbox()

if __name__ == "__main__":
    main()
