#!/usr/bin/env python3
"""
🧪 Nexus Validator - Quality validation tool for processed documents

Validates that all chunks in the system meet the 9.5/10 quality standard.
"""

import sys
from pathlib import Path
import json
import argparse
from typing import List, Dict, Any
from datetime import datetime
import statistics

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.quality.quality_scorer import QualityAnalyzer

console = Console()

class NexusValidator:
    """
    Validates document quality in the Nexus system.
    
    Features:
    - Scan processed documents for quality scores
    - Re-validate quality scores
    - Generate quality reports
    - Identify chunks needing improvement
    """
    
    def __init__(self, quality_threshold: float = 9.5):
        self.quality_threshold = quality_threshold
        self.quality_analyzer = QualityAnalyzer()
        self.validation_results = []
        
    def validate_directory(self, directory: Path) -> Dict[str, Any]:
        """Validate all documents in a directory"""
        console.print(Panel.fit(
            f"[bold cyan]🧪 Nexus Quality Validator[/bold cyan]\n"
            f"Validating: {directory}\n"
            f"Quality Threshold: {self.quality_threshold}/10",
            border_style="cyan"
        ))
        
        # Find all JSON chunk files
        chunk_files = list(directory.glob("**/*.json"))
        
        if not chunk_files:
            console.print("[yellow]No chunk files found to validate[/yellow]")
            return {}
        
        console.print(f"\n[bold]Found {len(chunk_files)} files to validate[/bold]\n")
        
        # Validate each file
        for chunk_file in track(chunk_files, description="Validating chunks..."):
            self._validate_file(chunk_file)
        
        # Generate summary
        summary = self._generate_summary()
        
        # Display results
        self._display_results(summary)
        
        # Save validation report
        self._save_report(summary)
        
        return summary
    
    def validate_chunk(self, chunk_text: str, metadata: Dict = None) -> Dict[str, Any]:
        """Validate a single chunk"""
        # Analyze quality
        quality_result = self.quality_analyzer.analyze(chunk_text)
        
        # Check against threshold
        passes = quality_result['overall_score'] >= self.quality_threshold
        
        validation = {
            'passes': passes,
            'score': quality_result['overall_score'],
            'threshold': self.quality_threshold,
            'dimensions': quality_result['dimension_scores'],
            'suggestions': quality_result.get('suggestions', []),
            'metadata': metadata or {}
        }
        
        return validation
    
    def _validate_file(self, file_path: Path):
        """Validate chunks in a file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Handle different file formats
            chunks = []
            if isinstance(data, list):
                chunks = data
            elif isinstance(data, dict):
                if 'chunks' in data:
                    chunks = data['chunks']
                elif 'documents' in data:
                    chunks = data['documents']
                else:
                    # Single chunk
                    chunks = [data]
            
            # Validate each chunk
            for chunk in chunks:
                # Extract text and metadata
                text = chunk.get('text', '') or chunk.get('content', '') or chunk.get('page_content', '')
                metadata = chunk.get('metadata', {})
                
                if text:
                    # Validate
                    validation = self.validate_chunk(text, metadata)
                    validation['file'] = str(file_path)
                    validation['chunk_id'] = metadata.get('chunk_id', f'unknown_{len(self.validation_results)}')
                    
                    # Check stored quality score
                    stored_score = metadata.get('final_quality_score') or metadata.get('quality_score')
                    if stored_score:
                        validation['stored_score'] = stored_score
                        validation['score_match'] = abs(stored_score - validation['score']) < 0.1
                    
                    self.validation_results.append(validation)
        
        except Exception as e:
            console.print(f"[red]Error validating {file_path}: {e}[/red]")
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate validation summary"""
        if not self.validation_results:
            return {
                'total_chunks': 0,
                'passing_chunks': 0,
                'failing_chunks': 0,
                'average_score': 0,
                'validation_errors': []
            }
        
        scores = [r['score'] for r in self.validation_results]
        passing = [r for r in self.validation_results if r['passes']]
        failing = [r for r in self.validation_results if not r['passes']]
        
        # Dimension analysis
        dimension_scores = {}
        for result in self.validation_results:
            for dim, score in result['dimensions'].items():
                if dim not in dimension_scores:
                    dimension_scores[dim] = []
                dimension_scores[dim].append(score)
        
        dimension_averages = {
            dim: statistics.mean(scores) 
            for dim, scores in dimension_scores.items()
        }
        
        # Find worst performers
        worst_chunks = sorted(
            failing, 
            key=lambda x: x['score']
        )[:10]  # Top 10 worst
        
        return {
            'total_chunks': len(self.validation_results),
            'passing_chunks': len(passing),
            'failing_chunks': len(failing),
            'pass_rate': len(passing) / len(self.validation_results) if self.validation_results else 0,
            'average_score': statistics.mean(scores),
            'min_score': min(scores),
            'max_score': max(scores),
            'std_dev': statistics.stdev(scores) if len(scores) > 1 else 0,
            'dimension_averages': dimension_averages,
            'worst_chunks': worst_chunks,
            'validation_timestamp': datetime.utcnow().isoformat()
        }
    
    def _display_results(self, summary: Dict[str, Any]):
        """Display validation results"""
        # Overall summary
        console.print("\n" + "="*60)
        
        if summary['total_chunks'] == 0:
            console.print("[yellow]No chunks found to validate[/yellow]")
            return
        
        # Pass/Fail summary
        pass_rate = summary['pass_rate'] * 100
        status_color = "green" if pass_rate >= 95 else "yellow" if pass_rate >= 80 else "red"
        
        console.print(Panel(
            f"[bold]Validation Summary[/bold]\n\n"
            f"Total Chunks: {summary['total_chunks']}\n"
            f"Passing (≥{self.quality_threshold}): {summary['passing_chunks']} "
            f"[{status_color}]({pass_rate:.1f}%)[/{status_color}]\n"
            f"Failing (<{self.quality_threshold}): {summary['failing_chunks']}\n\n"
            f"Average Score: {summary['average_score']:.2f}/10\n"
            f"Score Range: {summary['min_score']:.2f} - {summary['max_score']:.2f}\n"
            f"Std Deviation: {summary['std_dev']:.2f}",
            border_style=status_color
        ))
        
        # Dimension breakdown
        console.print("\n[bold]Quality Dimension Breakdown:[/bold]")
        
        dim_table = Table(show_header=True)
        dim_table.add_column("Dimension", style="cyan")
        dim_table.add_column("Average Score", style="green")
        dim_table.add_column("Status")
        
        for dim, avg in summary['dimension_averages'].items():
            status = "✅" if avg >= 8.0 else "⚠️" if avg >= 7.0 else "❌"
            dim_table.add_row(
                dim.replace('_', ' ').title(),
                f"{avg:.2f}/10",
                status
            )
        
        console.print(dim_table)
        
        # Worst performers
        if summary['worst_chunks']:
            console.print("\n[bold red]Chunks Requiring Attention:[/bold red]")
            
            worst_table = Table(show_header=True)
            worst_table.add_column("Chunk ID", style="cyan", max_width=20)
            worst_table.add_column("Score", style="red")
            worst_table.add_column("Issues", style="yellow", max_width=40)
            worst_table.add_column("File", style="dim", max_width=30)
            
            for chunk in summary['worst_chunks']:
                # Find worst dimension
                worst_dim = min(
                    chunk['dimensions'].items(), 
                    key=lambda x: x[1]
                )
                issues = f"{worst_dim[0]}: {worst_dim[1]:.1f}"
                
                worst_table.add_row(
                    chunk['chunk_id'][:20],
                    f"{chunk['score']:.2f}/10",
                    issues,
                    Path(chunk['file']).name
                )
            
            console.print(worst_table)
        
        # Success message
        if pass_rate >= 95:
            console.print("\n[bold green]✅ Excellent! System maintains high quality standards![/bold green]")
        elif pass_rate >= 80:
            console.print("\n[bold yellow]⚠️  Good, but some chunks need improvement[/bold yellow]")
        else:
            console.print("\n[bold red]❌ Quality standards not met. Review and enhance failing chunks.[/bold red]")
    
    def _save_report(self, summary: Dict[str, Any]):
        """Save detailed validation report"""
        report_dir = Path("data/quality_reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary
        summary_file = report_dir / f"validation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed results if there are failures
        if summary['failing_chunks'] > 0:
            details_file = report_dir / f"validation_details_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            failing_details = [
                r for r in self.validation_results 
                if not r['passes']
            ]
            
            with open(details_file, 'w') as f:
                json.dump({
                    'validation_timestamp': summary['validation_timestamp'],
                    'quality_threshold': self.quality_threshold,
                    'failing_chunks': failing_details
                }, f, indent=2)
            
            console.print(f"\n[dim]Detailed reports saved to {report_dir}/[/dim]")

def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="Nexus Quality Validator - Validate chunk quality"
    )
    parser.add_argument(
        'path',
        type=str,
        help='Path to validate (file or directory)'
    )
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=9.5,
        help='Quality threshold (default: 9.5)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output report file'
    )
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = NexusValidator(quality_threshold=args.threshold)
    
    # Validate path
    path = Path(args.path)
    if path.is_dir():
        summary = validator.validate_directory(path)
    elif path.is_file():
        # Validate single file
        validator._validate_file(path)
        summary = validator._generate_summary()
        validator._display_results(summary)
    else:
        console.print(f"[red]Error: Path not found: {args.path}[/red]")
        sys.exit(1)
    
    # Save custom output if requested
    if args.output and summary:
        with open(args.output, 'w') as f:
            json.dump(summary, f, indent=2)
        console.print(f"\n[green]Report saved to {args.output}[/green]")

if __name__ == "__main__":
    main()