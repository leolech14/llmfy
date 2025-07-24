from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import json

from langchain.schema import Document
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from .config import Config
from .data_assessor import DataAssessor

console = Console()

class ProcessingPlanner:
    """Creates optimized processing plans based on data assessment"""
    
    def __init__(self):
        self.config = Config()
        self.assessor = DataAssessor()
    
    def create_processing_plan(self, documents: List[Document], 
                             assessment: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a comprehensive processing plan"""
        
        # Run assessment if not provided
        if not assessment:
            assessment = self.assessor.assess_documents(documents)
        
        console.print(Panel.fit("[bold green]📋 Creating Optimized Processing Plan[/bold green]"))
        
        plan = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'total_documents': len(documents),
                'plan_version': '1.0'
            },
            'document_groups': self._group_documents(documents, assessment),
            'processing_stages': self._create_processing_stages(assessment),
            'chunking_configs': self._create_chunking_configs(assessment),
            'metadata_enrichment': self._create_enrichment_plan(assessment),
            'batch_configuration': self._create_batch_config(assessment),
            'quality_checks': self._create_quality_checks(assessment)
        }
        
        return plan
    
    def _group_documents(self, documents: List[Document], 
                        assessment: Dict[str, Any]) -> Dict[str, List[int]]:
        """Group documents by processing strategy"""
        groups = {
            'documentation': [],
            'source_code': [],
            'configuration': [],
            'tests': [],
            'examples': [],
            'other': []
        }
        
        for idx, doc in enumerate(documents):
            file_type = doc.metadata.get('file_type', '')
            filename = doc.metadata.get('filename', '').lower()
            source = doc.metadata.get('source', '').lower()
            
            # Categorize documents
            if file_type in ['.md', '.markdown', '.rst', '.txt'] or 'readme' in filename:
                groups['documentation'].append(idx)
            elif file_type in ['.py', '.js', '.ts', '.tsx', '.jsx', '.java', '.cpp', '.go']:
                if 'test' in source or 'spec' in source:
                    groups['tests'].append(idx)
                elif 'example' in source or 'demo' in source:
                    groups['examples'].append(idx)
                else:
                    groups['source_code'].append(idx)
            elif filename in ['package.json', 'requirements.txt', 'setup.py', 'cargo.toml', 
                            'pom.xml', 'build.gradle', '.env.example']:
                groups['configuration'].append(idx)
            else:
                groups['other'].append(idx)
        
        # Remove empty groups
        return {k: v for k, v in groups.items() if v}
    
    def _create_processing_stages(self, assessment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create ordered processing stages"""
        stages = []
        
        # Stage 1: High-priority documentation
        stages.append({
            'stage_id': 1,
            'name': 'Core Documentation',
            'description': 'Process README files and main documentation',
            'groups': ['documentation'],
            'priority': 'high',
            'chunking_profile': 'documentation',
            'enrichments': ['headers', 'links', 'code_examples']
        })
        
        # Stage 2: Configuration and setup
        stages.append({
            'stage_id': 2,
            'name': 'Configuration Files',
            'description': 'Process configuration to understand project structure',
            'groups': ['configuration'],
            'priority': 'high',
            'chunking_profile': 'structured_data',
            'enrichments': ['dependencies', 'versions', 'settings']
        })
        
        # Stage 3: Main source code
        stages.append({
            'stage_id': 3,
            'name': 'Source Code',
            'description': 'Process main application code',
            'groups': ['source_code'],
            'priority': 'medium',
            'chunking_profile': 'code',
            'enrichments': ['functions', 'classes', 'imports', 'complexity']
        })
        
        # Stage 4: Tests and examples
        stages.append({
            'stage_id': 4,
            'name': 'Tests and Examples',
            'description': 'Process test files and example code',
            'groups': ['tests', 'examples'],
            'priority': 'low',
            'chunking_profile': 'code',
            'enrichments': ['test_names', 'assertions', 'example_usage']
        })
        
        # Stage 5: Other files
        if assessment['summary']['file_type_distribution'].get('other', 0) > 0:
            stages.append({
                'stage_id': 5,
                'name': 'Miscellaneous Files',
                'description': 'Process remaining files',
                'groups': ['other'],
                'priority': 'low',
                'chunking_profile': 'adaptive',
                'enrichments': ['basic']
            })
        
        return stages
    
    def _create_chunking_configs(self, assessment: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Create specific chunking configurations for different content types"""
        configs = {}
        
        # Documentation chunking
        configs['documentation'] = {
            'strategy': 'recursive_character',
            'chunk_size': 512,
            'chunk_overlap': 50,
            'separators': ["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " "],
            'preserve_headers': True,
            'min_chunk_size': 100
        }
        
        # Code chunking
        configs['code'] = {
            'strategy': 'semantic_code',
            'chunk_size': 768,  # Larger for code context
            'chunk_overlap': 128,
            'separators': ["\nclass ", "\ndef ", "\nasync def ", "\nfunction ", "\n\n", "\n"],
            'preserve_structure': True,
            'include_imports': True,
            'min_chunk_size': 200
        }
        
        # Structured data (JSON, YAML, etc.)
        configs['structured_data'] = {
            'strategy': 'structural',
            'chunk_size': 1024,
            'chunk_overlap': 0,  # No overlap for structured data
            'preserve_structure': True,
            'parse_format': True
        }
        
        # Adaptive chunking for mixed content
        configs['adaptive'] = {
            'strategy': 'adaptive',
            'chunk_size': 512,
            'chunk_overlap': 50,
            'auto_detect_type': True,
            'fallback_strategy': 'recursive_character'
        }
        
        return configs
    
    def _create_enrichment_plan(self, assessment: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Create metadata enrichment plan"""
        enrichments = {}
        
        # Documentation enrichments
        enrichments['headers'] = {
            'extractor': 'regex',
            'pattern': r'^(#{1,6})\s+(.+)$',
            'multiline': True,
            'output_format': 'hierarchy'
        }
        
        enrichments['links'] = {
            'extractor': 'regex',
            'pattern': r'\[([^\]]+)\]\(([^)]+)\)',
            'output_format': 'list'
        }
        
        enrichments['code_examples'] = {
            'extractor': 'regex',
            'pattern': r'```(\w*)\n([\s\S]*?)```',
            'output_format': 'structured'
        }
        
        # Code enrichments
        enrichments['functions'] = {
            'extractor': 'ast',
            'languages': ['python', 'javascript', 'typescript'],
            'extract': ['name', 'parameters', 'docstring']
        }
        
        enrichments['classes'] = {
            'extractor': 'ast',
            'languages': ['python', 'javascript', 'typescript'],
            'extract': ['name', 'methods', 'inheritance']
        }
        
        enrichments['imports'] = {
            'extractor': 'regex',
            'patterns': {
                'python': r'(?:from\s+([\w.]+)\s+)?import\s+([\w\s,]+)',
                'javascript': r'import\s+(?:{[^}]+}|[\w]+)\s+from\s+["\']([^"\']+)["\']'
            }
        }
        
        enrichments['complexity'] = {
            'extractor': 'analysis',
            'metrics': ['cyclomatic_complexity', 'lines_of_code', 'comment_ratio']
        }
        
        # Test enrichments
        enrichments['test_names'] = {
            'extractor': 'regex',
            'patterns': {
                'python': r'def\s+test_(\w+)',
                'javascript': r'(?:test|it)\s*\(["\']([^"\']+)["\']'
            }
        }
        
        return enrichments
    
    def _create_batch_config(self, assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Create batch processing configuration"""
        total_docs = assessment['summary']['total_documents']
        total_tokens = assessment['summary']['total_tokens']
        
        # Calculate optimal batch size
        if total_docs < 100:
            batch_size = 50
        elif total_docs < 1000:
            batch_size = 100
        else:
            batch_size = 250
        
        # Estimate processing time
        processing_rate = 10  # documents per second (conservative estimate)
        total_time_seconds = total_docs / processing_rate
        
        return {
            'batch_size': batch_size,
            'parallel_batches': 2,  # Process 2 batches in parallel
            'rate_limiting': {
                'enabled': True,
                'requests_per_minute': 3000,  # OpenAI rate limit
                'tokens_per_minute': 1_000_000
            },
            'error_handling': {
                'max_retries': 3,
                'retry_delay': 1.0,
                'continue_on_error': True
            },
            'progress_tracking': {
                'checkpoint_interval': 100,  # Save progress every 100 docs
                'enable_resume': True
            },
            'estimated_time': {
                'total_seconds': total_time_seconds,
                'human_readable': f"{total_time_seconds/60:.1f} minutes"
            }
        }
    
    def _create_quality_checks(self, assessment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create quality check procedures"""
        checks = []
        
        # Chunking quality
        checks.append({
            'name': 'chunk_size_validation',
            'description': 'Ensure chunks are within size limits',
            'parameters': {
                'min_tokens': 50,
                'max_tokens': 1000,
                'warn_threshold': 0.1  # Warn if >10% chunks are outside range
            }
        })
        
        # Embedding quality
        checks.append({
            'name': 'embedding_validation',
            'description': 'Validate embedding generation',
            'parameters': {
                'check_dimensions': True,
                'check_normalization': True,
                'sample_similarity_check': True
            }
        })
        
        # Metadata completeness
        checks.append({
            'name': 'metadata_completeness',
            'description': 'Ensure all required metadata is present',
            'required_fields': ['source', 'file_type', 'chunk_index', 'timestamp'],
            'recommended_fields': ['section', 'complexity', 'language']
        })
        
        # Deduplication
        checks.append({
            'name': 'deduplication_check',
            'description': 'Check for duplicate content',
            'parameters': {
                'similarity_threshold': 0.95,
                'check_exact_matches': True
            }
        })
        
        return checks
    
    def save_plan(self, plan: Dict[str, Any], output_path: Optional[Path] = None) -> Path:
        """Save processing plan to file"""
        if not output_path:
            output_path = self.config.DATA_DIR / f"processing_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_path, 'w') as f:
            json.dump(plan, f, indent=2)
        
        console.print(f"[green]✓ Processing plan saved to: {output_path}[/green]")
        return output_path
    
    def display_plan_summary(self, plan: Dict[str, Any]):
        """Display a summary of the processing plan"""
        console.print("\n[bold]Processing Plan Summary:[/bold]")
        
        # Document groups
        console.print("\n[yellow]Document Groups:[/yellow]")
        for group, indices in plan['document_groups'].items():
            console.print(f"  • {group}: {len(indices)} documents")
        
        # Processing stages
        console.print("\n[yellow]Processing Stages:[/yellow]")
        for stage in plan['processing_stages']:
            console.print(f"  {stage['stage_id']}. {stage['name']} "
                         f"[dim]({', '.join(stage['groups'])})[/dim] - "
                         f"Priority: {stage['priority']}")
        
        # Batch configuration
        batch_config = plan['batch_configuration']
        console.print("\n[yellow]Batch Configuration:[/yellow]")
        console.print(f"  • Batch size: {batch_config['batch_size']} documents")
        console.print(f"  • Estimated time: {batch_config['estimated_time']['human_readable']}")
        console.print(f"  • Error handling: {batch_config['error_handling']['max_retries']} retries")
        
        # Quality checks
        console.print("\n[yellow]Quality Checks:[/yellow]")
        for check in plan['quality_checks']:
            console.print(f"  • {check['name']}: {check['description']}")
    
    def execute_plan(self, plan: Dict[str, Any], documents: List[Document], pipeline) -> Dict[str, Any]:
        """Execute the processing plan (to be integrated with main pipeline)"""
        results = {
            'stages_completed': [],
            'documents_processed': 0,
            'errors': [],
            'quality_report': {}
        }
        
        console.print("\n[bold green]Executing Processing Plan...[/bold green]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            for stage in plan['processing_stages']:
                task = progress.add_task(f"Stage {stage['stage_id']}: {stage['name']}")
                
                # Get documents for this stage
                stage_doc_indices = []
                for group in stage['groups']:
                    stage_doc_indices.extend(plan['document_groups'].get(group, []))
                
                stage_docs = [documents[i] for i in stage_doc_indices]
                
                if stage_docs:
                    # Apply chunking configuration
                    chunking_config = plan['chunking_configs'][stage['chunking_profile']]
                    
                    # Process documents with stage-specific settings
                    # (This would integrate with the actual pipeline)
                    console.print(f"  Processing {len(stage_docs)} documents with "
                                 f"{stage['chunking_profile']} chunking profile...")
                    
                    results['stages_completed'].append(stage['name'])
                    results['documents_processed'] += len(stage_docs)
                
                progress.update(task, completed=True)
        
        return results
