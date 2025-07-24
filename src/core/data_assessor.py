from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from collections import defaultdict, Counter
import json
import re
from datetime import datetime

from langchain.schema import Document
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
import tiktoken

from .config import Config

console = Console()

class DataAssessor:
    """Analyzes data before processing to optimize chunking and retrieval strategies"""
    
    def __init__(self):
        self.config = Config()
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        
    def assess_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """Comprehensive assessment of documents to optimize processing"""
        console.print(Panel.fit("[bold blue]📊 Assessing Document Collection[/bold blue]"))
        
        assessment = {
            'summary': self._generate_summary(documents),
            'content_analysis': self._analyze_content(documents),
            'structure_analysis': self._analyze_structure(documents),
            'optimal_strategies': self._recommend_strategies(documents),
            'metadata_plan': self._plan_metadata_enrichment(documents),
            'estimated_metrics': self._estimate_processing_metrics(documents),
            'timestamp': datetime.now().isoformat()
        }
        
        return assessment
    
    def _generate_summary(self, documents: List[Document]) -> Dict[str, Any]:
        """Generate high-level summary of the document collection"""
        total_tokens = sum(len(self.encoding.encode(doc.page_content)) for doc in documents)
        
        file_types = Counter()
        sources = set()
        
        for doc in documents:
            file_types[doc.metadata.get('file_type', 'unknown')] += 1
            sources.add(doc.metadata.get('source', 'unknown'))
        
        return {
            'total_documents': len(documents),
            'total_tokens': total_tokens,
            'unique_sources': len(sources),
            'file_type_distribution': dict(file_types),
            'average_tokens_per_doc': total_tokens // len(documents) if documents else 0
        }
    
    def _analyze_content(self, documents: List[Document]) -> Dict[str, Any]:
        """Analyze content characteristics"""
        content_types = defaultdict(list)
        language_distribution = defaultdict(int)
        
        for doc in documents:
            content = doc.page_content
            file_type = doc.metadata.get('file_type', 'unknown')
            
            # Detect content characteristics
            characteristics = {
                'has_code': bool(re.search(r'```[\s\S]*?```|def\s+\w+|function\s+\w+|class\s+\w+', content)),
                'has_tables': bool(re.search(r'\|[\s\S]*?\|', content)),
                'has_lists': bool(re.search(r'^\s*[-*+]\s+|\d+\.\s+', content, re.MULTILINE)),
                'has_headers': bool(re.search(r'^#+\s+', content, re.MULTILINE)),
                'has_urls': bool(re.search(r'https?://\S+', content)),
                'avg_line_length': self._calculate_avg_line_length(content),
                'complexity_score': self._calculate_complexity_score(content)
            }
            
            content_types[file_type].append(characteristics)
            
            # Simple language detection
            if file_type in ['.py', '.pyw']:
                language_distribution['python'] += 1
            elif file_type in ['.js', '.jsx', '.ts', '.tsx']:
                language_distribution['javascript'] += 1
            elif file_type in ['.md', '.markdown']:
                language_distribution['markdown'] += 1
            else:
                language_distribution['other'] += 1
        
        return {
            'content_characteristics': self._aggregate_characteristics(content_types),
            'language_distribution': dict(language_distribution),
            'complexity_analysis': self._analyze_complexity_distribution(documents)
        }
    
    def _analyze_structure(self, documents: List[Document]) -> Dict[str, Any]:
        """Analyze document structure and relationships"""
        structure_info = {
            'directory_tree': self._build_directory_tree(documents),
            'file_relationships': self._analyze_file_relationships(documents),
            'naming_patterns': self._analyze_naming_patterns(documents),
            'size_distribution': self._analyze_size_distribution(documents)
        }
        
        return structure_info
    
    def _recommend_strategies(self, documents: List[Document]) -> Dict[str, Any]:
        """Recommend optimal processing strategies based on analysis"""
        strategies = {
            'chunking_strategy': self._recommend_chunking_strategy(documents),
            'namespace_strategy': self._recommend_namespace_strategy(documents),
            'metadata_strategy': self._recommend_metadata_strategy(documents),
            'processing_order': self._recommend_processing_order(documents)
        }
        
        return strategies
    
    def _recommend_chunking_strategy(self, documents: List[Document]) -> Dict[str, Any]:
        """Recommend optimal chunking parameters"""
        # Analyze document characteristics
        code_docs = sum(1 for doc in documents if doc.metadata.get('file_type', '') in ['.py', '.js', '.ts', '.tsx', '.jsx'])
        markdown_docs = sum(1 for doc in documents if doc.metadata.get('file_type', '') in ['.md', '.markdown'])
        
        recommendations = []
        
        if code_docs > len(documents) * 0.5:
            recommendations.append({
                'type': 'code',
                'chunk_size': 768,  # Larger for code
                'chunk_overlap': 128,
                'strategy': 'semantic',
                'reason': 'Majority code files benefit from larger chunks to preserve context'
            })
        
        if markdown_docs > len(documents) * 0.3:
            recommendations.append({
                'type': 'markdown',
                'chunk_size': 512,
                'chunk_overlap': 50,
                'strategy': 'recursive_character',
                'reason': 'Markdown files work well with standard recursive splitting'
            })
        
        # Default recommendation
        if not recommendations:
            recommendations.append({
                'type': 'mixed',
                'chunk_size': 512,
                'chunk_overlap': 50,
                'strategy': 'adaptive',
                'reason': 'Mixed content types require adaptive chunking'
            })
        
        return {
            'recommendations': recommendations,
            'adaptive_chunking': code_docs > 0 and markdown_docs > 0
        }
    
    def _recommend_namespace_strategy(self, documents: List[Document]) -> Dict[str, Any]:
        """Recommend namespace organization"""
        namespaces = defaultdict(list)
        
        for doc in documents:
            file_type = doc.metadata.get('file_type', '')
            source = doc.metadata.get('source', '')
            
            # Categorize by type
            if file_type in ['.py', '.js', '.ts', '.tsx', '.jsx']:
                namespaces['code'].append(source)
            elif file_type in ['.md', '.markdown']:
                namespaces['documentation'].append(source)
            elif 'test' in source.lower() or 'spec' in source.lower():
                namespaces['tests'].append(source)
            else:
                namespaces['general'].append(source)
        
        return {
            'recommended_namespaces': {k: len(v) for k, v in namespaces.items()},
            'rationale': 'Separate namespaces improve search relevance and allow targeted queries'
        }
    
    def _recommend_metadata_strategy(self, documents: List[Document]) -> List[Dict[str, str]]:
        """Recommend metadata enrichment strategies"""
        strategies = []
        
        # Check for code files
        if any(doc.metadata.get('file_type', '') in ['.py', '.js', '.ts'] for doc in documents):
            strategies.append({
                'field': 'code_context',
                'description': 'Extract function/class names and imports',
                'purpose': 'Enable searching by specific code elements'
            })
        
        # Check for documentation
        if any(doc.metadata.get('file_type', '') in ['.md', '.markdown'] for doc in documents):
            strategies.append({
                'field': 'doc_sections',
                'description': 'Extract header hierarchy and sections',
                'purpose': 'Enable navigation through documentation structure'
            })
        
        # Always recommend these
        strategies.extend([
            {
                'field': 'semantic_tags',
                'description': 'Auto-generate tags based on content',
                'purpose': 'Improve semantic search accuracy'
            },
            {
                'field': 'cross_references',
                'description': 'Identify references between documents',
                'purpose': 'Enable relationship-based retrieval'
            }
        ])
        
        return strategies
    
    def _recommend_processing_order(self, documents: List[Document]) -> List[str]:
        """Recommend optimal processing order"""
        # Process in order of importance and dependencies
        order = []
        
        # 1. Documentation first (provides context)
        doc_files = [d for d in documents if d.metadata.get('file_type', '') in ['.md', '.markdown']]
        if doc_files:
            order.append("1. Documentation files (README, guides)")
        
        # 2. Configuration files
        config_files = [d for d in documents if d.metadata.get('filename', '').lower() in 
                       ['package.json', 'requirements.txt', 'setup.py', 'cargo.toml']]
        if config_files:
            order.append("2. Configuration files (understand dependencies)")
        
        # 3. Main source code
        order.append("3. Core source code files")
        
        # 4. Tests and examples
        test_files = [d for d in documents if 'test' in d.metadata.get('source', '').lower()]
        if test_files:
            order.append("4. Test files and examples")
        
        return order
    
    def _plan_metadata_enrichment(self, documents: List[Document]) -> Dict[str, Any]:
        """Plan metadata enrichment for better retrieval"""
        enrichment_plan = {
            'automatic_enrichments': [],
            'extraction_patterns': {},
            'relationship_mapping': {}
        }
        
        # Plan automatic enrichments
        for doc in documents[:10]:  # Sample first 10 docs
            file_type = doc.metadata.get('file_type', '')
            
            if file_type in ['.py']:
                enrichment_plan['extraction_patterns']['python'] = {
                    'functions': r'def\s+(\w+)\s*\(',
                    'classes': r'class\s+(\w+)\s*[\(:]',
                    'imports': r'(?:from\s+[\w.]+\s+)?import\s+[\w\s,]+',
                }
            elif file_type in ['.js', '.ts']:
                enrichment_plan['extraction_patterns']['javascript'] = {
                    'functions': r'function\s+(\w+)|const\s+(\w+)\s*=\s*(?:\([^)]*\)|async)',
                    'classes': r'class\s+(\w+)',
                    'exports': r'export\s+(?:default\s+)?(?:class|function|const)',
                }
            elif file_type in ['.md']:
                enrichment_plan['extraction_patterns']['markdown'] = {
                    'headers': r'^(#+)\s+(.+)$',
                    'links': r'\[([^\]]+)\]\(([^)]+)\)',
                    'code_blocks': r'```(\w*)\n([\s\S]*?)```',
                }
        
        enrichment_plan['automatic_enrichments'] = [
            'file_category',
            'complexity_level',
            'has_examples',
            'has_documentation',
            'dependencies_mentioned'
        ]
        
        return enrichment_plan
    
    def _estimate_processing_metrics(self, documents: List[Document]) -> Dict[str, Any]:
        """Estimate processing time and costs"""
        total_tokens = sum(len(self.encoding.encode(doc.page_content)) for doc in documents)
        
        # Estimate based on document characteristics
        estimated_chunks = total_tokens // 400  # Average chunk size
        processing_time_seconds = len(documents) * 0.5 + estimated_chunks * 0.1
        embedding_cost = (total_tokens / 1_000_000) * 0.10  # OpenAI pricing
        
        return {
            'total_tokens': total_tokens,
            'estimated_chunks': estimated_chunks,
            'estimated_processing_time': f"{processing_time_seconds:.1f} seconds",
            'estimated_embedding_cost': f"${embedding_cost:.4f}",
            'storage_requirements': {
                'chromadb_mb': (estimated_chunks * 2) / 1000,  # Rough estimate
                'pinecone_vectors': estimated_chunks
            }
        }
    
    def _calculate_avg_line_length(self, content: str) -> float:
        """Calculate average line length"""
        lines = content.split('\n')
        if not lines:
            return 0
        return sum(len(line) for line in lines) / len(lines)
    
    def _calculate_complexity_score(self, content: str) -> float:
        """Calculate content complexity score (0-1)"""
        # Simple heuristic based on various factors
        factors = {
            'long_lines': len([l for l in content.split('\n') if len(l) > 100]) / max(len(content.split('\n')), 1),
            'special_chars': len(re.findall(r'[^a-zA-Z0-9\s]', content)) / max(len(content), 1),
            'nested_structures': len(re.findall(r'[{(\[]', content)) / max(len(content) / 100, 1)
        }
        
        return min(sum(factors.values()) / len(factors), 1.0)
    
    def _aggregate_characteristics(self, content_types: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Aggregate content characteristics"""
        aggregated = {}
        
        for file_type, characteristics_list in content_types.items():
            if characteristics_list:
                aggregated[file_type] = {
                    'has_code': sum(c['has_code'] for c in characteristics_list) / len(characteristics_list),
                    'has_tables': sum(c['has_tables'] for c in characteristics_list) / len(characteristics_list),
                    'avg_complexity': sum(c['complexity_score'] for c in characteristics_list) / len(characteristics_list)
                }
        
        return aggregated
    
    def _analyze_complexity_distribution(self, documents: List[Document]) -> Dict[str, int]:
        """Analyze complexity distribution"""
        distribution = {'low': 0, 'medium': 0, 'high': 0}
        
        for doc in documents:
            score = self._calculate_complexity_score(doc.page_content)
            if score < 0.3:
                distribution['low'] += 1
            elif score < 0.7:
                distribution['medium'] += 1
            else:
                distribution['high'] += 1
        
        return distribution
    
    def _build_directory_tree(self, documents: List[Document]) -> Dict[str, Any]:
        """Build directory tree structure"""
        tree = {}
        
        for doc in documents:
            source = doc.metadata.get('source', '')
            if source:
                parts = Path(source).parts
                current = tree
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                
                # Add file
                current[parts[-1]] = {
                    'type': doc.metadata.get('file_type', 'unknown'),
                    'size': len(doc.page_content)
                }
        
        return tree
    
    def _analyze_file_relationships(self, documents: List[Document]) -> Dict[str, List[str]]:
        """Analyze relationships between files"""
        relationships = defaultdict(list)
        
        for doc in documents:
            content = doc.page_content
            source = doc.metadata.get('source', '')
            
            # Look for imports/references
            imports = re.findall(r'(?:from|import)\s+([\w.]+)', content)
            for imp in imports:
                relationships[source].append(f"imports: {imp}")
            
            # Look for file references
            file_refs = re.findall(r'["\']([^"\']+\.\w+)["\']', content)
            for ref in file_refs:
                if ref != source:
                    relationships[source].append(f"references: {ref}")
        
        return dict(relationships)
    
    def _analyze_naming_patterns(self, documents: List[Document]) -> Dict[str, int]:
        """Analyze file naming patterns"""
        patterns = Counter()
        
        for doc in documents:
            filename = doc.metadata.get('filename', '')
            
            # Common patterns
            if filename.startswith('test_') or filename.endswith('_test.py'):
                patterns['test_files'] += 1
            elif filename.startswith('__'):
                patterns['private_files'] += 1
            elif filename.lower() in ['readme.md', 'readme.txt', 'readme']:
                patterns['readme_files'] += 1
            elif 'config' in filename.lower():
                patterns['config_files'] += 1
            elif filename.endswith('.example'):
                patterns['example_files'] += 1
        
        return dict(patterns)
    
    def _analyze_size_distribution(self, documents: List[Document]) -> Dict[str, int]:
        """Analyze document size distribution"""
        distribution = {
            'tiny (<1KB)': 0,
            'small (1-10KB)': 0,
            'medium (10-50KB)': 0,
            'large (50-100KB)': 0,
            'huge (>100KB)': 0
        }
        
        for doc in documents:
            size_kb = len(doc.page_content) / 1024
            
            if size_kb < 1:
                distribution['tiny (<1KB)'] += 1
            elif size_kb < 10:
                distribution['small (1-10KB)'] += 1
            elif size_kb < 50:
                distribution['medium (10-50KB)'] += 1
            elif size_kb < 100:
                distribution['large (50-100KB)'] += 1
            else:
                distribution['huge (>100KB)'] += 1
        
        return distribution
    
    def generate_assessment_report(self, assessment: Dict[str, Any]) -> str:
        """Generate a formatted assessment report"""
        report = []
        
        # Summary section
        report.append("# Document Collection Assessment Report")
        report.append(f"\nGenerated: {assessment['timestamp']}")
        report.append("\n## Summary")
        summary = assessment['summary']
        report.append(f"- Total Documents: {summary['total_documents']}")
        report.append(f"- Total Tokens: {summary['total_tokens']:,}")
        report.append(f"- Average Tokens per Document: {summary['average_tokens_per_doc']:,}")
        report.append(f"- Unique Sources: {summary['unique_sources']}")
        
        # File type distribution
        report.append("\n### File Type Distribution")
        for file_type, count in summary['file_type_distribution'].items():
            report.append(f"- {file_type}: {count}")
        
        # Content analysis
        report.append("\n## Content Analysis")
        content = assessment['content_analysis']
        report.append("\n### Language Distribution")
        for lang, count in content['language_distribution'].items():
            report.append(f"- {lang}: {count}")
        
        report.append("\n### Complexity Distribution")
        for level, count in content['complexity_analysis'].items():
            report.append(f"- {level}: {count}")
        
        # Recommended strategies
        report.append("\n## Recommended Processing Strategies")
        strategies = assessment['optimal_strategies']
        
        report.append("\n### Chunking Strategy")
        for rec in strategies['chunking_strategy']['recommendations']:
            report.append(f"- **{rec['type']}**: chunk_size={rec['chunk_size']}, overlap={rec['chunk_overlap']}")
            report.append(f"  - Strategy: {rec['strategy']}")
            report.append(f"  - Reason: {rec['reason']}")
        
        report.append("\n### Namespace Strategy")
        for ns, count in strategies['namespace_strategy']['recommended_namespaces'].items():
            report.append(f"- {ns}: {count} documents")
        
        report.append("\n### Processing Order")
        for step in strategies['processing_order']:
            report.append(f"- {step}")
        
        # Estimated metrics
        report.append("\n## Estimated Processing Metrics")
        metrics = assessment['estimated_metrics']
        report.append(f"- Estimated Chunks: {metrics['estimated_chunks']:,}")
        report.append(f"- Processing Time: {metrics['estimated_processing_time']}")
        report.append(f"- Embedding Cost: {metrics['estimated_embedding_cost']}")
        report.append(f"- ChromaDB Storage: ~{metrics['storage_requirements']['chromadb_mb']:.1f} MB")
        report.append(f"- Pinecone Vectors: {metrics['storage_requirements']['pinecone_vectors']:,}")
        
        return "\n".join(report)
    
    def display_assessment(self, assessment: Dict[str, Any]):
        """Display assessment results in the console"""
        # Summary table
        summary_table = Table(title="Document Collection Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        
        summary = assessment['summary']
        summary_table.add_row("Total Documents", str(summary['total_documents']))
        summary_table.add_row("Total Tokens", f"{summary['total_tokens']:,}")
        summary_table.add_row("Avg Tokens/Doc", f"{summary['average_tokens_per_doc']:,}")
        
        console.print(summary_table)
        
        # Strategy recommendations
        console.print("\n[bold]Recommended Strategies:[/bold]")
        strategies = assessment['optimal_strategies']
        
        for rec in strategies['chunking_strategy']['recommendations']:
            console.print(f"  • {rec['type']}: [yellow]{rec['strategy']}[/yellow] "
                         f"(size={rec['chunk_size']}, overlap={rec['chunk_overlap']})")
        
        # Estimated metrics
        metrics = assessment['estimated_metrics']
        console.print(f"\n[bold]Estimated Processing:[/bold]")
        console.print(f"  • Time: [yellow]{metrics['estimated_processing_time']}[/yellow]")
        console.print(f"  • Cost: [yellow]{metrics['estimated_embedding_cost']}[/yellow]")
        console.print(f"  • Chunks: [yellow]{metrics['estimated_chunks']:,}[/yellow]")