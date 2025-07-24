#!/usr/bin/env python3
"""
🤖 Smart Ingestion System - Intelligent document analysis and planning

Analyzes documents before processing to:
1. Determine document type and structure
2. Assess current quality
3. Plan optimal chunking strategy
4. Predict enhancement needs
5. Estimate costs and time
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import statistics

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

@dataclass
class DocumentProfile:
    """Profile of a document based on analysis"""
    doc_type: str  # technical, tutorial, reference, narrative, mixed
    structure: str  # linear, hierarchical, fragmented
    quality_estimate: float
    language: str  # programming language if code-heavy
    topics: List[str]
    estimated_chunks: int
    special_features: List[str]  # tables, code blocks, images, etc.
    
@dataclass 
class IngestionPlan:
    """Plan for how to process a document"""
    profile: DocumentProfile
    chunking_strategy: str
    chunk_size: int
    overlap: int
    quality_threshold: float
    enhancement_strategies: List[str]
    estimated_cost: float
    estimated_time: float
    warnings: List[str]
    recommendations: List[str]

class SmartIngestionPlanner:
    """
    Analyzes documents and creates intelligent processing plans
    """
    
    def __init__(self):
        self.console = Console()
        
        # Document type patterns
        self.doc_patterns = {
            'technical': {
                'keywords': ['api', 'function', 'class', 'method', 'parameter', 'implementation'],
                'structures': [r'```\w+', r'def\s+\w+', r'class\s+\w+', r'interface\s+\w+'],
                'weight': 0
            },
            'tutorial': {
                'keywords': ['step', 'first', 'next', 'tutorial', 'guide', 'how to', 'example'],
                'structures': [r'\d+\.\s+', r'Step\s+\d+', r'##\s+.*Example'],
                'weight': 0
            },
            'reference': {
                'keywords': ['reference', 'documentation', 'spec', 'definition', 'syntax'],
                'structures': [r'##\s+\w+', r'###\s+Parameters', r'###\s+Returns'],
                'weight': 0
            },
            'narrative': {
                'keywords': ['story', 'experience', 'journey', 'learned', 'discovered'],
                'structures': [r'\b(I|we|they)\s+\w+ed\b'],
                'weight': 0
            }
        }
        
    def analyze_document(self, file_path: Path) -> Tuple[DocumentProfile, IngestionPlan]:
        """Analyze document and create processing plan"""
        
        console.print(Panel.fit(
            f"[bold cyan]🤖 Smart Document Analysis[/bold cyan]\n"
            f"Analyzing: {file_path.name}",
            border_style="cyan"
        ))
        
        # Read document
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # Step 1: Profile document
            task = progress.add_task("Profiling document type...", total=None)
            profile = self._profile_document(content, file_path)
            progress.remove_task(task)
            
            # Step 2: Analyze quality
            task = progress.add_task("Assessing quality...", total=None)
            quality_assessment = self._assess_quality(content)
            profile.quality_estimate = quality_assessment['average_score']
            progress.remove_task(task)
            
            # Step 3: Create plan
            task = progress.add_task("Creating ingestion plan...", total=None)
            plan = self._create_plan(profile, content, quality_assessment)
            progress.remove_task(task)
        
        # Display results
        self._display_analysis(profile, plan)
        
        return profile, plan
    
    def _profile_document(self, content: str, file_path: Path) -> DocumentProfile:
        """Create document profile"""
        
        # Detect document type
        for doc_type, patterns in self.doc_patterns.items():
            score = 0
            
            # Check keywords
            for keyword in patterns['keywords']:
                score += content.lower().count(keyword)
            
            # Check structures
            for pattern in patterns['structures']:
                matches = re.findall(pattern, content, re.MULTILINE)
                score += len(matches) * 2  # Weight structure matches higher
            
            self.doc_patterns[doc_type]['weight'] = score
        
        # Get primary type
        primary_type = max(self.doc_patterns.items(), key=lambda x: x[1]['weight'])[0]
        
        # Detect structure
        structure = self._detect_structure(content)
        
        # Detect special features
        special_features = self._detect_features(content)
        
        # Extract topics
        topics = self._extract_topics(content)
        
        # Detect programming language if technical
        language = self._detect_language(content) if primary_type == 'technical' else 'none'
        
        # Estimate chunks
        estimated_chunks = self._estimate_chunks(content, primary_type)
        
        return DocumentProfile(
            doc_type=primary_type,
            structure=structure,
            quality_estimate=0,  # Will be filled later
            language=language,
            topics=topics[:5],  # Top 5 topics
            estimated_chunks=estimated_chunks,
            special_features=special_features
        )
    
    def _detect_structure(self, content: str) -> str:
        """Detect document structure"""
        
        # Check for hierarchical headers
        headers = re.findall(r'^#+\s+.+$', content, re.MULTILINE)
        if len(headers) > 5:
            # Check if headers follow hierarchy
            header_levels = [len(h.split()[0]) for h in headers]
            if header_levels == sorted(header_levels):
                return 'hierarchical'
        
        # Check for numbered lists
        numbered = re.findall(r'^\d+\.\s+', content, re.MULTILINE)
        if len(numbered) > 3:
            return 'linear'
        
        # Check for consistent paragraphs
        paragraphs = content.split('\n\n')
        if len(paragraphs) > 10:
            para_lengths = [len(p.split()) for p in paragraphs if p.strip()]
            if para_lengths:
                avg_length = statistics.mean(para_lengths)
                std_dev = statistics.stdev(para_lengths) if len(para_lengths) > 1 else 0
                if std_dev < avg_length * 0.5:  # Consistent paragraph sizes
                    return 'linear'
        
        return 'fragmented'
    
    def _detect_features(self, content: str) -> List[str]:
        """Detect special features in document"""
        features = []
        
        # Code blocks
        if '```' in content:
            features.append('code_blocks')
        
        # Tables (markdown)
        if re.search(r'\|.+\|.+\|', content):
            features.append('tables')
        
        # Links
        if re.search(r'\[.+\]\(.+\)', content):
            features.append('links')
        
        # Images
        if re.search(r'!\[.*\]\(.+\)', content):
            features.append('images')
        
        # Lists
        if re.search(r'^[*-]\s+', content, re.MULTILINE):
            features.append('bullet_lists')
        
        # Quotes
        if re.search(r'^>\s+', content, re.MULTILINE):
            features.append('quotes')
        
        # Math (LaTeX)
        if '$' in content or '\\[' in content:
            features.append('math')
        
        return features
    
    def _extract_topics(self, content: str) -> List[str]:
        """Extract main topics from document"""
        topics = []
        
        # Extract from headers
        headers = re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)
        topics.extend([h.strip() for h in headers[:10]])
        
        # Extract capitalized phrases (likely important terms)
        cap_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b', content)
        
        # Count frequency
        topic_freq = {}
        for phrase in cap_phrases:
            if len(phrase) > 3 and phrase not in ['The', 'This', 'That', 'These']:
                topic_freq[phrase] = topic_freq.get(phrase, 0) + 1
        
        # Get top topics
        top_topics = sorted(topic_freq.items(), key=lambda x: x[1], reverse=True)
        topics.extend([t[0] for t in top_topics[:5]])
        
        return list(dict.fromkeys(topics))  # Remove duplicates, preserve order
    
    def _detect_language(self, content: str) -> str:
        """Detect programming language from code blocks"""
        
        # Look for code blocks with language hints
        code_blocks = re.findall(r'```(\w+)', content)
        if code_blocks:
            # Return most common
            from collections import Counter
            return Counter(code_blocks).most_common(1)[0][0]
        
        # Check for language-specific patterns
        if 'import ' in content or 'from ' in content or 'def ' in content:
            return 'python'
        elif 'function ' in content or 'const ' in content or '=>' in content:
            return 'javascript'
        elif 'public class' in content or 'private ' in content:
            return 'java'
        
        return 'mixed'
    
    def _estimate_chunks(self, content: str, doc_type: str) -> int:
        """Estimate number of chunks based on document type"""
        
        # Base chunk sizes by type
        chunk_sizes = {
            'technical': 1200,  # Smaller chunks for precise technical content
            'tutorial': 1500,   # Medium chunks for step-by-step
            'reference': 1000,  # Smaller chunks for quick lookup
            'narrative': 2000   # Larger chunks for story flow
        }
        
        chunk_size = chunk_sizes.get(doc_type, 1500)
        
        # Adjust for special features
        if 'code_blocks' in self._detect_features(content):
            chunk_size = int(chunk_size * 0.8)  # Smaller chunks for code
        
        return max(1, len(content) // chunk_size)
    
    def _assess_quality(self, content: str) -> Dict[str, Any]:
        """Quick quality assessment of content"""
        
        # Sample chunks for assessment
        chunk_size = 1500
        chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
        
        # Assess first few chunks
        scores = []
        issues = []
        
        for chunk in chunks[:5]:  # Sample first 5 chunks
            score, chunk_issues = self._score_chunk(chunk)
            scores.append(score)
            issues.extend(chunk_issues)
        
        return {
            'average_score': statistics.mean(scores) if scores else 0,
            'min_score': min(scores) if scores else 0,
            'max_score': max(scores) if scores else 0,
            'common_issues': list(set(issues))[:5]
        }
    
    def _score_chunk(self, chunk: str) -> Tuple[float, List[str]]:
        """Quick scoring of a chunk"""
        score = 10.0
        issues = []
        
        # Check length
        words = chunk.split()
        if len(words) < 50:
            score -= 2
            issues.append('too_short')
        
        # Check for context
        if chunk.startswith(('This', 'That', 'It', 'They')):
            score -= 1
            issues.append('missing_context')
        
        # Check for examples
        if not any(phrase in chunk.lower() for phrase in ['for example', 'such as', 'e.g.', 'instance']):
            score -= 1
            issues.append('no_examples')
        
        # Check for definitions
        technical_terms = re.findall(r'\b[A-Z]{2,}\b', chunk)
        if technical_terms and not any(term in chunk for term in ['stands for', 'means', 'refers to']):
            score -= 1
            issues.append('undefined_terms')
        
        # Check structure
        if '\n' not in chunk and len(chunk) > 500:
            score -= 0.5
            issues.append('no_structure')
        
        return max(0, score), issues
    
    def _create_plan(self, profile: DocumentProfile, content: str, quality: Dict) -> IngestionPlan:
        """Create ingestion plan based on analysis"""
        
        # Determine chunking strategy
        if profile.doc_type == 'technical' and 'code_blocks' in profile.special_features:
            chunking_strategy = 'code_aware'  # Preserve code blocks
            chunk_size = 1200
            overlap = 100
        elif profile.doc_type == 'tutorial':
            chunking_strategy = 'step_aware'  # Preserve steps
            chunk_size = 1500  
            overlap = 200
        elif profile.structure == 'hierarchical':
            chunking_strategy = 'header_aware'  # Respect headers
            chunk_size = 1500
            overlap = 150
        else:
            chunking_strategy = 'semantic'  # Default
            chunk_size = 1500
            overlap = 200
        
        # Determine enhancement needs
        enhancement_strategies = []
        if quality['average_score'] < 9.5:
            if 'missing_context' in quality['common_issues']:
                enhancement_strategies.append('add_context')
            if 'no_examples' in quality['common_issues']:
                enhancement_strategies.append('add_examples')
            if 'undefined_terms' in quality['common_issues']:
                enhancement_strategies.append('define_terms')
            if 'too_short' in quality['common_issues']:
                enhancement_strategies.append('expand_content')
        
        # Estimate costs
        if os.getenv('LLMFY_ENV', 'development') == 'development':
            estimated_cost = 0  # Free local embeddings
        else:
            # ~$0.0001 per 1K tokens, ~250 tokens per chunk
            estimated_cost = (profile.estimated_chunks * 250 / 1000) * 0.0001
        
        # Estimate time
        estimated_time = profile.estimated_chunks * 0.5  # ~0.5 seconds per chunk
        
        # Warnings and recommendations
        warnings = []
        recommendations = []
        
        if profile.estimated_chunks > 100:
            warnings.append(f"Large document will create ~{profile.estimated_chunks} chunks")
            recommendations.append("Consider processing in batches")
        
        if quality['average_score'] < 7:
            warnings.append(f"Low quality score ({quality['average_score']:.1f}/10)")
            recommendations.append("Manual review recommended after enhancement")
        
        if 'images' in profile.special_features:
            warnings.append("Document contains images")
            recommendations.append("Enable multi-modal processing for best results")
        
        return IngestionPlan(
            profile=profile,
            chunking_strategy=chunking_strategy,
            chunk_size=chunk_size,
            overlap=overlap,
            quality_threshold=9.5,
            enhancement_strategies=enhancement_strategies,
            estimated_cost=estimated_cost,
            estimated_time=estimated_time,
            warnings=warnings,
            recommendations=recommendations
        )
    
    def _display_analysis(self, profile: DocumentProfile, plan: IngestionPlan):
        """Display analysis results"""
        
        # Document Profile
        console.print("\n[bold]Document Profile:[/bold]")
        profile_table = Table(show_header=False, box=None)
        profile_table.add_column("Property", style="cyan")
        profile_table.add_column("Value", style="white")
        
        profile_table.add_row("Type", profile.doc_type.title())
        profile_table.add_row("Structure", profile.structure.title())
        profile_table.add_row("Quality Estimate", f"{profile.quality_estimate:.1f}/10")
        profile_table.add_row("Language", profile.language.title() if profile.language != 'none' else 'N/A')
        profile_table.add_row("Topics", ", ".join(profile.topics[:3]) if profile.topics else 'General')
        profile_table.add_row("Special Features", ", ".join(profile.special_features) if profile.special_features else 'None')
        profile_table.add_row("Estimated Chunks", str(profile.estimated_chunks))
        
        console.print(profile_table)
        
        # Ingestion Plan
        console.print("\n[bold]Ingestion Plan:[/bold]")
        plan_table = Table(show_header=False, box=None)
        plan_table.add_column("Property", style="cyan")
        plan_table.add_column("Value", style="white")
        
        plan_table.add_row("Chunking Strategy", plan.chunking_strategy.replace('_', ' ').title())
        plan_table.add_row("Chunk Size", f"{plan.chunk_size} chars")
        plan_table.add_row("Overlap", f"{plan.overlap} chars")
        plan_table.add_row("Quality Threshold", f"{plan.quality_threshold}/10")
        
        if plan.enhancement_strategies:
            strategies = ", ".join([s.replace('_', ' ') for s in plan.enhancement_strategies])
            plan_table.add_row("Enhancements", strategies)
        else:
            plan_table.add_row("Enhancements", "None needed")
        
        plan_table.add_row("Estimated Cost", f"${plan.estimated_cost:.4f}" if plan.estimated_cost > 0 else "Free (local)")
        plan_table.add_row("Estimated Time", f"{plan.estimated_time:.1f} seconds")
        
        console.print(plan_table)
        
        # Warnings and Recommendations
        if plan.warnings:
            console.print("\n[bold yellow]Warnings:[/bold yellow]")
            for warning in plan.warnings:
                console.print(f"  ⚠️  {warning}")
        
        if plan.recommendations:
            console.print("\n[bold green]Recommendations:[/bold green]")
            for rec in plan.recommendations:
                console.print(f"  💡 {rec}")
    
    def save_plan(self, plan: IngestionPlan, file_path: Path) -> Path:
        """Save ingestion plan for later use"""
        
        plan_dir = Path("data/ingestion_plans")
        plan_dir.mkdir(exist_ok=True)
        
        plan_file = plan_dir / f"plan_{file_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        plan_data = {
            'created_at': datetime.utcnow().isoformat(),
            'source_file': str(file_path),
            'profile': {
                'doc_type': plan.profile.doc_type,
                'structure': plan.profile.structure,
                'quality_estimate': plan.profile.quality_estimate,
                'language': plan.profile.language,
                'topics': plan.profile.topics,
                'estimated_chunks': plan.profile.estimated_chunks,
                'special_features': plan.profile.special_features
            },
            'plan': {
                'chunking_strategy': plan.chunking_strategy,
                'chunk_size': plan.chunk_size,
                'overlap': plan.overlap,
                'quality_threshold': plan.quality_threshold,
                'enhancement_strategies': plan.enhancement_strategies,
                'estimated_cost': plan.estimated_cost,
                'estimated_time': plan.estimated_time,
                'warnings': plan.warnings,
                'recommendations': plan.recommendations
            }
        }
        
        with open(plan_file, 'w') as f:
            json.dump(plan_data, f, indent=2)
        
        return plan_file
