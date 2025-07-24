#!/usr/bin/env python3
"""
Analyze quality of good files and identify fixes for bad ones
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any

class KnowledgeQualityAnalyzer:
    def __init__(self, librarian_root: str | None = None):
        """Initialize the analyzer.

        Parameters
        ----------
        librarian_root : str | None, optional
            Base directory for processed documents. Defaults to ``./data/librarian``
            relative to the current working directory.
        """

        if librarian_root is None:
            librarian_root = Path.cwd() / "data" / "librarian"

        self.root = Path(librarian_root)
        self.vector_dir = self.root / "vectors"
        
        # Quality scoring criteria (0-10 scale)
        self.quality_metrics = {
            'clarity': {'weight': 0.2, 'score': 0},
            'completeness': {'weight': 0.2, 'score': 0},
            'structure': {'weight': 0.15, 'score': 0},
            'examples': {'weight': 0.15, 'score': 0},
            'definitions': {'weight': 0.15, 'score': 0},
            'relationships': {'weight': 0.15, 'score': 0}
        }
    
    def analyze_all_documents(self):
        """Analyze all documents and rank by quality"""
        
        print("📊 ANALYZING DOCUMENT QUALITY\n")
        
        document_scores = []
        
        # Analyze each document
        for vector_file in self.vector_dir.glob("*_data.json"):
            with open(vector_file) as f:
                doc_data = json.load(f)
            
            score, analysis = self._analyze_document_quality(doc_data)
            
            document_scores.append({
                'name': doc_data.get('doc_id', vector_file.stem),
                'score': score,
                'analysis': analysis,
                'chunks': doc_data.get('chunk_count', 0),
                'category': doc_data.get('category', 'unknown')
            })
        
        # Sort by score
        document_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Identify good, average, and bad
        good_docs = [d for d in document_scores if d['score'] >= 7.0]
        average_docs = [d for d in document_scores if 4.0 <= d['score'] < 7.0]
        bad_docs = [d for d in document_scores if d['score'] < 4.0]
        
        return good_docs, average_docs, bad_docs
    
    def _analyze_document_quality(self, doc_data: Dict) -> Tuple[float, Dict]:
        """Analyze quality of a single document"""
        
        analysis = {
            'clarity': 0,
            'completeness': 0,
            'structure': 0,
            'examples': 0,
            'definitions': 0,
            'relationships': 0,
            'issues': [],
            'strengths': []
        }
        
        chunks = doc_data.get('chunks', [])
        if not chunks:
            return 0, analysis
        
        # Sample chunks for analysis
        sample_size = min(10, len(chunks))
        sample_chunks = chunks[:sample_size]
        
        # Analyze each metric
        for chunk in sample_chunks:
            text = chunk.get('text', '')
            
            # Clarity (0-10)
            clarity_score = self._score_clarity(text)
            analysis['clarity'] += clarity_score
            
            # Completeness (0-10)
            completeness_score = self._score_completeness(text, chunk)
            analysis['completeness'] += completeness_score
            
            # Structure (0-10)
            structure_score = self._score_structure(text)
            analysis['structure'] += structure_score
            
            # Examples (0-10)
            examples_score = self._score_examples(text)
            analysis['examples'] += examples_score
            
            # Definitions (0-10)
            definitions_score = self._score_definitions(text)
            analysis['definitions'] += definitions_score
            
            # Relationships (0-10)
            relationships_score = self._score_relationships(text)
            analysis['relationships'] += relationships_score
        
        # Average scores
        for metric in analysis:
            if metric not in ['issues', 'strengths'] and sample_size > 0:
                analysis[metric] = analysis[metric] / sample_size
        
        # Calculate weighted total score
        total_score = 0
        for metric, data in self.quality_metrics.items():
            total_score += analysis[metric] * data['weight']
        
        # Identify specific issues and strengths
        self._identify_issues_and_strengths(analysis, sample_chunks)
        
        return total_score, analysis
    
    def _score_clarity(self, text: str) -> float:
        """Score text clarity (0-10)"""
        score = 10.0
        
        # Deduct for unclear references
        unclear_refs = len(re.findall(r'\b(this|that|it|they)\b(?!\s+(?:is|are|was|were))', text))
        score -= min(unclear_refs * 0.5, 3.0)
        
        # Deduct for overly complex sentences
        sentences = text.split('.')
        long_sentences = [s for s in sentences if len(s.split()) > 30]
        score -= min(len(long_sentences) * 0.5, 2.0)
        
        # Deduct for jargon without explanation
        jargon_pattern = r'\b[A-Z]{3,}\b'
        unexplained_jargon = len(re.findall(jargon_pattern, text))
        score -= min(unexplained_jargon * 0.3, 2.0)
        
        return max(0, score)
    
    def _score_completeness(self, text: str, chunk: Dict) -> float:
        """Score content completeness (0-10)"""
        score = 5.0  # Start neutral
        
        # Add points for self-contained chunks
        if text.strip() and not text.startswith(('and ', 'but ', 'or ', 'however ')):
            score += 2.0
        
        # Add points for proper introduction
        if re.match(r'^[A-Z].*[.!?]', text[:100] if len(text) > 100 else text):
            score += 1.0
        
        # Add points for conclusion/summary
        if any(phrase in text.lower() for phrase in ['in summary', 'therefore', 'this means']):
            score += 1.0
        
        # Add points for metadata
        if chunk.get('metadata', {}).get('section'):
            score += 1.0
        
        return min(10, score)
    
    def _score_structure(self, text: str) -> float:
        """Score structural quality (0-10)"""
        score = 0
        
        # Headers
        if re.search(r'^#{1,6}\s+', text, re.MULTILINE):
            score += 3.0
        
        # Lists
        if re.search(r'^[-*\d]\.\s+', text, re.MULTILINE):
            score += 2.0
        
        # Code blocks
        if '```' in text:
            score += 2.0
        
        # Paragraphs (multiple \n\n)
        if text.count('\n\n') > 2:
            score += 1.5
        
        # Bold/emphasis
        if re.search(r'\*\*[^*]+\*\*', text):
            score += 1.5
        
        return min(10, score)
    
    def _score_examples(self, text: str) -> float:
        """Score example presence and quality (0-10)"""
        score = 0
        
        # Explicit examples
        example_phrases = ['for example', 'e.g.', 'such as', 'example:', 'for instance']
        example_count = sum(1 for phrase in example_phrases if phrase.lower() in text.lower())
        score += min(example_count * 2.5, 5.0)
        
        # Code examples
        if '```' in text:
            score += 3.0
        
        # Practical scenarios
        if any(phrase in text.lower() for phrase in ['consider', 'suppose', 'imagine', 'let\'s say']):
            score += 2.0
        
        return min(10, score)
    
    def _score_definitions(self, text: str) -> float:
        """Score definition quality (0-10)"""
        score = 0
        
        # Explicit definitions
        definition_patterns = [
            r'\b\w+\s+(?:is|are|means|refers to)\s+',
            r':\s*[A-Z][^.]+\.',
            r'\([^)]+\)',  # Parenthetical explanations
            r'"[^"]+"',    # Quoted terms
        ]
        
        for pattern in definition_patterns:
            matches = len(re.findall(pattern, text))
            score += min(matches * 1.5, 3.0)
        
        return min(10, score)
    
    def _score_relationships(self, text: str) -> float:
        """Score relationship clarity (0-10)"""
        score = 0
        
        # Relationship indicators
        relationship_phrases = [
            'relates to', 'connected to', 'depends on', 'similar to',
            'different from', 'compared to', 'in contrast', 'builds on'
        ]
        
        for phrase in relationship_phrases:
            if phrase in text.lower():
                score += 1.5
        
        # Cross-references
        if re.search(r'see\s+(?:also|section|chapter)', text, re.IGNORECASE):
            score += 2.0
        
        # Causal relationships
        if any(word in text.lower() for word in ['because', 'therefore', 'thus', 'consequently']):
            score += 2.0
        
        return min(10, score)
    
    def _identify_issues_and_strengths(self, analysis: Dict, chunks: List[Dict]):
        """Identify specific issues and strengths"""
        
        # Identify strengths
        if analysis['clarity'] > 7:
            analysis['strengths'].append("Clear and unambiguous language")
        if analysis['examples'] > 7:
            analysis['strengths'].append("Rich with practical examples")
        if analysis['structure'] > 7:
            analysis['strengths'].append("Well-structured content")
        
        # Identify issues
        if analysis['clarity'] < 5:
            analysis['issues'].append("Contains unclear references and complex language")
        if analysis['completeness'] < 5:
            analysis['issues'].append("Chunks lack self-contained context")
        if analysis['examples'] < 3:
            analysis['issues'].append("Needs more concrete examples")
        if analysis['definitions'] < 3:
            analysis['issues'].append("Technical terms not properly defined")
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze a single chunk of text and return quality scores"""
        
        # Create a fake chunk dict for compatibility
        chunk = {'text': text}
        
        # Get individual scores
        clarity = self._score_clarity(text)
        completeness = self._score_completeness(text, chunk)
        structure = self._score_structure(text)
        examples = self._score_examples(text)
        definitions = self._score_definitions(text)
        relationships = self._score_relationships(text)
        
        # Calculate overall score
        dimension_scores = {
            'clarity': clarity,
            'completeness': completeness,
            'structure': structure,
            'examples': examples,
            'definitions': definitions,
            'relationships': relationships
        }
        
        # Weighted average
        overall_score = 0
        for metric, data in self.quality_metrics.items():
            overall_score += dimension_scores[metric] * data['weight']
        
        return {
            'overall_score': overall_score,
            'dimension_scores': dimension_scores,
            'issues': [],
            'suggestions': []
        }


def generate_improvement_plan(bad_docs: List[Dict], average_docs: List[Dict]):
    """Generate specific improvement plans for bad documents"""
    
    print("\n🔧 IMPROVEMENT PLANS FOR LOW-QUALITY DOCUMENTS\n")
    
    for doc in bad_docs[:5]:  # Top 5 worst
        print(f"📄 {doc['name'][:50]}... (Score: {doc['score']:.1f}/10)")
        print(f"   Category: {doc['category']}")
        
        analysis = doc['analysis']
        
        # Generate specific fixes
        print("   Fixes needed:")
        
        if analysis['clarity'] < 5:
            print("   ✏️  Clarity fixes:")
            print("      - Replace pronouns with specific nouns")
            print("      - Break long sentences (>25 words)")
            print("      - Define acronyms on first use")
        
        if analysis['completeness'] < 5:
            print("   ✏️  Completeness fixes:")
            print("      - Add context headers to each chunk")
            print("      - Include topic sentences")
            print("      - Add chunk summaries")
        
        if analysis['examples'] < 3:
            print("   ✏️  Example fixes:")
            print("      - Add code snippets for technical concepts")
            print("      - Include 'For example:' sections")
            print("      - Show before/after scenarios")
        
        if analysis['structure'] < 5:
            print("   ✏️  Structure fixes:")
            print("      - Add markdown headers (##)")
            print("      - Use bullet points for lists")
            print("      - Separate with clear paragraphs")
        
        print()


def show_good_examples(good_docs: List[Dict]):
    """Show what makes good documents good"""
    
    print("\n✨ WHAT MAKES GOOD DOCUMENTS GOOD\n")
    
    for doc in good_docs[:3]:  # Top 3
        print(f"🌟 {doc['name'][:50]}... (Score: {doc['score']:.1f}/10)")
        print(f"   Strengths: {', '.join(doc['analysis']['strengths'])}")
        
        print("   Quality factors:")
        analysis = doc['analysis']
        for metric in ['clarity', 'completeness', 'structure', 'examples']:
            if analysis[metric] > 7:
                print(f"   • {metric.capitalize()}: {analysis[metric]:.1f}/10")
        print()


# Main execution
if __name__ == "__main__":
    analyzer = KnowledgeQualityAnalyzer()
    
    # Analyze all documents
    good_docs, average_docs, bad_docs = analyzer.analyze_all_documents()
    
    print(f"📊 Quality Distribution:")
    print(f"   🌟 Good (7+/10): {len(good_docs)} documents")
    print(f"   📊 Average (4-7/10): {len(average_docs)} documents")
    print(f"   ⚠️  Poor (<4/10): {len(bad_docs)} documents")
    
    # Show good examples
    show_good_examples(good_docs)
    
    # Generate improvement plans
    generate_improvement_plan(bad_docs, average_docs)
    
    # Overall recommendations
    print("\n💡 UNIVERSAL IMPROVEMENTS:")
    print("1. Add this header to EVERY chunk:")
    print("   [Source: {filename} | Section: {section} | Topic: {topic}]")
    print("\n2. Define terms inline:")
    print("   'We use RRF (Reciprocal Rank Fusion, formula: Σ 1/(k+rank)) to...'")
    print("\n3. Make chunks self-contained:")
    print("   Start with context, end with summary")
    print("\n4. Add examples after concepts:")
    print("   'Concept explanation... For example: [concrete case]'")
    print("\n5. Link related concepts:")
    print("   'This relates to hybrid retrieval (see section X) by...'")