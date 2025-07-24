#!/usr/bin/env python3
"""
🎯 Quality Scorer V2 - Based on Quality Control Research

Implements retrieval-oriented quality scoring based on insights from
"Quality Control Methods for LLM Knowledge Bases" research.
"""

import re
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import statistics

@dataclass
class QualityDimensions:
    """Quality dimensions that actually matter for RAG performance"""
    context_independence: float = 0.0  # Can chunk stand alone?
    information_density: float = 0.0   # Specific facts, numbers, details
    semantic_coherence: float = 0.0    # Single topic focus
    factual_grounding: float = 0.0     # References, sources, specifics
    clarity: float = 0.0               # Readability and structure
    relevance_potential: float = 0.0   # Likely to answer questions

class ImprovedQualityAnalyzer:
    """
    Quality analyzer based on research insights:
    - Focus on retrieval performance, not formatting
    - Recognize valuable content regardless of structure
    - Use semantic signals over syntactic patterns
    """
    
    def __init__(self):
        # Weights based on importance for RAG
        self.dimension_weights = {
            'context_independence': 0.25,
            'information_density': 0.20,
            'semantic_coherence': 0.20,
            'factual_grounding': 0.15,
            'clarity': 0.10,
            'relevance_potential': 0.10
        }
        
    def analyze(self, text: str, metadata: Dict = None) -> Dict[str, Any]:
        """Analyze chunk quality with improved metrics"""
        
        # First check if this is a PDF artifact chunk
        if self._is_artifact_chunk(text):
            return {
                'overall_score': 0.0,
                'dimension_scores': {dim: 0.0 for dim in self.dimension_weights},
                'strengths': [],
                'weaknesses': ['PDF artifact or non-content chunk'],
                'recommendations': ['Remove this chunk - it contains no meaningful content']
            }
        
        dimensions = QualityDimensions()
        
        # Score each dimension
        dimensions.context_independence = self._score_context_independence(text)
        dimensions.information_density = self._score_information_density(text)
        dimensions.semantic_coherence = self._score_semantic_coherence(text)
        dimensions.factual_grounding = self._score_factual_grounding(text)
        dimensions.clarity = self._score_clarity(text)
        dimensions.relevance_potential = self._score_relevance_potential(text)
        
        # Calculate weighted overall score
        dimension_scores = {
            'context_independence': dimensions.context_independence,
            'information_density': dimensions.information_density,
            'semantic_coherence': dimensions.semantic_coherence,
            'factual_grounding': dimensions.factual_grounding,
            'clarity': dimensions.clarity,
            'relevance_potential': dimensions.relevance_potential
        }
        
        overall_score = sum(
            score * self.dimension_weights[dim] 
            for dim, score in dimension_scores.items()
        )
        
        # Boost for continuity markers
        if self._has_continuity_markers(text):
            overall_score = min(10, overall_score + 0.5)
        
        # Identify strengths and weaknesses
        strengths = [dim for dim, score in dimension_scores.items() if score >= 8.0]
        weaknesses = [dim for dim, score in dimension_scores.items() if score < 6.0]
        
        return {
            'overall_score': overall_score,
            'dimension_scores': dimension_scores,
            'strengths': strengths,
            'weaknesses': weaknesses,
            'recommendations': self._get_recommendations(dimension_scores)
        }
    
    def _score_context_independence(self, text: str) -> float:
        """Can this chunk be understood without external context?"""
        score = 10.0
        
        # Check for undefined pronouns at start
        if re.match(r'^(This|That|It|They|These|Those)\s+', text):
            score -= 3.0
        
        # Check for proper nouns or specific subjects
        proper_nouns = len(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text))
        if proper_nouns > 0:
            score = min(10, score + proper_nouns * 0.5)
        
        # Check for explicit topic statement
        if any(phrase in text.lower() for phrase in 
               ['this section', 'this guide', 'we will', 'this document']):
            score = min(10, score + 1.0)
            
        return max(0, score)
    
    def _score_information_density(self, text: str) -> float:
        """How much specific, actionable information?"""
        score = 5.0  # Start neutral
        
        # Technical specifics (hex codes, measurements, numbers)
        technical_patterns = [
            r'#[0-9A-Fa-f]{3,6}\b',  # Hex colors
            r'\b\d+(?:\.\d+)?(?:px|em|rem|%|ms|s)\b',  # Measurements
            r'\b\d+x\d+\b',  # Dimensions
            r'(?:^|[^.])\d+(?:\.\d+)?',  # Numbers
        ]
        
        tech_count = sum(len(re.findall(pattern, text)) for pattern in technical_patterns)
        score += min(3.0, tech_count * 0.3)
        
        # Code or technical terms
        if re.search(r'`[^`]+`|"[^"]+"|\'[^\']+\'', text):
            score += 1.0
            
        # Specific names, tools, frameworks
        if re.search(r'\b(React|Vue|Angular|CSS|HTML|JavaScript|Python|API)\b', text, re.I):
            score += 0.5
            
        # Examples and specifics
        if any(phrase in text.lower() for phrase in 
               ['for example', 'such as', 'specifically', 'in particular']):
            score += 1.0
            
        return min(10, score)
    
    def _score_semantic_coherence(self, text: str) -> float:
        """Is the chunk about a single, focused topic?"""
        score = 8.0  # Start high, deduct for issues
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) > 1:
            # Check topic consistency (simple heuristic)
            # Count unique "topic words" (nouns/verbs)
            topic_words = set()
            for sentence in sentences:
                words = re.findall(r'\b[a-zA-Z]{4,}\b', sentence.lower())
                topic_words.update(words[:3])  # First few meaningful words
            
            # High variety of topics = lower coherence
            topic_variety = len(topic_words) / max(len(sentences), 1)
            if topic_variety > 5:
                score -= 2.0
                
        # Multiple transition words suggest multiple topics
        transitions = len(re.findall(
            r'\b(however|meanwhile|additionally|furthermore|moreover|nevertheless)\b', 
            text, re.I
        ))
        score -= min(2.0, transitions * 0.5)
        
        return max(0, score)
    
    def _score_factual_grounding(self, text: str) -> float:
        """Does the chunk provide sources, references, or specific claims?"""
        score = 5.0
        
        # References to specific sources
        if re.search(r'\b(Google|Apple|Microsoft|AWS|Material Design|HIG)\b', text):
            score += 2.0
            
        # URLs or citations
        if re.search(r'https?://|www\.|\.com|\.org', text):
            score += 1.5
            
        # Specific claims with evidence
        if re.search(r'\b(research shows|studies indicate|according to|recommended by)\b', text, re.I):
            score += 1.5
            
        # Version numbers, specifications
        if re.search(r'v?\d+\.\d+|\b(RFC|ISO|IEEE)\s*\d+', text):
            score += 1.0
            
        return min(10, score)
    
    def _score_clarity(self, text: str) -> float:
        """Is the text clear and well-structured?"""
        score = 8.0
        
        # Sentence length variation
        sentences = re.split(r'[.!?]+', text)
        if sentences:
            lengths = [len(s.split()) for s in sentences if s.strip()]
            if lengths:
                avg_length = statistics.mean(lengths)
                if avg_length > 30:  # Very long sentences
                    score -= 2.0
                elif avg_length < 5:  # Too choppy
                    score -= 1.0
        
        # Clear structure indicators
        if re.search(r'(first|second|finally|step \d+|^\d+\.)', text, re.I | re.M):
            score += 1.0
            
        # Jargon without explanation
        jargon_pattern = r'\b[A-Z]{3,}\b'
        jargon_count = len(re.findall(jargon_pattern, text))
        unexplained = jargon_count - text.count('(')  # Rough heuristic
        if unexplained > 2:
            score -= 1.0
            
        return max(0, min(10, score))
    
    def _score_relevance_potential(self, text: str) -> float:
        """How likely is this chunk to answer user questions?"""
        score = 5.0
        
        # Question-answering patterns
        qa_patterns = [
            'how to', 'what is', 'why', 'when to use',
            'the purpose', 'this means', 'in order to',
            'the benefit', 'the advantage', 'solves'
        ]
        
        qa_matches = sum(1 for pattern in qa_patterns if pattern in text.lower())
        score += min(3.0, qa_matches * 0.5)
        
        # Actionable content
        if re.search(r'\b(use|create|implement|apply|configure|set)\b', text, re.I):
            score += 1.0
            
        # Explanatory content
        if re.search(r'\b(because|therefore|thus|means that|results in)\b', text, re.I):
            score += 1.0
            
        return min(10, score)
    
    def _is_artifact_chunk(self, text: str) -> bool:
        """
        Detect if chunk is just PDF artifacts with no real content.
        Based on blind test findings.
        """
        stripped = text.strip()
        
        # Check for chunks that are just numbers/bullets
        if re.match(r'^[\d\s•·∙◦▪▫◘○●□■☐☑✓✗×,;:.]+$', stripped):
            return True
            
        # Check for very short chunks with no words
        if len(stripped) < 20 and not re.search(r'[a-zA-Z]{3,}', stripped):
            return True
            
        # Check for chunks that are mostly numbers with minimal text
        text_chars = len(re.findall(r'[a-zA-Z]', stripped))
        total_chars = len(stripped)
        if total_chars > 0 and text_chars / total_chars < 0.3:
            return True
            
        return False
    
    def _has_continuity_markers(self, text: str) -> bool:
        """Check if chunk has overlap/continuity markers"""
        continuity_patterns = [
            r'\[\.\.\..*?\]',  # Overlap before marker
            r'\[.*?\.\.\.\]',  # Overlap after marker  
            r'\[continued',  # Continuation marker
            r'\[Section:',  # Section marker
            r'\[Context:',  # Context header
        ]
        
        return any(re.search(pattern, text, re.I) for pattern in continuity_patterns)
    
    def _get_recommendations(self, scores: Dict[str, float]) -> List[str]:
        """Provide specific improvement recommendations"""
        recommendations = []
        
        if scores['context_independence'] < 7:
            recommendations.append("Add explicit topic/context at chunk beginning")
            
        if scores['information_density'] < 7:
            recommendations.append("Include more specific examples, measurements, or technical details")
            
        if scores['semantic_coherence'] < 7:
            recommendations.append("Focus on a single topic per chunk")
            
        if scores['factual_grounding'] < 7:
            recommendations.append("Add sources, references, or specific evidence")
            
        if scores['clarity'] < 7:
            recommendations.append("Simplify sentence structure and define technical terms")
            
        return recommendations


# Compatibility wrapper for existing code
class KnowledgeQualityAnalyzer(ImprovedQualityAnalyzer):
    """Backward compatibility wrapper"""
    pass


if __name__ == "__main__":
    # Test with examples
    analyzer = ImprovedQualityAnalyzer()
    
    # Test the Architect's Guide chunk
    test_chunk = """
    Instead, top designers use off-black and tinted dark palettes. For example, 
    Google's Material Design dark theme recommends a very dark gray (#121212) as 
    the base surface color. This softer black reduces eye strain in low-light 
    conditions and prevents the high contrast issues of pure black.
    """
    
    result = analyzer.analyze(test_chunk)
    print(f"Score: {result['overall_score']:.1f}/10")
    print(f"Strengths: {', '.join(result['strengths'])}")
    print(f"Recommendations: {result['recommendations']}")