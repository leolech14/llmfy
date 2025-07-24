"""
Chunk Optimizer - Post-processing for 10/10 quality
Analyzes chunk relationships and optimizes for perfect context
"""

import logging
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ..quality.quality_scorer_v2 import ImprovedQualityAnalyzer
from ..quality.quality_enhancer_v2 import AdvancedQualityEnhancer
from ..embeddings.hybrid_embedder import HybridEmbedder

logger = logging.getLogger(__name__)

class ChunkOptimizer:
    def __init__(self, quality_threshold: float = 9.0):
        self.quality_analyzer = ImprovedQualityAnalyzer()
        self.quality_enhancer = AdvancedQualityEnhancer()
        self.embedder = HybridEmbedder()
        self.quality_threshold = quality_threshold
        
    def optimize_chunks(self, chunks: List[Dict]) -> Tuple[List[Dict], Dict]:
        """
        Optimize chunks for 10/10 quality
        Returns optimized chunks and optimization report
        """
        logger.info(f"Optimizing {len(chunks)} chunks for perfection...")
        
        # Step 1: Analyze current quality
        chunk_scores = []
        for chunk in chunks:
            result = self.quality_analyzer.analyze(chunk['text'])
            score = result['overall_score']
            chunk_scores.append((chunk, score))
        
        # Step 2: Identify problematic chunks
        problematic = [(i, chunk, score) for i, (chunk, score) in enumerate(chunk_scores) 
                      if score < self.quality_threshold]
        
        logger.info(f"Found {len(problematic)} chunks below {self.quality_threshold} threshold")
        
        # Step 3: Analyze relationships and suggest merges
        merge_suggestions = self._analyze_chunk_relationships(chunk_scores)
        
        # Step 4: Apply optimizations
        optimized_chunks = self._apply_optimizations(
            chunks, chunk_scores, merge_suggestions
        )
        
        # Step 5: Final enhancement pass
        final_chunks = self._final_enhancement_pass(optimized_chunks)
        
        # Generate report
        report = self._generate_optimization_report(
            chunks, final_chunks, merge_suggestions
        )
        
        return final_chunks, report
    
    def _analyze_chunk_relationships(self, chunk_scores: List[Tuple[Dict, float]]) -> Dict:
        """Analyze semantic relationships between chunks"""
        relationships = {
            'continuations': [],
            'references': [],
            'split_concepts': [],
            'merge_candidates': []
        }
        
        for i in range(len(chunk_scores) - 1):
            current_chunk, current_score = chunk_scores[i]
            next_chunk, next_score = chunk_scores[i + 1]
            
            # Check for direct continuations
            if self._is_continuation(current_chunk['text'], next_chunk['text']):
                relationships['continuations'].append((i, i + 1))
            
            # Check for reference relationships
            if self._has_reference_relationship(current_chunk['text'], next_chunk['text']):
                relationships['references'].append((i, i + 1))
            
            # Check for split concepts
            if self._is_split_concept(current_chunk['text'], next_chunk['text']):
                relationships['split_concepts'].append((i, i + 1))
            
            # Determine merge candidates
            if self._should_merge_chunks(
                current_chunk, next_chunk, current_score, next_score
            ):
                relationships['merge_candidates'].append((i, i + 1))
        
        # Extend merge candidates to groups
        relationships['merge_groups'] = self._extend_to_merge_groups(
            relationships['merge_candidates']
        )
        
        return relationships
    
    def _is_continuation(self, text1: str, text2: str) -> bool:
        """Check if text2 is a direct continuation of text1"""
        # Check for incomplete sentence
        if not text1.rstrip().endswith('.') and text2[0].islower():
            return True
        
        # Check for continuation words
        continuation_starters = [
            'however', 'therefore', 'moreover', 'furthermore',
            'additionally', 'consequently', 'thus', 'hence'
        ]
        first_word = text2.split()[0].lower() if text2.split() else ''
        return first_word in continuation_starters
    
    def _has_reference_relationship(self, text1: str, text2: str) -> bool:
        """Check if text2 references content from text1"""
        # Extract key entities from text1
        entities = self._extract_key_entities(text1)
        
        # Check for pronouns or references in text2
        reference_patterns = ['it', 'they', 'this', 'that', 'these', 'those', 'such']
        text2_start = ' '.join(text2.split()[:10]).lower()
        
        return any(ref in text2_start for ref in reference_patterns) and bool(entities)
    
    def _is_split_concept(self, text1: str, text2: str) -> bool:
        """Check if a concept is split across chunks"""
        # Look for framework/concept indicators
        concept_words = [
            'framework', 'principle', 'pattern', 'approach', 'method',
            'technique', 'strategy', 'system', 'model', 'architecture'
        ]
        
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        # Check if concept is introduced in text1 but explained in text2
        for concept in concept_words:
            if concept in text1_lower and (
                'include' in text1_lower or 
                'consist' in text1_lower or
                ':' in text1[-50:]
            ):
                return True
        
        return False
    
    def _should_merge_chunks(self, chunk1: Dict, chunk2: Dict, 
                           score1: float, score2: float) -> bool:
        """Determine if chunks should be merged"""
        # Don't merge if both are already excellent
        if score1 >= 8.5 and score2 >= 8.5:
            return False
        
        # Don't merge if combined would be too large
        combined_length = len(chunk1['text']) + len(chunk2['text'])
        if combined_length > 2000:  # Character limit
            return False
        
        # Check various merge criteria
        if self._is_continuation(chunk1['text'], chunk2['text']):
            return True
        
        if self._has_reference_relationship(chunk1['text'], chunk2['text']):
            return True
        
        if self._is_split_concept(chunk1['text'], chunk2['text']):
            return True
        
        # Check embedding similarity
        if self._are_semantically_close(chunk1, chunk2):
            return True
        
        return False
    
    def _are_semantically_close(self, chunk1: Dict, chunk2: Dict, 
                               threshold: float = 0.92) -> bool:
        """Check if chunks are semantically similar"""
        # Generate embeddings if not present
        if 'embedding' not in chunk1:
            _, emb1 = self.embedder.embed_documents([chunk1['text']])[0]
            chunk1['embedding'] = emb1
        
        if 'embedding' not in chunk2:
            _, emb2 = self.embedder.embed_documents([chunk2['text']])[0]
            chunk2['embedding'] = emb2
        
        # Calculate cosine similarity
        similarity = cosine_similarity(
            [chunk1['embedding']], 
            [chunk2['embedding']]
        )[0][0]
        
        return similarity > threshold
    
    def _extend_to_merge_groups(self, merge_pairs: List[Tuple[int, int]]) -> List[List[int]]:
        """Extend merge pairs to groups"""
        if not merge_pairs:
            return []
        
        # Build adjacency graph
        graph = defaultdict(set)
        for i, j in merge_pairs:
            graph[i].add(j)
            graph[j].add(i)
        
        # Find connected components
        visited = set()
        groups = []
        
        for node in graph:
            if node not in visited:
                group = []
                stack = [node]
                
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        group.append(current)
                        stack.extend(graph[current] - visited)
                
                groups.append(sorted(group))
        
        return groups
    
    def _apply_optimizations(self, chunks: List[Dict], 
                           chunk_scores: List[Tuple[Dict, float]], 
                           relationships: Dict) -> List[Dict]:
        """Apply optimization strategies"""
        optimized = chunks.copy()
        
        # Apply merges (in reverse order to maintain indices)
        for group in sorted(relationships['merge_groups'], reverse=True):
            if len(group) > 1:
                merged_text = self._merge_chunk_group(
                    [chunks[i] for i in group]
                )
                
                # Replace first chunk with merged, remove others
                optimized[group[0]] = {
                    **chunks[group[0]],
                    'text': merged_text,
                    'merged_from': group
                }
                
                # Mark others for removal
                for i in group[1:]:
                    optimized[i] = None
        
        # Remove None entries
        optimized = [chunk for chunk in optimized if chunk is not None]
        
        # Apply enhancements to remaining chunks
        for i, chunk in enumerate(optimized):
            if 'merged_from' not in chunk:  # Don't re-enhance merged chunks yet
                context = {
                    'previous_chunk': optimized[i-1]['text'] if i > 0 else None,
                    'next_chunk': optimized[i+1]['text'] if i < len(optimized)-1 else None,
                    'position': i,
                    'total': len(optimized)
                }
                
                result = self.quality_enhancer.enhance_for_perfection(
                    chunk['text'], context
                )
                
                if result.confidence > 0.75:
                    optimized[i] = {
                        **chunk,
                        'text': result.enhanced_text,
                        'enhancement': result.enhancement_type
                    }
        
        return optimized
    
    def _merge_chunk_group(self, chunks: List[Dict]) -> str:
        """Merge a group of chunks intelligently"""
        texts = [chunk['text'] for chunk in chunks]
        
        # Use quality enhancer's merge function
        return self.quality_enhancer.merge_chunks(texts, list(range(len(texts))))
    
    def _final_enhancement_pass(self, chunks: List[Dict]) -> List[Dict]:
        """Final pass to ensure all chunks meet quality standards"""
        final_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Re-score the chunk
            result = self.quality_analyzer.analyze(chunk['text'])
            score = result['overall_score']
            
            if score < self.quality_threshold:
                # One more enhancement attempt
                context = {
                    'previous_chunk': chunks[i-1]['text'] if i > 0 else None,
                    'next_chunk': chunks[i+1]['text'] if i < len(chunks)-1 else None,
                    'section': chunk.get('metadata', {}).get('section', 'Unknown'),
                    'position': i,
                    'total': len(chunks),
                    'current_score': score
                }
                
                result = self.quality_enhancer.enhance_for_perfection(
                    chunk['text'], context
                )
                
                chunk['text'] = result.enhanced_text
                chunk['final_enhancement'] = True
                result_analysis = self.quality_analyzer.analyze(result.enhanced_text)
                chunk['final_score'] = result_analysis['overall_score']
            else:
                chunk['final_score'] = score
            
            final_chunks.append(chunk)
        
        return final_chunks
    
    def _extract_key_entities(self, text: str) -> List[str]:
        """Extract key entities/concepts from text"""
        import re
        
        # Simple entity extraction - could be enhanced with NER
        entities = []
        
        # Find capitalized phrases
        cap_phrases = re.findall(r'(?:^|\s)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', text)
        entities.extend(cap_phrases)
        
        # Find quoted terms
        quoted = re.findall(r'"([^"]+)"', text)
        entities.extend(quoted)
        
        # Find technical terms (camelCase, snake_case)
        technical = re.findall(r'\b[a-z]+_[a-z]+\b|\b[a-z]+[A-Z][a-zA-Z]+\b', text)
        entities.extend(technical)
        
        return list(set(entities))
    
    def _generate_optimization_report(self, original_chunks: List[Dict], 
                                    optimized_chunks: List[Dict], 
                                    relationships: Dict) -> Dict:
        """Generate detailed optimization report"""
        # Calculate quality improvements
        original_scores = [
            self.quality_analyzer.analyze(chunk['text'])['overall_score'] 
            for chunk in original_chunks
        ]
        
        optimized_scores = [
            chunk.get('final_score', self.quality_analyzer.analyze(chunk['text'])['overall_score'])
            for chunk in optimized_chunks
        ]
        
        report = {
            'summary': {
                'original_chunks': len(original_chunks),
                'optimized_chunks': len(optimized_chunks),
                'chunks_merged': len(original_chunks) - len(optimized_chunks),
                'average_score_before': np.mean(original_scores),
                'average_score_after': np.mean(optimized_scores),
                'perfect_chunks': sum(1 for s in optimized_scores if s >= 9.5),
                'improvement_percentage': (
                    (np.mean(optimized_scores) - np.mean(original_scores)) / 
                    np.mean(original_scores) * 100
                )
            },
            'relationships': {
                'continuations_found': len(relationships['continuations']),
                'references_resolved': len(relationships['references']),
                'split_concepts_merged': len(relationships['split_concepts']),
                'merge_groups': relationships['merge_groups']
            },
            'quality_distribution': {
                'before': {
                    '9.5-10': sum(1 for s in original_scores if s >= 9.5),
                    '8.5-9.5': sum(1 for s in original_scores if 8.5 <= s < 9.5),
                    '7.5-8.5': sum(1 for s in original_scores if 7.5 <= s < 8.5),
                    'below_7.5': sum(1 for s in original_scores if s < 7.5)
                },
                'after': {
                    '9.5-10': sum(1 for s in optimized_scores if s >= 9.5),
                    '8.5-9.5': sum(1 for s in optimized_scores if 8.5 <= s < 9.5),
                    '7.5-8.5': sum(1 for s in optimized_scores if 7.5 <= s < 8.5),
                    'below_7.5': sum(1 for s in optimized_scores if s < 7.5)
                }
            },
            'optimization_actions': {
                'enhancements_applied': sum(
                    1 for chunk in optimized_chunks 
                    if 'enhancement' in chunk
                ),
                'chunks_merged': sum(
                    1 for chunk in optimized_chunks 
                    if 'merged_from' in chunk
                ),
                'final_enhancements': sum(
                    1 for chunk in optimized_chunks 
                    if chunk.get('final_enhancement', False)
                )
            }
        }
        
        return report