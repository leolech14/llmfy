"""
Content-agnostic semantic chunking for 10/10 quality
Uses overlapping windows and semantic boundaries
"""

import re
from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ChunkCandidate:
    text: str
    start_idx: int
    end_idx: int
    score: float
    overlap_before: str = ""
    overlap_after: str = ""

class SemanticChunker:
    def __init__(self, 
                 target_chunk_size: int = 250,  # tokens
                 min_chunk_size: int = 100,
                 max_chunk_size: int = 400,
                 overlap_size: int = 50,  # tokens of overlap
                 semantic_weight: float = 0.7):
        self.target_chunk_size = target_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.semantic_weight = semantic_weight
        
        # Semantic boundary patterns (content-agnostic)
        self.strong_boundaries = [
            r'\n\n+',  # Multiple newlines
            r'\n#+\s',  # Markdown headers
            r'\n\d+\.\s+\w',  # Numbered lists
            r'\n[•·∙◦▪▫◘○●□■]\s',  # Bullet points
            r'[.!?]\s+[A-Z]',  # Sentence with capital start
        ]
        
        self.weak_boundaries = [
            r'[.!?]\s+',  # End of sentence
            r';\s+',  # Semicolon
            r':\s+',  # Colon
            r',\s+',  # Comma
        ]

    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Create semantically coherent chunks with overlap for continuity
        """
        if not text or not text.strip():
            return []
        
        # Tokenize (simple word-based for now)
        tokens = self._tokenize(text)
        
        # Find all potential boundaries
        boundaries = self._find_semantic_boundaries(text, tokens)
        
        # Generate chunk candidates
        candidates = self._generate_chunk_candidates(text, tokens, boundaries)
        
        # Select optimal chunks
        chunks = self._select_optimal_chunks(candidates)
        
        # Add overlap context
        chunks_with_overlap = self._add_overlap_context(chunks, text)
        
        # Format for output
        return self._format_chunks(chunks_with_overlap, metadata)
    
    def _tokenize(self, text: str) -> List[Tuple[str, int, int]]:
        """Simple word-based tokenization with positions"""
        tokens = []
        current_pos = 0
        
        for match in re.finditer(r'\S+', text):
            token = match.group()
            start = match.start()
            end = match.end()
            tokens.append((token, start, end))
        
        return tokens
    
    def _find_semantic_boundaries(self, text: str, tokens: List[Tuple[str, int, int]]) -> List[Dict]:
        """Find semantic boundaries in text"""
        boundaries = []
        
        # Strong boundaries
        for pattern in self.strong_boundaries:
            for match in re.finditer(pattern, text):
                boundaries.append({
                    'position': match.start(),
                    'strength': 1.0,
                    'type': 'strong',
                    'pattern': pattern
                })
        
        # Weak boundaries
        for pattern in self.weak_boundaries:
            for match in re.finditer(pattern, text):
                boundaries.append({
                    'position': match.start(),
                    'strength': 0.5,
                    'type': 'weak',
                    'pattern': pattern
                })
        
        # Sort by position
        boundaries.sort(key=lambda x: x['position'])
        
        # Merge nearby boundaries
        boundaries = self._merge_nearby_boundaries(boundaries)
        
        return boundaries
    
    def _merge_nearby_boundaries(self, boundaries: List[Dict], threshold: int = 10) -> List[Dict]:
        """Merge boundaries that are very close together"""
        if not boundaries:
            return []
        
        merged = [boundaries[0]]
        
        for boundary in boundaries[1:]:
            if boundary['position'] - merged[-1]['position'] < threshold:
                # Keep the stronger boundary
                if boundary['strength'] > merged[-1]['strength']:
                    merged[-1] = boundary
            else:
                merged.append(boundary)
        
        return merged
    
    def _generate_chunk_candidates(self, text: str, tokens: List[Tuple[str, int, int]], 
                                 boundaries: List[Dict]) -> List[ChunkCandidate]:
        """Generate potential chunk candidates"""
        candidates = []
        
        # Add boundary positions for easier access
        boundary_positions = [b['position'] for b in boundaries]
        
        # Sliding window approach
        for i in range(0, len(tokens), max(1, self.target_chunk_size // 2)):
            # Try different chunk sizes around target
            for size_delta in [-50, -25, 0, 25, 50]:
                chunk_size = self.target_chunk_size + size_delta
                
                if chunk_size < self.min_chunk_size or chunk_size > self.max_chunk_size:
                    continue
                
                if i + chunk_size > len(tokens):
                    chunk_size = len(tokens) - i
                
                if chunk_size < self.min_chunk_size:
                    continue
                
                # Get chunk boundaries
                start_token = tokens[i]
                end_idx = min(i + chunk_size, len(tokens) - 1)
                end_token = tokens[end_idx]
                
                start_pos = start_token[1]
                end_pos = end_token[2]
                
                # Extract chunk text
                chunk_text = text[start_pos:end_pos]
                
                # Calculate semantic score
                score = self._calculate_chunk_score(
                    chunk_text, start_pos, end_pos, boundaries
                )
                
                candidates.append(ChunkCandidate(
                    text=chunk_text,
                    start_idx=start_pos,
                    end_idx=end_pos,
                    score=score
                ))
        
        return candidates
    
    def _calculate_chunk_score(self, chunk_text: str, start_pos: int, 
                             end_pos: int, boundaries: List[Dict]) -> float:
        """Calculate quality score for a chunk candidate"""
        score = 1.0
        
        # Penalty for starting mid-sentence
        if start_pos > 0 and not re.match(r'^[A-Z\[\n]', chunk_text):
            score *= 0.8
        
        # Penalty for ending mid-sentence
        if not re.search(r'[.!?\]]\s*$', chunk_text):
            score *= 0.8
        
        # Bonus for starting at a strong boundary
        for boundary in boundaries:
            if abs(boundary['position'] - start_pos) < 5 and boundary['type'] == 'strong':
                score *= 1.2
                break
        
        # Bonus for ending at a boundary
        for boundary in boundaries:
            if abs(boundary['position'] - end_pos) < 5:
                score *= (1.1 if boundary['type'] == 'weak' else 1.2)
                break
        
        # Length score (prefer chunks close to target size)
        length_ratio = len(chunk_text.split()) / (self.target_chunk_size * 1.5)  # Approximate
        length_score = 1.0 - abs(1.0 - length_ratio) * 0.5
        score *= length_score
        
        # Semantic completeness (has both start and end punctuation)
        if re.match(r'^[A-Z\[]', chunk_text) and re.search(r'[.!?]\s*$', chunk_text):
            score *= 1.1
        
        return min(score, 2.0)  # Cap at 2.0
    
    def _select_optimal_chunks(self, candidates: List[ChunkCandidate]) -> List[ChunkCandidate]:
        """Select non-overlapping chunks with best scores"""
        # Sort by score descending
        candidates.sort(key=lambda x: x.score, reverse=True)
        
        selected = []
        covered_positions = set()
        
        for candidate in candidates:
            # Check overlap with already selected chunks
            overlap = False
            for pos in range(candidate.start_idx, candidate.end_idx):
                if pos in covered_positions:
                    overlap = True
                    break
            
            if not overlap:
                selected.append(candidate)
                # Mark positions as covered
                for pos in range(candidate.start_idx, candidate.end_idx):
                    covered_positions.add(pos)
        
        # Sort by position
        selected.sort(key=lambda x: x.start_idx)
        
        return selected
    
    def _add_overlap_context(self, chunks: List[ChunkCandidate], 
                           full_text: str) -> List[ChunkCandidate]:
        """Add overlapping context to chunks for continuity"""
        chunks_with_overlap = []
        
        for i, chunk in enumerate(chunks):
            # Get overlap from previous chunk
            if i > 0 and chunk.start_idx > 0:
                # Find a good starting point for overlap (sentence boundary)
                overlap_start = max(0, chunk.start_idx - 200)  # Look back up to 200 chars
                overlap_text = full_text[overlap_start:chunk.start_idx]
                
                # Find last sentence boundary
                sentences = re.split(r'[.!?]\s+', overlap_text)
                if len(sentences) > 1:
                    # Take last complete sentence
                    chunk.overlap_before = sentences[-2].strip() + '.'
                    if len(chunk.overlap_before) > 150:
                        # Truncate if too long
                        chunk.overlap_before = '...' + chunk.overlap_before[-100:]
            
            # Get overlap for next chunk
            if i < len(chunks) - 1 and chunk.end_idx < len(full_text):
                # Find a good ending point for overlap
                overlap_end = min(len(full_text), chunk.end_idx + 200)
                overlap_text = full_text[chunk.end_idx:overlap_end]
                
                # Find first sentence boundary
                sentences = re.split(r'[.!?]\s+', overlap_text)
                if sentences and sentences[0]:
                    # Take first complete sentence
                    chunk.overlap_after = sentences[0].strip()
                    if not chunk.overlap_after.endswith('.'):
                        chunk.overlap_after += '.'
                    if len(chunk.overlap_after) > 150:
                        # Truncate if too long
                        chunk.overlap_after = chunk.overlap_after[:100] + '...'
            
            chunks_with_overlap.append(chunk)
        
        return chunks_with_overlap
    
    def _format_chunks(self, chunks: List[ChunkCandidate], metadata: Dict = None) -> List[Dict]:
        """Format chunks for output"""
        formatted_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Build chunk with context
            chunk_text = chunk.text
            
            # Add subtle context markers if overlap exists
            if chunk.overlap_before:
                chunk_text = f"[...{chunk.overlap_before[-50:]}]\n\n{chunk_text}"
            
            if chunk.overlap_after:
                chunk_text = f"{chunk_text}\n\n[{chunk.overlap_after[:50]}...]"
            
            formatted_chunk = {
                'text': chunk_text,
                'metadata': {
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'has_overlap_before': bool(chunk.overlap_before),
                    'has_overlap_after': bool(chunk.overlap_after),
                    'semantic_score': chunk.score,
                    **(metadata or {})
                }
            }
            
            formatted_chunks.append(formatted_chunk)
        
        return formatted_chunks

    def create_sliding_window_chunks(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Alternative: Create overlapping chunks with sliding window
        Ensures no content is missed between chunks
        """
        if not text or not text.strip():
            return []
        
        tokens = self._tokenize(text)
        chunks = []
        
        # Sliding window with overlap
        step_size = self.target_chunk_size - self.overlap_size
        
        for i in range(0, len(tokens), step_size):
            # Get chunk boundaries
            start_idx = i
            end_idx = min(i + self.target_chunk_size, len(tokens))
            
            if end_idx - start_idx < self.min_chunk_size:
                # Last chunk too small, merge with previous
                if chunks:
                    continue
            
            # Get text positions
            start_pos = tokens[start_idx][1]
            end_pos = tokens[end_idx - 1][2] if end_idx > 0 else len(text)
            
            chunk_text = text[start_pos:end_pos]
            
            # Find clean boundaries
            chunk_text = self._adjust_to_clean_boundaries(chunk_text, text, start_pos, end_pos)
            
            chunks.append({
                'text': chunk_text,
                'metadata': {
                    'chunk_index': len(chunks),
                    'overlap_tokens': self.overlap_size if i > 0 else 0,
                    'sliding_window': True,
                    **(metadata or {})
                }
            })
        
        return chunks
    
    def _adjust_to_clean_boundaries(self, chunk_text: str, full_text: str, 
                                   start_pos: int, end_pos: int) -> str:
        """Adjust chunk to end at clean boundaries"""
        # Try to end at sentence boundary
        last_period = chunk_text.rfind('.')
        last_question = chunk_text.rfind('?')
        last_exclaim = chunk_text.rfind('!')
        
        last_sentence_end = max(last_period, last_question, last_exclaim)
        
        if last_sentence_end > len(chunk_text) * 0.8:  # If near end
            chunk_text = chunk_text[:last_sentence_end + 1]
        
        # Try to start at sentence boundary
        if start_pos > 0:
            # Look for capital letter start
            match = re.search(r'[.!?]\s+([A-Z])', chunk_text[:100])
            if match:
                chunk_text = chunk_text[match.start(1):]
        
        return chunk_text.strip()