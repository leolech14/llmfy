"""
Advanced Quality Enhancer for achieving 10/10 scores
Implements:
1. Contextual enhancement for references
2. Framework continuity detection
3. Smart chunk merging for related content
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class EnhancementResult:
    enhanced_text: str
    enhancement_type: str
    confidence: float
    merged_chunks: List[int] = None

class AdvancedQualityEnhancer:
    def __init__(self):
        self.reference_patterns = {
            'values': r'^(values|these|those|such|the\s+\w+)\s',
            'numbered': r'^(\d+\.|\(\d+\)|•\s*\d+)',
            'continuation': r'^(however|therefore|thus|moreover|furthermore|additionally)',
            'pronoun': r'^(it|they|this|that|these|those)\s',
        }
        
        self.framework_indicators = [
            'framework', 'pattern', 'principle', 'approach', 'methodology',
            'technique', 'strategy', 'system', 'architecture', 'model'
        ]
        
        self.incomplete_endings = [
            r',\s*$', r':\s*$', r'–\s*$', r'such as\s*$', r'including\s*$',
            r'for example\s*$', r'e\.g\.\s*$', r'i\.e\.\s*$'
        ]

    def enhance_chunk(self, chunk: str, context: Dict = None) -> EnhancementResult:
        """
        Enhance a single chunk to improve its context independence
        """
        # Check if chunk needs reference resolution
        if self._needs_reference_resolution(chunk):
            enhanced = self._resolve_references(chunk, context)
            if enhanced != chunk:
                return EnhancementResult(
                    enhanced_text=enhanced,
                    enhancement_type="reference_resolution",
                    confidence=0.85
                )
        
        # Check if chunk is incomplete framework description
        if self._is_incomplete_framework(chunk):
            enhanced = self._enhance_framework_context(chunk, context)
            if enhanced != chunk:
                return EnhancementResult(
                    enhanced_text=enhanced,
                    enhancement_type="framework_completion",
                    confidence=0.80
                )
        
        # Check if chunk has abrupt ending
        if self._has_abrupt_ending(chunk):
            enhanced = self._smooth_ending(chunk)
            if enhanced != chunk:
                return EnhancementResult(
                    enhanced_text=enhanced,
                    enhancement_type="ending_smoothing",
                    confidence=0.75
                )
        
        return EnhancementResult(
            enhanced_text=chunk,
            enhancement_type="none",
            confidence=1.0
        )

    def _needs_reference_resolution(self, chunk: str) -> bool:
        """Check if chunk starts with unclear reference"""
        first_line = chunk.split('\n')[0].lower()
        for pattern in self.reference_patterns.values():
            if re.match(pattern, first_line, re.IGNORECASE):
                return True
        return False

    def _resolve_references(self, chunk: str, context: Dict) -> str:
        """Add context to resolve unclear references"""
        if not context or 'previous_chunk' not in context:
            return chunk
        
        # Extract the main topic from previous chunk
        prev_chunk = context.get('previous_chunk', '')
        main_topic = self._extract_main_topic(prev_chunk)
        
        if not main_topic:
            return chunk
        
        # Replace unclear references
        lines = chunk.split('\n')
        first_line = lines[0]
        
        # Handle "values" type references
        if re.match(r'^values\s', first_line, re.IGNORECASE):
            first_line = f"The {main_topic} values" + first_line[6:]
        elif re.match(r'^these\s', first_line, re.IGNORECASE):
            first_line = f"These {main_topic} elements" + first_line[5:]
        elif re.match(r'^it\s', first_line, re.IGNORECASE):
            first_line = f"The {main_topic}" + first_line[2:]
        
        lines[0] = first_line
        return '\n'.join(lines)

    def _is_incomplete_framework(self, chunk: str) -> bool:
        """Check if chunk contains incomplete framework description"""
        chunk_lower = chunk.lower()
        
        # Check for framework indicators
        has_framework = any(indicator in chunk_lower for indicator in self.framework_indicators)
        
        # Check for incomplete patterns
        has_incomplete = (
            'such as' in chunk_lower and chunk_lower.count('such as') > chunk_lower.count('.')
            or chunk.count(':') > chunk.count('\n') / 3  # Many colons suggest lists
            or bool(re.search(r'\d+\.\s+[A-Z]', chunk))  # Numbered items
        )
        
        return has_framework and has_incomplete

    def _enhance_framework_context(self, chunk: str, context: Dict) -> str:
        """Add contextual summary for framework descriptions"""
        if not self._has_complete_thought(chunk):
            # Add a contextual bridge
            framework_name = self._extract_framework_name(chunk)
            if framework_name:
                bridge = f"\n\n[This section describes {framework_name} - partial view]"
                return chunk + bridge
        return chunk

    def _has_abrupt_ending(self, chunk: str) -> bool:
        """Check if chunk ends abruptly"""
        last_line = chunk.strip().split('\n')[-1]
        for pattern in self.incomplete_endings:
            if re.search(pattern, last_line):
                return True
        return len(last_line.split()) < 5 and not last_line.endswith('.')

    def _smooth_ending(self, chunk: str) -> str:
        """Add transitional ending to abrupt chunks"""
        last_line = chunk.strip().split('\n')[-1]
        
        if re.search(r',\s*$', last_line):
            return chunk.rstrip() + " [continued in next section]"
        elif re.search(r':\s*$', last_line):
            return chunk.rstrip() + "\n[Details follow in subsequent sections]"
        elif not last_line.endswith('.'):
            return chunk.rstrip() + "..."
        
        return chunk

    def _extract_main_topic(self, text: str) -> Optional[str]:
        """Extract the main topic/subject from text"""
        # Look for capitalized noun phrases
        matches = re.findall(r'(?:^|\s)([A-Z]\w+(?:\s+[A-Z]\w+)*)', text)
        if matches:
            # Return the most frequent or longest match
            return max(matches, key=len)
        
        # Look for framework indicators
        for indicator in self.framework_indicators:
            pattern = rf'{indicator}\s+(\w+(?:\s+\w+)*)'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None

    def _extract_framework_name(self, text: str) -> Optional[str]:
        """Extract framework or pattern name from text"""
        for indicator in self.framework_indicators:
            pattern = rf'(\w+(?:\s+\w+)*)\s+{indicator}'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def _has_complete_thought(self, chunk: str) -> bool:
        """Check if chunk contains at least one complete thought"""
        sentences = re.split(r'[.!?]\s+', chunk)
        complete_sentences = [s for s in sentences if len(s.split()) > 7]
        return len(complete_sentences) >= 2

    def suggest_chunk_merges(self, chunks: List[Tuple[str, float]], threshold: float = 0.85) -> List[List[int]]:
        """
        Suggest which chunks should be merged for better context
        Returns list of chunk indices to merge
        """
        merge_groups = []
        
        for i in range(len(chunks) - 1):
            current_chunk, current_score = chunks[i]
            next_chunk, next_score = chunks[i + 1]
            
            # Check if chunks should be merged
            if self._should_merge(current_chunk, next_chunk, current_score, next_score):
                # Find or create merge group
                added = False
                for group in merge_groups:
                    if group[-1] == i:
                        group.append(i + 1)
                        added = True
                        break
                
                if not added:
                    merge_groups.append([i, i + 1])
        
        return merge_groups

    def _should_merge(self, chunk1: str, chunk2: str, score1: float, score2: float) -> bool:
        """Determine if two chunks should be merged"""
        # Don't merge if both are already high quality
        if score1 >= 8.5 and score2 >= 8.5:
            return False
        
        # Check for continuation patterns
        if self._needs_reference_resolution(chunk2):
            return True
        
        # Check for incomplete ending in first chunk
        if self._has_abrupt_ending(chunk1):
            return True
        
        # Check for framework split
        if self._is_incomplete_framework(chunk1) and any(
            indicator in chunk2.lower() for indicator in self.framework_indicators
        ):
            return True
        
        # Check for numbered list continuation
        if re.search(r'\d+\.\s*$', chunk1.strip()) and re.match(r'^\d+\.', chunk2.strip()):
            return True
        
        return False

    def merge_chunks(self, chunks: List[str], indices: List[int]) -> str:
        """Merge multiple chunks with proper transitions"""
        if len(indices) < 2:
            return chunks[indices[0]]
        
        merged = chunks[indices[0]]
        
        for i in range(1, len(indices)):
            current_chunk = chunks[indices[i]]
            
            # Add appropriate transition
            if self._needs_reference_resolution(current_chunk):
                # Add minimal transition
                merged += f"\n\n{current_chunk}"
            elif re.match(r'^\d+\.', current_chunk.strip()):
                # Continue numbered list
                merged += f"\n{current_chunk}"
            else:
                # Standard paragraph break
                merged += f"\n\n{current_chunk}"
        
        return merged

    def enhance_for_perfection(self, chunk: str, metadata: Dict = None) -> EnhancementResult:
        """
        Ultimate enhancement for 10/10 quality
        Combines all enhancement techniques
        """
        context = metadata or {}
        
        # Step 1: Resolve references
        enhanced = chunk
        if self._needs_reference_resolution(enhanced):
            result = self._resolve_references(enhanced, context)
            if result != enhanced:
                enhanced = result
        
        # Step 2: Enhance framework context
        if self._is_incomplete_framework(enhanced):
            result = self._enhance_framework_context(enhanced, context)
            if result != enhanced:
                enhanced = result
        
        # Step 3: Smooth endings
        if self._has_abrupt_ending(enhanced):
            result = self._smooth_ending(enhanced)
            if result != enhanced:
                enhanced = result
        
        # Step 4: Add section markers for clarity
        if metadata and 'section' in metadata:
            section_header = f"[Section: {metadata['section']}]\n"
            if not enhanced.startswith('['):
                enhanced = section_header + enhanced
        
        # Calculate enhancement confidence
        changes = self._calculate_changes(chunk, enhanced)
        confidence = 1.0 - (changes * 0.1)  # Reduce confidence with more changes
        
        return EnhancementResult(
            enhanced_text=enhanced,
            enhancement_type="comprehensive",
            confidence=max(0.7, confidence)
        )
    
    def _calculate_changes(self, original: str, enhanced: str) -> int:
        """Calculate number of significant changes"""
        changes = 0
        if len(enhanced) > len(original) * 1.2:
            changes += 1
        if enhanced.count('\n') > original.count('\n'):
            changes += 1
        if enhanced.count('[') > original.count('['):
            changes += 1
        return changes
