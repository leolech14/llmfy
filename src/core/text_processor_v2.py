#!/usr/bin/env python3
"""
🎯 Text Processor V2 - Retrieval-Optimized Document Processing

Based on industry best practices from OpenAI, Anthropic, and Pinecone.
Key improvements:
- Content-aware splitting at semantic boundaries
- Contextual headers for self-contained chunks
- Optimal chunk sizing (200-300 tokens)
- Smart overlap strategy
- Preprocessing for better retrieval
"""

from typing import List, Dict, Any, Optional, Tuple
import re
from pathlib import Path
from dataclasses import dataclass
import hashlib

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import tiktoken
from rich.console import Console
from rich.progress import track

from .config import Config
from .semantic_chunker import SemanticChunker

console = Console()

@dataclass
class ChunkingConfig:
    """Configuration for smart chunking"""
    chunk_size: int = 250  # tokens (research: 200-300 optimal)
    chunk_overlap: int = 50  # 20% overlap
    min_chunk_size: int = 50
    max_chunk_size: int = 500
    
    # Semantic boundaries to respect
    section_separators: List[str] = None
    
    def __post_init__(self):
        if self.section_separators is None:
            self.section_separators = [
                "\n## ",  # Markdown h2
                "\n### ", # Markdown h3
                "\n\n",   # Paragraphs
                "\n",     # Lines
                ". ",     # Sentences
                ", ",     # Clauses
                " "       # Words
            ]


class TextProcessorV2:
    """
    Improved text processor with retrieval-optimized chunking.
    
    Key features:
    - Content-aware splitting
    - Contextual headers
    - Smart preprocessing
    - Semantic boundary detection
    """
    
    def __init__(self, config: Optional[ChunkingConfig] = None, use_semantic_chunking: bool = True):
        self.config = config or ChunkingConfig()
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.use_semantic_chunking = use_semantic_chunking
        
        # Initialize semantic chunker
        if self.use_semantic_chunking:
            self.semantic_chunker = SemanticChunker(
                target_chunk_size=self.config.chunk_size,
                min_chunk_size=self.config.min_chunk_size,
                max_chunk_size=self.config.max_chunk_size,
                overlap_size=self.config.chunk_overlap
            )
        
        # Initialize splitter with research-based settings (fallback)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=self._token_length,
            separators=self.config.section_separators
        )
        
        # Boilerplate patterns to remove
        self.boilerplate_patterns = [
            r'Page \d+ of \d+',
            r'Copyright.*\d{4}.*?(?:\.|$)',
            r'All rights reserved.*?(?:\.|$)',
            r'^\s*Table of Contents\s*$',
            r'^\s*Index\s*$',
            r'^\s*References\s*$'
        ]
    
    def _token_length(self, text: str) -> int:
        """Calculate token count"""
        return len(self.encoding.encode(text))
    
    def preprocess_for_chunking(self, text: str, doc_type: Optional[str] = None) -> str:
        """
        Preprocess text for optimal chunking based on research insights.
        
        Steps:
        1. Clean PDF artifacts
        2. Normalize whitespace while preserving structure
        3. Remove boilerplate
        4. Mark semantic boundaries
        5. Handle special content
        """
        
        # 1. Clean PDF artifacts first
        text = self._clean_pdf_artifacts(text)
        
        # 2. Normalize whitespace but preserve paragraph structure
        text = re.sub(r'\n{4,}', '\n\n\n', text)  # Max 3 newlines
        text = re.sub(r'[ \t]+', ' ', text)       # Single spaces
        text = re.sub(r'^\s+', '', text, flags=re.MULTILINE)  # Leading spaces
        
        # 2. Remove boilerplate
        for pattern in self.boilerplate_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
        
        # 3. Mark special content boundaries (for splitting logic)
        # Code blocks
        text = re.sub(r'```(\w*)\n', r'[CODE_BLOCK_START:\1]\n', text)
        text = re.sub(r'```', '[CODE_BLOCK_END]', text)
        
        # Tables (simple markdown)
        text = re.sub(r'(\n\|.*\|.*\n)', r'[TABLE_START]\1[TABLE_END]', text)
        
        # Lists (preserve structure)
        text = re.sub(r'^(\s*[-*+]\s+)', r'[LIST_ITEM]\1', text, flags=re.MULTILINE)
        text = re.sub(r'^(\s*\d+\.\s+)', r'[LIST_ITEM]\1', text, flags=re.MULTILINE)
        
        return text
    
    def add_contextual_headers(self, chunk: str, metadata: Dict[str, Any]) -> str:
        """
        Add context to make chunks self-contained.
        Based on Anthropic's Contextual Retrieval approach.
        
        Reduces failed retrievals by up to 67% according to research.
        """
        context_parts = []
        
        # Document-level context
        if metadata.get('document_title'):
            context_parts.append(f"Document: {metadata['document_title']}")
        elif metadata.get('filename'):
            # Fallback to filename without extension
            title = Path(metadata['filename']).stem.replace('_', ' ').title()
            context_parts.append(f"Document: {title}")
        
        # Section context
        if metadata.get('section'):
            context_parts.append(f"Section: {metadata['section']}")
        
        # Category/Type
        if metadata.get('category'):
            context_parts.append(f"Category: {metadata['category']}")
        
        # Date context (if available)
        if metadata.get('date'):
            context_parts.append(f"Date: {metadata['date']}")
        
        # Build context header
        if context_parts:
            context_header = " | ".join(context_parts)
            # Add separator
            return f"[Context: {context_header}]\n\n{chunk}"
        
        return chunk
    
    def smart_chunk_document(self, document: Document) -> List[Document]:
        """
        Smart chunking that respects content structure.
        """
        # Determine document type
        is_code = document.metadata.get('file_type', '').lower() in {
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c',
            '.h', '.hpp', '.cs', '.rb', '.go', '.rs', '.swift'
        }
        
        # Preprocess based on type
        if is_code:
            processed_text = self._process_code_document(document.page_content)
        else:
            processed_text = self.preprocess_for_chunking(
                document.page_content, 
                document.metadata.get('doc_type')
            )
        
        # Skip if too short
        if self._token_length(processed_text) < self.config.min_chunk_size:
            console.print(f"[yellow]Skipping short document: {document.metadata.get('filename', 'Unknown')}[/yellow]")
            return []
        
        # Smart chunking based on content
        if self._has_clear_structure(processed_text):
            chunks = self._structure_aware_chunking(processed_text, document.metadata)
        else:
            chunks = self._semantic_chunking(processed_text, document.metadata)
        
        # Add contextual headers to each chunk
        contextualized_chunks = []
        for i, chunk in enumerate(chunks):
            # Extract section info if available
            section = self._extract_section_from_chunk(chunk.page_content)
            if section:
                chunk.metadata['section'] = section
            
            # Add context
            chunk.page_content = self.add_contextual_headers(
                chunk.page_content, 
                chunk.metadata
            )
            
            # Add chunk-specific metadata
            chunk.metadata.update({
                'chunk_index': i,
                'chunk_total': len(chunks),
                'chunk_tokens': self._token_length(chunk.page_content),
                'chunking_method': 'structure_aware' if self._has_clear_structure(processed_text) else 'semantic'
            })
            
            contextualized_chunks.append(chunk)
        
        return contextualized_chunks
    
    def _process_code_document(self, text: str) -> str:
        """Special handling for code files"""
        # Remove excessive blank lines but preserve structure
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Mark function/class boundaries
        text = re.sub(r'^(class\s+\w+.*?:)', r'[CLASS_START]\n\1', text, flags=re.MULTILINE)
        text = re.sub(r'^(def\s+\w+.*?:)', r'[FUNCTION_START]\n\1', text, flags=re.MULTILINE)
        
        return text
    
    def _has_clear_structure(self, text: str) -> bool:
        """Check if document has clear structural markers"""
        structure_indicators = [
            r'^#{1,6}\s+',  # Markdown headers
            r'^\d+\.\s+',    # Numbered sections
            r'\[.*_START\]', # Our markers
            r'^Chapter\s+\d+', # Book-style
            r'^Section\s+\d+', # Academic-style
        ]
        
        for pattern in structure_indicators:
            if re.search(pattern, text, re.MULTILINE):
                return True
        return False
    
    def _structure_aware_chunking(self, text: str, metadata: Dict) -> List[Document]:
        """Chunk based on document structure"""
        # Split on major boundaries first
        major_sections = re.split(r'\n(?=#{1,3}\s+)', text)
        
        chunks = []
        for section in major_sections:
            if self._token_length(section) <= self.config.max_chunk_size:
                # Section is small enough
                chunks.append(Document(
                    page_content=section,
                    metadata=metadata.copy()
                ))
            else:
                # Need to split further
                sub_chunks = self.text_splitter.create_documents(
                    texts=[section],
                    metadatas=[metadata.copy()]
                )
                chunks.extend(sub_chunks)
        
        return chunks
    
    def _semantic_chunking(self, text: str, metadata: Dict) -> List[Document]:
        """Advanced semantic chunking with overlap for continuity"""
        if self.use_semantic_chunking and self.semantic_chunker:
            # Use advanced semantic chunker
            chunks = self.semantic_chunker.chunk_text(text, metadata)
            
            # Convert to Document objects
            documents = []
            for chunk_data in chunks:
                doc = Document(
                    page_content=chunk_data['text'],
                    metadata={**metadata.copy(), **chunk_data.get('metadata', {})}
                )
                documents.append(doc)
            
            return documents
        else:
            # Fallback to default chunking
            return self.text_splitter.create_documents(
                texts=[text],
                metadatas=[metadata.copy()]
            )
    
    def _clean_pdf_artifacts(self, text: str) -> str:
        """
        Clean common PDF extraction artifacts that create noise.
        Based on blind test findings.
        """
        # Remove standalone numbers (likely footnotes/page numbers)
        text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)
        
        # Remove standalone bullet points and special characters
        text = re.sub(r'^[•·∙◦▪▫◘○●□■☐☑✓✗×]$', '', text, flags=re.MULTILINE)
        
        # Remove lines with only numbers and spaces (like "1 2 3 4")
        text = re.sub(r'^[\d\s]+$', '', text, flags=re.MULTILINE)
        
        # Remove orphaned punctuation
        text = re.sub(r'^[,;:.]$', '', text, flags=re.MULTILINE)
        
        # Clean up excessive newlines created by removals
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        
        # Remove common PDF header/footer patterns
        text = re.sub(r'^\d+\s*\|\s*Page$', '', text, flags=re.MULTILINE)
        
        return text
    
    def _extract_section_from_chunk(self, chunk_text: str) -> Optional[str]:
        """Extract section heading from chunk if present"""
        # Look for markdown headers
        header_match = re.search(r'^#{1,6}\s+(.+)$', chunk_text, re.MULTILINE)
        if header_match:
            return header_match.group(1).strip()
        
        # Look for other section indicators
        section_match = re.search(r'^(?:Chapter|Section)\s+[\d\w]+[:\s]+(.+)$', chunk_text, re.MULTILINE)
        if section_match:
            return section_match.group(1).strip()
        
        return None
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Process multiple documents with improved chunking"""
        all_chunks = []
        
        chunking_method = "semantic" if self.use_semantic_chunking else "smart"
        console.print(f"[blue]Processing {len(documents)} documents with {chunking_method} chunking...[/blue]")
        
        for doc in track(documents, description=f"{chunking_method.capitalize()} chunking..."):
            chunks = self.smart_chunk_document(doc)
            all_chunks.extend(chunks)
        
        console.print(f"[green]Created {len(all_chunks)} chunks from {len(documents)} documents[/green]")
        
        # Log chunking statistics
        if all_chunks:
            avg_tokens = sum(c.metadata.get('chunk_tokens', 0) for c in all_chunks) / len(all_chunks)
            console.print(f"[cyan]Average chunk size: {avg_tokens:.0f} tokens[/cyan]")
        
        # Remove near-duplicates (not just exact)
        unique_chunks = self._remove_near_duplicate_chunks(all_chunks)
        
        if len(unique_chunks) < len(all_chunks):
            console.print(f"[yellow]Removed {len(all_chunks) - len(unique_chunks)} duplicate/near-duplicate chunks[/yellow]")
        
        return unique_chunks
    
    def _remove_near_duplicate_chunks(self, chunks: List[Document]) -> List[Document]:
        """Remove duplicate and near-duplicate chunks"""
        seen_hashes = set()
        unique_chunks = []
        
        for chunk in chunks:
            # Create normalized version for comparison
            normalized = re.sub(r'\s+', ' ', chunk.page_content.lower()).strip()
            content_hash = hashlib.sha256(normalized.encode()).hexdigest()
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_chunks.append(chunk)
        
        return unique_chunks


# Backward compatibility
TextProcessor = TextProcessorV2

# Create a variant with sliding window for maximum continuity
class TextProcessorV2SlidingWindow(TextProcessorV2):
    """Text processor using sliding window approach for maximum continuity"""
    
    def _semantic_chunking(self, text: str, metadata: Dict) -> List[Document]:
        """Use sliding window chunking for maximum overlap"""
        if self.use_semantic_chunking and self.semantic_chunker:
            # Use sliding window variant
            chunks = self.semantic_chunker.create_sliding_window_chunks(text, metadata)
            
            # Convert to Document objects
            documents = []
            for chunk_data in chunks:
                doc = Document(
                    page_content=chunk_data['text'],
                    metadata={**metadata.copy(), **chunk_data.get('metadata', {})}
                )
                documents.append(doc)
            
            return documents
        else:
            return super()._semantic_chunking(text, metadata)
