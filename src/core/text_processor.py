"""
Legacy TextProcessor - Maintained for backward compatibility
This is now a wrapper around TextProcessorV2
"""

from typing import List, Optional
from langchain.schema import Document

from .text_processor_v2 import TextProcessorV2, ChunkingConfig

class TextProcessor:
    """
    Legacy TextProcessor interface maintained for backward compatibility.
    Internally uses TextProcessorV2 with the new implementation.
    """
    
    def __init__(self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None):
        # Create config for V2
        config = ChunkingConfig(
            chunk_size=chunk_size or 1000,  # Legacy default
            chunk_overlap=chunk_overlap or 200,  # Legacy default
            min_chunk_size=50
        )
        
        # Use V2 implementation internally
        self._processor = TextProcessorV2(config=config, use_semantic_chunking=False)
        
        # Store legacy settings for compatibility
        self.chunk_size = chunk_size or 1000
        self.chunk_overlap = chunk_overlap or 200
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Process documents using the improved V2 processor"""
        return self._processor.process_documents(documents)
    
    def split_text(self, text: str) -> List[str]:
        """Legacy method - splits text into chunks"""
        # Create a temporary document
        doc = Document(page_content=text, metadata={})
        chunks = self._processor.smart_chunk_document(doc)
        return [chunk.page_content for chunk in chunks]
    
    # Any other legacy methods can be added here as needed
