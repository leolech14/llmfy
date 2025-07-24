from langchain.schema import Document
from src.core.text_processor_v2 import TextProcessorV2


def test_simple_chunking():
    processor = TextProcessorV2()
    doc = Document(page_content="This is a test document. " * 20, metadata={"filename": "test.txt"})
    chunks = processor.smart_chunk_document(doc)
    assert len(chunks) > 0
