import pytest
from langchain.schema import Document

from src.core.text_processor_v2 import TextProcessorV2, ChunkingConfig

@pytest.fixture
def sample_document():
    text = (
        "## Intro\nThis is introduction.\n\n"
        "## Details\nMore details about things.\n\n"
        "## Conclusion\nThe end."
    )
    return Document(page_content=text, metadata={"filename": "sample.md"})


def test_smart_chunk_document_returns_expected_chunks(sample_document):
    config = ChunkingConfig(min_chunk_size=1, chunk_size=100, chunk_overlap=0)
    processor = TextProcessorV2(config=config, use_semantic_chunking=False)
    chunks = processor.smart_chunk_document(sample_document)

    assert len(chunks) == 3
    assert [c.metadata["chunk_index"] for c in chunks] == [0, 1, 2]
    assert all(c.metadata["chunk_total"] == 3 for c in chunks)
    assert all(c.page_content.startswith("[Context: Document: Sample") for c in chunks)

