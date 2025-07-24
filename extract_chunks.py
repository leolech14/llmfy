#!/usr/bin/env python3
"""
Extract processed chunks from ChromaDB for blind testing
"""

import chromadb
import json
from pathlib import Path

# Initialize ChromaDB
client = chromadb.PersistentClient(path="data/vector_db/chroma_db")
collection = client.get_or_create_collection("knowledge_base")

# Get all documents
results = collection.get(
    include=["documents", "metadatas"],
    limit=1000  # Get all
)

# Filter for Quality Control Methods chunks
quality_chunks = []
for i, doc in enumerate(results['documents']):
    metadata = results['metadatas'][i]
    if metadata.get('filename', '').startswith('Quality Control Methods'):
        quality_chunks.append({
            'chunk_index': metadata.get('chunk_index', i),
            'content': doc,
            'metadata': {
                'chunk_tokens': metadata.get('chunk_tokens'),
                'chunk_total': metadata.get('chunk_total'),
                'quality_score': metadata.get('final_quality_score'),
                'section': metadata.get('section', 'Unknown')
            }
        })

# Sort by chunk index
quality_chunks.sort(key=lambda x: x['chunk_index'])

print(f"Found {len(quality_chunks)} chunks from Quality Control Methods document")

# Save to blind test directory
blind_dir = Path("/Users/lech/02_knowledge/llmfy/blind_test")
blind_dir.mkdir(exist_ok=True)

# Save as JSON for easy reading
output_file = blind_dir / "quality_control_chunks.json"
with open(output_file, 'w') as f:
    json.dump(quality_chunks, f, indent=2)

print(f"Saved chunks to: {output_file}")

# Also save as text file for easier reading
text_file = blind_dir / "quality_control_chunks.txt"
with open(text_file, 'w') as f:
    f.write("QUALITY CONTROL METHODS - PROCESSED CHUNKS\n")
    f.write("=" * 50 + "\n\n")
    
    for chunk in quality_chunks:
        f.write(f"CHUNK {chunk['chunk_index'] + 1}/{chunk['metadata']['chunk_total']}\n")
        f.write(f"Quality Score: {chunk['metadata']['quality_score']:.1f}/10\n")
        f.write(f"Tokens: {chunk['metadata']['chunk_tokens']}\n")
        f.write("-" * 30 + "\n")
        f.write(chunk['content'])
        f.write("\n\n" + "=" * 50 + "\n\n")

print(f"Saved readable version to: {text_file}")

# Create a README for the blind test
readme_file = blind_dir / "README_BLIND_TEST.md"
with open(readme_file, 'w') as f:
    f.write("""# Blind Context Test

## Instructions for New LLM Session

1. Open a NEW Claude/LLM session (no context from previous conversation)
2. Ask it to read the `quality_control_chunks.txt` file
3. Ask it to create a comprehensive summary of what the document is about
4. Compare with the original PDF to see if context was preserved

## What to Look For

- Can the LLM understand what each chunk is about?
- Do the contextual headers help?
- Is the document's structure and flow preserved?
- Are key concepts and relationships clear?

## Files

- `quality_control_chunks.txt` - Human-readable chunks
- `quality_control_chunks.json` - JSON format with metadata
""")

print(f"\nBlind test ready! Instructions in: {readme_file}")