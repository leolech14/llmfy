#!/usr/bin/env python3
"""Debug chunk extraction"""

import chromadb

client = chromadb.PersistentClient(path="data/vector_db/chroma_db")
collection = client.get_or_create_collection("knowledge_base")

print(f"Collection count: {collection.count()}")

# Get sample documents
if collection.count() > 0:
    results = collection.get(
        include=["documents", "metadatas"],
        limit=5
    )
    
    print("\nSample documents:")
    for i, doc in enumerate(results['documents']):
        metadata = results['metadatas'][i]
        print(f"\nDocument {i+1}:")
        print(f"  Filename: {metadata.get('filename', 'Unknown')}")
        print(f"  Content preview: {doc[:100]}...")
        print(f"  Metadata keys: {list(metadata.keys())}")
else:
    print("Collection is empty!")
    
# Try to get by filename pattern
print("\n\nTrying to search for Quality Control documents...")
try:
    # Get all and filter
    all_results = collection.get(limit=1000)
    quality_docs = []
    
    for i, metadata in enumerate(all_results['metadatas']):
        filename = metadata.get('filename', '').lower()
        if 'quality' in filename or 'control' in filename:
            quality_docs.append(i)
            
    print(f"Found {len(quality_docs)} documents with 'quality' or 'control' in filename")
    
except Exception as e:
    print(f"Error: {e}")
