#!/usr/bin/env python3
"""List ChromaDB collections"""

import chromadb

client = chromadb.PersistentClient(path="data/chroma")
collections = client.list_collections()

print("Available collections:")
for col in collections:
    print(f"  - {col.name}")
    print(f"    Count: {col.count()}")