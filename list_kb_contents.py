#!/usr/bin/env python3
"""
List all documents in the knowledge base with detailed statistics
"""

import chromadb
from pathlib import Path
from collections import defaultdict

def main():
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path='data/vector_db/chroma_db')
    
    try:
        collection = client.get_collection('knowledge_base')
        print(f"✅ Connected to knowledge base")
    except Exception as e:
        print(f"❌ Error connecting to knowledge base: {e}")
        return
    
    # Get all documents with metadata
    all_docs = collection.get(limit=1000, include=['metadatas'])
    
    # Analyze files
    files_info = defaultdict(lambda: {
        'count': 0,
        'quality_scores': [],
        'chunk_indices': [],
        'original_filenames': set()
    })
    
    for metadata in all_docs['metadatas']:
        filename = metadata.get('filename', 'Unknown')
        files_info[filename]['count'] += 1
        files_info[filename]['chunk_indices'].append(metadata.get('chunk_index', 0))
        
        if 'final_quality_score' in metadata:
            files_info[filename]['quality_scores'].append(metadata['final_quality_score'])
        
        if 'original_filename' in metadata:
            files_info[filename]['original_filenames'].add(metadata['original_filename'])
    
    # Print detailed statistics
    print(f'\n📊 Detailed Knowledge Base Contents:')
    print(f'Total Documents: {len(files_info)}')
    print(f'Total Chunks: {sum(info["count"] for info in files_info.values())}')
    print(f'\n📁 Documents in Knowledge Base:\n')
    
    for filename, info in sorted(files_info.items()):
        avg_quality = sum(info['quality_scores']) / len(info['quality_scores']) if info['quality_scores'] else 0
        chunk_indices = sorted(info['chunk_indices'])
        
        print(f'  📄 {filename}')
        print(f'     • Chunks: {info["count"]}')
        print(f'     • Average Quality: {avg_quality:.1f}/10')
        print(f'     • Chunk Range: {min(chunk_indices)} - {max(chunk_indices)}')
        
        if info['original_filenames']:
            print(f'     • Original Names: {", ".join(info["original_filenames"])}')
        print()
    
    # Get sample chunks from each document
    print('\n📝 Sample Content from Each Document:\n')
    
    for filename in sorted(files_info.keys()):
        # Query for chunks from this specific file
        results = collection.get(
            where={"filename": filename},
            limit=1,
            include=["documents", "metadatas"]
        )
        
        if results['documents']:
            content = results['documents'][0][:200] + '...' if len(results['documents'][0]) > 200 else results['documents'][0]
            print(f'  📄 {filename}:')
            print(f'     "{content}"')
            print()

if __name__ == "__main__":
    main()