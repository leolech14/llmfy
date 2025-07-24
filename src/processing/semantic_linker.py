#!/usr/bin/env python3
"""
🔗 Semantic Linker - Post-processing phase to create semantic relationships between chunks
"""

import json
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import chromadb
from dataclasses import dataclass
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
import openai
import os
from dotenv import load_dotenv

load_dotenv()
console = Console()

@dataclass
class SemanticLink:
    """Represents a semantic relationship between chunks"""
    source_id: str
    target_id: str
    relationship_type: str  # 'continues', 'references', 'elaborates', 'contrasts', 'implements'
    strength: float  # 0.0-1.0
    description: str

class SemanticLinker:
    """Creates semantic links between chunks using AI analysis"""
    
    def __init__(self, db_path: str = "data/vector_db/chroma_db"):
        self.db_path = Path(db_path)
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        self.links: List[SemanticLink] = []
        
    def analyze_chunks(self, collection_name: str = "knowledge_base_hybrid") -> Dict[str, Any]:
        """Analyze all chunks and create semantic links"""
        console.print("[cyan]🔗 Starting semantic link analysis...[/cyan]")
        
        # Get collection
        try:
            collection = self.client.get_collection(collection_name)
            total_chunks = collection.count()
            console.print(f"[green]Found {total_chunks} chunks to analyze[/green]")
        except Exception as e:
            console.print(f"[red]Error getting collection: {e}[/red]")
            return {}
        
        # Get all chunks with metadata
        results = collection.get(include=["documents", "metadatas", "embeddings"])
        chunks = []
        for i in range(len(results['ids'])):
            chunks.append({
                'id': results['ids'][i],
                'content': results['documents'][i],
                'metadata': results['metadatas'][i],
                'embedding': results['embeddings'][i]
            })
        
        # Analyze relationships
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Analyzing chunk relationships...", total=len(chunks))
            
            # Group chunks by source file for efficient processing
            chunks_by_file = self._group_by_file(chunks)
            
            for filename, file_chunks in chunks_by_file.items():
                console.print(f"\n[dim]Processing {filename} ({len(file_chunks)} chunks)[/dim]")
                
                # Analyze relationships within the same document
                self._analyze_document_relationships(file_chunks)
                
                # Skip cross-references for now to speed up processing
                # self._find_cross_references(file_chunks, chunks)
                
                progress.update(task, advance=len(file_chunks))
        
        # Save links
        self._save_links()
        
        return self._generate_report()
    
    def _group_by_file(self, chunks: List[Dict]) -> Dict[str, List[Dict]]:
        """Group chunks by source file"""
        grouped = {}
        for chunk in chunks:
            filename = chunk['metadata'].get('filename', 'unknown')
            if filename not in grouped:
                grouped[filename] = []
            grouped[filename].append(chunk)
        
        # Sort chunks within each file by chunk_index
        for filename in grouped:
            grouped[filename].sort(key=lambda x: x['metadata'].get('chunk_index', 0))
        
        return grouped
    
    def _analyze_document_relationships(self, chunks: List[Dict]):
        """Analyze relationships between chunks in the same document"""
        # Sequential relationships
        for i in range(len(chunks) - 1):
            curr_chunk = chunks[i]
            next_chunk = chunks[i + 1]
            
            # Check if chunks are truly sequential (no gap)
            curr_index = curr_chunk['metadata'].get('chunk_index', -1)
            next_index = next_chunk['metadata'].get('chunk_index', -1)
            
            if next_index == curr_index + 1:
                # Analyze continuation relationship
                relationship = self._analyze_continuation(curr_chunk, next_chunk)
                if relationship:
                    self.links.append(relationship)
        
        # Conceptual relationships within document
        self._find_conceptual_links(chunks)
    
    def _analyze_continuation(self, chunk1: Dict, chunk2: Dict) -> Optional[SemanticLink]:
        """Analyze if chunk2 continues from chunk1 using embeddings"""
        # Simple heuristic approach - no AI calls needed
        chunk1_end = chunk1['content'][-100:].lower()
        chunk2_start = chunk2['content'][:100].lower()
        
        # Check for continuation indicators
        continues = False
        strength = 0.0
        relationship = "new_topic"
        
        # Strong continuation signals
        if chunk1_end.rstrip().endswith((':',  ',', ';', '-')):
            continues = True
            strength = 0.9
            relationship = "continues"
        # Sentence fragment continuation
        elif chunk1_end.rstrip()[-1:].isalpha() and chunk2_start[0].islower():
            continues = True
            strength = 0.8
            relationship = "continues"
        # List or enumeration continuation
        elif any(chunk2_start.strip().startswith(p) for p in ['• ', '- ', '* ', '\d+. ', '\d+) ']):
            continues = True
            strength = 0.7
            relationship = "elaborates"
        
        # Use embedding similarity as additional signal
        if 'embedding' in chunk1 and 'embedding' in chunk2:
            embedding1 = np.array(chunk1['embedding'])
            embedding2 = np.array(chunk2['embedding'])
            similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
            
            if similarity > 0.9:
                strength = max(strength, similarity)
                if not continues:
                    relationship = "references"
        
        if strength > 0.6:
            return SemanticLink(
                source_id=chunk1['id'],
                target_id=chunk2['id'],
                relationship_type=relationship,
                strength=strength,
                description=f"Sequential chunks with {relationship} relationship"
            )
        
        return None
    
    def _find_conceptual_links(self, chunks: List[Dict]):
        """Find conceptual relationships within a document"""
        # Extract key concepts from each chunk
        chunk_concepts = []
        
        for chunk in chunks:
            concepts = self._extract_concepts(chunk['content'])
            chunk_concepts.append({
                'chunk': chunk,
                'concepts': concepts
            })
        
        # Find chunks with overlapping concepts
        for i in range(len(chunk_concepts)):
            for j in range(i + 2, len(chunk_concepts)):  # Skip adjacent chunks
                shared = set(chunk_concepts[i]['concepts']) & set(chunk_concepts[j]['concepts'])
                
                if len(shared) >= 2:  # At least 2 shared concepts
                    # Create reference link
                    self.links.append(SemanticLink(
                        source_id=chunk_concepts[i]['chunk']['id'],
                        target_id=chunk_concepts[j]['chunk']['id'],
                        relationship_type='references',
                        strength=min(len(shared) / 5, 1.0),  # Normalize strength
                        description=f"Shares concepts: {', '.join(list(shared)[:3])}"
                    ))
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text using simple NLP"""
        import re
        
        concepts = []
        text_lower = text.lower()
        
        # Technical patterns
        # CamelCase terms
        camel_case = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', text)
        concepts.extend(camel_case[:3])
        
        # Acronyms
        acronyms = re.findall(r'\b[A-Z]{2,}\b', text)
        concepts.extend(acronyms[:2])
        
        # Common technical terms
        tech_terms = [
            'embedding', 'chunk', 'vector', 'semantic', 'quality',
            'dimension', 'model', 'agent', 'mcp', 'protocol',
            'api', 'database', 'search', 'retrieval', 'context'
        ]
        
        for term in tech_terms:
            if term in text_lower:
                concepts.append(term)
                if len(concepts) >= 5:
                    break
        
        # Quoted terms
        quoted = re.findall(r'"([^"]+)"', text)
        concepts.extend([q for q in quoted if len(q.split()) <= 3][:2])
        
        # Deduplicate and limit
        seen = set()
        unique_concepts = []
        for c in concepts:
            c_lower = c.lower()
            if c_lower not in seen:
                seen.add(c_lower)
                unique_concepts.append(c)
                if len(unique_concepts) >= 5:
                    break
        
        return unique_concepts
    
    def _find_cross_references(self, file_chunks: List[Dict], all_chunks: List[Dict]):
        """Find references to other documents"""
        # This is expensive, so only do it for chunks that mention other files or external concepts
        for chunk in file_chunks:
            if any(keyword in chunk['content'].lower() for keyword in ['see', 'refer to', 'as described in', 'mentioned in']):
                # Use embeddings to find similar chunks in other documents
                similar = self._find_similar_chunks(chunk, all_chunks, same_file=False)
                
                for similar_chunk, similarity in similar[:3]:  # Top 3 matches
                    if similarity > 0.85:
                        self.links.append(SemanticLink(
                            source_id=chunk['id'],
                            target_id=similar_chunk['id'],
                            relationship_type='references',
                            strength=similarity,
                            description=f"Cross-document reference (similarity: {similarity:.2f})"
                        ))
    
    def _find_similar_chunks(self, chunk: Dict, all_chunks: List[Dict], same_file: bool = True) -> List[Tuple[Dict, float]]:
        """Find similar chunks using embeddings"""
        chunk_embedding = np.array(chunk['embedding'])
        current_file = chunk['metadata'].get('filename')
        
        similarities = []
        for other in all_chunks:
            # Skip self and optionally same file
            if other['id'] == chunk['id']:
                continue
            if not same_file and other['metadata'].get('filename') == current_file:
                continue
            
            # Calculate cosine similarity
            other_embedding = np.array(other['embedding'])
            similarity = np.dot(chunk_embedding, other_embedding) / (np.linalg.norm(chunk_embedding) * np.linalg.norm(other_embedding))
            
            similarities.append((other, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities
    
    def _save_links(self):
        """Save semantic links to file"""
        output_path = self.db_path / "semantic_links.json"
        
        link_data = [
            {
                'source_id': link.source_id,
                'target_id': link.target_id,
                'type': link.relationship_type,
                'strength': link.strength,
                'description': link.description
            }
            for link in self.links
        ]
        
        with open(output_path, 'w') as f:
            json.dump({
                'links': link_data,
                'total_links': len(link_data),
                'relationship_types': list(set(l.relationship_type for l in self.links))
            }, f, indent=2)
        
        console.print(f"[green]✅ Saved {len(self.links)} semantic links to {output_path}[/green]")
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate analysis report"""
        report = {
            'total_links': len(self.links),
            'relationship_types': {},
            'avg_strength': np.mean([l.strength for l in self.links]) if self.links else 0
        }
        
        # Count relationship types
        for link in self.links:
            if link.relationship_type not in report['relationship_types']:
                report['relationship_types'][link.relationship_type] = 0
            report['relationship_types'][link.relationship_type] += 1
        
        # Display report
        console.print("\n[bold cyan]📊 Semantic Link Analysis Report[/bold cyan]")
        console.print(f"Total Links Created: {report['total_links']}")
        console.print(f"Average Link Strength: {report['avg_strength']:.2f}")
        console.print("\nRelationship Types:")
        for rel_type, count in report['relationship_types'].items():
            console.print(f"  • {rel_type}: {count}")
        
        return report


def main():
    """Run semantic linking as standalone"""
    linker = SemanticLinker()
    linker.analyze_chunks()

if __name__ == "__main__":
    main()