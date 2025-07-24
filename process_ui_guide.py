#!/usr/bin/env python3
"""
Process the UI/UX Architecture Guide through Nexus
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import our modules
from src.quality.quality_scorer import KnowledgeQualityAnalyzer as QualityAnalyzer
from src.quality.quality_enhancer import QualityEnhancer

print("🎨 Processing Architect's Guide to Master-Class UI/UX\n")

# Read the document
doc_path = Path('data/inbox/architects_guide_ui_ux.md')
if not doc_path.exists():
    print(f"❌ Error: {doc_path} not found")
    sys.exit(1)

with open(doc_path, 'r') as f:
    content = f.read()

print(f"📚 Document loaded: {len(content)} characters\n")

# Initialize quality tools
analyzer = QualityAnalyzer()
enhancer = QualityEnhancer()

# Let's analyze a sample chunk (first 1500 chars)
sample_chunk = content[:1500]
print("🔍 Analyzing sample chunk quality...\n")

# Analyze quality
quality_result = analyzer.analyze(sample_chunk)
print(f"📊 Quality Score: {quality_result['overall_score']:.2f}/10\n")

# Show dimension scores
print("📋 Quality Dimensions:")
for dim, score in quality_result['dimension_scores'].items():
    status = "✅" if score >= 8 else "⚠️" if score >= 7 else "❌"
    print(f"  {dim.replace('_', ' ').title()}: {score:.1f}/10 {status}")

print("\n📝 Suggestions:")
for suggestion in quality_result.get('suggestions', []):
    print(f"  - {suggestion}")

# Check if enhancement needed
if quality_result['overall_score'] < 9.5:
    print(f"\n🔧 Score {quality_result['overall_score']:.2f} is below 9.5 threshold")
    print("Would be enhanced automatically during processing")
else:
    print(f"\n✅ Score {quality_result['overall_score']:.2f} meets quality threshold!")

# Check how it would be chunked
print("\n📦 Document would be processed into multiple chunks")
print(f"Estimated chunks: ~{len(content) // 1500} chunks")

print("\n🚀 To process the full document, run:")
print("python -m src.core.nexus_pipeline --input data/inbox/architects_guide_ui_ux.md")
print("\nThis will:")
print("1. Split into optimal chunks")
print("2. Assess quality of each chunk")
print("3. Enhance any chunks below 9.5/10")
print("4. Generate embeddings (local in dev mode)")
print("5. Store in vector database")
print("6. Create quality report")
