#!/usr/bin/env python3
"""
Test the new quality scorer with real examples
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.quality.quality_scorer import KnowledgeQualityAnalyzer as OldAnalyzer
from src.quality.quality_scorer_v2 import ImprovedQualityAnalyzer as NewAnalyzer

# Test chunk from Architect's Guide (previously scored 4.7/10)
architect_chunk = """
Instead, top designers use off-black and tinted dark palettes. For example, 
Google's Material Design dark theme recommends a very dark gray (#121212) as 
the base surface color. This softer black reduces eye strain in low-light 
conditions and prevents the high contrast issues of pure black.
"""

# Test chunk with good formatting (for comparison)
formatted_chunk = """
## Dark Theme Color Selection

When implementing dark themes, consider these guidelines:

1. **Base Colors**: Use #121212 (Material Design) or #1a1a1a
2. **Text Colors**: High contrast white (#FFFFFF) for primary text
3. **Accent Colors**: Muted versions of brand colors

### Example Implementation:
```css
.dark-theme {
  background: #121212;
  color: #FFFFFF;
}
```

This approach ensures readability while reducing eye strain.
"""

# Test code documentation chunk
code_doc_chunk = """
## Installation
1. Install dependencies: `npm install`
2. Configure environment: `cp .env.example .env`
3. Run development server: `npm run dev`
"""

print("🔬 QUALITY SCORER COMPARISON TEST\n")
print("=" * 60)

# Initialize analyzers
old_analyzer = OldAnalyzer()
new_analyzer = NewAnalyzer()

# Test each chunk
test_chunks = [
    ("Architect's Guide (specific, technical)", architect_chunk),
    ("Well-formatted documentation", formatted_chunk),
    ("Code installation instructions", code_doc_chunk)
]

for name, chunk in test_chunks:
    print(f"\n📄 {name}")
    print("-" * 40)
    
    # Old scorer
    old_result = old_analyzer.analyze(chunk)
    old_score = old_result['overall_score']
    
    # New scorer
    new_result = new_analyzer.analyze(chunk)
    new_score = new_result['overall_score']
    
    print(f"Old Pattern-Based Score: {old_score:.1f}/10")
    print(f"New Retrieval-Based Score: {new_score:.1f}/10")
    print(f"Difference: {new_score - old_score:+.1f}")
    
    # Show dimension breakdown for new scorer
    print("\nNew Scorer Dimensions:")
    for dim, score in new_result['dimension_scores'].items():
        print(f"  - {dim}: {score:.1f}/10")
    
    if new_result.get('strengths'):
        print(f"\nStrengths: {', '.join(new_result['strengths'])}")
    
    if new_result.get('recommendations'):
        print("\nRecommendations:")
        for rec in new_result['recommendations']:
            print(f"  • {rec}")

print("\n" + "=" * 60)
print("\n🎯 KEY INSIGHTS:")
print("• The new scorer recognizes valuable content regardless of format")
print("• Focuses on retrieval-relevant qualities like specificity and context")
print("• Better aligned with what makes chunks useful for RAG systems")

