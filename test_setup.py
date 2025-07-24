#!/usr/bin/env python3
"""
Simple test to verify Nexus setup
"""

import os
import sys
from pathlib import Path

print("🧠 Nexus AI Library - Setup Test\n")

# Check Python version
print(f"✅ Python version: {sys.version}")

# Check directory structure
nexus_dir = Path(__file__).parent
required_dirs = ['src', 'config', 'data', 'data/inbox', 'data/processed']

print("\n📁 Directory structure:")
for dir_name in required_dirs:
    dir_path = nexus_dir / dir_name
    if dir_path.exists():
        print(f"  ✅ {dir_name}/")
    else:
        print(f"  ❌ {dir_name}/ (missing)")

# Check key files
print("\n📄 Key files:")
key_files = [
    'nexus_quickstart.py',
    'nexus_validator.py',
    'src/core/nexus_pipeline.py',
    'src/quality/quality_scorer.py',
    'src/quality/quality_enhancer.py',
    'src/embeddings/hybrid_embedder.py',
    'config/nexus_config.yaml',
    '.env'
]

for file_name in key_files:
    file_path = nexus_dir / file_name
    if file_path.exists():
        print(f"  ✅ {file_name}")
    else:
        print(f"  ❌ {file_name} (missing)")

# Check API keys
print("\n🔑 API Keys:")
if (nexus_dir / '.env').exists():
    # Load .env file
    with open(nexus_dir / '.env', 'r') as f:
        env_content = f.read()
    
    if 'OPENAI_API_KEY=' in env_content and 'sk-' in env_content:
        print("  ✅ OpenAI API key configured")
    else:
        print("  ❌ OpenAI API key missing")
    
    if 'PINECONE_API_KEY=' in env_content and 'pcsk_' in env_content:
        print("  ✅ Pinecone API key configured")
    else:
        print("  ❌ Pinecone API key missing")
else:
    print("  ❌ .env file missing")

# Check test document
test_doc = nexus_dir / 'data/inbox/test_document.md'
if test_doc.exists():
    print("\n📝 Test document:")
    print(f"  ✅ {test_doc.name} ({test_doc.stat().st_size} bytes)")
else:
    print("\n📝 Test document:")
    print("  ❌ test_document.md missing")

print("\n🚀 Next steps:")
print("1. Create virtual environment: python3 -m venv venv")
print("2. Activate it: source venv/bin/activate")
print("3. Install dependencies: pip install langchain chromadb sentence-transformers openai pinecone-client python-dotenv rich pyyaml")
print("4. Run quickstart: python nexus_quickstart.py")
print("\nOr simply run: python3 nexus_quickstart.py (it will do everything for you!)")
