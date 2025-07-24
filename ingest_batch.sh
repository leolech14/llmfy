#!/bin/bash
# Batch ingestion script for knowledge documents

echo "🚀 Starting batch ingestion of knowledge documents..."

# Create array of documents to ingest
documents=(
    # MCP and AI Agent Architecture
    "/Users/lech/Downloads/Advanced Technical Analysis_ MCP-Based AI Agent Ar.pdf"
    "/Users/lech/Downloads/Bridging MCPs and AI Agents_ Practical Pathways to.pdf"
    "/Users/lech/Downloads/Building a Terminal-Based AI Agent with MCP Integr.pdf"
    
    # AI Systems and Knowledge Management
    "/Users/lech/Downloads/Building a Natural AI Specialist System Locally_ A.pdf"
    "/Users/lech/Downloads/Building a Top-Level AI Library Knowledge System U.md"
    "/Users/lech/Downloads/Advanced Strategies for AI-Friendly Directory Syst.pdf"
    
    # Development Tools and Design
    "/Users/lech/Downloads/Blueprint for a \"Best-of-Breed\" Mermaid Editor.pdf"
    "/Users/lech/Downloads/Building a Personal Embedding System with Pinecone.md"
)

# Copy documents to inbox
echo "📁 Copying documents to inbox..."
for doc in "${documents[@]}"; do
    if [ -f "$doc" ]; then
        echo "  • Copying: $(basename "$doc")"
        cp "$doc" data/inbox/
    else
        echo "  ⚠️  Not found: $doc"
    fi
done

echo "✅ Documents copied to inbox"
echo ""
echo "📊 Inbox contents:"
ls -la data/inbox/

echo ""
echo "Ready to process! Run: ./llmfy process"