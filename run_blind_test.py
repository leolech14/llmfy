#!/usr/bin/env python3
"""
Quick script to run blind test on already processed documents
"""

import sys
from pathlib import Path

from src.evaluation.blind_test import BlindTestEvaluator

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_blind_test.py <document_pattern>")
        print("Example: python run_blind_test.py 'Quality Control'")
        sys.exit(1)
    
    document_pattern = sys.argv[1]
    
    print(f"Running blind test for documents matching: {document_pattern}")
    
    evaluator = BlindTestEvaluator()
    report = evaluator.run_blind_test(document_pattern)
    
    if report:
        print(f"\n✅ Blind test complete!")
        print(f"Reconstruction score: {report['evaluation']['reconstruction_score']}/10")
    else:
        print("\n❌ Blind test failed - no documents found")

if __name__ == "__main__":
    main()

