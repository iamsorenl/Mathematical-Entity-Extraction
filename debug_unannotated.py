#!/usr/bin/env python3

from few_shot_tagger import FewShotMathTagger
from pathlib import Path
import time

def test_single_file():
    """Test on just the smallest file first"""
    
    tagger = FewShotMathTagger()
    mmd_files = list(Path('A2-NLP_244/unannotated_mmds').glob('*.mmd.filtered'))
    
    if not mmd_files:
        print("No files found!")
        return
    
    # Find smallest file
    smallest_file = min(mmd_files, key=lambda f: f.stat().st_size)
    print(f"Testing smallest file: {smallest_file.name} ({smallest_file.stat().st_size:,} bytes)")
    
    with open(smallest_file, 'r') as f:
        text = f.read()
    
    print(f"File content: {len(text):,} chars")
    print(f"Preview: {text[:200]}...")
    
    # Test with tiny chunk first
    tiny_text = text[:1000]
    print(f"\n=== TESTING TINY CHUNK (1000 chars) ===")
    
    start_time = time.time()
    tokens, tags = tagger.predict_tags(tiny_text)
    elapsed = time.time() - start_time
    
    print(f"✅ Completed in {elapsed:.1f}s")
    print(f"Tokens: {len(tokens)}")
    print(f"Non-O tags: {len([t for t in tags if t != 'O'])}")
    
    if elapsed < 30:  # If tiny chunk works quickly
        print(f"\n=== TESTING LARGER CHUNK (10000 chars) ===")
        larger_text = text[:10000]
        
        start_time = time.time()
        tokens, tags = tagger.predict_tags(larger_text)
        elapsed = time.time() - start_time
        
        print(f"✅ Completed in {elapsed:.1f}s")
        print(f"Tokens: {len(tokens)}")
        print(f"Non-O tags: {len([t for t in tags if t != 'O'])}")

if __name__ == "__main__":
    test_single_file()