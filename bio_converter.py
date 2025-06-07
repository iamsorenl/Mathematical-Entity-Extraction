import pandas as pd
import json
import re

def tokenize_mathematical_text(text):
    """Simple tokenizer that handles LaTeX notation"""
    # Split on whitespace but keep LaTeX commands together
    tokens = re.findall(r'\\\([^)]*\\\)|\\\[[^\]]*\\\]|[^\s]+', text)
    tokens = [t for t in tokens if t.strip()]  # Remove empty tokens
    return tokens

def char_to_token_mapping(text, tokens):
    """Map character positions to token indices"""
    char_to_token = {}
    char_pos = 0
    
    for token_idx, token in enumerate(tokens):
        start_pos = text.find(token, char_pos)
        if start_pos != -1:
            for i in range(start_pos, start_pos + len(token)):
                char_to_token[i] = token_idx
            char_pos = start_pos + len(token)
    
    return char_to_token

def convert_spans_to_bio(text, annotations):
    """Convert character spans to BIO tags"""
    tokens = tokenize_mathematical_text(text)
    char_to_token = char_to_token_mapping(text, tokens)
    
    # Initialize all tokens as 'O'
    bio_tags = ['O'] * len(tokens)
    
    for anno in annotations:
        start_char, end_char = anno['start'], anno['end']
        tag = anno['tag']
        
        # Find which tokens this annotation spans
        token_indices = set()
        for char_idx in range(start_char, end_char):
            if char_idx in char_to_token:
                token_indices.add(char_to_token[char_idx])
        
        # Assign BIO tags
        for token_idx in token_indices:
            if bio_tags[token_idx] == 'O':
                bio_tags[token_idx] = f'B-{tag}'
            else:
                # Multi-label: add to existing tags
                bio_tags[token_idx] += f',{tag}'
    
    return tokens, bio_tags