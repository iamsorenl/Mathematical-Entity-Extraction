from sklearn.metrics import classification_report, f1_score
import numpy as np

def evaluate_bio_predictions(true_tags, pred_tags):
    """Evaluate BIO tag predictions with multi-label support"""
    # Handle multi-label tags
    all_labels = set()
    for tag_list in true_tags + pred_tags:
        if tag_list != 'O':
            # Fix: handle both string and list cases
            if isinstance(tag_list, str):
                all_labels.update(tag_list.split(','))
            else:
                all_labels.update(tag_list)
    
    # Convert to binary matrices for each label
    results = {}
    for label in all_labels:
        true_binary = []
        pred_binary = []
        
        for true_tag, pred_tag in zip(true_tags, pred_tags):
            # Check if label is in true tags
            true_has_label = label in (true_tag.split(',') if true_tag != 'O' else [])
            pred_has_label = label in (pred_tag.split(',') if pred_tag != 'O' else [])
            
            true_binary.append(1 if true_has_label else 0)
            pred_binary.append(1 if pred_has_label else 0)
        
        if sum(true_binary) > 0:  # Only calculate F1 if label exists in true data
            f1 = f1_score(true_binary, pred_binary, average='binary')
            results[label] = f1
    
    return results