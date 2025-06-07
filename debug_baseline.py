import pandas as pd
import json
from few_shot_tagger import FewShotMathTagger
from bio_converter import convert_spans_to_bio
from evaluator import evaluate_bio_predictions

def debug_part1():
    """Debug the Part 1 baseline system"""
    
    # Load data
    train_df = pd.read_json('A2-NLP_244/train.json')
    val_df = pd.read_json('A2-NLP_244/val.json')
    
    with open('A2-NLP_244/file_contents.json', 'r') as f:
        file_contents = json.load(f)
    
    # Test on just ONE file first
    sample_fileid = val_df.iloc[0]['fileid']
    text = file_contents[sample_fileid]
    file_annotations = val_df[val_df['fileid'] == sample_fileid]
    
    print(f"=== DEBUGGING SINGLE FILE ===")
    print(f"File ID: {sample_fileid}")
    print(f"Text length: {len(text)}")
    print(f"Number of annotations: {len(file_annotations)}")
    print(f"First 200 chars: {text[:200]}")
    
    # Debug BIO conversion
    print(f"\n=== DEBUGGING BIO CONVERSION ===")
    annotations_list = file_annotations.to_dict('records')
    print(f"Annotations: {annotations_list}")
    
    try:
        true_tokens, true_tags = convert_spans_to_bio(text, annotations_list)
        print(f"Number of tokens: {len(true_tokens)}")
        print(f"Number of tags: {len(true_tags)}")
        print(f"First 10 tokens: {true_tokens[:10]}")
        print(f"First 10 tags: {true_tags[:10]}")
        
        # Count non-O tags
        non_o_tags = [tag for tag in true_tags if tag != 'O']
        print(f"Non-O tags: {len(non_o_tags)} out of {len(true_tags)}")
        print(f"Sample non-O tags: {non_o_tags[:5]}")
        
    except Exception as e:
        print(f"ERROR in BIO conversion: {e}")
        return
    
    # Debug prediction
    print(f"\n=== DEBUGGING PREDICTIONS ===")
    tagger = FewShotMathTagger()
    
    try:
        pred_tokens, pred_tags = tagger.predict_tags(text)
        print(f"Predicted tokens: {len(pred_tokens)}")
        print(f"Predicted tags: {len(pred_tags)}")
        print(f"First 10 pred tokens: {pred_tokens[:10]}")
        print(f"First 10 pred tags: {pred_tags[:10]}")
        
        # Count non-O predictions
        non_o_preds = [tag for tag in pred_tags if tag != 'O']
        print(f"Non-O predictions: {len(non_o_preds)} out of {len(pred_tags)}")
        
    except Exception as e:
        print(f"ERROR in prediction: {e}")
        return
    
    # Check token alignment
    print(f"\n=== TOKEN ALIGNMENT CHECK ===")
    if len(true_tokens) != len(pred_tokens):
        print(f"WARNING: Token length mismatch! True: {len(true_tokens)}, Pred: {len(pred_tokens)}")
        
        # Show differences
        min_len = min(len(true_tokens), len(pred_tokens))
        for i in range(min_len):
            if true_tokens[i] != pred_tokens[i]:
                print(f"  Diff at {i}: '{true_tokens[i]}' vs '{pred_tokens[i]}'")
                if i > 5:  # Don't spam too much
                    break

if __name__ == "__main__":
    debug_part1()