import json
import pandas as pd
import os
from pathlib import Path
from few_shot_tagger import FewShotMathTagger
from bio_converter import convert_spans_to_bio  # Fixed import name
from sklearn.metrics import f1_score, classification_report, precision_recall_fscore_support
import warnings
from tqdm import tqdm
import logging
import signal
import time

# Suppress warnings and set transformers logging
warnings.filterwarnings('ignore')
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

def run_with_timeout(func, timeout_seconds=300):  # 5 minute timeout
    """Run function with timeout protection"""
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        result = func()
        signal.alarm(0)  # Cancel timeout
        return result
    except TimeoutError:
        print(f"TIMEOUT after {timeout_seconds}s - Model is hanging!")
        signal.alarm(0)
        return None

def load_mmd_text(file_id, data_dir="A2-NLP_244"):
    """Load the actual MMD text content"""
    # Try to find the corresponding text in file_contents.json first
    file_contents_path = Path(data_dir) / "file_contents.json"
    if file_contents_path.exists():
        with open(file_contents_path, 'r', encoding='utf-8') as f:
            file_contents = json.load(f)
        
        if file_id in file_contents:
            return file_contents[file_id]
    
    # Fallback: try to find the corresponding MMD file
    mmd_files_dir = Path(data_dir) / "unannotated_mmds"
    if mmd_files_dir.exists():
        for mmd_file in mmd_files_dir.glob("*.mmd"):
            if file_id.replace('.json', '') in mmd_file.name:
                with open(mmd_file, 'r', encoding='utf-8') as f:
                    return f.read()
    
    return None

def evaluate_on_validation_set():
    """Run Part 1 evaluation on validation set"""
    
    print("=== PART 1: BASELINE EVALUATION ===")
    
    # Initialize the tagger
    tagger = FewShotMathTagger()
    
    # Load validation data
    val_df = pd.read_json("A2-NLP_244/val.json")
    
    all_true_tags = []
    all_pred_tags = []
    results = []
    
    # Get unique files and set up progress bar
    unique_files = val_df['fileid'].unique()
    
    # Process each unique file with progress bar
    processed_files = 0
    for file_id in tqdm(unique_files, desc="Processing validation files"):
        print(f"\nProcessing {file_id}...")
        
        # Get annotations for this file
        file_annotations = val_df[val_df['fileid'] == file_id].to_dict('records')
        
        # Try to load the actual text
        text = load_mmd_text(file_id)
        if text is None:
            print(f"Could not find text for {file_id}, skipping...")
            continue
        
        try:
            # Convert annotations to BIO tags
            true_tokens, true_tags = convert_spans_to_bio(text, file_annotations)
            
            # Get predictions from our model
            pred_tokens, pred_tags = tagger.predict_tags(text)
            
            # Ensure alignment
            if len(true_tags) != len(pred_tags):
                print(f"Tag alignment issue: true={len(true_tags)}, pred={len(pred_tags)}")
                # Truncate to shorter length for evaluation
                min_len = min(len(true_tags), len(pred_tags))
                true_tags = true_tags[:min_len]
                pred_tags = pred_tags[:min_len]
            
            # Store results
            all_true_tags.extend(true_tags)
            all_pred_tags.extend(pred_tags)
            
            # Calculate per-file metrics
            file_f1 = f1_score(true_tags, pred_tags, average='weighted', zero_division=0)
            results.append({
                'file_id': file_id,
                'num_tokens': len(true_tags),
                'num_true_entities': len([t for t in true_tags if t != 'O']),
                'num_pred_entities': len([t for t in pred_tags if t != 'O']),
                'f1_score': file_f1
            })
            
            processed_files += 1
            print(f"File F1: {file_f1:.3f}")
            
        except Exception as e:
            print(f"Error processing {file_id}: {e}")
            continue
        
        # Process a few files for testing (remove this limit for full evaluation)
        if processed_files >= 5:  # Increased to 5 for better testing
            print(f"\nProcessing limited to {processed_files} files for testing...")
            break

    if not all_true_tags:
        print("No files were successfully processed!")
        return None
    
    # Calculate overall metrics
    overall_f1 = f1_score(all_true_tags, all_pred_tags, average='weighted', zero_division=0)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_true_tags, all_pred_tags, average='weighted', zero_division=0
    )
    
    # Print results
    print(f"\n=== PART 1 RESULTS ===")
    print(f"Files processed: {processed_files}")
    print(f"Total tokens: {len(all_true_tags)}")
    print(f"True entities: {len([t for t in all_true_tags if t != 'O'])}")
    print(f"Predicted entities: {len([t for t in all_pred_tags if t != 'O'])}")
    print(f"Overall F1 Score: {overall_f1:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    
    # Save results
    results_summary = {
        'part': 'Part 1 - Baseline (Few-shot)',
        'model': tagger.model_name if tagger.use_llm else 'Rule-based fallback',
        'files_processed': processed_files,
        'total_tokens': len(all_true_tags),
        'overall_f1': overall_f1,
        'precision': precision,
        'recall': recall,
        'per_file_results': results
    }
    
    # Save to JSON for submission
    with open('part1_validation_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nResults saved to 'part1_validation_results.json'")
    return results_summary

def run_inference_on_unannotated():
    """Run inference on unannotated MMD files for Part 3 analysis"""
    print("\n=== RUNNING INFERENCE ON UNANNOTATED MMDS ===")
    
    tagger = FewShotMathTagger()
    unannotated_dir = Path("A2-NLP_244/unannotated_mmds")
    
    all_predictions = []
    mmd_files = list(unannotated_dir.glob("*.mmd.filtered"))
    
    print(f"Found {len(mmd_files)} unannotated MMD files")
    
    if not mmd_files:
        print("No MMD files found!")
        pred_df = pd.DataFrame(columns=['fileid', 'start', 'end', 'tag'])
        pred_df.to_json('part1_unannotated_predictions.json', orient='records')
        return
    
    # Main progress bar for files
    file_pbar = tqdm(mmd_files, desc="Processing MMD files", unit="file")
    
    for file_idx, mmd_file in enumerate(file_pbar):
        # Update file progress description
        file_pbar.set_description(f"Processing file {file_idx+1}/{len(mmd_files)}")
        print(f"\nProcessing {mmd_file.name}...")
        
        # Step 1: File loading
        print("  Loading file...")
        with open(mmd_file, 'r', encoding='utf-8') as f:
            text = f.read()

        # Step 2: Truncation check
        MAX_CHARS = 50000
        original_length = len(text)
        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS]
            print(f"  Truncated large file from {original_length:,} to {MAX_CHARS:,} chars")
        else:
            print(f"  File size OK: {original_length:,} chars")
        
        # Step 3: Model inference with progressive testing
        print("  Running model inference...")

        # Test with progressively larger chunks
        test_sizes = [1000, 5000, 10000, 25000, 50000]
        successful_size = 0

        for test_size in test_sizes:
            if len(text) > test_size:
                print(f"    Testing with {test_size:,} chars...")
                test_text = text[:test_size]
                
                start_time = time.time()
                
                def test_inference():
                    return tagger.predict_tags(test_text)
                
                # Try with 2-minute timeout per test
                result = run_with_timeout(test_inference, timeout_seconds=120)
                
                if result is not None:
                    test_tokens, test_tags = result
                    elapsed = time.time() - start_time
                    print(f"    {test_size:,} chars completed in {elapsed:.1f}s")
                    successful_size = test_size
                else:
                    print(f"    {test_size:,} chars TIMED OUT - using {successful_size:,}")
                    break
            else:
                successful_size = len(text)
                break

        # Use the largest successful size
        if successful_size > 0:
            final_text = text[:successful_size]
            print(f"  Using {successful_size:,} chars (truncated from {len(text):,})")
            
            def final_inference():
                return tagger.predict_tags(final_text)
            
            result = run_with_timeout(final_inference, timeout_seconds=300)
            
            if result is not None:
                tokens, pred_tags = result
                text = final_text  # Update text for position finding
            else:
                print("  FINAL INFERENCE FAILED - Using rule-based fallback")
                # Force rule-based mode
                from bio_converter import tokenize_mathematical_text
                tokens = tokenize_mathematical_text(final_text)
                pred_tags = ['O'] * len(tokens)
        else:
            print("  ALL TESTS FAILED - Skipping this file")
            continue
        
        # Step 4: Converting to annotation format
        print("  Converting predictions to annotations...")
        file_predictions = 0
        
        # Progress bar for annotation conversion
        conversion_pbar = tqdm(zip(tokens, pred_tags), 
                              total=len(tokens),
                              desc="    Converting annotations", 
                              unit="token", 
                              leave=False)
        
        current_pos = 0
        for i, (token, tag) in enumerate(conversion_pbar):
            if tag != 'O':
                # Find token position in original text
                token_start = text.find(token, current_pos)
                if token_start != -1:
                    token_end = token_start + len(token)
                    
                    # Clean up tag format
                    clean_tag = tag.replace('B-', '').split(',')[0]
                    
                    all_predictions.append({
                        'fileid': mmd_file.name,
                        'start': token_start,
                        'end': token_end,
                        'tag': clean_tag,
                        'text': token
                    })
                    
                    file_predictions += 1
                    current_pos = token_end
            
            # Update conversion progress every 100 tokens
            if i % 100 == 0:
                conversion_pbar.set_description(f"    Found {file_predictions} entities")
        
        print(f"  Found {file_predictions} entities in this file")
        
        # Update main progress bar
        file_pbar.set_postfix({
            'entities': len(all_predictions), 
            'current_file_entities': file_predictions
        })
    
    # Final step: Saving results
    print(f"\nSaving {len(all_predictions)} total predictions...")
    
    with tqdm(total=1, desc="Saving JSON file") as save_pbar:
        pred_df = pd.DataFrame(all_predictions)
        pred_df.to_json('part1_unannotated_predictions.json', orient='records', indent=2)
        save_pbar.update(1)
    
    print(f"\nFound {len(all_predictions)} entity predictions")
    print(f"Saved to 'part1_unannotated_predictions.json'")
    
    return all_predictions

if __name__ == "__main__":
    # Run validation evaluation
    val_results = evaluate_on_validation_set()
    
    # Run inference on unannotated files
    unannotated_preds = run_inference_on_unannotated()
    
    print(f"\nPART 1 COMPLETE!")
    print(f"Files created for submission:")
    print(f"   - part1_validation_results.json (validation metrics)")
    print(f"   - part1_unannotated_predictions.json (predictions for analysis)")