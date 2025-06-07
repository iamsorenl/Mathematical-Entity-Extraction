import json
import pandas as pd
from collections import Counter

def generate_part1_report():
    """Generate a summary report for Part 1 submission"""
    
    print("=== GENERATING PART 1 REPORT ===")
    
    # Load results
    try:
        with open('part1_validation_results.json', 'r') as f:
            val_results = json.load(f)
        
        pred_df = pd.read_json('part1_unannotated_predictions.json')
    except FileNotFoundError as e:
        print(f"❌ Missing results file: {e}")
        print("Run 'python run_part1_evaluation.py' first!")
        return
    
    # Generate report
    report = f"""
# Part 1 - Baseline Model Report

## Model Description
- **Method**: Few-shot prompting with {val_results['model']}
- **Approach**: BIO tagging with rule-based fallback
- **Input**: Raw MMD mathematical text
- **Output**: Token-level tags (definition, theorem, proof, example, name, reference)

## Validation Results
- **Files Processed**: {val_results['files_processed']}
- **Total Tokens**: {val_results['total_tokens']:,}
- **F1 Score**: {val_results['overall_f1']:.3f}
- **Precision**: {val_results['precision']:.3f}
- **Recall**: {val_results['recall']:.3f}

## Predictions on Unannotated Data
- **Total Predictions**: {len(pred_df):,}
- **Files Processed**: {pred_df['fileid'].nunique()}

### Entity Distribution:
"""
    
    # Add entity distribution
    if len(pred_df) > 0:
        entity_counts = pred_df['tag'].value_counts()
        for tag, count in entity_counts.items():
            report += f"- **{tag}**: {count} ({count/len(pred_df)*100:.1f}%)\n"
    else:
        report += "- No entities predicted\n"
    
    report += f"""
## Files Generated for Submission
1. `part1_validation_results.json` - Detailed validation metrics
2. `part1_unannotated_predictions.json` - Predictions in required format
3. `few_shot_tagger.py` - Main model implementation
4. `bio_converter.py` - Data conversion utilities

## Technical Details
- **Tokenization**: Custom mathematical text tokenizer
- **Prompting**: Llama-3.1-8B-Instruct with BIO format examples
- **Fallback**: Enhanced rule-based system for reliability
- **Output Format**: Pandas DataFrame → JSON with fields: fileid, start, end, tag
"""
    
    # Save report
    with open('Part1_Report.md', 'w') as f:
        f.write(report)
    
    print("✅ Report saved to 'Part1_Report.md'")
    print("\n" + "="*50)
    print(report)

if __name__ == "__main__":
    generate_part1_report()