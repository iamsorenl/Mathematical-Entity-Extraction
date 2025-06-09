import json
import pandas as pd
from collections import Counter
from datetime import datetime

def generate_part1_report():
    """Generate comprehensive Part 1 report"""
    
    print("=== GENERATING PART 1 REPORT ===")
    
    # Load results
    with open('part1_validation_results.json', 'r') as f:
        val_results = json.load(f)
    
    # Load predictions (handle empty case)
    pred_df = pd.read_json('part1_unannotated_predictions.json')
    
    # Handle empty predictions DataFrame
    if pred_df.empty:
        num_files_processed = 0
        num_predictions = 0
        unique_tags = []
    else:
        num_files_processed = pred_df['fileid'].nunique() if 'fileid' in pred_df.columns else 0
        num_predictions = len(pred_df)
        unique_tags = pred_df['tag'].unique().tolist() if 'tag' in pred_df.columns else []
    
    report = f"""# Part 1 Report: Mathematical Entity Extraction Baseline

## Model Description

**Model**: Few-shot prompting with Llama-3.1-8B-Instruct + Rule-based fallback

### Architecture
- **Primary**: Meta Llama-3.1-8B-Instruct with few-shot prompting
- **Fallback**: Enhanced rule-based system using regex patterns
- **Tokenization**: NLTK word tokenizer with LaTeX-aware processing
- **Output**: BIO tagging with multi-label support

### Data Processing
1. **Tokenization**: NLTK word tokenizer preserves mathematical notation
2. **BIO Conversion**: Span annotations → token-level BIO tags
3. **Multi-label Support**: Handles overlapping entities (e.g., "definition,name")
4. **LaTeX Handling**: Preserves mathematical expressions as single tokens

## Validation Scores

### Overall Performance
- **Overall F1 Score**: {val_results['overall_f1']:.3f}
- **Precision**: {val_results['precision']:.3f}  
- **Recall**: {val_results['recall']:.3f}
- **Files Processed**: {val_results['files_processed']}
- **Total Tokens**: {val_results['total_tokens']:,}

### Per-File Results
| File | Tokens | F1 Score |
|------|--------|----------|"""

    # Add per-file results table
    for result in val_results['per_file_results']:
        file_name = result['file_id'].split('/')[-1][:50] + "..." if len(result['file_id']) > 50 else result['file_id']
        report += f"\n| {file_name} | {result['num_tokens']} | {result['f1_score']:.3f} |"

    report += f"""

## Inference Results on Unannotated Data

### Summary Statistics
- **Files Processed**: {num_files_processed}
- **Total Predictions**: {num_predictions}
- **Unique Tags Found**: {len(unique_tags)}
- **Tag Distribution**: {', '.join(unique_tags) if unique_tags else 'No predictions made'}

### Notes on Low Prediction Count
The current model produced {num_predictions} predictions on unannotated files. This suggests:
1. Conservative prediction threshold (rule-based fallback may be too strict)
2. Domain shift between validation and unannotated data
3. Potential need for prompt engineering improvements

## Instructions for Reproduction

### Requirements
```bash
pip install -r requirements.txt
```

### Running Validation Evaluation
```bash
CUDA_VISIBLE_DEVICES=X python run_part1_evaluation.py
```

### Running Inference on Single File
```python
from few_shot_tagger import FewShotMathTagger

tagger = FewShotMathTagger()
with open('your_file.mmd', 'r') as f:
    text = f.read()

tokens, tags = tagger.predict_tags(text)
print(f"Found {{len([t for t in tags if t != 'O'])}} entities")
```

### Output Format
- **Validation**: `part1_validation_results.json`
- **Predictions**: `part1_unannotated_predictions.json` 
- **Format**: `{{"fileid": str, "start": int, "end": int, "tag": str}}`

## Technical Details

### Few-Shot Prompting Strategy
- 6 examples covering all entity types
- BIO format with multi-label support  
- Handles mathematical notation preservation
- Graceful fallback to rule-based system

### Rule-Based Fallback Patterns
- **Definitions**: "define", "definition", "let", "suppose"
- **Theorems**: "theorem", "proposition", "lemma", "corollary"
- **Names**: Mathematical variables and notation
- **References**: Citation patterns and cross-references

### Known Limitations
1. **Token Alignment**: Some alignment issues between true/predicted sequences
2. **Mathematical Notation**: Complex LaTeX expressions may be under-tokenized
3. **Context Window**: Limited by model's context length for very long documents
4. **Domain Adaptation**: May need fine-tuning for specific mathematical domains

---
*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    # Save report
    with open('Part1_Report.md', 'w') as f:
        f.write(report)
    
    print("Report saved to 'Part1_Report.md'")
    print("Summary:")
    print(f"   - Validation F1: {val_results['overall_f1']:.3f}")
    print(f"   - Files processed: {val_results['files_processed']}")
    print(f"   - Predictions made: {num_predictions}")
    
    return report

if __name__ == "__main__":
    generate_part1_report()