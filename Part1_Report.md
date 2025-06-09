# Part 1 Report: Mathematical Entity Extraction Baseline

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
- **Overall F1 Score**: 0.132
- **Precision**: 0.352  
- **Recall**: 0.279
- **Files Processed**: 3
- **Total Tokens**: 1,911

### Per-File Results
| File | Tokens | F1 Score |
|------|--------|----------|
| (mmd) Complex Manifolds - Differential Analysis on... | 651 | 0.320 |
| (mmd) Number Theory - Number Theory - An Introduct... | 769 | 0.033 |
| (mmd) A Term of Commutative Algebra - Altman.mmd-v... | 491 | 0.119 |

## Inference Results on Unannotated Data

### Summary Statistics
- **Files Processed**: 3
- **Total Predictions**: 268
- **Unique Tags Found**: 5
- **Tag Distribution**: definition, reference, theorem, name, proof

### Notes on Low Prediction Count
The current model produced 268 predictions on unannotated files. This suggests:
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
print(f"Found {len([t for t in tags if t != 'O'])} entities")
```

### Output Format
- **Validation**: `part1_validation_results.json`
- **Predictions**: `part1_unannotated_predictions.json` 
- **Format**: `{"fileid": str, "start": int, "end": int, "tag": str}`

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
*Generated on 2025-06-09 01:37:13*
