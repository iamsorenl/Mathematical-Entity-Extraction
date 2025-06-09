# Mathematical Entity Extraction

A comprehensive system for extracting mathematical entities (definitions, theorems, proofs, examples, names, and references) from mathematical textbook content using few-shot prompting with Llama-3.1-8B-Instruct.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

## 🔍 Overview

This project implements a baseline system for mathematical entity extraction using:
- **Few-shot prompting** with Meta's Llama-3.1-8B-Instruct
- **BIO tagging** with multi-label support
- **Rule-based fallback** for robustness
- **Mathematical notation handling** for LaTeX expressions

The system achieves **F1: 0.203** on validation data with **precision: 0.381** and **recall: 0.300**.

## ✨ Features

- 🤖 **Few-shot prompting** with 6 carefully crafted examples
- 🏷️ **Multi-label BIO tagging** (e.g., tokens can be both "definition" and "name")
- 📐 **LaTeX-aware tokenization** preserving mathematical expressions
- 🛡️ **Rule-based fallback** for cases when LLM fails
- 📊 **Comprehensive evaluation** with token-level F1 scores
- 📝 **Detailed error analysis** and improvement recommendations

## 🛠️ Installation

### Prerequisites

- **Python 3.8+**
- **CUDA-capable GPU** (recommended, 16GB+ VRAM for Llama-3.1-8B)
- **Git LFS** for model downloads
- **Hugging Face account** for model access

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/Mathematical-Entity-Extraction.git
cd Mathematical-Entity-Extraction
```

### Step 2: Create Python Environment

```bash
# Using conda (recommended)
conda create -n math-extraction python=3.10
conda activate math-extraction

# OR using venv
python -m venv math-extraction
source math-extraction/bin/activate  # Linux/Mac
# math-extraction\Scripts\activate    # Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Hugging Face CLI

```bash
# Install Hugging Face CLI (if not already included)
pip install huggingface_hub

# Login to Hugging Face (required for Llama access)
huggingface-cli login
```

**Note**: You'll need to:
1. Create a Hugging Face account at https://huggingface.co/
2. Request access to Meta's Llama models at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
3. Get approval (usually instant)
4. Use your HF token when prompted by `huggingface-cli login`

### Step 5: Download NLTK Data

```bash
python -c "import nltk; nltk.download('punkt')"
```

## 🚀 Quick Start

### Run Complete Evaluation Pipeline

```bash
# Set GPU device (adjust as needed)
export CUDA_VISIBLE_DEVICES=0

# Run Part 1 evaluation (few-shot baseline)
python run_part1_evaluation.py

# Generate Part 3 error analysis
python generate_part3_analysis.py

# Generate Part 1 report
python generate_part1_report.py
```

### Expected Outputs

After running the pipeline, you'll have:
- `part1_validation_results.json` - Validation metrics and F1 scores
- `part1_unannotated_predictions.json` - Predictions on unannotated data
- `Part1_Report.md` - Comprehensive Part 1 report
- `Part3_ErrorAnalysis.md` - Detailed error analysis

## 📖 Usage

### Individual Components

#### 1. Run Few-Shot Tagging on Single File

```python
from few_shot_tagger import FewShotMathTagger

# Initialize tagger
tagger = FewShotMathTagger()

# Load mathematical text
with open('your_math_file.mmd', 'r') as f:
    text = f.read()

# Get predictions
tokens, tags = tagger.predict_tags(text)

# Count entities found
entity_count = len([t for t in tags if t != 'O'])
print(f"Found {entity_count} entities")
```

#### 2. Convert Annotations to BIO Format

```python
from bio_converter import convert_spans_to_bio

# Convert span annotations to BIO tags
tokens, bio_tags = convert_spans_to_bio(text, annotations)
```

#### 3. Evaluation Metrics

```python
from sklearn.metrics import f1_score, classification_report

# Calculate F1 score
f1 = f1_score(true_tags, pred_tags, average='weighted')
print(f"F1 Score: {f1:.3f}")
```

### Configuration Options

#### GPU Memory Management

```bash
# For smaller GPUs, try these settings:
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# For multiple GPUs
export CUDA_VISIBLE_DEVICES=0,1
```

#### Model Parameters

Edit `few_shot_tagger.py` to adjust:
- **Temperature**: `temperature=0.1` (default, lower = more deterministic)
- **Max tokens**: `max_new_tokens=1000` (adjust for longer texts)
- **Model**: Change `model_name` to use different models

## 📊 Results

### Baseline Performance

| Metric | Score |
|--------|-------|
| **Overall F1** | 0.203 |
| **Precision** | 0.381 |
| **Recall** | 0.300 |
| **Files Processed** | 3/3 (100%) |

### Per-File Results

| File | Tokens | F1 Score | Notes |
|------|--------|----------|-------|
| Complex Manifolds | 651 | 0.350 | Good performance |
| Number Theory | 454 | 0.015 | Severe underperformance |
| Commutative Algebra | 347 | 0.225 | High false negatives |

### Key Findings

- ✅ **Conservative but accurate**: High precision (38.1%) when confident
- ⚠️ **High false negative rate**: Missing 59% of true entities
- 🔍 **Domain variance**: Performance varies significantly across mathematical domains
- 🚫 **Zero unannotated predictions**: Suggests domain shift or overly strict thresholds

## 📁 Project Structure

```
Mathematical-Entity-Extraction/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── few_shot_tagger.py                # Main few-shot prompting implementation
├── bio_converter.py                  # BIO tagging utilities
├── run_part1_evaluation.py           # Complete evaluation pipeline
├── generate_part1_report.py          # Part 1 report generator
├── generate_part3_analysis.py        # Part 3 error analysis
├── custom.bib                         # Bibliography for LaTeX report
├── assignment_report.tex              # LaTeX academic report
├── A2-NLP_244/                       # Dataset directory
│   ├── train.json                    # Training annotations
│   ├── val.json                      # Validation annotations
│   ├── file_contents.json            # Text content mapping
│   └── unannotated_mmds/             # Unannotated MMD files
├── outputs/                          # Generated results
│   ├── part1_validation_results.json
│   ├── part1_unannotated_predictions.json
│   ├── Part1_Report.md
│   └── Part3_ErrorAnalysis.md
└── logs/                             # Execution logs
```

## 🔧 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

```bash
# Reduce batch size or use smaller model
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Consider using Llama-3.1-1B instead:
# Edit few_shot_tagger.py: model_name = "meta-llama/Llama-3.1-1B-Instruct"
```

#### 2. Hugging Face Authentication

```bash
# Re-login to Hugging Face
huggingface-cli logout
huggingface-cli login

# Verify access to Llama models
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')"
```

#### 3. NLTK Download Issues

```bash
# Manual NLTK data download
python -c "
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
"
```

#### 4. Permission Errors on Dataset

```bash
# Check file permissions
ls -la A2-NLP_244/
chmod +r A2-NLP_244/*.json
```

### Performance Optimization

#### For Faster Development/Testing

1. **Limit files processed**: Edit `run_part1_evaluation.py` line 103
   ```python
   if processed_files >= 2:  # Process only 2 files for testing
       break
   ```

2. **Use smaller model**: In `few_shot_tagger.py`
   ```python
   model_name = "meta-llama/Llama-3.1-1B-Instruct"  # Faster, less memory
   ```

3. **CPU-only mode**: Set `device="cpu"` in `few_shot_tagger.py`

### Validation

#### Verify Installation

```bash
# Test core components
python -c "
from few_shot_tagger import FewShotMathTagger
from bio_converter import convert_spans_to_bio
print('✅ All imports successful')
"

# Test model loading (requires GPU/significant RAM)
python -c "
from few_shot_tagger import FewShotMathTagger
tagger = FewShotMathTagger()
print('✅ Model loaded successfully')
"
```

#### Check Dataset

```bash
# Verify dataset structure
python -c "
import json
import pandas as pd
val_df = pd.read_json('A2-NLP_244/val.json')
print(f'✅ Validation set: {len(val_df)} annotations')
print(f'✅ Unique files: {val_df[\"fileid\"].nunique()}')
"
```

## 🎯 Reproducing Results

To exactly reproduce the reported results:

1. **Use the exact environment**:
   ```bash
   pip install -r requirements.txt  # Exact versions
   ```

2. **Set random seeds** (already configured in code):
   ```python
   temperature=0.1  # Deterministic generation
   ```

3. **Use same GPU setup**:
   ```bash
   export CUDA_VISIBLE_DEVICES=0  # Single GPU
   ```

4. **Run complete pipeline**:
   ```bash
   python run_part1_evaluation.py
   python generate_part3_analysis.py
   ```
