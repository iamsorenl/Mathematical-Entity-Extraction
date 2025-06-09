# Mathematical Entity Extraction

A system for extracting mathematical entities (definitions, theorems, proofs, examples, names, and references) from mathematical textbook content using few-shot prompting with Llama-3.1-8B-Instruct.

## Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Installation](#installation)
* [Quick Start](#quick-start)
* [Usage](#usage)
* [Results](#results)
* [Project Structure](#project-structure)
* [Troubleshooting](#troubleshooting)
* [Reproducing Results](#reproducing-results)
* [Deliverables Status](#deliverables-status)

## Overview

This project implements a baseline system for mathematical entity extraction using:

* **Few-shot prompting** with Meta's Llama-3.1-8B-Instruct
* **BIO tagging** with multi-label support
* **Rule-based fallback** for robustness
* **Mathematical notation handling** for LaTeX expressions

## Features

* Few-shot prompting with 6 crafted examples
* Multi-label BIO tagging (tokens can be both "definition" and "name")
* LaTeX-aware tokenization preserving mathematical expressions
* Rule-based fallback for edge cases
* Comprehensive evaluation with token-level F1 scores
* Detailed error analysis

## Installation

### Prerequisites

* Python 3.8+
* CUDA-capable GPU (16GB+ VRAM recommended for Llama-3.1-8B)
* Git LFS
* Hugging Face account with Llama-3 access

### Setup

```bash
git clone https://github.com/yourusername/Mathematical-Entity-Extraction.git
cd Mathematical-Entity-Extraction

# Using conda
conda create -n math-extraction python=3.10
conda activate math-extraction

# Install dependencies
pip install -r requirements.txt

# Hugging Face login
pip install huggingface_hub
huggingface-cli login

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"
```

## Quick Start

### Run Evaluation Pipeline

```bash
export CUDA_VISIBLE_DEVICES=0

# Run Part 1 evaluation (few-shot baseline)
python run_part1_evaluation.py

# Generate Part 3 error analysis
python generate_part3_analysis.py

# Generate Part 1 report
python generate_part1_report.py
```

### Outputs

* `part1_validation_results.json` — Validation metrics and F1 scores
* `part1_unannotated_predictions.json` — Predictions on unannotated data
* `Part1_Report.md` — Part 1 report
* `Part3_ErrorAnalysis.md` — Error analysis

## Usage

### Run Few-Shot Tagging on Single File

```python
from few_shot_tagger import FewShotMathTagger

tagger = FewShotMathTagger()

with open('your_math_file.mmd', 'r') as f:
    text = f.read()

tokens, tags = tagger.predict_tags(text)
print(f"Found {len([t for t in tags if t != 'O'])} entities")
```

### Convert Annotations to BIO Format

```python
from bio_converter import convert_spans_to_bio

tokens, bio_tags = convert_spans_to_bio(text, annotations)
```

## Results

### Baseline Performance

| Metric          | Score |
| --------------- | ----- |
| Overall F1      | 0.132 |
| Precision       | 0.352 |
| Recall          | 0.279 |
| Files Processed | 3/3   |
| Total Tokens    | 1,911 |

### Per-File Results

| File                | F1 Score |
| ------------------- | -------- |
| Complex Manifolds   | 0.320    |
| Number Theory       | 0.033    |
| Commutative Algebra | 0.119    |

### Unannotated Data

* Files Processed: 3/3
* Total Predictions: 268 entities
* Entity Types Found: definition, theorem, proof, reference, name

## Project Structure

```
Mathematical-Entity-Extraction/
├── README.md
├── requirements.txt
├── few_shot_tagger.py
├── bio_converter.py
├── run_part1_evaluation.py
├── generate_part1_report.py
├── generate_part3_analysis.py
├── DataExploration_Summary.txt
├── Assignment 2.pdf
├── A2-NLP_244/
│   ├── train.json
│   ├── val.json
│   ├── test.txt
│   ├── val.txt
│   ├── math-atlas-paper.pdf
│   ├── file_contents.json
│   └── unannotated_mmds/
│       ├── (mmd) Algebra - Lang.mmd.filtered
│       ├── (mmd) Categorical Homotopy Theory - Riehl.mmd.filtered
│       ├── (mmd) Course of Analytical Geometry - Sharipov.mmd.filtered
├── Part1_Report.md
├── Part3_ErrorAnalysis.md
├── part1_validation_results.json
├── part1_unannotated_predictions.json
├── data_exploration.py
├── debug_baseline.py
├── debug_unannotated.py
├── evaluator.py
├── prompt_designer.py
├── .gitignore
```

## Troubleshooting

### CUDA Out of Memory

```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
# Or use Llama-3.1-1B in few_shot_tagger.py
```

### Hugging Face Authentication

```bash
huggingface-cli login
```

### NLTK Download Issues

```bash
python -c "import nltk; nltk.download('punkt')"
```

## Reproducing Results

1. Install exact environment:

   ```bash
   pip install -r requirements.txt
   ```

2. Set random seeds and config:

   ```python
   temperature=0.1  # In few_shot_tagger.py
   ```

3. Run pipeline:

   ```bash
   python run_part1_evaluation.py
   python generate_part3_analysis.py
   ```

## Deliverables Status

| Deliverable                                                    | Status     |
| -------------------------------------------------------------- | ---------- |
| Source Code                                                    | ✅ Complete |
| Part 1 Report (`Part1_Report.md`)                              | ✅ Complete |
| Part 3 Error Analysis (`Part3_ErrorAnalysis.md`)               | ✅ Complete |
| Validation Results (`part1_validation_results.json`)           | ✅ Complete |
| Unannotated Predictions (`part1_unannotated_predictions.json`) | ✅ Complete |

---
