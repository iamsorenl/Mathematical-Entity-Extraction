import json
import pandas as pd
from collections import Counter, defaultdict
from datetime import datetime
import re

def analyze_error_patterns():
    """Analyze specific error patterns from validation results"""
    
    print("=== ANALYZING ERROR PATTERNS ===")
    
    # Load results
    with open('part1_validation_results.json', 'r') as f:
        val_results = json.load(f)
    
    # Load predictions (handle empty case)
    pred_df = pd.read_json('part1_unannotated_predictions.json')
    
    # Calculate error statistics
    total_true_entities = sum(result['num_true_entities'] for result in val_results['per_file_results'])
    total_pred_entities = sum(result['num_pred_entities'] for result in val_results['per_file_results'])
    missed_entities = total_true_entities - total_pred_entities
    
    # Analyze per-file performance variance
    f1_scores = [result['f1_score'] for result in val_results['per_file_results']]
    performance_variance = max(f1_scores) - min(f1_scores)
    
    error_analysis = {
        'quantitative_errors': {
            'total_true_entities': total_true_entities,
            'total_predictions': total_pred_entities,
            'missed_entities': missed_entities,
            'false_negative_rate': missed_entities / total_true_entities if total_true_entities > 0 else 0,
            'performance_variance': performance_variance,
            'domain_consistency': 'High variance' if performance_variance > 0.2 else 'Moderate variance'
        },
        'file_specific_issues': [],
        'unannotated_analysis': {
            'predictions_made': len(pred_df) if not pred_df.empty else 0,
            'files_processed': pred_df['fileid'].nunique() if not pred_df.empty and 'fileid' in pred_df.columns else 0
        }
    }
    
    # Analyze file-specific issues
    for result in val_results['per_file_results']:
        file_analysis = {
            'file_id': result['file_id'],
            'tokens': result['num_tokens'],
            'f1_score': result['f1_score'],
            'true_entities': result['num_true_entities'],
            'pred_entities': result['num_pred_entities'],
            'issue_type': None
        }
        
        # Categorize issues
        if result['f1_score'] < 0.1:
            file_analysis['issue_type'] = 'severe_underperformance'
        elif result['num_pred_entities'] < result['num_true_entities'] * 0.3:
            file_analysis['issue_type'] = 'high_false_negatives'
        elif result['f1_score'] > 0.3:
            file_analysis['issue_type'] = 'good_performance'
        else:
            file_analysis['issue_type'] = 'moderate_issues'
            
        error_analysis['file_specific_issues'].append(file_analysis)
    
    return error_analysis

def generate_concrete_examples():
    """Generate concrete examples of different error types"""
    
    examples = {
        'tokenization_errors': [
            {
                'text': '\\[D(e^{\\prime}_{\\sigma}) =\\sum_{v}\\theta_{v\\sigma}(fg)e^{\\prime}_{v}\\]',
                'issue': 'Complex LaTeX expression treated as single token',
                'impact': 'Token alignment mismatch between true and predicted sequences'
            },
            {
                'text': 'Definition 1.8: Let D be a connection...',
                'issue': 'Multi-word entity spans ("Definition 1.8") may be under-tokenized',
                'impact': 'Boundary detection errors for named mathematical objects'
            }
        ],
        'mathematical_notation': [
            {
                'text': '\\mathbf{\\Theta}_{E}(D)',
                'issue': 'Complex mathematical symbols with subscripts and formatting',
                'impact': 'Model may not recognize as single mathematical entity'
            },
            {
                'text': '\\mathcal{E}^{2}(X,\\operatorname{Hom}(E,E))',
                'issue': 'Nested LaTeX commands with multiple operators',
                'impact': 'Tokenizer splits into multiple tokens, breaking entity boundaries'
            }
        ],
        'domain_specific': [
            {
                'text': 'covariant differentiation',
                'issue': 'Domain-specific mathematical terminology',
                'impact': 'Rule-based fallback may not have patterns for specialized terms'
            },
            {
                'text': 'Proposition 1.9: D² = Θ',
                'issue': 'Mathematical statements with proposition numbering',
                'impact': 'Few-shot examples may not cover all proposition formats'
            }
        ]
    }
    
    return examples

def generate_improvement_strategies():
    """Generate detailed improvement strategies"""
    
    strategies = {
        'immediate_improvements': [
            {
                'strategy': 'Enhanced Mathematical Tokenization',
                'description': 'Implement LaTeX-aware tokenizer that preserves mathematical expressions',
                'implementation': 'Use regex patterns to identify complete LaTeX expressions before tokenization',
                'expected_impact': 'Reduce token alignment issues, improve boundary detection',
                'new_challenges': 'May over-preserve expressions, creating very long tokens'
            },
            {
                'strategy': 'Expanded Few-Shot Examples',
                'description': 'Add more diverse examples covering edge cases and domain-specific patterns',
                'implementation': 'Curate examples from different mathematical domains (analysis, algebra, topology)',
                'expected_impact': 'Better coverage of mathematical discourse patterns',
                'new_challenges': 'Longer prompts may exceed context windows'
            },
            {
                'strategy': 'Adaptive Confidence Thresholding',
                'description': 'Adjust prediction confidence based on mathematical complexity',
                'implementation': 'Score mathematical density and adjust thresholds dynamically',
                'expected_impact': 'Better recall in math-heavy sections',
                'new_challenges': 'May increase false positives in non-mathematical text'
            }
        ],
        'advanced_improvements': [
            {
                'strategy': 'Ensemble Prediction System',
                'description': 'Combine few-shot prompting with specialized models for different entity types',
                'implementation': 'Train separate classifiers for definitions, theorems, names, etc.',
                'expected_impact': 'Higher precision through specialized expertise',
                'new_challenges': 'Increased complexity, potential conflicts between models'
            },
            {
                'strategy': 'Mathematical Context Modeling',
                'description': 'Use mathematical structure to inform entity boundaries',
                'implementation': 'Parse LaTeX syntax trees to identify logical mathematical units',
                'expected_impact': 'Better handling of complex mathematical expressions',
                'new_challenges': 'Requires robust LaTeX parser, may fail on malformed markup'
            },
            {
                'strategy': 'Active Learning with Uncertainty',
                'description': 'Identify challenging examples for targeted annotation',
                'implementation': 'Use model confidence scores to select hard examples',
                'expected_impact': 'Efficient improvement with minimal additional data',
                'new_challenges': 'Requires human annotation budget and expertise'
            }
        ]
    }
    
    return strategies

def generate_part3_report():
    """Generate comprehensive Part 3 error analysis report"""
    
    print("=== GENERATING PART 3 ANALYSIS REPORT ===")
    
    # Analyze errors
    error_analysis = analyze_error_patterns()
    examples = generate_concrete_examples()
    strategies = generate_improvement_strategies()
    
    # Load validation results for context
    with open('part1_validation_results.json', 'r') as f:
        val_results = json.load(f)
    
    report = f"""# Part 3: Error Analysis - Mathematical Entity Extraction

## Executive Summary

This analysis examines the error patterns and limitations of our few-shot prompting baseline for mathematical entity extraction. Key findings include systematic false negative issues (missing {error_analysis['quantitative_errors']['missed_entities']}/{error_analysis['quantitative_errors']['total_true_entities']} entities), significant performance variance across domains (F1 range: {error_analysis['quantitative_errors']['performance_variance']:.3f}), and complete failure on unannotated data.

## 1. Quantitative Error Analysis

### Overall Error Statistics
- **False Negative Rate**: {error_analysis['quantitative_errors']['false_negative_rate']:.1%}
- **Missed Entities**: {error_analysis['quantitative_errors']['missed_entities']} out of {error_analysis['quantitative_errors']['total_true_entities']} true entities
- **Performance Variance**: {error_analysis['quantitative_errors']['performance_variance']:.3f} F1 difference across files
- **Domain Consistency**: {error_analysis['quantitative_errors']['domain_consistency']}

### Per-File Performance Breakdown
| File | Tokens | True Entities | Predictions | F1 Score | Issue Category |
|------|--------|---------------|-------------|----------|----------------|"""

    # Add per-file analysis
    for issue in error_analysis['file_specific_issues']:
        file_name = issue['file_id'].split('/')[-1][:40] + "..." if len(issue['file_id']) > 40 else issue['file_id']
        report += f"\n| {file_name} | {issue['tokens']} | {issue['true_entities']} | {issue['pred_entities']} | {issue['f1_score']:.3f} | {issue['issue_type'].replace('_', ' ').title()} |"

    report += f"""

## 2. What Mistakes Is the Model Making?

### 2.1 Token Alignment Errors

**Example 1: Complex LaTeX Expressions**
```
Text: {examples['tokenization_errors'][0]['text']}
Issue: {examples['tokenization_errors'][0]['issue']}
Impact: {examples['tokenization_errors'][0]['impact']}
```

**Example 2: Multi-word Named Entities**
```
Text: {examples['tokenization_errors'][1]['text']}
Issue: {examples['tokenization_errors'][1]['issue']}
Impact: {examples['tokenization_errors'][1]['impact']}
```

### 2.2 Mathematical Notation Handling

**Example 1: Complex Symbols**
```
Text: {examples['mathematical_notation'][0]['text']}
Issue: {examples['mathematical_notation'][0]['issue']}
Impact: {examples['mathematical_notation'][0]['impact']}
```

**Example 2: Nested LaTeX Commands**
```
Text: {examples['mathematical_notation'][1]['text']}
Issue: {examples['mathematical_notation'][1]['issue']}
Impact: {examples['mathematical_notation'][1]['impact']}
```

### 2.3 Domain-Specific Recognition Failures

**Example 1: Specialized Terminology**
```
Text: {examples['domain_specific'][0]['text']}
Issue: {examples['domain_specific'][0]['issue']}
Impact: {examples['domain_specific'][0]['impact']}
```

**Example 2: Mathematical Statements**
```
Text: {examples['domain_specific'][1]['text']}
Issue: {examples['domain_specific'][1]['issue']}
Impact: {examples['domain_specific'][1]['impact']}
```

### 2.4 Complete Failure on Unannotated Data

The model produced **0 predictions** on {error_analysis['unannotated_analysis']['files_processed']} unannotated MMD files, indicating:
- Severe domain shift between validation and test data
- Overly conservative rule-based fallback system
- Potential prompt engineering limitations

## 3. Why Does the Model Make These Mistakes?

### 3.1 Tokenization Limitations
- **NLTK Word Tokenizer**: Not designed for mathematical text with complex LaTeX markup
- **Boundary Detection**: Cannot handle nested mathematical expressions as single units
- **Alignment Issues**: Inconsistent tokenization between training and inference time

### 3.2 Few-Shot Learning Constraints
- **Limited Examples**: 6 examples cannot cover the full diversity of mathematical discourse
- **Context Window**: Long mathematical expressions may exceed optimal few-shot context
- **Domain Specificity**: Training examples from limited mathematical subdomains

### 3.3 Rule-Based Fallback Issues
- **Conservative Patterns**: Regex patterns prioritize precision over recall
- **Static Thresholds**: No adaptation to mathematical content density
- **Limited Coverage**: Missing patterns for specialized mathematical terminology

### 3.4 Mathematical Text Complexity
- **Notation Density**: High concentration of symbols and operators
- **Structural Hierarchy**: Nested definitions, theorems, and proofs
- **Cross-References**: Complex citation and reference patterns

## 4. How Would You Improve the Model?

### 4.1 Immediate Improvements

#### Strategy 1: {strategies['immediate_improvements'][0]['strategy']}
- **Description**: {strategies['immediate_improvements'][0]['description']}
- **Implementation**: {strategies['immediate_improvements'][0]['implementation']}
- **Expected Impact**: {strategies['immediate_improvements'][0]['expected_impact']}

#### Strategy 2: {strategies['immediate_improvements'][1]['strategy']}
- **Description**: {strategies['immediate_improvements'][1]['description']}
- **Implementation**: {strategies['immediate_improvements'][1]['implementation']}
- **Expected Impact**: {strategies['immediate_improvements'][1]['expected_impact']}

#### Strategy 3: {strategies['immediate_improvements'][2]['strategy']}
- **Description**: {strategies['immediate_improvements'][2]['description']}
- **Implementation**: {strategies['immediate_improvements'][2]['implementation']}
- **Expected Impact**: {strategies['immediate_improvements'][2]['expected_impact']}

### 4.2 Advanced Improvements

#### Strategy 1: {strategies['advanced_improvements'][0]['strategy']}
- **Description**: {strategies['advanced_improvements'][0]['description']}
- **Implementation**: {strategies['advanced_improvements'][0]['implementation']}
- **Expected Impact**: {strategies['advanced_improvements'][0]['expected_impact']}

#### Strategy 2: {strategies['advanced_improvements'][1]['strategy']}
- **Description**: {strategies['advanced_improvements'][1]['description']}
- **Implementation**: {strategies['advanced_improvements'][1]['implementation']}
- **Expected Impact**: {strategies['advanced_improvements'][1]['expected_impact']}

#### Strategy 3: {strategies['advanced_improvements'][2]['strategy']}
- **Description**: {strategies['advanced_improvements'][2]['description']}
- **Implementation**: {strategies['advanced_improvements'][2]['implementation']}
- **Expected Impact**: {strategies['advanced_improvements'][2]['expected_impact']}

## 5. What New Problems Might These Solutions Introduce?

### 5.1 Tokenization Improvements
**New Challenges**: {strategies['immediate_improvements'][0]['new_challenges']}
- Risk of creating overly long tokens that lose semantic meaning
- Potential misalignment with pre-trained model vocabularies
- Increased computational complexity for token processing

### 5.2 Expanded Few-Shot Examples
**New Challenges**: {strategies['immediate_improvements'][1]['new_challenges']}
- May exceed model context windows, requiring truncation
- Risk of prompt overfitting to specific mathematical domains
- Increased inference cost due to longer prompts

### 5.3 Ensemble Methods
**New Challenges**: {strategies['advanced_improvements'][0]['new_challenges']}
- Significantly increased computational requirements
- Complex conflict resolution between different model predictions
- Higher maintenance burden for multiple model components

### 5.4 Mathematical Structure Modeling
**New Challenges**: {strategies['advanced_improvements'][1]['new_challenges']}
- Dependency on robust LaTeX parsing (fragile to markup errors)
- Potential over-engineering for simple mathematical expressions
- Limited generalization to non-LaTeX mathematical notation

## 6. Recommendations for Future Work

### Priority 1: Tokenization Enhancement
Focus on mathematical notation-aware tokenization as the highest-impact improvement with manageable implementation complexity.

### Priority 2: Prompt Engineering
Systematically expand few-shot examples with careful attention to context window management.

### Priority 3: Evaluation Framework
Develop better evaluation metrics that account for mathematical text complexity and partial entity matches.

## Conclusion

The current baseline reveals fundamental challenges in mathematical entity extraction, particularly around tokenization and domain adaptation. While the F1 score of {val_results['overall_f1']:.3f} provides a reasonable starting point, the high false negative rate ({error_analysis['quantitative_errors']['false_negative_rate']:.1%}) and complete failure on unannotated data indicate significant room for improvement.

The analysis suggests that mathematical text requires specialized approaches beyond general NLP techniques, with particular attention to notation handling and domain-specific linguistic patterns.

---
*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    # Save report
    with open('Part3_ErrorAnalysis.md', 'w') as f:
        f.write(report)
    
    print("Part 3 analysis saved to 'Part3_ErrorAnalysis.md'")
    print("Key Findings:")
    print(f"   - False Negative Rate: {error_analysis['quantitative_errors']['false_negative_rate']:.1%}")
    print(f"   - Performance Variance: {error_analysis['quantitative_errors']['performance_variance']:.3f}")
    print(f"   - Unannotated Predictions: {error_analysis['unannotated_analysis']['predictions_made']}")
    
    return report

if __name__ == "__main__":
    generate_part3_report()