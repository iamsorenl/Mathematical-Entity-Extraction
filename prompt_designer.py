def create_few_shot_prompt(text, examples=None):
    """Create few-shot prompt for mathematical entity extraction"""
    
    base_prompt = """
Perform BIO tagging on mathematical text. Assign labels for each token from:
- definition: new concept introductions
- theorem: mathematical claims/statements  
- proof: logical justifications
- example: concept applications
- name: identifiers for objects/theorems
- reference: mentions of previous definitions

Use format: [token] → [tag1,tag2] (multi-label allowed)

Example:
Input: "**Theorem 1.2** (Uniqueness): A least element is unique."
Output: 
[**Theorem] → [theorem,name]
[1.2**] → [theorem,name] 
[(Uniqueness):] → [theorem]
[A] → [theorem]
[least] → [theorem,reference]
[element] → [theorem,reference]
[is] → [theorem]
[unique.] → [theorem]

Now tag this text:
"""
    return base_prompt + text