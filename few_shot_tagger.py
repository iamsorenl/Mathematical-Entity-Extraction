import json
import re
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from bio_converter import tokenize_mathematical_text

class FewShotMathTagger:
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct"):
        self.model_name = model_name
        self.pipe = None
        self.use_llm = True
        
        try:
            print(f"Loading {model_name}...")
            # Llama-specific configuration
            self.pipe = pipeline(
                "text-generation",
                model=model_name,
                max_new_tokens=300,
                do_sample=False,
                temperature=0.1,
                device_map="auto",
                torch_dtype="auto"
            )
            print(f"✅ Successfully loaded: {model_name}")
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            print("Falling back to rule-based system")
            self.use_llm = False
    
    def create_prompt(self, text):
        """Create Llama-optimized instruction prompt"""
        
        # Truncate text if too long
        short_text = text[:800] if len(text) > 800 else text
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert at mathematical text analysis. Your task is to perform BIO tagging on mathematical text using these labels: definition, theorem, proof, example, name, reference.

Output format: [label] for each token, [O] for no label. Multiple labels can be combined like [definition,name].

<|eot_id|><|start_header_id|>user<|end_header_id|>

Example:
Input: **Definition 1.8:** Let D be a connection
Output: [definition,name] [definition,name] [definition] [definition] [definition] [definition] [definition]

Now tag this text:
Input: {short_text}
Output:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return prompt
    
    def predict_tags_with_hf(self, text):
        """Use Llama model for prediction"""
        prompt = self.create_prompt(text)
        
        try:
            response = self.pipe(prompt, max_new_tokens=300)[0]['generated_text']
            
            # Extract just the assistant's response
            assistant_start = response.rfind("<|start_header_id|>assistant<|end_header_id|>")
            if assistant_start != -1:
                assistant_start += len("<|start_header_id|>assistant<|end_header_id|>")
                prediction = response[assistant_start:].strip()
            else:
                # Fallback: look for Output: pattern
                output_start = response.rfind("Output:")
                if output_start != -1:
                    prediction = response[output_start + 7:].strip()
                else:
                    prediction = response.split("Output:")[-1].strip()
            
            return self.parse_bio_output(prediction)
        except Exception as e:
            print(f"Llama model failed: {e}")
            return None
    
    def parse_bio_output(self, output):
        """Parse model output to extract BIO tags"""
        # Look for patterns like [definition,name] [theorem] [O] etc.
        tag_pattern = r'\[([^\]]*)\]'
        matches = re.findall(tag_pattern, output)
        
        tags = []
        for match in matches:
            if match.strip() in ['', 'O']:
                tags.append('O')
            else:
                # Convert to B- format for consistency
                tag_parts = match.split(',')
                bio_tag = 'B-' + ',B-'.join(part.strip() for part in tag_parts)
                tags.append(bio_tag)
        
        return tags
    
    def predict_tags(self, text):
        """Main prediction method with chunking for long texts"""
        tokens = tokenize_mathematical_text(text)
        
        # Handle long texts by chunking
        if len(text) > 1000:
            return self.predict_tags_chunked(text, tokens)
        
        if self.use_llm and self.pipe:
            pred_tags = self.predict_tags_with_hf(text)
            if pred_tags and len(pred_tags) == len(tokens):
                return tokens, pred_tags
            else:
                print(f"Tag count mismatch or no tags: pred={len(pred_tags) if pred_tags else 0}, tokens={len(tokens)}")
        
        # Fallback to enhanced rule-based system
        pred_tags = self.enhanced_rule_based_tagger(tokens)
        return tokens, pred_tags
    
    def predict_tags_chunked(self, text, tokens):
        """Handle long texts by processing in chunks"""
        chunk_size = 800
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        all_pred_tags = []
        
        for chunk in chunks:
            chunk_tokens = tokenize_mathematical_text(chunk)
            
            if self.use_llm and self.pipe:
                chunk_pred_tags = self.predict_tags_with_hf(chunk)
                if not chunk_pred_tags:
                    chunk_pred_tags = self.enhanced_rule_based_tagger(chunk_tokens)
            else:
                chunk_pred_tags = self.enhanced_rule_based_tagger(chunk_tokens)
            
            all_pred_tags.extend(chunk_pred_tags)
        
        # Ensure we return exactly the right number of tags
        return tokens, all_pred_tags[:len(tokens)]
    
    def enhanced_rule_based_tagger(self, tokens):
        """Enhanced rule-based system for fallback"""
        tags = []
        
        for i, token in enumerate(tokens):
            tag = 'O'
            
            # Mathematical statement headers
            if '**' in token:
                if any(kw in token.lower() for kw in ['theorem', 'proposition', 'lemma', 'corollary']):
                    tag = 'B-theorem,B-name'
                elif 'definition' in token.lower():
                    tag = 'B-definition,B-name'
            
            # Proof indicators
            elif (token.lower().startswith('_proof') or 
                  'proof:' in token.lower() or 
                  token.lower() == '_proof:_'):
                tag = 'B-proof'
            
            # Definition indicators
            elif ('define' in token.lower() or 'definition' in token.lower()) and len(token) > 5:
                tag = 'B-definition'
            
            # LaTeX mathematical symbols
            elif re.match(r'\\[a-zA-Z]+', token) and len(token) < 25:
                tag = 'B-name'
            
            # Mathematical concepts
            elif any(term in token.lower() for term in [
                'curvature', 'manifold', 'algebra', 'differentiation', 
                'mapping', 'tensor', 'bundle', 'connection']):
                tag = 'B-reference'
            
            # Named mathematical objects
            elif re.search(r'(Theorem|Proposition|Lemma|Definition)\s*\d+\.\d+', token):
                tag = 'B-name'
            
            tags.append(tag)
        
        return tags