import json
import re
import time
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from bio_converter import tokenize_mathematical_text

class FewShotMathTagger:
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct"):
        self.model_name = model_name
        self.pipe = None
        self.use_llm = True
        
        # Silence warnings
        import warnings
        import logging
        import os
        warnings.filterwarnings('ignore')
        logging.getLogger("transformers").setLevel(logging.ERROR)
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        
        try:
            print(f"Loading {model_name}...")
            self.pipe = pipeline(
                "text-generation",
                model=model_name,
                max_new_tokens=300,
                do_sample=False,
                temperature=0.1,
                device_map="auto",
                torch_dtype="auto"
            )
            print(f"Successfully loaded: {model_name}")
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
        
        # Handle long texts by chunking (increased threshold)
        if len(text) > 3000:  # ← Changed from 1000 to 3000
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
        """Handle long texts with efficient batch processing"""
        chunk_size = 2000  # Smaller chunks for better GPU utilization
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        print(f"Processing {len(chunks)} chunks in batches...")
        
        # Batch process chunks
        all_prompts = [self.create_prompt(chunk) for chunk in chunks]
        
        try:
            # Process all prompts in one batch call
            print("  🚀 Running batch inference...")
            responses = self.pipe(all_prompts, max_new_tokens=300)
            
            all_pred_tags = []
            for i, (chunk, response) in enumerate(zip(chunks, responses)):
                chunk_tokens = tokenize_mathematical_text(chunk)
                
                # Extract assistant response
                if isinstance(response, list):
                    generated = response[0]['generated_text']
                else:
                    generated = response['generated_text']
                
                assistant_start = generated.rfind("<|start_header_id|>assistant<|end_header_id|>")
                if assistant_start != -1:
                    assistant_start += len("<|start_header_id|>assistant<|end_header_id|>")
                    prediction = generated[assistant_start:].strip()
                else:
                    prediction = generated.split("Output:")[-1].strip()
                
                chunk_pred_tags = self.parse_bio_output(prediction)
                
                # Fallback to rule-based if parsing fails
                if not chunk_pred_tags or len(chunk_pred_tags) != len(chunk_tokens):
                    chunk_pred_tags = self.enhanced_rule_based_tagger(chunk_tokens)
                
                all_pred_tags.extend(chunk_pred_tags)
            
            return tokens, all_pred_tags[:len(tokens)]
            
        except Exception as e:
            print(f"  Batch processing failed: {e}")
            print("  Falling back to sequential processing...")
            return self.predict_tags_chunked_sequential(text, tokens)

    def predict_tags_chunked_sequential(self, text, tokens):
        """Fallback sequential processing"""
        chunk_size = 2000
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        all_pred_tags = []
        for i, chunk in enumerate(chunks):
            print(f"  Processing chunk {i+1}/{len(chunks)}...")
            chunk_tokens = tokenize_mathematical_text(chunk)
            
            if len(chunk) < 3000:  # Only use LLM for reasonable sizes
                chunk_pred_tags = self.predict_tags_with_hf(chunk)
                if not chunk_pred_tags:
                    chunk_pred_tags = self.enhanced_rule_based_tagger(chunk_tokens)
            else:
                chunk_pred_tags = self.enhanced_rule_based_tagger(chunk_tokens)
            
            all_pred_tags.extend(chunk_pred_tags)
        
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
    
    def debug_predict_tags(self, text, max_chars=50000):
        """Debug version with detailed logging"""
        
        if len(text) > max_chars:
            text = text[:max_chars]
            print(f"    Truncated to {max_chars:,} chars")
        
        # Step 1: Tokenization
        print(f"    Tokenizing {len(text):,} chars...")
        start_time = time.time()
        tokens = tokenize_mathematical_text(text)
        elapsed = time.time() - start_time
        print(f"    Tokenized to {len(tokens):,} tokens in {elapsed:.1f}s")
        
        # Step 2: Check if we should use LLM
        if self.use_llm and len(tokens) < 2000:  # Only use LLM for reasonable sizes
            print(f"    Trying LLM inference...")
            start_time = time.time()
            
            try:
                # Create prompt
                prompt = self.create_prompt(text)
                print(f"    Created prompt: {len(prompt):,} chars")
                
                # LLM inference with monitoring
                print(f"    Running Llama inference...")
                response = self.pipe(prompt, max_new_tokens=300)[0]['generated_text']
                
                elapsed = time.time() - start_time
                print(f"    LLM completed in {elapsed:.1f}s")
                
                # Parse response
                tags = self.parse_bio_output(response)
                if len(tags) == len(tokens):
                    print(f"    LLM tags aligned perfectly")
                    return tokens, tags
                else:
                    print(f"    LLM tag mismatch: {len(tags)} vs {len(tokens)}")
            
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"    LLM failed after {elapsed:.1f}s: {e}")
        
        # Step 3: Rule-based fallback
        print(f"    🔧 Using rule-based fallback...")
        start_time = time.time()
        tags = self.enhanced_rule_based_tagger(tokens)
        elapsed = time.time() - start_time
        print(f"    Rule-based completed in {elapsed:.1f}s")
        
        return tokens, tags