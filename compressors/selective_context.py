import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .base import BaseCompressor

class SelectiveContextCompressor(BaseCompressor):
    def __init__(self, proxy_model_name='gpt2'):
        print("Initializing SelectiveContextCompressor with proxy model 'gpt2'...")
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(proxy_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(proxy_model_name).to(device)
        self.device = device
        print("SelectiveContextCompressor initialized successfully.")

    def compress(self, prompt: str, target_ratio: float) -> str:
        # Split into words and work at word level instead of subword tokens
        words = prompt.split()
        num_original_words = len(words)
        num_target_words = max(1, int(num_original_words * target_ratio))

        if num_target_words >= num_original_words:
            return prompt

        # For each word, calculate its importance by encoding it and getting perplexity
        word_importance = []
        for i, word in enumerate(words):
            # Give higher importance to numbers and key mathematical terms
            if any(char.isdigit() for char in word):
                # Numbers are very important - give them maximum importance
                word_importance.append((i, float('inf')))
                continue
            elif word.lower() in ['calculate', 'compute', 'find', 'determine', 'solve', 'what', 'how', 'if', 'then', 'and', 'or', 'but', 'the', 'a', 'an']:
                # Key instruction words and connecting words are very important
                word_importance.append((i, 100.0))
                continue
            
            # For regular words, use perplexity-based importance
            word_tokens = self.tokenizer.encode(word, add_special_tokens=False)
            if not word_tokens:
                # Handle empty tokens (punctuation, etc.)
                word_importance.append((i, 1.0))  # Low but not zero importance for punctuation
                continue

            word_tensor = torch.tensor([word_tokens]).to(self.device)

            with torch.no_grad():
                outputs = self.model(word_tensor, labels=word_tensor)
                # Calculate average perplexity for this word
                loss = torch.nn.functional.cross_entropy(
                    outputs.logits.view(-1, outputs.logits.size(-1)),
                    word_tensor.view(-1)
                )
                word_importance.append((i, loss.item()))

        # Sort words by importance (higher loss = more important)
        sorted_words_by_importance = sorted(word_importance, key=lambda x: x[1], reverse=True)

        # Keep the most important words
        num_words_to_keep = num_target_words
        indices_to_keep = {idx for idx, _ in sorted_words_by_importance[:num_words_to_keep]}

        # Always keep the first word to maintain context
        indices_to_keep.add(0)

        # Reconstruct the compressed prompt
        remaining_words = [word for i, word in enumerate(words) if i in indices_to_keep]

        return ' '.join(remaining_words)
