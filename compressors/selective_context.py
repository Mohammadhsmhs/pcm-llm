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
        tokens = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        num_original_tokens = tokens.shape[1]
        num_target_tokens = int(num_original_tokens * target_ratio)

        if num_target_tokens >= num_original_tokens:
            return prompt

        with torch.no_grad():
            outputs = self.model(tokens, labels=tokens)
            # Calculate perplexity for each token
            loss_per_token = torch.nn.functional.cross_entropy(
                outputs.logits.view(-1, outputs.logits.size(-1)),
                tokens.view(-1),
                reduction='none'
            ).view(tokens.shape)

        # Self-information is the negative log likelihood (which is the loss)
        self_information = loss_per_token.squeeze().tolist()

        # Create a list of (index, importance_score) tuples
        # We can't remove the first token as it anchors the context
        token_importance = list(enumerate(self_information, 0))
        token_importance[0] = (0, float('inf')) # Set importance to infinity to never remove it

        # Sort tokens by their importance (lower self-information means less important)
        sorted_tokens_by_importance = sorted(token_importance, key=lambda x: x[1])
        
        num_tokens_to_remove = num_original_tokens - num_target_tokens
        
        # Get the indices of the least important tokens to remove
        indices_to_remove = {idx for idx, _ in sorted_tokens_by_importance[:num_tokens_to_remove]}

        # Build the list of token IDs to keep
        original_token_ids = tokens.squeeze().tolist()
        remaining_token_ids = [token_id for i, token_id in enumerate(original_token_ids) if i not in indices_to_remove]
        
        return self.tokenizer.decode(remaining_token_ids, skip_special_tokens=True)
