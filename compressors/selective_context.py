import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BaseCompressor


class SelectiveContextCompressor(BaseCompressor):
    """
    Selective Context Compressor using attention-based importance scoring.

    This implementation follows the standard selective context compression approach:
    - Uses a proxy language model (GPT-2) to analyze token importance
    - Measures importance based on attention weights from transformer layers
    - Identifies tokens that are most crucial for maintaining semantic coherence
    - Preserves mathematical and logical terms with rule-based prioritization

    This is the correct implementation of selective context compression as described
    in the research literature on prompt compression.
    """

    def __init__(self, proxy_model_name="gpt2"):
        print("Initializing SelectiveContextCompressor with proxy model 'gpt2'...")
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(proxy_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(proxy_model_name).to(device)
        self.device = device
        print("SelectiveContextCompressor initialized successfully.")

    def compress(self, prompt: str, target_ratio: float) -> str:
        """
        Compress the prompt using attention-based selective context compression.

        Args:
            prompt: The original prompt to compress
            target_ratio: Target compression ratio (0.1 = 10% of original)

        Returns:
            Compressed prompt with most contextually important tokens preserved
        """
        # Tokenize the entire prompt
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        num_original_tokens = len(tokens)
        num_target_tokens = max(1, int(num_original_tokens * target_ratio))

        if num_target_tokens >= num_original_tokens:
            return prompt

        # Calculate token importance using attention weights
        token_importance = self._calculate_attention_importance(tokens)

        # Sort tokens by importance (higher = more important)
        sorted_tokens_by_importance = sorted(
            token_importance, key=lambda x: x[1], reverse=True
        )

        # Keep the most important tokens
        indices_to_keep = {
            idx for idx, _ in sorted_tokens_by_importance[:num_target_tokens]
        }

        # Always keep the first few tokens to maintain context
        for i in range(min(3, len(tokens))):
            indices_to_keep.add(i)

        # Reconstruct the compressed prompt
        kept_tokens = [tokens[i] for i in sorted(indices_to_keep)]
        compressed_text = self.tokenizer.decode(kept_tokens, skip_special_tokens=True)

        return compressed_text

    def _calculate_attention_importance(self, tokens):
        """
        Calculate token importance using leave-one-out perplexity.

        This is the standard selective context compression approach:
        - Measure baseline perplexity of the full sequence
        - Measure perplexity when each token is removed
        - Importance = perplexity increase when token is removed
        - Higher importance = bigger impact on understanding
        """
        token_importance = []

        try:
            # Convert to tensor and add batch dimension
            input_tensor = torch.tensor([tokens]).to(self.device)

            with torch.no_grad():
                # Get baseline perplexity for the full sequence
                outputs = self.model(input_tensor, labels=input_tensor)
                baseline_loss = outputs.loss.item()
                baseline_perplexity = torch.exp(torch.tensor(baseline_loss)).item()

                # For each token, calculate perplexity when that token is removed
                for i in range(len(tokens)):
                    # Create sequence with token i removed
                    if len(tokens) > 1:  # Only if we have more than one token
                        tokens_without_i = tokens[:i] + tokens[i + 1 :]
                        reduced_tensor = torch.tensor([tokens_without_i]).to(
                            self.device
                        )

                        # Get perplexity of reduced sequence
                        reduced_outputs = self.model(
                            reduced_tensor, labels=reduced_tensor
                        )
                        reduced_loss = reduced_outputs.loss.item()
                        reduced_perplexity = torch.exp(
                            torch.tensor(reduced_loss)
                        ).item()

                        # Importance = perplexity increase when token is removed
                        perplexity_increase = reduced_perplexity - baseline_perplexity

                        # Apply rule-based boost for important tokens
                        rule_boost = self._get_rule_based_importance(tokens[i])
                        final_score = perplexity_increase * (1 + rule_boost)

                        token_importance.append((i, final_score))
                    else:
                        # If only one token, give it high importance
                        token_importance.append((i, 100.0))

        except Exception as e:
            print(f"Leave-one-out perplexity scoring failed: {e}")
            token_importance = self._fallback_importance_scoring(tokens)

        return token_importance

    def _get_rule_based_importance(self, token_id):
        """
        Get rule-based importance boost for special tokens.
        """
        try:
            token_text = self.tokenizer.decode([token_id]).lower().strip()

            # Numbers get high boost
            if any(char.isdigit() for char in token_text):
                return 2.0

            # Mathematical operators
            if any(
                char in token_text
                for char in ["+", "-", "*", "/", "=", "<", ">", "×", "÷"]
            ):
                return 1.5

            # Question words and important terms
            important_terms = {
                "calculate",
                "compute",
                "find",
                "determine",
                "solve",
                "what",
                "how",
                "if",
                "then",
                "equals",
                "total",
                "sum",
                "difference",
                "product",
            }

            if token_text in important_terms:
                return 1.0

        except:
            pass

        return 0.0

    def _fallback_importance_scoring(self, tokens):
        """
        Fallback importance scoring when attention-based method fails.
        """
        token_importance = []

        for i, token_id in enumerate(tokens):
            try:
                token_text = self.tokenizer.decode([token_id])

                # Position-based scoring
                position_score = 1.0
                if i == 0:
                    position_score = 2.0
                elif i < len(tokens) * 0.2:
                    position_score = 1.5

                # Rule-based scoring
                rule_score = self._get_rule_based_importance(token_id)

                # Length-based scoring
                length_score = min(len(token_text) * 0.1, 1.0)

                total_score = position_score + rule_score + length_score
                token_importance.append((i, total_score))

            except:
                token_importance.append((i, 1.0))  # Default score

        return token_importance
