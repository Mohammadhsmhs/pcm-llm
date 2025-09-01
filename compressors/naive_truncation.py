from transformers import AutoTokenizer
from .base import BaseCompressor


class NaiveTruncationCompressor(BaseCompressor):
    """
    Naive Truncation Compressor that simply cuts off tokens from the end.

    This is a simple baseline compression method that:
    - Tokenizes the input text
    - Keeps only the first N tokens based on target_ratio
    - Decodes back to text

    Fast and simple, but may cut off important information at the end.
    """

    def __init__(self, model_name: str = None):
        """
        Initialize the naive truncation compressor.

        Args:
            model_name: HuggingFace model name for tokenizer (default: from config)
        """
        # Import config here to avoid circular imports
        from config import NAIVE_TRUNCATION_MODEL

        self.model_name = model_name or NAIVE_TRUNCATION_MODEL
        print(f"Initializing NaiveTruncationCompressor with tokenizer '{self.model_name}'...")
        self.tokenizer = None
        self._ensure_tokenizer()
        print("NaiveTruncationCompressor initialized successfully.")

    def _ensure_tokenizer(self):
        """Lazy loading of tokenizer."""
        if self.tokenizer is None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                # Set pad token if not exists
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            except Exception as e:
                raise RuntimeError(f"Failed to load tokenizer for {self.model_name}: {e}")

    def compress(self, prompt: str, target_ratio: float) -> str:
        """
        Perform naive truncation on the input text.

        Args:
            prompt: Input text to compress
            target_ratio: Target compression ratio (e.g., 0.8 = keep 80% of tokens)

        Returns:
            Compressed text as string
        """
        self._ensure_tokenizer()

        # Tokenize the input text
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        original_length = len(input_ids)

        # Calculate target length based on ratio
        target_length = max(1, int(original_length * target_ratio))

        # Perform naive truncation (keep from beginning)
        if original_length <= target_length:
            truncated_ids = input_ids
        else:
            truncated_ids = input_ids[:target_length]

        # Decode back to text
        truncated_text = self.tokenizer.decode(truncated_ids, skip_special_tokens=True)

        return truncated_text
