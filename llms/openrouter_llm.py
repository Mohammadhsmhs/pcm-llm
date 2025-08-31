from .base import BaseLLM


class OpenRouter_LLM(BaseLLM):
    """
    A class to interact with OpenRouter's language models via their API.
    Uses OpenAI-compatible interface with custom base URL and proper headers for rankings.
    
    Features:
    - Full OpenRouter API compatibility
    - Automatic referrer headers for better rankings
    - Configurable model selection (DeepSeek R1, etc.)
    - High token limits for long-form content
    """

    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name)
        if not api_key or api_key == "YOUR_API_KEY_HERE" or api_key == "your_openrouter_api_key_here":
            raise ValueError("OpenRouter API key is missing. Please update it in api_keys.py or set OPENROUTER_API_KEY env var.")
        try:
            from openai import OpenAI  # type: ignore
        except Exception as import_error:
            raise ImportError(
                "OpenAI client not available. Please install requirements or select a different provider."
            ) from import_error

        # Use OpenRouter's base URL instead of OpenAI's
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        print(f"Initialized OpenRouter LLM with model: {self.model_name}")

    def get_response(self, prompt: str) -> str:
        """Sends a prompt to the specified OpenRouter model."""
        print(f"\n--- Sending to OpenRouter model '{self.model_name}' ---")

        try:
            response = self.client.chat.completions.create(
                extra_headers={
                    # "HTTP-Referer": "https://github.com/Mohammadhsmhs/pcm-llm",  # GitHub repo for rankings
                    "X-Title": "PCM-LLM Benchmark System",  # Project name for rankings
                },
                extra_body={},  # For future extensibility
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,  # More creative responses
                max_tokens=32768,  # Very high limit for long responses
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ OpenRouter API Error: {e}")
            return f"Error calling OpenRouter API: {e}"
