
from .base import BaseLLM


class OpenAI_LLM(BaseLLM):
    """
    A class to interact with OpenAI's language models via their API.
    """

    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name)
        if not api_key:
            raise ValueError("OpenAI API key is required. Please set the OPENAI_API_KEY environment variable.")
        try:
            from openai import OpenAI  # type: ignore
        except Exception as import_error:
            raise ImportError(
                "OpenAI client not available. Please install requirements or select a different provider."
            ) from import_error

        self.client = OpenAI(api_key=api_key)
        print(f"Initialized OpenAI LLM with model: {self.model_name}")

    def get_response(self, prompt: str) -> str:
        """Sends a prompt to the specified OpenAI model."""
        print(f"\n--- Sending to OpenAI model '{self.model_name}' ---")
        
        # Add structured output instruction to the prompt
        structured_prompt = prompt + "\n\nPlease provide your final answer in this exact format: #### [final_answer_number]"
        
        # Import settings locally to avoid circular imports
        from core.config import settings
        
        # Set max_tokens based on unlimited mode
        max_tokens = settings.performance.max_tokens if hasattr(settings.performance, 'max_tokens') else 4096
        if hasattr(settings.evaluation, 'unlimited_mode') and settings.evaluation.unlimited_mode:
            max_tokens = 16384
            print(f"🔓 Unlimited mode: Extended max_tokens to {max_tokens}")
        
        temperature = settings.generation.temperature if hasattr(settings.generation, 'temperature') else 0.1
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": structured_prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error calling OpenAI API: {e}"


