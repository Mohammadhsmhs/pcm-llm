
from .base import BaseLLM


class OpenAI_LLM(BaseLLM):
    """
    A class to interact with OpenAI's language models via their API.
    """

    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name)
        if not api_key or api_key == "YOUR_API_KEY_HERE":
            raise ValueError("OpenAI API key is missing. Please update it in config.py or set OPENAI_API_KEY env var.")
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
        structured_prompt = prompt + "\n\nIMPORTANT: End your response with the final answer in this exact format: #### [final_answer_number]"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": structured_prompt}],
                temperature=0.1,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error calling OpenAI API: {e}"


