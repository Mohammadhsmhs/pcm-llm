
from llms.base.base import BaseLLM
from core.config import LLMConfig


class OpenAILLM(BaseLLM):
    """
    A class to interact with OpenAI's language models via their API.
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config.model_name)
        self.config = config
        if not self.config.api_key:
            raise ValueError("OpenAI API key is required. Please set the OPENAI_API_KEY environment variable.")
        try:
            from openai import OpenAI
        except ImportError as import_error:
            raise ImportError(
                "OpenAI client not available. Please install requirements or select a different provider."
            ) from import_error

        self.client = OpenAI(api_key=self.config.api_key)
        print(f"Initialized OpenAI LLM with model: {self.model_name}")

    def get_response(self, prompt: str) -> str:
        """Sends a prompt to the specified OpenAI model."""
        print(f"\n--- Sending to OpenAI model '{self.model_name}' ---")
        
        # Add structured output instruction to the prompt
        structured_prompt = prompt + "\n\nPlease provide your final answer in this exact format: #### [final_answer_number]"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": structured_prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error calling OpenAI API: {e}"


