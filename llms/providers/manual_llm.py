from core.config import LLMConfig
from llms.base.base import BaseLLM
from utils.prompt_utils import add_structured_instructions


class ManualLLM(BaseLLM):
    """
    A handler that facilitates a manual copy-paste workflow. It prompts the user
    to take the generated prompt, use it in any external chat platform (like ChatGPT),
    and then paste the response back into the terminal.
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config.model_name or "manual-input")
        print("Initialized Manual LLM handler.")
        print("The script will pause and wait for you to provide the LLM response.")

    def get_response(self, prompt: str, task_type: str = "reasoning") -> str:
        """Guides the user through the manual copy-paste process."""
        # Add task-specific structured instructions
        structured_prompt = add_structured_instructions(prompt, task_type)

        print("\n" + "=" * 50)
        print("--- MANUAL ACTION REQUIRED ---")
        print("1. Copy the following prompt (text between the lines):")
        print("-" * 50)
        print(structured_prompt)
        print("-" * 50)

        print("\n2. Paste the prompt into your chosen chat platform (e.g., ChatGPT).")
        print("3. Copy the full response from the chat platform.")

        # Wait for the user to paste the response.
        response = input("\n4. Paste the response here and press Enter:\n> ")
        print("=" * 50)
        return response.strip()
