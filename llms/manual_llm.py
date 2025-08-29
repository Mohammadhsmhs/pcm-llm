
from .base import BaseLLM


class ManualLLM(BaseLLM):
    """
    A handler that facilitates a manual copy-paste workflow. It prompts the user
    to take the generated prompt, use it in any external chat platform (like ChatGPT),
    and then paste the response back into the terminal.
    """

    def __init__(self, model_name: str = "manual-input"):
        super().__init__(model_name)
        print("Initialized Manual LLM handler.")
        print("The script will pause and wait for you to provide the LLM response.")

    def get_response(self, prompt: str) -> str:
        """Guides the user through the manual copy-paste process."""
        print("\n" + "=" * 50)
        print("--- MANUAL ACTION REQUIRED ---")
        print("1. Copy the following prompt (text between the lines):")
        print("-" * 50)
        print(prompt)
        print("-" * 50)

        print("\n2. Paste the prompt into your chosen chat platform (e.g., ChatGPT).")
        print("3. Copy the full response from the chat platform.")

        # Wait for the user to paste the response.
        response = input("\n4. Paste the response here and press Enter:\n> ")
        print("=" * 50)
        return response.strip()


