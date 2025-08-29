import time
from .base import BaseLLM


class MockLLM(BaseLLM):
    """
    A mock class to simulate responses from a Large Language Model for development
    and testing purposes.
    """

    def __init__(self, model_name: str = "mock-model"):
        super().__init__(model_name)
        print(f"Initialized mock LLM: {self.model_name}")

    def get_response(self, prompt: str) -> str:
        """Simulates generating a response from a prompt."""
        print(f"\n--- Sending to mock LLM '{self.model_name}' ---")
        print(f"Prompt (first 100 chars): '{prompt[:100]}...'")

        # Simulate network latency to mimic real-world conditions
        time.sleep(1)

        # This response is hardcoded for the default GSM8K sample
        mock_response = "Based on the calculation, Sarah should receive $7 in change."
        print(f"Mock Response: '{mock_response}'")
        return mock_response


