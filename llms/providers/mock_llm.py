import time

from core.config import LLMConfig
from llms.base.base import BaseLLM


class MockLLM(BaseLLM):
    """
    A mock class to simulate responses from a Large Language Model for development
    and testing purposes.
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config.model_name or "mock-model")
        print(f"Initialized mock LLM: {self.model_name}")

    def get_response(self, prompt: str, task_type: str = "reasoning") -> str:
        """Simulates generating a response from a prompt."""
        print(f"\n--- Sending to mock LLM '{self.model_name}' ---")
        print(f"Prompt (first 100 chars): '{prompt[:100]}...'")

        # Simulate network latency to mimic real-world conditions
        time.sleep(1)

        # Return task-specific mock responses
        if task_type == "reasoning":
            mock_response = (
                "Based on the calculation, Sarah should receive $7 in change. #### 7"
            )
        elif task_type == "classification":
            mock_response = "This review expresses positive sentiment about the movie. #### positive"
        elif task_type == "summarization":
            mock_response = "The article discusses the growing acceptance of medical marijuana and its potential benefits. #### The article explores the revolution in medical marijuana acceptance, highlighting changing public opinion, scientific research, and policy shifts toward legalization."
        else:
            mock_response = "Mock response for unknown task type. #### mock_answer"

        print(f"Mock Response: '{mock_response}'")
        return mock_response
