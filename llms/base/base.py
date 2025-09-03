from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """
    Abstract base class for Large Language Model handlers.
    Ensures that any new LLM integration will have a consistent interface.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def get_response(self, prompt: str, task_type: str = "reasoning") -> str:
        """
        Takes a prompt string and returns the LLM's response string.
        
        Args:
            prompt: The input prompt for the model
            task_type: The type of task (reasoning, summarization, classification)
        """
        raise NotImplementedError


