from abc import ABC, abstractmethod


class BaseCompressor(ABC):
    """
    Abstract base class for all prompt compression methods.
    This enforces a common interface for all compression algorithms.
    """

    @abstractmethod
    def compress(self, prompt: str, target_ratio: float) -> str:
        """
        Compresses a given prompt to a target ratio of its original size.

        Args:
            prompt: The original prompt string.
            target_ratio: The desired ratio of tokens to keep (e.g., 0.25 for 4x compression).

        Returns:
            The compressed prompt string.
        """
        raise NotImplementedError
