from compressors.selective_context import SelectiveContextCompressor
from .base import BaseCompressor
from .llmlingua2 import LLMLingua2Compressor


class CompressorFactory:
    """
    The Factory pattern encapsulates the logic for creating objects.
    This makes it easy to add new compression methods to the framework
    without changing the main application logic.
    """

    @staticmethod
    def create(method_name: str) -> BaseCompressor:
        """
        Creates an instance of a specified compression method.
        """
        if method_name == "llmlingua2":
            print("Initializing LLMLingua2Compressor...")
            return LLMLingua2Compressor()
        elif method_name == "selective_context":
            print("Initializing SelectiveContextCompressor...")
            return SelectiveContextCompressor()
        # Example of how you would add another method:
        # elif method_name == "cpc":
        #     return CPCCompressor()
        else:
            raise ValueError(f"Unknown compression method: {method_name}")


