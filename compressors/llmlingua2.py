import torch
from llmlingua import PromptCompressor
from compressors.base import BaseCompressor

class LLMLingua2Compressor(BaseCompressor):
    """
    A compressor that uses the official llmlingua library to perform
    prompt compression with the LLMLingua-2 model.
    """
    def __init__(self):
        print("Initializing LLMLingua2Compressor...")
        
        # Determine the appropriate device for the current hardware
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
            
        print(f"LLMLingua will use device: {self.device}")

        # This will download the model the first time it's run.
        self.processor = PromptCompressor(
            model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
            use_llmlingua2=True,
            device_map=self.device # Use the detected device
        )
        print("LLMLingua2Compressor initialized successfully.")

    def compress(self, prompt: str, target_ratio: float) -> str:
        """
        Compresses the prompt using the LLMLingua-2 model.
        """
        print(f"\nCompressing with real LLMLingua-2 to {target_ratio:.0%} of original size...")
        
        result = self.processor.compress_prompt(
            prompt,
            rate=target_ratio,
            force_tokens=['\n', '?']
        )
        
        compressed_prompt = result['compressed_prompt']
        return compressed_prompt