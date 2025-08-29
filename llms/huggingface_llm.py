
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

from llms.base import BaseLLM

class HuggingFace_LLM(BaseLLM):
    """
    A unified class to run open-source models via the Transformers library.
    It automatically detects and uses the best available hardware:
    NVIDIA GPU (in Colab) > Apple Silicon (on Mac) > CPU.
    """
    def __init__(self, model_name: str):
        super().__init__(model_name)
        
        # Smart device detection
        if torch.cuda.is_available():
            self.device = "cuda"
            self.torch_dtype = torch.bfloat16
        elif torch.backends.mps.is_available():
            self.device = "mps"
            self.torch_dtype = torch.float16 # MPS works better with float16
        else:
            self.device = "cpu"
            self.torch_dtype = torch.float32

        print(f"Detected device: {self.device}. Initializing Hugging Face model...")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            trust_remote_code=True,
            torch_dtype=self.torch_dtype,
            # attn_implementation="eager" # This is needed for MPS
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model.to(self.device)

        # The pipeline's device mapping is slightly different
        pipeline_device = 0 if self.device == "cuda" else self.device
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=pipeline_device,
            # attn_implementation="eager",
            use_cache=False
        )
        print(f"Successfully loaded model '{self.model_name}' on {self.device}.")

    def get_response(self, prompt: str) -> str:
        """Generates a response from the loaded Hugging Face model."""
        print(f"\n--- Sending to Hugging Face model '{self.model_name}' ---")
        messages = [{"role": "user", "content": prompt}]
        templated_prompt = self.pipeline.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        try:
            outputs = self.pipeline(
                templated_prompt,
                max_new_tokens=256,
                eos_token_id=self.pipeline.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
            )
            return outputs[0]["generated_text"][len(templated_prompt) :].strip()
        except Exception as e:
            return f"Error during model inference: {e}"


