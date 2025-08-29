
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, BitsAndBytesConfig
from llms.base import BaseLLM

class HuggingFace_LLM(BaseLLM):
    """
    A unified class to run open-source models via the Transformers library.
    It automatically detects and uses the best available hardware:
    NVIDIA GPU (in Colab) > Apple Silicon (on Mac) > CPU.
    """
    def __init__(self, model_name: str, quantization: str = "none"):
        super().__init__(model_name)
        self.quantization = quantization.lower()
        
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
            # Quantization is not supported on CPU
            if self.quantization != "none":
                print("Warning: Quantization is not supported on CPU. Using full precision.")
                self.quantization = "none"

        print(f"Detected device: {self.device}. Initializing Hugging Face model with {self.quantization} quantization...")

        # Configure quantization if requested and supported
        quantization_config = None
        if self.quantization != "none":
            if torch.cuda.is_available():
                # BitsAndBytes quantization only works on CUDA
                if self.quantization == "4bit":
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    print("Using 4-bit quantization for fastest inference and lowest memory usage.")
                elif self.quantization == "8bit":
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0
                    )
                    print("Using 8-bit quantization for faster inference and reduced memory usage.")
                else:
                    print(f"Unknown quantization mode: {self.quantization}. Using full precision.")
                    quantization_config = None
            elif torch.backends.mps.is_available():
                # BitsAndBytes doesn't support MPS - fall back to full precision
                print(f"⚠️  BitsAndBytes quantization not supported on MPS. Using full precision instead of {self.quantization}.")
                self.quantization = "none"
            else:
                # CPU - no quantization support
                print(f"⚠️  Quantization not supported on CPU. Using full precision instead of {self.quantization}.")
                self.quantization = "none"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            trust_remote_code=True,
            torch_dtype=self.torch_dtype,
            quantization_config=quantization_config,
            # attn_implementation="eager" # This is needed for MPS
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Only move to device if not using quantization (quantization handles device placement)
        if quantization_config is None:
            self.model.to(self.device)
        
        # Verify quantization is working
        self._verify_quantization()

        # The pipeline's device mapping is slightly different
        # pipeline_device = 0 if self.device == "cuda" else self.device
        # self.pipeline = pipeline(
        #     "text-generation",
        #     model=self.model,
        #     tokenizer=self.tokenizer,
        #     device=pipeline_device,
        #     # attn_implementation="eager",
        #     use_cache=False
        # )
        print(f"Successfully loaded model '{self.model_name}' on {self.device} with {self.quantization} quantization.")

    def _verify_quantization(self):
        """Verify that quantization is actually applied by checking parameter data types."""
        if self.quantization == "none":
            print("ℹ️  Using full precision (no quantization applied)")
            return
            
        print("Verifying quantization...")
        
        # Check a few parameters to see their data types
        total_params = 0
        quantized_params = 0
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:  # Only count trainable parameters
                total_params += param.numel()
                if param.dtype in [torch.int8, torch.uint8]:
                    quantized_params += param.numel()
        
        if total_params > 0:
            quantization_ratio = quantized_params / total_params
            print(f"Quantization verification: {quantized_params}/{total_params} parameters quantized ({quantization_ratio:.1%})")
            
            if self.quantization == "8bit" and quantization_ratio < 0.5:
                print("⚠️  Warning: Expected higher quantization ratio for 8-bit quantization")
            elif self.quantization == "4bit" and quantization_ratio < 0.8:
                print("⚠️  Warning: Expected higher quantization ratio for 4-bit quantization")
            else:
                print("✅ Quantization verification passed")
        else:
            print("ℹ️  No trainable parameters found to verify quantization")

    def get_response(self, prompt: str) -> str:
        """Generates a response from the loaded Hugging Face model."""
        print(f"\n--- Sending to Hugging Face model '{self.model_name}' ---")
        messages = [{"role": "user", "content": prompt}]
        templated_prompt = self.tokenizer.apply_chat_template(
             messages, tokenize=False, add_generation_prompt=True
         )
        inputs = self.tokenizer(templated_prompt, return_tensors="pt").to(self.device)
        
       # Use a streamer to print tokens as they are generated for real-time feedback
        streamer = TextStreamer(self.tokenizer, skip_prompt=True)
         # The generate call now streams output to the console
        output = self.model.generate(
           **inputs, 
           streamer=streamer,
           max_new_tokens=256, 
           eos_token_id=self.tokenizer.eos_token_id, 
           do_sample=True, 
           temperature=0.1, 
           top_p=0.9,
           use_cache=False # Crucial fix for MPS devices
       )
       
       # Decode the full output for logging purposes (the streamer only prints)
        full_response = self.tokenizer.decode(output[0], skip_special_tokens=True)
       
       # Extract only the newly generated part for the return value
        return full_response[len(templated_prompt):].strip()


