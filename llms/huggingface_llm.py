
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
        """Verify that quantization is actually applied by checking various indicators."""
        if self.quantization == "none":
            print("ℹ️  Using full precision (no quantization applied)")
            return
            
        print("Verifying quantization...")
        
        # Method 1: Check for BitsAndBytes quantized modules
        quantized_modules = []
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight') and hasattr(module.weight, 'quant_state'):
                quantized_modules.append(name)
        
        # Method 2: Check memory usage (quantized models use less memory)
        total_memory = 0
        for param in self.model.parameters():
            total_memory += param.numel() * param.element_size()
        
        # Method 3: Check for quantization-specific layer types
        has_quantized_layers = any(
            'Bnb' in str(type(module)) or 'Quantized' in str(type(module))
            for module in self.model.modules()
        )
        
        # Method 4: Check parameter data types (more comprehensive)
        total_params = 0
        int8_params = 0
        int4_params = 0
        
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            if hasattr(param, 'dtype'):
                if param.dtype == torch.int8:
                    int8_params += param.numel()
                elif hasattr(param, 'quant_state'):  # 4-bit quantized
                    int4_params += param.numel()
        
        print(f"Quantization verification results:")
        print(f"  - Quantized modules found: {len(quantized_modules)}")
        print(f"  - Has quantized layer types: {has_quantized_layers}")
        print(f"  - Total model memory: {total_memory / 1024**3:.2f} GB")
        print(f"  - Int8 parameters: {int8_params}/{total_params} ({int8_params/total_params*100:.1f}%)")
        print(f"  - Int4 parameters: {int4_params}/{total_params} ({int4_params/total_params*100:.1f}%)")
        
        # Determine if quantization is working
        quantization_working = (
            len(quantized_modules) > 0 or
            has_quantized_layers or
            int8_params > 0 or
            int4_params > 0
        )
        
        if quantization_working:
            print("✅ Quantization verification passed - model appears to be quantized")
        else:
            print("⚠️  Warning: No clear signs of quantization detected")
            
        # Additional memory efficiency check
        if self.quantization == "8bit" and total_memory > 10 * 1024**3:  # More than 10GB
            print("⚠️  Warning: Model memory seems high for 8-bit quantization")
        elif self.quantization == "4bit" and total_memory > 5 * 1024**3:  # More than 5GB
            print("⚠️  Warning: Model memory seems high for 4-bit quantization")

    def get_response(self, prompt: str) -> str:
        """Generates a response from the loaded Hugging Face model."""
        print(f"\n--- Sending to Hugging Face model '{self.model_name}' ---")
        
        # Add structured output instruction to the prompt
        structured_prompt = prompt + "\n\nIMPORTANT: End your response with the final answer in this exact format: #### [final_answer_number]"
        
        # Truncate if too long (Colab optimization)
        if len(structured_prompt) > 4000:  # Rough limit for 4K models
            structured_prompt = structured_prompt[:4000] + "...\n\nIMPORTANT: End your response with the final answer in this exact format: #### [final_answer_number]"
            print("Prompt truncated to fit model context length")
        
        messages = [{"role": "user", "content": structured_prompt}]
        templated_prompt = self.tokenizer.apply_chat_template(
             messages, tokenize=False, add_generation_prompt=True
         )
        
        # Tokenize with length limit
        inputs = self.tokenizer(
            templated_prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=512,  # Limit input length for memory efficiency
            padding=False
        ).to(self.device)
        
        # Clear any cached memory before generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
       # Use a streamer to print tokens as they are generated for real-time feedback
        streamer = TextStreamer(self.tokenizer, skip_prompt=True)
        
         # The generate call now streams output to the console
        output = self.model.generate(
           **inputs, 
           streamer=streamer,
           max_new_tokens=512,  # Reasonable limit for complete responses without timeout
           eos_token_id=self.tokenizer.eos_token_id, 
           do_sample=True, 
           temperature=0.3,  # Slightly increased for better reasoning
           top_p=0.95,       # Slightly increased for more diverse responses
           use_cache=False,  # Crucial fix for MPS devices and memory efficiency
           pad_token_id=self.tokenizer.eos_token_id,  # Avoid padding issues
           repetition_penalty=1.1,  # Add repetition penalty to avoid loops
           length_penalty=1.0       # Neutral length penalty
       )
       
       # Decode the full output for logging purposes (the streamer only prints)
        full_response = self.tokenizer.decode(output[0], skip_special_tokens=True)
       
       # Extract only the newly generated part for the return value
        response = full_response[len(templated_prompt):].strip()
        
        # Clear memory after generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
            
        return response

    def get_batch_responses(self, prompts: list) -> list:
        """Generate responses for multiple prompts - DISABLED due to string indexing bug."""
        raise NotImplementedError("Batch processing is temporarily disabled due to string indexing error")


