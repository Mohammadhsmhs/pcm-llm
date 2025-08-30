import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from llms.base import BaseLLM
from config import STREAM_TOKENS, UNLIMITED_MODE

# Local safe defaults (previously imported from config, but not guaranteed present)
USE_METAL_CACHE = True

class HuggingFace_LLM(BaseLLM):
    """
    A unified class to run open-source models via the Transformers library.
    Optimized for Apple Silicon M2 with Metal Performance Shaders.
    """
    def __init__(self, model_name: str, quantization: str = "none"):
        super().__init__(model_name)
        self.quantization = quantization.lower()

        # Smart device detection optimized for M2
        if torch.cuda.is_available():
            self.device = "cuda"
            self.torch_dtype = torch.bfloat16
        elif torch.backends.mps.is_available():
            self.device = "mps"
            self.torch_dtype = torch.float16  # MPS works better with float16
            print("🎯 Apple Silicon detected - using Metal Performance Shaders (MPS)")
        else:
            self.device = "cpu"
            self.torch_dtype = torch.float32

        print(f"📱 Device: {self.device} | Requested model: {model_name}")

        # Respect requested model but optionally suggest an optimized alternative on MPS/CPU
        self.model_name = model_name
        if self.device in ("mps", "cpu"):
            # If the requested model name suggests a bnb-quantized repo, switch to a safe FP16/FP32 repo
            requested_lower = self.model_name.lower()
            if any(tag in requested_lower for tag in ["-bnb-", "bnb-", "-4bit", "-8bit", "gguf"]):
                safe_default = "microsoft/Phi-3-mini-4k-instruct"
                print(
                    f"⚠️  Requested repo '{self.model_name}' appears to be a quantized checkpoint incompatible with {self.device}.\n"
                    f"   Switching to a safe default model: {safe_default}"
                )
                self.model_name = safe_default
            if self.quantization in {"4bit", "8bit"}:
                print("⚠️  BitsAndBytes quantization isn't supported on MPS/CPU. Falling back to full precision.")
                self.quantization = "none"

        # Prepare optional quantization config (CUDA only)
        quantization_config = None
        if self.device == "cuda" and self.quantization in {"4bit", "8bit"}:
            try:
                from transformers import BitsAndBytesConfig  # type: ignore
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=(self.quantization == "4bit"),
                    load_in_8bit=(self.quantization == "8bit"),
                )
                print(f"🧮 Using bitsandbytes quantization: {self.quantization}")
            except Exception:
                print("⚠️  bitsandbytes not available; proceeding without quantization.")
                quantization_config = None

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_cache=USE_METAL_CACHE,
            quantization_config=quantization_config,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Move to appropriate device
        self.model.to(self.device)

        # On MPS, prefer eager attention to avoid cache impl incompatibilities
        if self.device == "mps":
            try:
                # Newer Transformers expose attn_implementation in config
                self.model.config.attn_implementation = "eager"
                print("Using attn_implementation='eager' on MPS to improve stability.")
            except Exception:
                pass

        print(f"✅ Loaded {self.model_name} on {self.device}")
        print(f"   Model size: ~{self._get_model_size():.1f}GB")
        print(f"   Device memory: {self._get_device_memory():.1f}GB available")

    def _get_model_size(self):
        """Get model size in GB"""
        total_params = sum(p.numel() for p in self.model.parameters())
        return total_params * 2 / (1024**3)  # Rough estimate for float16

    def _get_device_memory(self):
        """Get available device memory in GB"""
        if self.device == "mps":
            # M2 typically has 8GB or 16GB unified memory
            return 8.0  # Conservative estimate for M2
        elif self.device == "cuda":
            return torch.cuda.get_device_properties(0).total_memory / (1024**3)
        else:
            return 4.0  # Conservative CPU estimate

    def get_response(self, prompt: str) -> str:
        """Generates a response optimized for M2 performance."""
        print(f"\n🤖 Generating response on {self.device}...")

        # Add structured output instruction
        structured_prompt = (
            prompt
            + "\n\nPlease provide your final answer in this exact format: #### [final_answer_number]"
        )

        # Build an input string using chat template if available; otherwise fall back to raw prompt
        templated_prompt = None
        try:
            # Only use chat template if the tokenizer supports it
            if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
                messages = [{"role": "user", "content": structured_prompt}]
                templated_prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        except Exception:
            # If anything goes wrong, we'll just use the structured prompt directly
            templated_prompt = None

        if not templated_prompt:
            templated_prompt = structured_prompt

        # Tokenize safely; ensure pad token is defined
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        inputs = self.tokenizer(templated_prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate with M2-optimized settings and stream tokens to stdout
        streamer = None
        if STREAM_TOKENS:
            streamer = TextStreamer(
                self.tokenizer,
                skip_special_tokens=True,
                skip_prompt=True,
            )
        
        # Set max_new_tokens based on unlimited mode
        max_tokens = 16384 if UNLIMITED_MODE else 8192
        if UNLIMITED_MODE:
            print(f"🔓 Unlimited mode: Extended max_tokens to {max_tokens}")
        
        # Default generation args
        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            # On MPS, disable KV cache to avoid DynamicCache issues
            use_cache=False if self.device == "mps" else USE_METAL_CACHE,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                streamer=streamer,
        )
        if self.device == "mps":
            # Greedy decoding is more stable on MPS if sampling causes NaNs
            gen_kwargs.update(dict(do_sample=False))
        else:
            gen_kwargs.update(dict(do_sample=True, temperature=0.3, top_p=0.9, repetition_penalty=1.1))

        with torch.no_grad():  # Reduce memory usage
            try:
                output = self.model.generate(**inputs, **gen_kwargs)
            except Exception as e:
                # Fallback for sampling-related NaN/inf errors
                msg = str(e).lower()
                if "probability tensor" in msg or "nan" in msg or "inf" in msg:
                    print("⚠️  Sampling failed due to NaN/inf probs; retrying with greedy decoding.")
                    safe_kwargs = dict(gen_kwargs)
                    safe_kwargs.update(dict(do_sample=False))
                    # Remove sampling-specific fields if present
                    for k in ("temperature", "top_p", "repetition_penalty"):
                        safe_kwargs.pop(k, None)
                    output = self.model.generate(**inputs, **safe_kwargs)
                else:
                    raise

        # Decode only the newly generated portion (safer than slicing by string length)
        input_length = inputs["input_ids"].shape[-1]
        generated_only = output[0][input_length:]
        response = self.tokenizer.decode(generated_only, skip_special_tokens=True).strip()

        return response


