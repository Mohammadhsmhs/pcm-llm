import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from llms.base import BaseLLM

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
        
"""
Refactored HuggingFace LLM following SOLID principles.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from llms.base import BaseLLM
from core.config import LLMConfig
from core.device_service import DeviceService, DeviceInfo


class HuggingFaceLLM(BaseLLM):
    """
    Refactored HuggingFace LLM implementation following SOLID principles.
    """

    def __init__(self, config: LLMConfig, device_service: DeviceService):
        super().__init__(config.model_name)
        self.config = config
        self.device_service = device_service

        # Detect optimal device
        self.device_info = self.device_service.get_optimal_device()
        self.device_info = self.device_service.validate_model_compatibility(
            self.device_info, config.model_name
        )

        print(f"📱 Device: {self.device_info.device_type} | Requested model: {config.model_name}")

        # Prepare quantization config
        quantization_config = self._create_quantization_config()

        # Load model with optimized settings
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            torch_dtype=self.device_info.torch_dtype,
            use_cache=self._should_use_cache(),
            quantization_config=quantization_config,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        # Move to device
        self.model.to(self.device_info.device_type)

        # Configure model for device compatibility
        self._configure_model_for_device()

        print(f"✅ Loaded {config.model_name} on {self.device_info.device_type}")
        print(f"   Model size: ~{self._get_model_size():.1f}GB")
        print(f"   Device memory: {self.device_info.memory_gb:.1f}GB available")

    def _create_quantization_config(self):
        """Create quantization configuration if needed."""
        if (self.device_info.supports_quantization and
            self.config.quantization in {"4bit", "8bit"}):
            try:
                from transformers import BitsAndBytesConfig
                return BitsAndBytesConfig(
                    load_in_4bit=(self.config.quantization == "4bit"),
                    load_in_8bit=(self.config.quantization == "8bit"),
                )
            except ImportError:
                print("⚠️  bitsandbytes not available; proceeding without quantization.")
        return None

    def _should_use_cache(self) -> bool:
        """Determine if KV cache should be used."""
        # Disable cache on MPS to avoid DynamicCache issues
        return self.device_info.device_type != "mps"

    def _configure_model_for_device(self):
        """Configure model settings for device compatibility."""
        if self.device_info.device_type == "mps":
            try:
                self.model.config.attn_implementation = "eager"
                print("Using attn_implementation='eager' on MPS to improve stability.")
            except Exception:
                pass

            try:
                self.model.config.use_cache = False
                print("Disabled model-level KV cache on MPS to prevent DynamicCache issues.")
            except Exception:
                pass

    def _get_model_size(self) -> float:
        """Get model size in GB."""
        total_params = sum(p.numel() for p in self.model.parameters())
        return total_params * 2 / (1024**3)  # Rough estimate for float16

    def get_response(self, prompt: str) -> str:
        """Generate response with device-optimized settings."""
        print(f"🤖 Generating response on {self.device_info.device_type}...")

        # Prepare prompt
        templated_prompt = self._prepare_prompt(prompt)

        # Tokenize
        inputs = self._tokenize_input(templated_prompt)

        # Generate response
        streamer = self._create_streamer()
        gen_kwargs = self._create_generation_kwargs(streamer)

        with torch.no_grad():
            try:
                output = self.model.generate(**inputs, **gen_kwargs)
            except Exception as e:
                output = self._handle_generation_error(e, inputs, gen_kwargs)

        # Decode response
        response = self._decode_output(output, inputs)
        return response

    def _prepare_prompt(self, prompt: str) -> str:
        """Prepare prompt with chat template if available."""
        structured_prompt = (
            prompt + "\n\nPlease provide your final answer in this exact format: #### [final_answer_number]"
        )

        try:
            if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
                messages = [{"role": "user", "content": structured_prompt}]
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        except Exception:
            pass

        return structured_prompt

    def _tokenize_input(self, prompt: str):
        """Tokenize input with proper padding."""
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        inputs = self.tokenizer(prompt, return_tensors="pt")
        return {k: v.to(self.device_info.device_type) for k, v in inputs.items()}

    def _create_streamer(self):
        """Create text streamer if streaming is enabled."""
        if self.config.stream_tokens:
            return TextStreamer(
                self.tokenizer,
                skip_special_tokens=True,
                skip_prompt=True,
            )
        return None

    def _create_generation_kwargs(self, streamer):
        """Create generation arguments optimized for device."""
        max_tokens = 16384 if self.config.unlimited_mode else self.config.max_tokens

        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": self._should_use_cache(),
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "streamer": streamer,
        }

        if self.device_info.device_type == "mps":
            gen_kwargs.update({"do_sample": False})  # Greedy decoding for MPS stability
        else:
            gen_kwargs.update({
                "do_sample": True,
                "temperature": self.config.temperature,
                "top_p": 0.9,
                "repetition_penalty": 1.1
            })

        return gen_kwargs

    def _handle_generation_error(self, error, inputs, gen_kwargs):
        """Handle generation errors with fallback strategies."""
        msg = str(error).lower()
        if any(keyword in msg for keyword in ["probability tensor", "nan", "inf"]):
            print("⚠️  Sampling failed due to NaN/inf probs; retrying with greedy decoding.")
            safe_kwargs = dict(gen_kwargs)
            safe_kwargs.update({"do_sample": False})
            for k in ("temperature", "top_p", "repetition_penalty"):
                safe_kwargs.pop(k, None)
            return self.model.generate(**inputs, **safe_kwargs)
        else:
            raise error

    def _decode_output(self, output, inputs):
        """Decode generated output."""
        input_length = inputs["input_ids"].shape[-1]
        generated_only = output[0][input_length:]
        return self.tokenizer.decode(generated_only, skip_special_tokens=True).strip()


# Factory function for backward compatibility
def HuggingFace_LLM(model_name: str, quantization: str = "none"):
    """Factory function for backward compatibility."""
    from core.config import config_provider

    config = config_provider.get_llm_config("huggingface")
    config.model_name = model_name
    config.quantization = quantization

    device_service = DeviceService()
    return HuggingFaceLLM(config, device_service)


