"""
Refactored HuggingFace LLM following SOLID principles.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

from core.config import LLMConfig
from core.device_service import DeviceInfo, DeviceService
from llms.base.base import BaseLLM
from utils.prompt_utils import add_structured_instructions


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

        print(
            f"📱 Device: {self.device_info.device_type} | Requested model: {config.model_name}"
        )

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
        if self.device_info.supports_quantization and self.config.quantization in {
            "4bit",
            "8bit",
        }:
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
                print(
                    "Disabled model-level KV cache on MPS to prevent DynamicCache issues."
                )
            except Exception:
                pass

    def _get_model_size(self) -> float:
        """Get model size in GB."""
        total_params = sum(p.numel() for p in self.model.parameters())
        return total_params * 2 / (1024**3)  # Rough estimate for float16

    def get_response(self, prompt: str, task_type: str = "reasoning") -> str:
        """Generate response with device-optimized settings."""
        print(f"🤖 Generating response on {self.device_info.device_type}...")

        # Prepare prompt
        templated_prompt = self._prepare_prompt(prompt, task_type)

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

    def _prepare_prompt(self, prompt: str, task_type: str = "reasoning") -> str:
        """Prepare prompt with chat template if available."""
        # Add task-specific structured instructions
        structured_prompt = add_structured_instructions(prompt, task_type)

        try:
            if (
                hasattr(self.tokenizer, "chat_template")
                and self.tokenizer.chat_template
            ):
                messages = [{"role": "user", "content": structured_prompt}]
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        except Exception:
            pass

        return structured_prompt

    def _tokenize_input(self, prompt: str):
        """Tokenize input with proper padding."""
        if (
            self.tokenizer.pad_token_id is None
            and self.tokenizer.eos_token_id is not None
        ):
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
            gen_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": self.config.temperature,
                    "top_p": 0.9,
                    "repetition_penalty": 1.1,
                }
            )

        return gen_kwargs

    def _handle_generation_error(self, error, inputs, gen_kwargs):
        """Handle generation errors with fallback strategies."""
        msg = str(error).lower()
        if any(keyword in msg for keyword in ["probability tensor", "nan", "inf"]):
            print(
                "⚠️  Sampling failed due to NaN/inf probs; retrying with greedy decoding."
            )
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
def HuggingFace_LLM(config: LLMConfig):
    """Factory function for backward compatibility."""
    device_service = DeviceService()
    return HuggingFaceLLM(config, device_service)
