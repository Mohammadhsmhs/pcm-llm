"""
Ollama LLM implementation using the official Ollama Python library.
Based on: https://www.cohorte.co/blog/using-ollama-with-python-step-by-step-guide
"""

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("⚠️  Warning: ollama package not installed. Install with: pip install ollama")

from typing import Optional, Dict, Any
from llms.base import BaseLLM


class Ollama_LLM(BaseLLM):
    """
    LLM backend using the official Ollama Python library.
    - Uses the official ollama Python package for better integration
    - Supports streaming responses and advanced features (streaming enabled by default)
    - Optimized for local LLM inference with Ollama
    - Supports function calling and tool usage
    """

    def __init__(
        self,
        model_name_or_config,
        temperature: float = 0.0,
        top_k: int = 1,
        num_ctx: int = 4096,
        repeat_penalty: float = 1.1,
        repeat_last_n: int = 64,
        stream: bool = True,  # Enable streaming by default for real-time output
    ):
        # Handle both config object and direct parameters
        if hasattr(model_name_or_config, 'model_name'):
            # It's a config object
            config = model_name_or_config
            model_name = config.model_name
            temperature = getattr(config, 'temperature', temperature)
            num_ctx = getattr(config, 'max_tokens', num_ctx)
        else:
            # It's a direct model name string
            model_name = model_name_or_config
        
        if not OLLAMA_AVAILABLE:
            raise ImportError("ollama package is required. Install with: pip install ollama")

        super().__init__(model_name)

        self.model_name = model_name
        self.temperature = temperature
        self.top_k = top_k
        self.num_ctx = num_ctx
        self.repeat_penalty = repeat_penalty
        self.repeat_last_n = repeat_last_n
        self.stream = stream

        # Test connection and model availability
        self._validate_setup()

        print(
            f"✅ Initialized Ollama LLM: {model_name}\n"
            f"   Temperature: {temperature} | Context: {num_ctx} | Stream: {stream}"
        )

    def _validate_setup(self):
        """Validate Ollama setup and model availability."""
        try:
            # Check if Ollama service is running and get available models
            models = ollama.list()
            # Handle both dict and object responses
            if hasattr(models, 'models'):
                available_models = [model.model for model in models.models]
            elif isinstance(models, dict):
                available_models = [model['name'] for model in models.get('models', [])]
            else:
                available_models = []

            if not available_models:
                raise ConnectionError("No models found. Make sure Ollama is running and you have pulled models.")

            if self.model_name not in available_models:
                print(f"⚠️  Warning: Model '{self.model_name}' not found locally.")
                print(f"   Available models: {', '.join(available_models)}")
                print(f"   To pull the model, run: ollama pull {self.model_name}")
                if available_models:
                    print("   Using first available model for now...")
                    self.model_name = available_models[0]
                    print(f"   Switched to: {self.model_name}")

        except Exception as e:
            print(f"⚠️  Warning: Could not connect to Ollama: {e}")
            print("   Make sure Ollama is installed and running:")
            print("   1. Install: curl -fsSL https://ollama.ai/install.sh | sh")
            print("   2. Start: ollama serve")
            print("   3. Pull model: ollama pull llama2")
            raise ConnectionError(f"Ollama setup validation failed: {e}")

    def get_response(self, prompt: str) -> str:
        """
        Generate a response using Ollama's generate API.

        Args:
            prompt: The input prompt for the model

        Returns:
            The generated response text
        """
        print(f"\n🤖 Generating response with Ollama ({self.model_name})...")

        try:
            # Prepare generation options
            from core.config import settings
            
            options = {
                "temperature": self.temperature,
                "top_k": self.top_k,
                "num_ctx": self.num_ctx,
                "repeat_penalty": self.repeat_penalty,
                "repeat_last_n": self.repeat_last_n,
                "num_predict": getattr(settings.performance, 'max_tokens', 1024) if hasattr(settings.performance, 'max_tokens') else 1024,
            }

            # Generate response using official library
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                stream=self.stream,
                options=options
            )

            # Handle streaming vs non-streaming responses
            if self.stream:
                full_response = ""
                for chunk in response:
                    if chunk.get('response'):
                        full_response += chunk['response']
                        print(chunk['response'], end='', flush=True)
                print()  # New line after streaming
                generated_text = full_response
            else:
                generated_text = response.get('response', '').strip()

            if not generated_text:
                return "Error: Empty response from Ollama"

            print(f"📝 Generated {len(generated_text)} characters")
            if not self.stream:  # Don't show preview if already streamed
                preview = generated_text[:200]
                print(f"📝 Response preview: {preview}{'...' if len(generated_text) > 200 else ''}")

            return generated_text

        except ollama.ResponseError as e:
            return f"Error: Ollama API error - {e}"
        except Exception as e:
            return f"Error in Ollama generation: {e}"

    def chat(self, messages: list, tools: Optional[list] = None) -> Dict[str, Any]:
        """
        Chat with the model using Ollama's chat API.
        Supports conversation history and tool calling.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            tools: Optional list of tool functions for the model to use

        Returns:
            Response dictionary from Ollama
        """
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                tools=tools,
                options={
                    "temperature": self.temperature,
                    "top_k": self.top_k,
                    "num_ctx": self.num_ctx,
                    "repeat_penalty": self.repeat_penalty,
                    "repeat_last_n": self.repeat_last_n,
                }
            )
            return response

        except Exception as e:
            return {"error": f"Chat error: {e}"}

    def list_available_models(self) -> list:
        """List all available models in Ollama."""
        try:
            models = ollama.list()
            # Handle both dict and object responses
            if hasattr(models, 'models'):
                return [model.model for model in models.models]
            elif isinstance(models, dict):
                return [model['name'] for model in models.get('models', [])]
            else:
                return []
        except Exception:
            return []

    def pull_model(self, model_name: str) -> bool:
        """Pull a model from the Ollama library."""
        try:
            print(f"📥 Pulling model: {model_name}")
            ollama.pull(model_name)
            print(f"✅ Successfully pulled model: {model_name}")
            return True
        except Exception as e:
            print(f"❌ Failed to pull model {model_name}: {e}")
            return False

    def get_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get information about a specific model."""
        model = model_name or self.model_name
        try:
            info = ollama.show(model)
            return info
        except Exception as e:
            return {"error": f"Could not get model info: {e}"}


# Convenience functions for common operations
def pull_model_if_needed(model_name: str) -> bool:
    """Pull a model if it's not already available."""
    try:
        models = ollama.list()
        # Handle both dict and object responses
        if hasattr(models, 'models'):
            available = [model.model for model in models.models]
        elif isinstance(models, dict):
            available = [m['name'] for m in models.get('models', [])]
        else:
            available = []

        if model_name not in available:
            print(f"Model {model_name} not found locally. Pulling...")
            ollama.pull(model_name)
            return True
        return False
    except Exception as e:
        print(f"Error checking/pulling model: {e}")
        return False


def list_local_models() -> list:
    """List all locally available models."""
    try:
        models = ollama.list()
        # Handle both dict and object responses
        if hasattr(models, 'models'):
            return [model.model for model in models.models]
        elif isinstance(models, dict):
            return [m['name'] for m in models.get('models', [])]
        else:
            return []
    except Exception:
        return []
