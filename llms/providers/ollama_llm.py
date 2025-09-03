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
from llms.base.base import BaseLLM
from core.config import LLMConfig
from utils.prompt_utils import add_structured_instructions


class OllamaLLM(BaseLLM):
    """
    LLM backend using the official Ollama Python library.
    - Uses the official ollama Python package for better integration
    - Supports streaming responses and advanced features (streaming enabled by default)
    - Optimized for local LLM inference with Ollama
    - Supports function calling and tool usage
    """

    def __init__(self, config: LLMConfig):
        if not OLLAMA_AVAILABLE:
            raise ImportError("ollama package is required. Install with: pip install ollama")

        super().__init__(config.model_name)
        self.config = config
        self._validate_setup()

        print(
            f"✅ Initialized Ollama LLM: {self.config.model_name}\n"
            f"   Temperature: {self.config.temperature} | Context: {self.config.max_tokens} | Stream: {self.config.stream_tokens}"
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

            if self.config.model_name not in available_models:
                print(f"⚠️  Warning: Model '{self.config.model_name}' not found locally.")
                print(f"   Available models: {', '.join(available_models)}")
                print(f"   To pull the model, run: ollama pull {self.config.model_name}")
                if available_models:
                    print("   Using first available model for now...")
                    self.config.model_name = available_models[0]
                    print(f"   Switched to: {self.config.model_name}")

        except Exception as e:
            print(f"⚠️  Warning: Could not connect to Ollama: {e}")
            print("   Make sure Ollama is installed and running:")
            print("   1. Install: curl -fsSL https://ollama.ai/install.sh | sh")
            print("   2. Start: ollama serve")
            print("   3. Pull model: ollama pull llama2")
            raise ConnectionError(f"Ollama setup validation failed: {e}")

    def get_response(self, prompt: str, task_type: str = "reasoning") -> str:
        """
        Generate a response using Ollama's generate API.

        Args:
            prompt: The input prompt for the model
            task_type: The type of task (reasoning, summarization, classification)

        Returns:
            The generated response text
        """
        print(f"\n🤖 Generating response with Ollama ({self.config.model_name})...")

        # Add task-specific structured output instructions
        # Add task-specific structured instructions
        structured_prompt = add_structured_instructions(prompt, task_type)

        try:
            # Prepare generation options
            options = {
                "temperature": self.config.temperature,
                "top_k": self.config.top_k,
                "num_ctx": self.config.max_tokens,
                "repeat_penalty": self.config.repetition_penalty,
                "repeat_last_n": 64,  # A common default
                "num_predict": self.config.max_tokens,
            }

            # Generate response using official library
            response = ollama.generate(
                model=self.config.model_name,
                prompt=structured_prompt,
                stream=self.config.stream_tokens,
                options=options
            )

            # Handle streaming vs non-streaming responses
            if self.config.stream_tokens:
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
            if not self.config.stream_tokens:  # Don't show preview if already streamed
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
                model=self.config.model_name,
                messages=messages,
                tools=tools,
                options={
                    "temperature": self.config.temperature,
                    "top_k": self.config.top_k,
                    "num_ctx": self.config.max_tokens,
                    "repeat_penalty": self.config.repetition_penalty,
                    "repeat_last_n": 64,
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
        model = model_name or self.config.model_name
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
