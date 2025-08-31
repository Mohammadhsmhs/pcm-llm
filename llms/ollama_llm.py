import requests
import json
from typing import Optional

from llms.base import BaseLLM


class Ollama_LLM(BaseLLM):
    """
    LLM backend using Ollama API.
    - Uses HTTP requests to Ollama server running locally
    - Optimized for preventing repetition with built-in parameters
    - Supports GGUF models loaded in Ollama
    """

    def __init__(
        self,
        model_name: str = "qwen3-14b",
        host: str = "http://localhost:11434",
        temperature: float = 0.0,
        top_k: int = 1,
        num_ctx: int = 4096,
        repeat_penalty: float = 1.1,
        repeat_last_n: int = 64,
    ):
        super().__init__(model_name)

        self.model_name = model_name
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.top_k = top_k
        self.num_ctx = num_ctx
        self.repeat_penalty = repeat_penalty
        self.repeat_last_n = repeat_last_n

        # Test connection
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code != 200:
                raise ConnectionError(f"Ollama server not responding: {response.status_code}")
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]
            if model_name not in model_names:
                print(f"⚠️  Warning: Model '{model_name}' not found in Ollama. Available: {model_names}")
        except Exception as e:
            print(f"⚠️  Warning: Could not connect to Ollama server: {e}")
            print("   Make sure Ollama is running with: ollama serve")

        print(
            f"Initialized Ollama LLM: {model_name}\n"
            f"  Host: {host} | Temperature: {temperature} | Context: {num_ctx}"
        )

    def get_response(self, prompt: str) -> str:
        print(f"\n🤖 Generating response with Ollama ({self.model_name})...")

        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "top_k": self.top_k,
                    "num_ctx": self.num_ctx,
                    "repeat_penalty": self.repeat_penalty,
                    "repeat_last_n": self.repeat_last_n,
                    "num_predict": 1024,  # Reasonable token limit
                }
            }

            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=300  # 5 minutes timeout
            )

            if response.status_code != 200:
                return f"Error: Ollama API returned {response.status_code}: {response.text}"

            result = response.json()
            generated_text = result.get("response", "").strip()

            if not generated_text:
                return "Error: Empty response from Ollama"

            print(f"📝 Generated {len(generated_text)} characters")
            print(f"📝 Response preview: {generated_text[:200]}...")

            return generated_text

        except requests.exceptions.Timeout:
            return "Error: Ollama request timed out (5 minutes)"
        except requests.exceptions.ConnectionError:
            return "Error: Could not connect to Ollama server. Make sure it's running."
        except Exception as e:
            return f"Error in Ollama generation: {e}"
