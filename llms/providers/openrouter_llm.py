import threading
import time

from core.config import LLMConfig
from llms.base.base import BaseLLM
from utils.prompt_utils import add_structured_instructions


class OpenRouterLLM(BaseLLM):
    """
    A class to interact with OpenRouter's language models via their API.
    Uses OpenAI-compatible interface with custom base URL and proper headers for rankings.

    Features:
    - Full OpenRouter API compatibility
    - Automatic referrer headers for better rankings
    - Configurable model selection (DeepSeek R1, etc.)
    - High token limits for long-form content
    - Built-in rate limiting to respect API limits
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config.model_name)
        self.config = config
        if not self.config.api_key:
            raise ValueError(
                "OpenRouter API key is required. Please set the OPENROUTER_API_KEY environment variable."
            )
        try:
            from openai import OpenAI
        except ImportError as import_error:
            raise ImportError(
                "OpenAI client not available. Please install requirements or select a different provider."
            ) from import_error

        # Use OpenRouter's base URL instead of OpenAI's
        self.client = OpenAI(
            api_key=self.config.api_key, base_url="https://openrouter.ai/api/v1"
        )

        # Rate limiting for free tier
        self.rate_limiter = RateLimiter(requests_per_minute=self.config.rate_limit_rpm)

        print(f"Initialized OpenRouter LLM with model: {self.model_name}")

    def get_response(self, prompt: str, task_type: str = "reasoning") -> str:
        """Sends a prompt to the specified OpenRouter model."""
        print(f"\n--- Sending to OpenRouter model '{self.model_name}' ---")

        # Add task-specific structured instructions
        structured_prompt = add_structured_instructions(prompt, task_type)

        # Wait for rate limit if necessary
        self.rate_limiter.wait_if_needed()

        for attempt in range(self.config.max_retries):
            try:
                response = self.client.chat.completions.create(
                    extra_headers={
                        "X-Title": "PCM-LLM Benchmark System",  # Project name for rankings
                    },
                    extra_body={},  # For future extensibility
                    model=self.model_name,
                    messages=[{"role": "user", "content": structured_prompt}],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                error_str = str(e).lower()
                if "rate limit" in error_str or "429" in error_str:
                    if attempt < self.config.max_retries - 1:
                        wait_time = 60
                        print(
                            f"⏱️  Rate limit hit. Waiting {wait_time} seconds before retry {attempt + 1}/{self.config.max_retries}..."
                        )
                        time.sleep(wait_time)
                        continue
                    else:
                        print(
                            f"❌ Rate limit persisted after {self.config.max_retries} attempts"
                        )

                print(f"❌ OpenRouter API Error: {e}")
                return f"Error calling OpenRouter API: {e}"
        return f"Error: Failed to get response after {self.config.max_retries} retries."


class RateLimiter:
    """
    Simple rate limiter to respect API limits.
    Uses a sliding window approach to ensure we don't exceed the rate limit.
    """

    def __init__(self, requests_per_minute: int = 16):
        self.requests_per_minute = requests_per_minute
        self.requests = []
        self.lock = threading.Lock()

    def wait_if_needed(self):
        """Wait if necessary to respect the rate limit."""
        with self.lock:
            current_time = time.time()

            # Remove requests older than 1 minute
            self.requests = [
                req_time for req_time in self.requests if current_time - req_time < 60
            ]

            # Check if we're at the limit
            if len(self.requests) >= self.requests_per_minute:
                # Calculate wait time until oldest request expires
                oldest_request = min(self.requests)
                wait_time = 60 - (current_time - oldest_request)

                if wait_time > 0:
                    print(f"⏱️  Rate limit reached. Waiting {wait_time:.1f} seconds...")
                    time.sleep(wait_time)

                    # Recheck after waiting
                    current_time = time.time()
                    self.requests = [
                        req_time
                        for req_time in self.requests
                        if current_time - req_time < 60
                    ]

            # Add current request
            self.requests.append(current_time)
