from .base import BaseLLM
import time
import threading
from config import OPENROUTER_RATE_LIMIT_RPM


class OpenRouter_LLM(BaseLLM):
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

    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name)
        if not api_key or api_key == "YOUR_API_KEY_HERE" or api_key == "your_openrouter_api_key_here":
            raise ValueError("OpenRouter API key is missing. Please update it in api_keys.py or set OPENROUTER_API_KEY env var.")
        try:
            from openai import OpenAI  # type: ignore
        except Exception as import_error:
            raise ImportError(
                "OpenAI client not available. Please install requirements or select a different provider."
            ) from import_error

        # Use OpenRouter's base URL instead of OpenAI's
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        
        # Rate limiting for free tier
        self.rate_limiter = RateLimiter(requests_per_minute=OPENROUTER_RATE_LIMIT_RPM)
        
        print(f"Initialized OpenRouter LLM with model: {self.model_name}")

    def get_response(self, prompt: str) -> str:
        """Sends a prompt to the specified OpenRouter model."""
        print(f"\n--- Sending to OpenRouter model '{self.model_name}' ---")

        # Wait for rate limit if necessary
        self.rate_limiter.wait_if_needed()
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    extra_headers={
                        # "HTTP-Referer": "https://github.com/Mohammadhsmhs/pcm-llm",  # GitHub repo for rankings
                        "X-Title": "PCM-LLM Benchmark System",  # Project name for rankings
                    },
                    extra_body={},  # For future extensibility
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,  # More creative responses
                    max_tokens=32768,  # Very high limit for long responses
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                error_str = str(e).lower()
                if "rate limit" in error_str or "429" in error_str:
                    if attempt < max_retries - 1:
                        wait_time = 60  # Wait 1 minute for rate limit reset
                        print(f"⏱️  Rate limit hit. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"❌ Rate limit persisted after {max_retries} attempts")
                
                print(f"❌ OpenRouter API Error: {e}")
                return f"Error calling OpenRouter API: {e}"


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
            self.requests = [req_time for req_time in self.requests 
                           if current_time - req_time < 60]
            
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
                    self.requests = [req_time for req_time in self.requests 
                                   if current_time - req_time < 60]
            
            # Add current request
            self.requests.append(current_time)
