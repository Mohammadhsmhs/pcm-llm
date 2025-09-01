from .base import BaseLLM
import time
import threading


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
        if not api_key:
            raise ValueError("OpenRouter API key is required. Please set the OPENROUTER_API_KEY environment variable.")
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
        
        # Import settings locally to avoid circular imports
        from core.config import settings
        
        # Rate limiting for free tier
        openrouter_config = settings.get_llm_config("openrouter")
        self.rate_limiter = RateLimiter(requests_per_minute=getattr(openrouter_config, 'rate_limit_rpm', 16))
        
        print(f"Initialized OpenRouter LLM with model: {self.model_name}")

    def get_response(self, prompt: str) -> str:
        """Sends a prompt to the specified OpenRouter model."""
        print(f"\n--- Sending to OpenRouter model '{self.model_name}' ---")

        # Wait for rate limit if necessary
        self.rate_limiter.wait_if_needed()
        
        # Import settings for configuration
        from core.config import settings
        
        max_retries = getattr(settings.performance, 'max_retries', 3) if hasattr(settings.performance, 'max_retries') else 3
        temperature = getattr(settings.generation, 'temperature', 0.7) if hasattr(settings.generation, 'temperature') else 0.7
        max_tokens = getattr(settings.performance, 'max_tokens', 32768) if hasattr(settings.performance, 'max_tokens') else 32768
        
        max_retries = getattr(settings.performance, 'max_retries', 3) if hasattr(settings.performance, 'max_retries') else 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    extra_headers={
                        "X-Title": "PCM-LLM Benchmark System",  # Project name for rankings
                    },
                    extra_body={},  # For future extensibility
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,  # More creative responses
                    max_tokens=max_tokens,  # Very high limit for long responses
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                error_str = str(e).lower()
                if "rate limit" in error_str or "429" in error_str:
                    if attempt < max_retries - 1:
                        wait_time = getattr(settings.performance, 'rate_limit_wait_time', 60) if hasattr(settings.performance, 'rate_limit_wait_time') else 60
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
