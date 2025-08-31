import os
from api_keys import OPENROUTER_API_KEY as API_OPENROUTER_KEY, OPENAI_API_KEY as API_OPENAI_KEY

# --- Benchmark Run Configuration ---
# Available tasks: "reasoning", "summarization", "classification"
SUPPORTED_TASKS = ["reasoning", "summarization", "classification"]
DEFAULT_TASK = "reasoning"

# Task-specific dataset configurations
TASK_CONFIGURATIONS = {
    "reasoning": {
        "dataset": "gsm8k",
        "config": "main",
        "description": "Mathematical reasoning with GSM8K dataset"
    },
    "summarization": {
        "dataset": "cnn_dailymail",
        "config": "3.0.0",
        "description": "News article summarization"
    },
    "classification": {
        "dataset": "imdb",
        "config": "plain_text",
        "description": "Sentiment classification on movie reviews"
    }
}

# Current task selection
DEFAULT_DATASET = TASK_CONFIGURATIONS[DEFAULT_TASK]["dataset"]
DEFAULT_DATASET_CONFIG = TASK_CONFIGURATIONS[DEFAULT_TASK]["config"]

# --- Sample Configuration ---
NUM_SAMPLES_TO_RUN = 7  # Increased for more realistic benchmarking results

# --- Compression Method Selection ---
# A list of all compression methods to run in the benchmark.
# Available options: "llmlingua2", "selective_context"
COMPRESSION_METHODS_TO_RUN = ["llmlingua2",]
# "selective_context"]
DEFAULT_TARGET_RATIO = 0.8 # Keep 80% of tokens

 

# --- LLM Provider Selection ---
# Available providers: "manual", "openai", "huggingface", "llamacpp", "openrouter"
DEFAULT_LLM_PROVIDER = "openrouter"

# --- HuggingFace Configuration ---
HUGGINGFACE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
HUGGINGFACE_QUANTIZATION = "4bit"  # Options: "none", "4bit", "8bit"

# --- OpenAI Configuration ---
OPENAI_API_KEY = API_OPENAI_KEY if API_OPENAI_KEY != "your_openai_api_key_here" else os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-3.5-turbo"

# --- OpenRouter Configuration ---
OPENROUTER_API_KEY = API_OPENROUTER_KEY if API_OPENROUTER_KEY != "your_openrouter_api_key_here" else os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = "deepseek/deepseek-r1:free"
OPENROUTER_RATE_LIMIT_RPM = 16  # Free tier limit: 16 requests per minute

# --- Llama.cpp Configuration ---
LLAMACPP_MODEL_PATH = ""  # Local model path (leave empty if using repo_id)
LLAMACPP_REPO_ID = "microsoft/Phi-3-mini-4k-instruct-gguf"
LLAMACPP_FILENAME = "Phi-3-mini-4k-instruct-q4.gguf"
LLAMACPP_N_CTX = 4096  # Increased context window for realistic benchmarking
LLAMACPP_N_GPU_LAYERS = 0  # Set to -1 for all layers on GPU
LLAMACPP_N_THREADS = 4

# --- Runtime Verbosity / Streaming ---
STREAM_TOKENS = True  # Disable streaming for better benchmarking performance

# --- Unlimited Mode Configuration ---
# When enabled, removes all timeouts and size limits for full-speed benchmarking
# WARNING: This may cause very long run times and high resource usage
UNLIMITED_MODE = True  # Set to True to disable all restrictions

# --- Optimized Benchmark Configuration ---
USE_JSONL = False  # Use CSV format for intermediate storage
ENABLE_CHECKPOINTING = True  # Save progress after each compression method
MAX_CONCURRENT_LOGGERS = 1  # Semaphore limit for thread-safe logging
COMPRESS_INTERMEDIATE = False  # No compression for intermediate files
ADAPTIVE_BATCH_SIZE = True  # Adjust batch size based on available memory
MEMORY_CHECKPOINT_INTERVAL = 50  # Memory monitoring interval (samples)
BATCH_SIZE_BASE = 5  # Base batch size for adaptive processing


