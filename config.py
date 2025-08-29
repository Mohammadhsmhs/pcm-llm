import os

# --- LLM Provider Selection ---
# Choose your desired LLM provider here. Options are:
# "huggingface": Use a free, open-source model in Colab/locally. (Recommended for free use)
# "openai"     : Use the paid OpenAI API. Requires an API key.
# "manual"     : Manually copy/paste the prompt and response from a chat platform.
# DEFAULT_LLM_PROVIDER = "manual"
DEFAULT_LLM_PROVIDER = "huggingface"

# --- API Keys (only needed for "openai" provider) ---
# IMPORTANT: Replace with your actual key if using the OpenAI provider.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE")

# --- Model Configuration ---
# This model is used if provider is "huggingface"
HUGGINGFACE_MODEL = "microsoft/phi-3-mini-4k-instruct"
# Alternative models optimized for Colab free tier:
# "microsoft/phi-2"                    # Smaller, faster (2.7B params)
# "microsoft/DialoGPT-small"           # Very small for testing
# "distilgpt2"                         # Distilled GPT-2 (much smaller)

# --- HuggingFace Quantization Configuration ---
# Quantization mode for HuggingFace models. Options:
# "none": No quantization (full precision, uses more memory)
# "8bit": 8-bit quantization using bitsandbytes (faster, less memory)
# "4bit": 4-bit quantization using bitsandbytes (fastest, least memory)
HUGGINGFACE_QUANTIZATION = "4bit"  # Changed to 4-bit for maximum memory efficiency on Colab

# This model is used if provider is "openai"
OPENAI_MODEL = "gpt-4o-mini"

# --- Benchmark Run Configuration ---
# How many samples from the dataset to run the benchmark on.
# Set to a small number (e.g., 5) for quick tests.
NUM_SAMPLES_TO_RUN = 3  # Keep small for Colab free tier

# --- Benchmark Task Configuration ---
DEFAULT_TASK = "reasoning"
DEFAULT_DATASET = "gsm8k"
DEFAULT_COMPRESSION_METHOD = "llmlingua2"
DEFAULT_DATASET_CONFIG = "main" # Some datasets have configs, e.g., 'main' for gsm8k
DEFAULT_TARGET_RATIO = 0.9  # Keep 90% of tokens 

# --- Colab Optimization Settings ---
# Memory management settings optimized for Colab free tier
ENABLE_MEMORY_MONITORING = True
MEMORY_WARNING_THRESHOLD = 0.8  # Warn when 80% of GPU memory is used
MAX_SEQUENCE_LENGTH = 512  # Limit input length to save memory
BATCH_SIZE = 4  # Increased from 1 for better GPU utilization
CLEAR_MEMORY_EVERY_N_SAMPLES = 2  # Less aggressive cleanup
ENABLE_PARALLEL_PROCESSING = True  # New: Enable parallel compression
MAX_CONCURRENT_PROCESSES = 2  # Parallel processes for compression
PRELOAD_MODEL = True  # Pre-load model for better performance
ENABLE_MODEL_WARMUP = True  # Warm up model before benchmark


