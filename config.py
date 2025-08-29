import os

# --- LLM Provider Selection ---
# Choose your desired LLM provider here. Options are:
# "huggingface": Use a free, open-source model in Colab/locally. (Recommended for free use)
# "openai"     : Use the paid OpenAI API. Requires an API key.
# "manual"     : Manually copy/paste the prompt and response from a chat platform.
# DEFAULT_LLM_PROVIDER = "manual"
DEFAULT_LLM_PROVIDER = "llamacpp"  # options: 'huggingface' | 'openai' | 'manual' | 'llamacpp'

# --- API Keys (only needed for "openai" provider) ---
# IMPORTANT: Replace with your actual key if using the OpenAI provider.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE")

# --- Model Configuration ---
# This model is used if provider is "huggingface"
# Prefer a smaller, MPS-friendly base to avoid heavy loads on macOS
HUGGINGFACE_MODEL = "microsoft/phi-2"
# Alternative models:
# "TheBloke/phi-3-mini-4k-instruct-4bit"  # 4-bit quantized (smaller, faster)
# "microsoft/phi-3-mini-4k-instruct"      # Full precision (larger, slower)

# --- HuggingFace Quantization Configuration ---
# Quantization mode for HuggingFace models. Options:
# "none": No quantization (full precision, uses more memory)
# "8bit": 8-bit quantization using bitsandbytes (faster, less memory)
# "4bit": 4-bit quantization using bitsandbytes (fastest, least memory)
# On MPS, bitsandbytes quantization is not supported; use full precision
HUGGINGFACE_QUANTIZATION = "none"

# This model is used if provider is "openai"
OPENAI_MODEL = "gpt-4o-mini"

# --- llama.cpp (GGUF) Configuration ---
# Set provider to 'llamacpp' to use these settings. Provide a local GGUF 4-bit file path.
# Example models (download separately):
# - TheBloke/Llama-2-7B-GGUF (Q4_K_M)
# - bartowski/Qwen2.5-7B-Instruct-GGUF (Q4_K_M)
# - TinyLlama/TinyLlama-1.1B-Chat-v1.0-GGUF (Q4)
LLAMACPP_MODEL_PATH = os.getenv("LLAMACPP_MODEL_PATH", "/path/to/model.Q4_K_M.gguf")
LLAMACPP_N_CTX = int(os.getenv("LLAMACPP_N_CTX", 4096))
LLAMACPP_N_GPU_LAYERS = int(os.getenv("LLAMACPP_N_GPU_LAYERS", -1))  # -1 to offload all layers to Metal if available
LLAMACPP_N_THREADS = int(os.getenv("LLAMACPP_N_THREADS", 0)) or None  # auto-detect if 0

# Optional: load from Hugging Face Hub directly (ignore MODEL_PATH when set)
# Example: repo_id="microsoft/Phi-3-mini-4k-instruct-gguf", filename="Phi-3-mini-4k-instruct-q4.gguf"
LLAMACPP_REPO_ID = "microsoft/Phi-3-mini-4k-instruct-gguf"
LLAMACPP_FILENAME = "Phi-3-mini-4k-instruct-q4.gguf"

# --- Benchmark Run Configuration ---
# How many samples from the dataset to run the benchmark on.
# Set to a small number (e.g., 5) for quick tests.
NUM_SAMPLES_TO_RUN = 3

# --- Benchmark Task Configuration ---
DEFAULT_TASK = "reasoning"
DEFAULT_DATASET = "gsm8k"
DEFAULT_COMPRESSION_METHOD = "llmlingua2"
DEFAULT_DATASET_CONFIG = "main" # Some datasets have configs, e.g., 'main' for gsm8k
DEFAULT_TARGET_RATIO = 0.9  # Keep 90% of tokens 


# --- Runtime Verbosity / Streaming ---
# When True, tokens stream to stdout during generation (can be noisy).
# Set to False to disable token-by-token printing and only return final text.
STREAM_TOKENS = os.getenv("STREAM_TOKENS", "0") not in {"0", "false", "False"}


