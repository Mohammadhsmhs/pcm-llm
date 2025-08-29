import os

# --- LLM Provider Selection ---
# Choose your desired LLM provider here. Options are:
# "huggingface": Use a free, open-source model in Colab/locally. (Recommended for free use)
# "openai"     : Use the paid OpenAI API. Requires an API key.
# "manual"     : Manually copy/paste the prompt and response from a chat platform.
DEFAULT_LLM_PROVIDER = "manual"
# DEFAULT_LLM_PROVIDER = "huggingface"

# --- API Keys (only needed for "openai" provider) ---
# IMPORTANT: Replace with your actual key if using the OpenAI provider.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE")

# --- Model Configuration ---
# This model is used if provider is "huggingface"
HUGGINGFACE_MODEL = "microsoft/phi-3-mini-4k-instruct"
# This model is used if provider is "openai"
OPENAI_MODEL = "gpt-4o-mini"

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


