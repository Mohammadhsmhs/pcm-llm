import os

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
NUM_SAMPLES_TO_RUN = 1  # Increased for more realistic benchmarking results

# --- Compression Method Selection ---
# A list of all compression methods to run in the benchmark.
# Available options: "llmlingua2", "selective_context"
COMPRESSION_METHODS_TO_RUN = ["llmlingua2", "selective_context"]
DEFAULT_TARGET_RATIO = 0.9 # Keep 80% of tokens

 

# --- LLM Provider Selection ---
DEFAULT_LLM_PROVIDER = "llamacpp"

# --- HuggingFace Configuration ---
HUGGINGFACE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
HUGGINGFACE_QUANTIZATION = "4bit"  # Options: "none", "4bit", "8bit"

# --- OpenAI Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-3.5-turbo"

# --- Llama.cpp Configuration ---
LLAMACPP_MODEL_PATH = ""  # Local model path (leave empty if using repo_id)
LLAMACPP_REPO_ID = "microsoft/Phi-3-mini-4k-instruct-gguf"
LLAMACPP_FILENAME = "Phi-3-mini-4k-instruct-q4.gguf"
LLAMACPP_N_CTX = 4096  # Increased context window for realistic benchmarking
LLAMACPP_N_GPU_LAYERS = 0  # Set to -1 for all layers on GPU
LLAMACPP_N_THREADS = 4

# --- Runtime Verbosity / Streaming ---
STREAM_TOKENS = False  # Disable streaming for better benchmarking performance

# --- Unlimited Mode Configuration ---
# When enabled, removes all timeouts and size limits for full-speed benchmarking
# WARNING: This may cause very long run times and high resource usage
UNLIMITED_MODE = False  # Set to True to disable all restrictions


