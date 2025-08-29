import os

# --- Benchmark Run Configuration ---
DEFAULT_TASK = "reasoning"
DEFAULT_DATASET = "gsm8k"
DEFAULT_DATASET_CONFIG = "main"

# --- Compression Method Selection ---
# A list of all compression methods to run in the benchmark.
# Available options: "llmlingua2", "selective_context"
COMPRESSION_METHODS_TO_RUN = ["llmlingua2", "selective_context"]
DEFAULT_TARGET_RATIO = 0.9 # Keep 80% of tokens

NUM_SAMPLES_TO_RUN = 10

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
LLAMACPP_N_CTX = 2048
LLAMACPP_N_GPU_LAYERS = 0  # Set to -1 for all layers on GPU
LLAMACPP_N_THREADS = 4

# --- Runtime Verbosity / Streaming ---
STREAM_TOKENS = os.getenv("STREAM_TOKENS", "0") not in {"0", "false", "False"}


