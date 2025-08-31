## Prompt Compression Benchmark (Starter)

This repository provides a unified framework to test and evaluate prompt compression methods for Large Language Models (LLMs). It includes a mock LLM and a placeholder LLMLingua-2 compressor so you can run end-to-end benchmarks without API costs.

### Structure

- `config.py`: Central configuration (model, dataset, compression method, etc.)
- `main.py`: Entry point to run a single benchmark
- `llms/`: LLM interfaces and implementations
  - `base.py`: `BaseLLM` abstract class
  - `manual_llm.py`: Manual copy-paste workflow
  - `openai_llm.py`: OpenAI API-backed LLM
  - `huggingface_llm.py`: Local/Colab Hugging Face LLM (with smart device detection)
  - `factory.py`: `LLMFactory` to select provider
- `compressors/`: Prompt compression algorithms
  - `base.py`: `BaseCompressor` abstract class
  - `llmlingua2.py`: Placeholder LLMLingua-2 implementation
  - `factory.py`: Factory to instantiate compressors
- `datasets/`: Data loading utilities
  - `loaders.py`: Simple dataset sample loader (GSM8K mock)
- `evaluation/`: Evaluation orchestration
  - `evaluator.py`: Latency + accuracy (reasoning) evaluation

### Quickstart

1. (Optional) Create and activate a virtual environment
   - macOS/Linux:
     ```bash
     python3 -m venv .venv && source .venv/bin/activate
     ```
   - Windows (PowerShell):
     ```powershell
     python -m venv .venv; .venv\\Scripts\\Activate.ps1
     ```
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Configure API keys
   - ### Compression Caching System

The system now includes intelligent caching to avoid recompressing the same prompts:

#### **Features:**
- ✅ **Automatic Caching**: Compressed prompts are saved to `compressed_cache/` directory
- ✅ **Smart Loading**: Checks cache before running compression
- ✅ **Persistent Storage**: Samples and compressed prompts are kept between runs
- ✅ **Integrity Checks**: Validates cache completeness and parameters

#### **Cache Structure:**
```
compressed_cache/
├── samples/           # Original dataset samples
│   ├── reasoning_100_samples.json
│   ├── summarization_100_samples.json
│   └── classification_100_samples.json
└── compressed/        # Compressed prompts
    ├── reasoning_llmlingua2_a1b2c3d4.json
    ├── summarization_llmlingua2_e5f6g7h8.json
    └── classification_llmlingua2_i9j0k1l2.json
```

**Note:** The `compressed_cache/` directory is preserved between runs and contains your processed data. It's not ignored by git so your cached work is maintained.

#### **Cache Management:**
```python
# Check cache status
from main import show_cache_info
show_cache_info()

# Clear entire cache
from main import clear_compression_cache
clear_compression_cache()

# Clear specific task cache
clear_compression_cache('reasoning')

# Clear specific method for a task
clear_compression_cache('reasoning', 'llmlingua2')
```

#### **Performance Benefits:**
- 🚀 **First Run**: Compresses all prompts and saves to cache
- ⚡ **Subsequent Runs**: Loads from cache instantly (no compression needed)
- 💾 **Disk Space**: Efficient JSON storage with metadata
- 🔄 **Automatic Updates**: Recompresses only when parameters change
4. Select your LLM provider in `config.py`
   - Set `DEFAULT_LLM_PROVIDER` to one of: `"manual"`, `"openai"`, `"huggingface"`, `"llamacpp"`, `"openrouter"`
   - For OpenAI: configure `OPENAI_API_KEY` in `api_keys.py`
   - For OpenRouter: configure `OPENROUTER_API_KEY` in `api_keys.py` (get from https://openrouter.ai/keys)
   - For Hugging Face: ensure you have sufficient RAM/VRAM
   - For Llama.cpp: download GGUF models or use HuggingFace repo IDs
5. Run the benchmark
   ```bash
   python main.py
   ```

#### **Command Line Options:**
```bash
# Show cache information
python main.py cache-info

# Clear entire cache
python main.py clear-cache

# Clear cache for specific task
python main.py clear-cache reasoning

# Clear specific compression method for a task
python main.py clear-cache reasoning llmlingua2

# Show help
python main.py help
```

### API Keys Configuration

API keys are now managed through the `api_keys.py` file for better security and organization:

1. **Edit `api_keys.py`** and replace placeholder values with your actual API keys
2. **The file is automatically excluded** from git (added to `.gitignore`)
3. **Fallback support**: If keys aren't set in `api_keys.py`, the system falls back to environment variables
4. **Never commit actual API keys** to version control

**Example `api_keys.py` setup:**
```python
# OpenRouter API key (get from https://openrouter.ai/keys)
OPENROUTER_API_KEY = "sk-or-v1-..."

# OpenAI API key (get from https://platform.openai.com/api-keys)
OPENAI_API_KEY = "sk-..."
```
```

### Notes

- The default provider is `manual`, which pauses and lets you paste responses from any chat platform.
- Use `openai` provider for API-backed runs (requires `OPENAI_API_KEY`).
- Use `huggingface` provider to run open-source models locally/Colab (may require GPU and large downloads).
- The HuggingFace implementation automatically handles device selection and optimization.
- Add additional datasets and tasks by extending `datasets/` and `evaluation/` modules.
- Add new compression methods by implementing `BaseCompressor` and registering them in `compressors/factory.py`.


