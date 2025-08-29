## Prompt Compression Benchmark (Starter)

This repository provides a unified framework to test and evaluate prompt compression methods for Large Language Models (LLMs). It includes a mock LLM and a placeholder LLMLingua-2 compressor so you can run end-to-end benchmarks without API costs.

### Structure

- `config.py`: Central configuration (model, dataset, compression method, etc.)
- `main.py`: Entry point to run a single benchmark
- `llms/`: LLM interfaces and implementations
  - `base.py`: `BaseLLM` abstract class
  - `manual_llm.py`: Manual copy-paste workflow
  - `openai_llm.py`: OpenAI API-backed LLM
  - `huggingface_llm.py`: Local/Colab Hugging Face LLM
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
3. Select your LLM provider in `config.py`
   - Set `DEFAULT_LLM_PROVIDER` to one of: `"manual"`, `"openai"`, `"huggingface"`
   - For OpenAI, set your key: `export OPENAI_API_KEY=sk-...`
   - For Hugging Face, ensure you have GPU (optional) and sufficient RAM/VRAM
4. Run the benchmark
   ```bash
   python main.py
   ```

### Notes

- The default provider is `manual`, which pauses and lets you paste responses from any chat platform.
- Use `openai` provider for API-backed runs (requires `OPENAI_API_KEY`).
- Use `huggingface` provider to run open-source models locally/Colab (may require GPU and large downloads).
- Add additional datasets and tasks by extending `datasets/` and `evaluation/` modules.
- Add new compression methods by implementing `BaseCompressor` and registering them in `compressors/factory.py`.


