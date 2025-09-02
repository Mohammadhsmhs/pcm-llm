# PCM-LLM: An Advanced Framework for Prompt Compression Benchmarking

This repository contains a comprehensive, production-ready benchmarking tool designed to evaluate various prompt compression methods across different Large Language Models (LLMs) and tasks. It provides a unified, extensible interface for testing compression algorithms, measuring performance metrics, and generating insightful analysis reports.

## 🚀 Key Features

- **Multi-LLM Support**: Seamless integration with OpenAI, HuggingFace, Llama.cpp, OpenRouter, and Ollama.
- **Advanced Compression Methods**: Includes LLMLingua-2, Selective Context, and Naive Truncation.
- **Diverse Task Evaluation**: Benchmarks against standard NLP tasks: reasoning, summarization, and classification.
- **Intelligent Caching**: Robust caching for compressed prompts and baseline LLM outputs to accelerate re-runs.
- **Production-Grade Architecture**: Built on SOLID principles with a clean, modular design, dependency injection, and a service-oriented architecture.
- **Real-time Monitoring**: Detailed logging, progress tracking, and memory usage monitoring.
- **In-depth Analysis**: A powerful, built-in benchmark analyzer that generates detailed Markdown reports.

## 🏗️ Architecture

The project follows a clean, modular architecture to ensure maintainability and extensibility.

```
pcm-llm/
├── core/                    # Core application logic
│   ├── bootstrap.py         # Dependency Injection (DI) setup
│   ├── cli.py               # Command-line interface (Typer)
│   ├── container.py         # Custom DI container
│   ├── config/              # Configuration management (settings.py)
│   ├── llm_factory.py       # Factory for creating LLM instances
│   └── pipeline/            # Core processing pipelines
│       ├── data_loader_pipeline.py
│       ├── compression_pipeline.py
│       └── evaluation_pipeline.py
│   └── benchmark_service.py # Main service orchestrating the pipelines
├── compressors/             # Implementations of prompt compression algorithms
├── llms/                    # LLM provider implementations (OpenAI, Ollama, etc.)
├── data_loaders/            # Utilities for loading datasets
├── evaluation/              # Performance evaluation and scoring logic
├── utils/                   # Shared utilities (caching, logging, etc.)
├── tests/                   # Unit and integration tests
├── results/                 # Raw benchmark output (CSV files)
├── analysis_output/         # Generated analysis reports (.md)
├── compressed_cache/        # Persistent cache for prompts and results
├── logs/                    # Application and run logs
├── main.py                  # Main entry point
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## 🔧 Core Components

### 1. Benchmark Service & Pipelines (`core/`)

The `BenchmarkService` orchestrates the workflow through a series of pipelines:

- **`DataLoaderPipeline`**: Loads and caches datasets for all specified tasks.
- **`CompressionPipeline`**: Applies compression methods to the data, using caching to speed up reruns.
- **`EvaluationPipeline`**: Evaluates the performance of compressed prompts against a baseline, calculating key metrics.

This architecture optimizes execution by:
1.  Loading all data once.
2.  Compressing all data for each method.
3.  Evaluating all results for each task.

### 2. LLM Factory (`core/llm_factory.py`)

A factory for creating LLM instances from different providers, abstracting away the specific implementation details.

### 3. Caching System (`utils/cache/cache_utils.py`)

An intelligent caching system for compressed prompts, baseline model outputs, and dataset samples. This dramatically speeds up subsequent benchmark runs.

## 📊 Benchmark Analyzer

The `benchmark_analyzer.py` script processes the raw CSV results, performs a comprehensive analysis, and generates a detailed Markdown report.

**Key Metrics Analyzed:**
- **Score Preservation**: How much of the original model's performance is retained.
- **Compression Ratio**: The percentage of tokens/characters removed.
- **Latency Overhead**: The change in generation time.
- **Efficiency Score**: A composite score balancing preservation and compression.

## 🛠️ Getting Started

### 1. Prerequisites
- Python 3.9+
- An OpenAI API key (if using OpenAI models)
- Ollama installed and running (if using local Ollama models)

### 2. Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/Mohammadhsmhs/pcm-llm.git
cd pcm-llm
pip install -r requirements.txt
```

### 3. Configuration

The main configuration is handled in `core/config/settings.py`. You can set your default LLM provider, model names, API keys, and other parameters.

**Example: Setting the default LLM to a local Ollama model**

In `core/config/settings.py`:

```python
class Settings:
    def __init__(self):
        # ...
        self.default_llm_provider = "ollama"
        # ...

    def _load_default_settings(self):
        # ...
        self.llm_configs = {
            "ollama": LLMSettings(
                provider="ollama",
                model_name="hf.co/microsoft/Phi-3-mini-4k-instruct-gguf:latest", # Or your preferred model
            ),
            # ... other providers
        }
```

### 4. Running the Benchmark

The application is controlled via `main.py`.

**Available Commands:**

- `all`: Run all available benchmarks.
- `reasoning`: Run the reasoning task benchmark.
- `summarization`: Run the summarization task benchmark.
- `classification`: Run the classification task benchmark.
- `cache-info`: Show the status of the cache.
- `clear-cache [task] [method]`: Clear the cache. Can be used for all, a specific task, or a specific method.
- `help`: Show the help message.

**Options:**

- `--sample N`: Specify the number of samples to run, overriding the config file.

**Examples:**

- **Run all benchmarks with 10 samples:**
  ```bash
  python main.py all --sample 10
  ```

- **Run only the reasoning benchmark:**
  ```bash
  python main.py reasoning
  ```

- **Clear the cache for the `llmlingua2` method on the `reasoning` task:**
  ```bash
  python main.py clear-cache reasoning llmlingua2
  ```

- **View cache information:**
  ```bash
  python main.py cache-info
  ```

### 5. Analyzing Results

After a benchmark run, a CSV file is generated in the `results/` directory. Use the `benchmark_analyzer.py` script to generate a detailed analysis report.

```bash
python benchmark_analyzer.py <path_to_csv_file>
```

This will create a Markdown report in the `analysis_output/` directory.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.
bash
git clone https://github.com/Mohammadhsmhs/pcm-llm.git
cd pcm-llm
pip install -r requirements.txt
```

### 3. Configuration

The application is configured via environment variables and the `core/config/settings.py` file. Key settings include:
- `PCM_DEFAULT_LLM_PROVIDER`: The primary LLM provider to use (e.g., `ollama`, `openai`).
- `PCM_LOG_LEVEL`: Logging verbosity.
- `PCM_CACHE_DIR`: Path to the cache directory.

### 4. Running the Benchmark

The command-line interface (`main.py`) provides a simple way to run benchmarks.

**Run all tasks with default settings:**
```bash
python main.py all
```

**Run a specific task with a limited number of samples:**
```bash
python main.py run-task --task-name summarization --sample 3
```

**Run the benchmark for all tasks and then analyze the results:**
```bash
python main.py all && python benchmark_analyzer.py
```

## 📈 Analyzing Results

After running a benchmark, use the `benchmark_analyzer.py` script to generate a detailed analysis report:

```bash
python benchmark_analyzer.py
```

This will process the latest CSV files in the `results/` directory and save a Markdown report and PNG visualizations to the `analysis_output/` directory.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any bugs, feature requests, or improvements.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.


#### Available Compressors

1. **LLMLingua-2** (`llmlingua2.py`)
   - State-of-the-art prompt compression using LLMLingua library
   - Semantic-aware compression with quality preservation
   - Configurable compression ratios

2. **Selective Context** (`selective_context.py`)
   - Intelligent context selection based on relevance scoring
   - Maintains semantic coherence while reducing length
   - Advanced heuristics for context importance

3. **Naive Truncation** (`naive_truncation.py`)
   - Simple token-based truncation using BERT tokenizer
   - Baseline comparison method
   - Fast and deterministic

### 4. Data Loading System (`data_loaders/`)

Flexible dataset management for different task types:

- **GSM8K**: Mathematical reasoning problems
- **CNN DailyMail**: News summarization articles
- **IMDB**: Sentiment classification reviews
- **Extensible**: Easy to add new datasets

```python
def load_benchmark_dataset(task_config: TaskConfig, num_samples: int) -> tuple:
    """Load and prepare dataset samples for benchmarking"""
```

### 6. Benchmark Analyzer (`benchmark_analyzer.py`)

Comprehensive analysis and reporting tool for benchmark results:

- **Automated Analysis**: Processes CSV results and generates detailed reports
- **Advanced Metrics**: Efficiency scores, preservation percentages, compression ratios
- **Visualization Support**: Optional matplotlib/seaborn plots (if available)
- **Method Comparison**: Side-by-side analysis of compression methods
- **Task Analysis**: Detailed breakdown by task type
- **Professional Reports**: Markdown-formatted analysis with rankings and recommendations

```python
from benchmark_analyzer import BenchmarkAnalyzer

analyzer = BenchmarkAnalyzer()
results = analyzer.run_complete_analysis()
# Generates comprehensive report with visualizations
```

### 7. Quick Analyzer (`quick_analyzer.py`)

Lightweight analysis tool for rapid result inspection:

- **Fast Analysis**: Quick CSV processing without full report generation
- **Summary Statistics**: Key metrics and performance indicators
- **CSV Export**: Easy data manipulation and further analysis

## 📊 Supported Tasks

### 1. Mathematical Reasoning (GSM8K)
- **Dataset**: GSM8K mathematical word problems
- **Evaluation**: Step-by-step reasoning accuracy
- **Metrics**: Answer correctness, solution completeness

### 2. Text Summarization (CNN DailyMail)
- **Dataset**: News articles with human-written summaries
- **Evaluation**: Summary quality and relevance with style-aware scoring
- **Metrics**: ROUGE scores, content preservation, style consistency
- **Advanced Features**: 
  - Style-aware scoring that rewards appropriate brevity
  - Qualitative analysis of content preservation and factual consistency
  - Length adjustment bonuses for responses matching ground truth style

### 3. Sentiment Classification (IMDB)
- **Dataset**: Movie review sentiment analysis
- **Evaluation**: Binary classification accuracy
- **Metrics**: Precision, recall, F1-score

## ⚙️ Configuration

### Modern Configuration System (`core/config/`)

The project uses a modern 12-factor configuration system with environment variables:

```python
# Environment Variables (recommended)
export PCM_DEFAULT_TASK=reasoning
export PCM_DEFAULT_LLM_PROVIDER=ollama
export PCM_NUM_SAMPLES=3
export PCM_OPENAI_API_KEY=your_key_here
export PCM_OPENROUTER_API_KEY=your_key_here

# Or use the modern settings API
from core.config import settings

# Task Configuration
settings.default_task = "reasoning"
settings.performance.num_samples = 3

# Compression Methods
settings.compression.methods = ["llmlingua2", "selective_context", "naive_truncation"]
settings.compression.target_ratio = 0.9

# LLM Provider Settings
settings.default_llm_provider = "ollama"
settings.llm_providers["huggingface"].model_name = "microsoft/Phi-3.5-mini-instruct"
settings.llm_providers["openai"].model_name = "gpt-3.5-turbo"

# Performance Settings
settings.evaluation.unlimited_mode = True
settings.performance.enable_checkpointing = True

# Evaluation Settings
settings.evaluation.enable_qualitative_analysis = True
settings.evaluation.enable_style_aware_scoring = True
```

### Environment Variables

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# OpenRouter
export OPENROUTER_API_KEY="sk-or-v1-..."

# HuggingFace (optional)
export HF_TOKEN="hf_..."
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd pcm-llm

# Create virtual environment
make venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\Activate.ps1  # Windows PowerShell

# Install dependencies
make install-dev  # For development with all tools
# OR
make install      # For production use only
```

### 2. Basic Usage

```bash
# Run default benchmark (reasoning task)
python main.py

# Run specific task
python main.py reasoning
python main.py summarization
python main.py classification

# Run multiple tasks
python main.py reasoning summarization classification
```

### 3. Advanced Commands

```bash
# Cache management
python main.py cache-info          # Show cache status
python main.py clear-cache         # Clear entire cache
python main.py clear-cache reasoning  # Clear specific task cache

# Analysis tools
python benchmark_analyzer.py       # Run comprehensive analysis with reports
python quick_analyzer.py           # Quick analysis of latest results

# Help and information
python main.py help                # Show all available commands
```

## 🔄 Compression Caching System

The intelligent caching system eliminates redundant compression work:

### Cache Structure
```
compressed_cache/
├── samples/           # Original dataset samples
│   ├── reasoning_100_samples.json
│   ├── summarization_100_samples.json
│   └── classification_100_samples.json
└── compressed/        # Compressed prompts with metadata
    ├── reasoning_llmlingua2_a1b2c3d4.json
    ├── summarization_selective_context_e5f6g7h8.json
    └── classification_naive_truncation_i9j0k1l2.json
```

### Cache Benefits
- **First Run**: Compresses all prompts and saves to cache
- **Subsequent Runs**: Loads from cache instantly
- **Parameter Validation**: Recompresses only when settings change
- **Persistent Storage**: Maintains cache between runs

## 📈 Results and Analysis

### Output Structure
```
results/
├── benchmark_[task]_[methods]_[timestamp].csv      # Raw benchmark data
├── benchmark_[task]_[methods]_[timestamp]_summary.json  # Aggregated metrics
├── analysis_[task]_[methods]_[timestamp].md        # Detailed analysis
├── task_log_[timestamp].csv                        # Task execution log
├── run_info_[timestamp].json                       # Run configuration
└── realtime_[timestamp].log                        # Real-time execution log

analysis_output/
├── benchmark_analysis_report_[timestamp].md        # Comprehensive analysis report
└── comprehensive_analysis.png                      # Performance visualizations (if matplotlib available)
```

### Analysis Tools

#### Benchmark Analyzer
```bash
python benchmark_analyzer.py
```
- **Comprehensive Reports**: Detailed markdown reports with method rankings
- **Performance Metrics**: Efficiency scores, preservation rates, compression ratios
- **Visualizations**: Charts and graphs (requires matplotlib/seaborn)
- **Method Comparison**: Side-by-side analysis across all tasks
- **Recommendations**: AI-powered suggestions based on results

#### Quick Analyzer
```bash
python quick_analyzer.py
```
- **Fast Analysis**: Quick processing of latest benchmark results
- **Summary Statistics**: Key performance indicators
- **CSV Export**: Ready for further analysis in Excel or other tools

## 🛠️ Development and Extension

### Adding New Compression Methods

1. **Implement BaseCompressor**:
```python
class MyCompressor(BaseCompressor):
    def compress(self, prompt: str, target_ratio: float) -> str:
        # Your compression logic here
        return compressed_prompt
```

2. **Register in Factory**:
```python
# compressors/factory.py
COMPRESSOR_REGISTRY = {
    "my_method": MyCompressor,
    # ... existing methods
}
```

### Adding New LLM Providers

1. **Implement BaseLLM**:
```python
class MyLLM(BaseLLM):
    def get_response(self, prompt: str) -> str:
        # Your LLM integration here
        return response
```

2. **Register in Factory**:
```python
# core/llm_factory.py
LLM_REGISTRY = {
    "my_provider": MyLLM,
    # ... existing providers
}
```

### Adding New Tasks

1. **Extend TaskConfig**:
```python
# config.py
TASK_CONFIGURATIONS["new_task"] = {
    "dataset": "new_dataset",
    "config": "main",
    "description": "Description of new task"
}
```

2. **Implement Evaluation Logic**:
```python
# evaluation/evaluator.py
def _calculate_performance(self, response: str, ground_truth: str) -> tuple:
    if self.task == "new_task":
        # Your evaluation logic here
        return score, extracted_answer
```

## 🔍 Troubleshooting

### Common Issues

1. **Memory Errors**: Enable checkpointing and reduce batch sizes
2. **Timeout Issues**: Adjust timeout settings or enable unlimited mode
3. **API Rate Limits**: Configure rate limiting for external providers
4. **Cache Corruption**: Clear cache and re-run compression

### Summarization-Specific Issues

5. **Low ROUGE Scores**: 
   - **Cause**: Style mismatch between generated summaries and ground truth
   - **Solution**: Updated prompts now request concise, headline-style summaries
   - **Alternative**: Enable style-aware scoring for better evaluation
   
6. **Verbose Summaries**:
   - **Cause**: Model generating detailed summaries instead of concise ones
   - **Solution**: Use improved prompts that specify single-sentence format
   - **Monitoring**: Check qualitative analysis output for style feedback

### Performance Optimization

- **Batch Processing**: Adjust batch sizes based on available memory
- **Checkpointing**: Save progress during long runs
- **Resource Monitoring**: Track memory usage and optimize accordingly
- **Concurrent Processing**: Configure logger concurrency limits

## 📚 Dependencies

### Core Dependencies
- **transformers** (≥4.40.0): HuggingFace model support and tokenization
- **torch** (≥2.1.0): PyTorch backend for model inference
- **accelerate** (≥0.29.0): HuggingFace optimization and distributed training
- **bitsandbytes** (≥0.41.0): 4-bit quantization support
- **einops** (≥0.7.0): Tensor operations and reshaping
- **openai** (≥1.3.0): OpenAI API integration
- **datasets** (≥2.19.0): Dataset loading and management from HuggingFace Hub
- **llmlingua** (≥0.2.2): Advanced prompt compression using LLMLingua
- **llama-cpp-python** (≥0.2.90): Local GGUF model inference via llama.cpp

### Optional Dependencies (for enhanced features)
- **matplotlib** (≥3.5.0): Data visualization and plotting
- **seaborn** (≥0.11.0): Statistical data visualization
- **pandas** (≥1.5.0): Data manipulation and analysis
- **numpy** (≥1.21.0): Numerical computing
- **scikit-learn** (≥1.0.0): Machine learning utilities

### Installation Options

```bash
# Minimal installation (core functionality only)
pip install -r requirements.txt

# Full installation (with analysis and visualization)
pip install -r requirements.txt matplotlib seaborn pandas numpy scikit-learn

# Development installation (with all tools)
make install-dev
```

## 🔄 Recent Updates

### Version Highlights
- **Enhanced Benchmark Analyzer**: Comprehensive analysis tool with detailed reports and visualizations
- **Improved Formatting**: Fixed f-string formatting issues in analysis reports
- **Better Error Handling**: Robust error recovery and graceful degradation
- **Updated Dependencies**: Modern Python packages with improved compatibility
- **Enhanced Documentation**: Comprehensive README with current project structure

### Key Improvements
- ✅ **Analysis Tools**: New benchmark_analyzer.py for comprehensive result analysis
- ✅ **Visualization Support**: Optional matplotlib/seaborn integration for charts
- ✅ **Professional Reports**: Markdown-formatted analysis with method rankings
- ✅ **Bug Fixes**: Resolved formatting issues in report generation
- ✅ **Documentation**: Updated README reflecting current architecture

### Development Workflow

The project includes a comprehensive development automation system:

```bash
# Quick development cycle (format, lint, type-check, test, run)
make dev-cycle

# Code quality checks
make format          # Format code with black and isort
make lint            # Run linting checks
make type-check      # Run type checking with mypy

# Testing
make test            # Run all tests
make test-unit       # Run unit tests only
make test-integration # Run integration tests only

# Cache management
make cache-info      # Show cache status
make clear-cache     # Clear cache directories

# Development setup
make dev-setup       # Install dev dependencies and pre-commit hooks
make pre-commit      # Run all pre-commit checks
```

### Development Guidelines
- Follow SOLID principles and clean architecture
- Maintain consistent error handling and logging
- Add comprehensive documentation for new features
- Ensure backward compatibility for configuration changes
- Use the dependency injection container for new services
- Follow the established configuration patterns
- Run `make pre-commit` before committing code

##  Acknowledgments

- **LLMLingua Team**: For the prompt compression research and implementation
- **HuggingFace**: For the transformers library and model hub
- **OpenAI**: For API access and model availability
- **Community Contributors**: For feedback, bug reports, and feature requests

## 📞 Support

- **Issues**: Report bugs and feature requests via GitHub Issues
- **Discussions**: Join community discussions for questions and ideas
- **Documentation**: Check inline code documentation and examples
- **Examples**: Review test files for usage patterns

---

**Built with ❤️ for the AI research community**


