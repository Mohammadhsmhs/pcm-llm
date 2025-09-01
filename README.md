# PCM-LLM: Prompt Compression Benchmark Tool

A comprehensive, production-ready benchmarking framework for evaluating prompt compression methods across different Large Language Models (LLMs) and tasks. This tool provides a unified interface to test compression algorithms, measure performance metrics, and generate detailed analysis reports.

## 🚀 Features

- **Multi-LLM Support**: OpenAI, HuggingFace, Llama.cpp, OpenRouter, Ollama, and manual modes
- **Advanced Compression Methods**: LLMLingua-2, Selective Context, and Naive Truncation
- **Comprehensive Evaluation**: Reasoning, summarization, and classification tasks
- **Intelligent Caching**: Persistent compression cache for efficient re-runs
- **Production Architecture**: SOLID principles, dependency injection, and clean interfaces
- **Real-time Monitoring**: Memory usage, progress tracking, and detailed logging
- **Unlimited Mode**: Configurable timeouts and resource limits for research use

## 🏗️ Architecture Overview

The project follows a clean, modular architecture with clear separation of concerns:

```
pcm-llm/
├── core/                    # Core business logic and services
│   ├── benchmark_service.py # Main benchmarking orchestration
│   ├── cli.py              # Command-line interface
│   ├── llm_factory.py      # LLM provider factory
│   └── config/             # Configuration management
├── compressors/             # Prompt compression algorithms
├── llms/                    # LLM provider implementations
├── data_loaders/            # Dataset loading utilities
├── evaluation/              # Performance evaluation logic
├── utils/                   # Shared utilities and helpers
└── results/                 # Benchmark results and analysis
```

## 🔧 Core Components

### 1. Benchmark Service (`core/benchmark_service.py`)

The central orchestrator that coordinates the entire benchmarking process:

- **Task Management**: Handles reasoning, summarization, and classification tasks
- **Compression Pipeline**: Applies multiple compression methods to prompts
- **Evaluation Orchestration**: Manages LLM interactions and metric calculation
- **Result Aggregation**: Collects and processes benchmark results

```python
class BenchmarkService(IBenchmarkService):
    def run_single_task_benchmark(self, task_name: str) -> List[BenchmarkResult]:
        # Load dataset → Compress prompts → Evaluate with LLM → Generate results
```

### 2. LLM Factory (`core/llm_factory.py`)

Factory pattern implementation for creating LLM instances:

- **Provider Selection**: Dynamically selects LLM implementation based on configuration
- **Configuration Management**: Handles provider-specific settings
- **Error Handling**: Graceful fallbacks and validation

Supported providers:
- `openai`: GPT models via OpenAI API
- `huggingface`: Local/remote HuggingFace models
- `llamacpp`: Local GGUF models via llama.cpp
- `openrouter`: Multiple models via OpenRouter API
- `ollama`: Local models via Ollama
- `manual`: Copy-paste workflow for any LLM

### 3. Compression Framework (`compressors/`)

Extensible framework for prompt compression algorithms:

#### Base Compressor Interface
```python
class BaseCompressor(ABC):
    @abstractmethod
    def compress(self, prompt: str, target_ratio: float) -> str:
        """Compress prompt to target token ratio"""
```

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

### 5. Evaluation Engine (`evaluation/`)

Comprehensive performance measurement:

- **Task-Specific Metrics**: Reasoning accuracy, summarization quality, classification precision
- **Latency Measurement**: Response time tracking with timeout protection
- **Memory Monitoring**: Resource usage tracking and optimization
- **Adaptive Timeouts**: Smart timeout configuration based on prompt length

## 📊 Supported Tasks

### 1. Mathematical Reasoning (GSM8K)
- **Dataset**: GSM8K mathematical word problems
- **Evaluation**: Step-by-step reasoning accuracy
- **Metrics**: Answer correctness, solution completeness

### 2. Text Summarization (CNN DailyMail)
- **Dataset**: News articles with human-written summaries
- **Evaluation**: Summary quality and relevance
- **Metrics**: ROUGE scores, content preservation

### 3. Sentiment Classification (IMDB)
- **Dataset**: Movie review sentiment analysis
- **Evaluation**: Binary classification accuracy
- **Metrics**: Precision, recall, F1-score

## ⚙️ Configuration

### Main Configuration (`config.py`)

```python
# Task Configuration
SUPPORTED_TASKS = ["reasoning", "summarization", "classification"]
DEFAULT_TASK = "reasoning"

# Compression Methods
COMPRESSION_METHODS_TO_RUN = ["llmlingua2", "selective_context", "naive_truncation"]
DEFAULT_TARGET_RATIO = 0.9  # Keep 90% of tokens

# LLM Provider Settings
DEFAULT_LLM_PROVIDER = "ollama"
HUGGINGFACE_MODEL = "microsoft/Phi-3.5-mini-instruct"
OPENAI_MODEL = "gpt-3.5-turbo"

# Performance Settings
UNLIMITED_MODE = True  # Disable timeouts for research
ENABLE_CHECKPOINTING = True  # Save progress during long runs
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
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\Activate.ps1  # Windows PowerShell

# Install dependencies
pip install -r requirements.txt
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
```

### Key Metrics
- **Compression Ratio**: Token reduction achieved
- **Performance Score**: Task-specific accuracy metrics
- **Latency**: Response time in seconds
- **Memory Usage**: Resource consumption tracking
- **Quality Preservation**: Semantic content retention

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

### Performance Optimization

- **Batch Processing**: Adjust batch sizes based on available memory
- **Checkpointing**: Save progress during long runs
- **Resource Monitoring**: Track memory usage and optimize accordingly
- **Concurrent Processing**: Configure logger concurrency limits

## 📚 Dependencies

### Core Dependencies
- **transformers** (≥4.40.0): HuggingFace model support
- **torch** (≥2.1.0): PyTorch backend
- **llama-cpp-python** (≥0.2.90): Local GGUF model inference
- **openai** (≥1.3.0): OpenAI API integration
- **datasets** (≥2.19.0): Dataset loading and management

### Optional Dependencies
- **llmlingua** (≥0.2.2): Advanced prompt compression
- **accelerate** (≥0.29.0): HuggingFace optimization
- **bitsandbytes** (≥0.41.0): Quantization support

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/new-feature`
3. **Implement your changes** following the existing architecture
4. **Add tests** for new functionality
5. **Submit a pull request** with detailed description

### Development Guidelines
- Follow SOLID principles and clean architecture
- Maintain consistent error handling and logging
- Add comprehensive documentation for new features
- Ensure backward compatibility for configuration changes

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

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


