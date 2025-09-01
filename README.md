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
- **Advanced Summarization Evaluation**: Style-aware scoring and qualitative analysis
- **Comprehensive Analysis Tools**: Built-in benchmark analyzer with visualizations and detailed reports

## 🏗️ Architecture Overview

The project follows a clean, modular architecture with clear separation of concerns and modern software engineering principles:

```
pcm-llm/
├── core/                    # Core business logic and services
│   ├── bootstrap.py         # Application bootstrap and DI container
│   ├── container.py         # Dependency injection container
│   ├── config/              # Configuration management
│   │   ├── settings.py      # Centralized settings (12-factor app)
│   │   └── config_manager.py # Configuration provider interface
│   ├── benchmark_service.py # Main benchmarking orchestration
│   ├── cli.py              # Command-line interface
│   └── llm_factory.py      # LLM provider factory
├── compressors/             # Prompt compression algorithms
├── llms/                    # LLM provider implementations
├── data_loaders/            # Dataset loading utilities
├── evaluation/              # Performance evaluation logic
├── utils/                   # Shared utilities and helpers
├── tests/                   # Test suite (unit, integration, functional)
├── results/                 # Benchmark results and analysis
├── analysis_output/         # Generated analysis reports and visualizations
├── compressed_cache/         # Compression cache storage
├── logs/                    # Application logs
├── models/                  # Local model storage
├── old_analyzers_backup/    # Legacy analyzer implementations
├── benchmark_analyzer.py    # Comprehensive analysis and reporting tool
├── quick_analyzer.py        # Quick analysis utilities
├── main.py                  # Main entry point
├── config.py                # Configuration file
├── pyproject.toml           # Modern Python packaging
├── requirements.txt         # Python dependencies
├── pytest.ini              # Test configuration
├── Makefile                 # Development automation
├── .pre-commit-config.yaml  # Code quality automation
└── README.md               # This file
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


