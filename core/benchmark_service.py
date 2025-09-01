"""
Refactored benchmark service following SOLID principles.
"""

from typing import List, Dict, Any, Protocol
from abc import ABC, abstractmethod
from dataclasses import dataclass

from core.config import IConfigProvider, BenchmarkConfig, TaskConfig
from core.llm_factory import ILLMFactory
from data_loaders.loaders import load_benchmark_dataset
from compressors.factory import CompressorFactory
from evaluation.evaluator import Evaluator
from utils.logger import BenchmarkLogger
from utils.run_info_logger import RunInfoLogger
from utils.system_utils import clear_memory, log_memory_usage


@dataclass
class BenchmarkResult:
    """Result of a benchmark execution."""
    task_name: str
    compression_method: str
    sample_id: int
    original_prompt: str
    compressed_prompt: str
    original_score: float
    compressed_score: float
    latency: float
    memory_usage: float


class IDataLoader(Protocol):
    """Protocol for data loading."""

    def load_dataset(self, task_config: TaskConfig, num_samples: int) -> tuple:
        """Load dataset for a task."""
        ...


class ICompressor(Protocol):
    """Protocol for compression."""

    def compress(self, text: str, ratio: float) -> str:
        """Compress text with given ratio."""
        ...


class IEvaluator(Protocol):
    """Protocol for evaluation."""

    def evaluate(self, prompt: str, ground_truth: str) -> Dict[str, Any]:
        """Evaluate prompt against ground truth."""
        ...


class ILogger(Protocol):
    """Protocol for logging."""

    def log_result(self, result_data: Dict[str, Any]) -> None:
        """Log a benchmark result."""
        ...

    def finalize_and_save(self) -> None:
        """Finalize and save logging data."""
        ...

    def generate_summary_report(self) -> str:
        """Generate summary report."""
        ...

    def export_analysis_report(self) -> str:
        """Export analysis report."""
        ...


class IRunInfoLogger(Protocol):
    """Protocol for run info logging."""

    def log_run_info(self, run_info: Dict[str, Any]) -> None:
        """Log run information."""
        ...


class IBenchmarkService(ABC):
    """Interface for benchmark services."""

    @abstractmethod
    def run_single_task_benchmark(self, task_name: str) -> List[BenchmarkResult]:
        """Run benchmark for a single task."""
        pass

    @abstractmethod
    def run_multi_task_benchmark(self, task_names: List[str]) -> Dict[str, List[BenchmarkResult]]:
        """Run benchmark for multiple tasks."""
        pass


class BenchmarkService(IBenchmarkService):
    """Benchmark service implementation following SOLID principles."""

    def __init__(
        self,
        config_provider: IConfigProvider,
        llm_factory: ILLMFactory,
        data_loader: IDataLoader,
        logger: ILogger,
        run_info_logger: IRunInfoLogger
    ):
        self.config_provider = config_provider
        self.llm_factory = llm_factory
        self.data_loader = data_loader
        self.logger = logger
        self.run_info_logger = run_info_logger

    def run_single_task_benchmark(self, task_name: str) -> List[BenchmarkResult]:
        """Run benchmark for a single task."""
        print(f"\n🔧 Running benchmark for task: {task_name}")
        print("=" * 60)

        # Get configurations
        benchmark_config = self.config_provider.get_benchmark_config()
        task_config = self.config_provider.get_task_config(task_name)

        # Load dataset
        prompts, ground_truths = self.data_loader.load_dataset(task_config, benchmark_config.num_samples)
        log_memory_usage("after dataset loading", self.run_info_logger)

        results = []

        # Create LLM instance
        benchmark_config = self.config_provider.get_benchmark_config()
        # Use Ollama for real LLM testing
        default_provider = "ollama"  # Use real Ollama LLM
        llm_config = self.config_provider.get_llm_config(default_provider)
        llm = self.llm_factory.create_llm(default_provider, llm_config)
        evaluator = Evaluator(task=task_name, llm=llm)

        # Process each compression method
        for compression_method in benchmark_config.compression_methods:
            compressor = CompressorFactory.create(compression_method)

            for i, (prompt, ground_truth) in enumerate(zip(prompts, ground_truths)):
                # Compress prompt
                compressed_prompt = compressor.compress(prompt, benchmark_config.target_ratio)

                # Evaluate both original and compressed
                original_metrics = evaluator.evaluate(prompt, ground_truth)
                compressed_metrics = evaluator.evaluate(compressed_prompt, ground_truth)

                # Create result
                result = BenchmarkResult(
                    task_name=task_name,
                    compression_method=compression_method,
                    sample_id=i,
                    original_prompt=prompt,
                    compressed_prompt=compressed_prompt,
                    original_score=original_metrics.get('score', 0.0),
                    compressed_score=compressed_metrics.get('score', 0.0),
                    latency=compressed_metrics.get('latency', 0.0),
                    memory_usage=0.0  # TODO: Implement memory tracking
                )

                results.append(result)

                # Log result
                self.logger.log_result({
                    "sample_id": i,
                    "task": task_name,
                    "compression_method": compression_method,
                    "original_prompt": prompt,
                    "compressed_prompt": compressed_prompt,
                    "original_score": result.original_score,
                    "compressed_score": result.compressed_score,
                    "latency": result.latency
                })

            # Clean up
            del compressor
            clear_memory()

        # Finalize logging
        self.logger.finalize_and_save()
        summary = self.logger.generate_summary_report()
        analysis_file = self.logger.export_analysis_report()

        print(f"✅ {task_name.upper()} benchmark completed!")
        print(f"   📊 Results saved to: {getattr(self.logger, 'log_dir', 'results')}")
        print(f"   📈 Analysis report: {analysis_file}")

        return results

    def run_multi_task_benchmark(self, task_names: List[str]) -> Dict[str, List[BenchmarkResult]]:
        """Run benchmark for multiple tasks."""
        results = {}

        for task_name in task_names:
            task_results = self.run_single_task_benchmark(task_name)
            results[task_name] = task_results

        return results


class DataLoaderAdapter:
    """Adapter for existing data loader."""

    def load_dataset(self, task_config: TaskConfig, num_samples: int) -> tuple:
        """Load dataset using existing loader."""
        dataset = load_benchmark_dataset(
            task_config.name,
            task_config.dataset,
            task_config.config,
            num_samples
        )

        # Extract prompts and ground truths based on task type
        prompts = []
        ground_truths = []

        if task_config.name == "reasoning":
            for item in dataset:
                prompts.append(item['question'])
                ground_truths.append(item['answer'])
        elif task_config.name == "summarization":
            for item in dataset:
                prompts.append(item['article'])
                ground_truths.append(item['highlights'])
        elif task_config.name == "classification":
            for item in dataset:
                prompts.append(item['text'])
                ground_truths.append(item['label'])

        return prompts, ground_truths
