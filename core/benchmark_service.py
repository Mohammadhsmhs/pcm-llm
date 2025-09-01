"""
Refactored benchmark service following SOLID principles with optimized execution sequence.
Clean, maintainable, and efficient implementation.
"""

from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

from core.config import settings
from utils import RunInfoLogger
from core.pipeline.data_loader_pipeline import DataLoaderPipeline
from core.pipeline.compression_pipeline import CompressionPipeline
from core.pipeline.evaluation_pipeline import EvaluationPipeline
from core.llm_factory import ILLMFactory


class IBenchmarkService(ABC):
    """Interface for the benchmark service, defining the contract for running benchmarks."""

    @abstractmethod
    def run_multi_task_benchmark(
        self,
        tasks_to_run: Optional[List[str]] = None,
        num_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Runs the full, optimized multi-task benchmark.
        """
        pass


class BenchmarkService(IBenchmarkService):
    """
    Orchestrates the multi-task benchmark by running a sequence of pipeline components.
    This service follows the Single Responsibility Principle by delegating the work
    of loading, compressing, and evaluating to specialized pipeline classes.
    """

    def __init__(self, run_info_logger: RunInfoLogger, llm_factory: ILLMFactory):
        self.run_info_logger = run_info_logger
        self.llm_factory = llm_factory

    def run_multi_task_benchmark(
        self,
        tasks_to_run: Optional[List[str]] = None,
        num_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Runs the full, optimized multi-task benchmark.

        This method preserves the critical execution order for efficiency:
        1. Load all data for all tasks.
        2. Loop through compressors, compressing data for all tasks with each one.
        3. Loop through tasks, evaluating all compressed versions for each one.

        Args:
            tasks_to_run: A list of task names to execute. Defaults to all supported tasks.
            num_samples: The number of samples to run for each task. Defaults to the config value.

        Returns:
            A dictionary containing the results of the benchmark execution.
        """
        tasks = tasks_to_run or settings.get_supported_tasks()
        samples = num_samples or settings.performance.num_samples
        compression_methods = settings.get_compression_methods()
        target_ratio = settings.get_target_ratio()

        self._print_benchmark_header(tasks, compression_methods, samples)

        # PHASE 1: Load all data
        data_loader = DataLoaderPipeline(tasks, samples, self.run_info_logger)
        all_samples_data = data_loader.run()

        # PHASE 2: Run compression for all tasks
        compressor = CompressionPipeline(tasks, compression_methods, target_ratio, all_samples_data)
        all_compressed_data, all_compression_metadata = compressor.run()

        # PHASE 3: Run evaluation for all tasks
        evaluator = EvaluationPipeline(
            tasks,
            all_samples_data,
            all_compressed_data,
            all_compression_metadata,
            self.run_info_logger,
            self.llm_factory,
        )
        all_task_results = evaluator.run()

        self._print_benchmark_footer(all_task_results)
        return all_task_results

    def _print_benchmark_header(self, tasks: List[str], methods: List[str], samples: int):
        """Prints the header for the benchmark run."""
        print("🚀 Starting ULTRA-OPTIMIZED Multi-Task Prompt Compression Benchmark")
        print("=" * 80)
        print(f"📋 Tasks to run: {', '.join(tasks)}")
        print(f"🗜️  Compression methods: {', '.join(methods)}")
        print(f"🤖 LLM Provider: {settings.default_llm_provider}")
        print(f"📊 Samples per task: {samples}")
        print("=" * 80)

    def _print_benchmark_footer(self, results: Dict[str, Any]):
        """Prints the footer summary for the benchmark run."""
        print("\n🎉 ALL TASKS COMPLETED!")
        print("=" * 80)
        for task_name, result_data in results.items():
            print(f"📄 {task_name.upper()} Report: {result_data.get('analysis_file', 'N/A')}")
        print("=" * 80)
