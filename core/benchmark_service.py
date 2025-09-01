"""
Refactored benchmark service following SOLID principles with optimized execution sequence.
"""

from typing import List, Dict, Any, Protocol, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import csv
import tempfile

from core.config import IConfigProvider, BenchmarkConfig, TaskConfig
from core.llm_factory import ILLMFactory
from data_loaders.loaders import load_benchmark_dataset
from compressors.factory import CompressorFactory
from evaluation.evaluator import Evaluator
from utils.logger import BenchmarkLogger
from utils.run_info_logger import RunInfoLogger
from utils.system_utils import clear_memory, log_memory_usage
from utils.cache_utils import (
    save_samples_to_cache, load_samples_from_cache,
    save_compressed_to_cache, load_compressed_from_cache, check_cache_status,
    save_baseline_to_cache, load_baseline_from_cache, check_baseline_cache_status
)
from utils.data_utils import extract_task_data, initialize_sample_result, get_model_name, write_intermediate_csv
from evaluation.utils import extract_gsm8k_answer


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
    def run_single_task_benchmark(self, task_name: str, num_samples: Optional[int] = None) -> List[BenchmarkResult]:
        """Run benchmark for a single task."""
        pass

    @abstractmethod
    def run_multi_task_benchmark(self, task_names: List[str], num_samples: Optional[int] = None) -> Dict[str, List[BenchmarkResult]]:
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

    def run_single_task_benchmark(self, task_name: str, num_samples: Optional[int] = None) -> List[BenchmarkResult]:
        """Run optimized benchmark for a single task following the correct sequence."""
        print(f"\n🔧 Running benchmark for task: {task_name}")
        print("=" * 60)

        # Get configurations
        benchmark_config = self.config_provider.get_benchmark_config()
        task_config = self.config_provider.get_task_config(task_name)

        # Use provided num_samples or fall back to config
        samples_to_run = num_samples if num_samples is not None else benchmark_config.num_samples

        # Create temporary directory for intermediate files
        with tempfile.TemporaryDirectory() as temp_dir:
            intermediate_file = os.path.join(temp_dir, f"{task_name}_intermediate.csv")

            # PHASE 1: Load and cache sample data
            print("\n📥 PHASE 1: Loading Sample Data")
            prompts, ground_truths = self._load_and_cache_samples(task_name, task_config, samples_to_run)
            log_memory_usage("after dataset loading", self.run_info_logger)

            # PHASE 2: Compression Pipeline (load each compressor once, compress all data, cache, unload)
            print("\n🗜️  PHASE 2: Compression Pipeline")
            compressed_data, compression_metadata = self._run_optimized_compression_pipeline(
                task_name, prompts, benchmark_config.compression_methods, benchmark_config.target_ratio
            )

            # PHASE 3: Memory cleanup before LLM loading
            print("\n🧹 PHASE 3: Memory Cleanup")
            clear_memory()
            log_memory_usage("after compression cleanup", self.run_info_logger)

            # PHASE 4: Evaluation Pipeline (load LLM once, process all cached data)
            print("\n🤖 PHASE 4: Evaluation Pipeline")
            results = self._run_optimized_evaluation_pipeline(
                task_name, prompts, ground_truths, compressed_data, compression_metadata,
                benchmark_config, intermediate_file
            )

            return results

    def _load_and_cache_samples(self, task_name: str, task_config: TaskConfig, num_samples: int) -> tuple:
        """Load dataset and cache samples if not already cached."""
        # Check cache first
        cached_samples = load_samples_from_cache(task_name, num_samples)
        if cached_samples:
            print(f"📖 Samples loaded from cache: {len(cached_samples)} samples")
            samples_data = cached_samples
            prompts = [sample['original_prompt'] for sample in samples_data]
            ground_truths = [sample['ground_truth'] for sample in samples_data]
        else:
            # Load fresh data
            prompts, ground_truths = self.data_loader.load_dataset(task_config, num_samples)

            # Cache the samples
            samples_data = []
            for i, (prompt, ground_truth) in enumerate(zip(prompts, ground_truths)):
                samples_data.append({
                    'sample_id': i,
                    'task': task_name,
                    'original_prompt': prompt,
                    'ground_truth': ground_truth
                })
            save_samples_to_cache(task_name, samples_data, num_samples)
            print(f"💾 Samples cached: {len(samples_data)} samples")

        return prompts, ground_truths

    def _run_optimized_compression_pipeline(self, task_name: str, prompts: List[str],
                                          compression_methods: List[str], target_ratio: float) -> tuple:
        """Run optimized compression pipeline: load compressor → compress all → cache → unload."""
        compressed_data = {}
        compression_metadata = {}

        for compression_method in compression_methods:
            print(f"   🔧 Processing {compression_method} for {task_name}...")

            # Check if compressed data is already cached
            if check_cache_status(task_name, compression_method, len(prompts), target_ratio):
                print(f"   🎯 CACHED {task_name} ({len(prompts)} samples) - {compression_method}")
                compressed_prompts, metadata = load_compressed_from_cache(
                    task_name, compression_method, len(prompts), target_ratio
                )
                compression_metadata[compression_method] = metadata
                compressed_data[compression_method] = compressed_prompts
                continue

            # Load compressor once for this method
            compressor = CompressorFactory.create(compression_method)
            compressed_prompts = []
            actual_ratios = []

            # Compress ALL prompts with this compressor
            for original_prompt in prompts:
                compressed_prompt = compressor.compress(original_prompt, target_ratio)
                compressed_prompts.append(compressed_prompt)

                # Calculate actual compression ratio
                original_tokens = len(original_prompt.split())
                compressed_tokens = len(compressed_prompt.split())
                ratio = original_tokens / compressed_tokens if compressed_tokens > 0 else 1.0
                actual_ratios.append(ratio)

            # Cache compressed prompts
            save_compressed_to_cache(
                task_name, compression_method, compressed_prompts,
                len(compressed_prompts), target_ratio, actual_ratios
            )

            # Store metadata
            compression_metadata[compression_method] = {
                "average_actual_ratio": sum(actual_ratios) / len(actual_ratios),
                "actual_ratios": actual_ratios
            }

            compressed_data[compression_method] = compressed_prompts

            # Unload compressor immediately after processing all data
            del compressor
            clear_memory()
            print(f"   ✅ {compression_method} completed and unloaded")

        return compressed_data, compression_metadata

    def _run_optimized_evaluation_pipeline(self, task_name: str, prompts: List[str], ground_truths: List[str],
                                         compressed_data: Dict[str, Any], compression_metadata: Dict[str, Any],
                                         benchmark_config: BenchmarkConfig, intermediate_file: str) -> List[BenchmarkResult]:
        """Run optimized evaluation pipeline: load LLM once → process all cached data."""
        # Create task-specific logger
        from core.config import settings
        task_logger = BenchmarkLogger(
            log_dir=settings.paths.logs_dir,
            results_dir=settings.paths.results_dir,
            task_name=task_name,
            compression_methods=benchmark_config.compression_methods
        )

        # Load LLM once for the entire evaluation phase
        default_provider = "ollama"
        llm_config = self.config_provider.get_llm_config(default_provider)
        llm = self.llm_factory.create_llm(default_provider, llm_config)
        evaluator = Evaluator(task=task_name, llm=llm)
        log_memory_usage("after LLM load", self.run_info_logger)

        # Initialize run info logger
        run_info_logger = RunInfoLogger(log_dir=settings.paths.logs_dir)
        run_config = {
            "task_name": task_name,
            "compression_methods": list(benchmark_config.compression_methods),
            "llm_provider": default_provider,
            "llm_model": llm_config.model_name,
            "num_samples": len(prompts),
            "target_compression_ratio": benchmark_config.target_ratio,
        }
        run_info_logger.update_run_config(run_config)

        results = []
        sample_results = {}

        # Check for cached baseline outputs
        llm_model_name = get_model_name(default_provider)
        baseline_cached = check_baseline_cache_status(task_name, default_provider, llm_model_name, len(prompts))

        if baseline_cached:
            print(f"📖 Baseline outputs loaded from cache: {len(prompts)} samples")
            cached_baseline_data = load_baseline_from_cache(task_name, default_provider, llm_model_name, len(prompts))
            baseline_cache_dict = {item['sample_id']: item for item in cached_baseline_data}
        else:
            print("🤖 Generating baseline outputs (not cached)...")
            baseline_cache_dict = {}

        # Process each sample with LLM once
        for i, (prompt, ground_truth) in enumerate(zip(prompts, ground_truths)):
            sample_results[i] = {
                "sample_id": i,
                "task": task_name,
                "original_prompt": prompt,
                "ground_truth_answer": ground_truth,
                "compression_methods": []
            }

            # Evaluate original prompt (baseline) - check cache first
            if baseline_cached and i in baseline_cache_dict:
                baseline_metrics = baseline_cache_dict[i]
                print(f"📋 Sample {i}: Using cached baseline output")
            else:
                # Format prompt based on task
                formatted_prompt = self._format_prompt_for_task(prompt, task_name)
                baseline_metrics = evaluator.evaluate(formatted_prompt, ground_truth)
                baseline_cache_dict[i] = baseline_metrics

            sample_results[i]["baseline_score"] = baseline_metrics.get('score', 0.0)
            sample_results[i]["baseline_latency"] = baseline_metrics.get('latency', 0.0)
            sample_results[i]["baseline_output"] = baseline_metrics.get('llm_response', '')
            sample_results[i]["baseline_extracted_answer"] = baseline_metrics.get('extracted_answer')

            # Process each compression method for this sample (using cached compressed data)
            for compression_method in benchmark_config.compression_methods:
                compressed_prompt = compressed_data[compression_method][i]

                # Format compressed prompt
                formatted_compressed_prompt = self._format_prompt_for_task(compressed_prompt, task_name)

                # Evaluate compressed prompt
                compressed_metrics = evaluator.evaluate(formatted_compressed_prompt, ground_truth)

                # Create result
                result = BenchmarkResult(
                    task_name=task_name,
                    compression_method=compression_method,
                    sample_id=i,
                    original_prompt=prompt,
                    compressed_prompt=compressed_prompt,
                    original_score=baseline_metrics.get('score', 0.0),
                    compressed_score=compressed_metrics.get('score', 0.0),
                    latency=compressed_metrics.get('latency', 0.0),
                    memory_usage=0.0
                )

                results.append(result)

                # Add to sample results for logging
                compression_data_entry = {
                    "method": compression_method,
                    "compressed_prompt": compressed_prompt,
                    "compressed_score": compressed_metrics.get('score', 0.0),
                    "compressed_latency": compressed_metrics.get('latency', 0.0),
                    "answers_match": compressed_metrics.get('answers_match', False),
                    "target_compression_ratio": benchmark_config.target_ratio,
                    "actual_compression_ratio": compression_metadata[compression_method]["actual_ratios"][i]
                }
                sample_results[i]["compression_methods"].append(compression_data_entry)

            # Log the complete sample result
            task_logger.log_result(sample_results[i])

        # Cache baseline outputs if not already cached
        if not baseline_cached:
            baseline_list = [baseline_cache_dict[i] for i in range(len(prompts))]
            save_baseline_to_cache(task_name, default_provider, llm_model_name, len(prompts), baseline_list)

        # Unload LLM after all evaluations are complete
        del llm
        del evaluator
        clear_memory()
        log_memory_usage("after LLM unload", self.run_info_logger)

        # Finalize logging
        task_logger.finalize_and_save()
        summary = task_logger.generate_summary_report()
        analysis_file = task_logger.export_analysis_report()

        print(f"✅ {task_name.upper()} benchmark completed!")
        print(f"   📊 Results saved to: {settings.paths.results_dir}")
        print(f"   📈 Analysis report: {analysis_file}")

        return results

    def _format_prompt_for_task(self, prompt: str, task_name: str) -> str:
        """Format prompt based on task type."""
        if task_name == "reasoning":
            return f"{prompt}\n\nSolve this and provide the final answer after #### with no extra words or characters."
        elif task_name == "classification":
            return f"{prompt}\n\nAnalyze the sentiment of this movie review. Your response MUST be ONLY the single word 'positive' or 'negative' and nothing else."
        elif task_name == "summarization":
            return f"{prompt}\n\nSummarize the entire article above in a single, concise sentence, capturing only the main headline or takeaway. Keep it brief and factual, similar to a news headline."
        return prompt

    def run_multi_task_benchmark(self, task_names: List[str], num_samples: Optional[int] = None) -> Dict[str, List[BenchmarkResult]]:
        """Run ULTRA-optimized benchmark for multiple tasks with single-phase execution."""
        print("🚀 Starting ULTRA-OPTIMIZED Multi-Task Prompt Compression Benchmark")
        print("=" * 80)
        print(f"📋 Tasks to run: {', '.join(task_names)}")
        print("=" * 80)

        # Get configurations
        benchmark_config = self.config_provider.get_benchmark_config()

        # Use provided num_samples or fall back to config
        samples_to_run = num_samples if num_samples is not None else benchmark_config.num_samples

        # PHASE 1: Load ALL sample data at once
        print("\n📥 PHASE 1: Loading ALL Sample Data")
        all_samples_data = {}
        for task_name in task_names:
            task_config = self.config_provider.get_task_config(task_name)
            prompts, ground_truths = self._load_and_cache_samples(task_name, task_config, samples_to_run)
            all_samples_data[task_name] = {
                'prompts': prompts,
                'ground_truths': ground_truths,
                'num_samples': len(prompts)
            }
        log_memory_usage("after loading all datasets", self.run_info_logger)

        # PHASE 2: Compression Pipeline (load each compressor once, compress ALL data, cache, unload)
        print("\n🗜️  PHASE 2: Compression Pipeline (ALL tasks)")
        all_compressed_data = {}
        all_compression_metadata = {}

        for compression_method in benchmark_config.compression_methods:
            print(f"   🔧 Processing {compression_method} for ALL tasks...")
            all_compressed_data[compression_method] = {}
            all_compression_metadata[compression_method] = {}

            # Load compressor once for this method
            compressor = CompressorFactory.create(compression_method)

            # Compress data for ALL tasks with this compressor
            for task_name in task_names:
                task_data = all_samples_data[task_name]
                prompts = task_data['prompts']

                # Check if compressed data is already cached
                if check_cache_status(task_name, compression_method, len(prompts), benchmark_config.target_ratio):
                    print(f"   🎯 CACHED {task_name} ({len(prompts)} samples) - {compression_method}")
                    compressed_prompts, metadata = load_compressed_from_cache(
                        task_name, compression_method, len(prompts), benchmark_config.target_ratio
                    )
                else:
                    compressed_prompts = []
                    actual_ratios = []

                    # Compress ALL prompts for this task
                    for original_prompt in prompts:
                        compressed_prompt = compressor.compress(original_prompt, benchmark_config.target_ratio)
                        compressed_prompts.append(compressed_prompt)

                        # Calculate actual compression ratio
                        original_tokens = len(original_prompt.split())
                        compressed_tokens = len(compressed_prompt.split())
                        ratio = original_tokens / compressed_tokens if compressed_tokens > 0 else 1.0
                        actual_ratios.append(ratio)

                    # Cache compressed prompts
                    save_compressed_to_cache(
                        task_name, compression_method, compressed_prompts,
                        len(compressed_prompts), benchmark_config.target_ratio, actual_ratios
                    )

                    metadata = {
                        "average_actual_ratio": sum(actual_ratios) / len(actual_ratios),
                        "actual_ratios": actual_ratios
                    }

                all_compressed_data[compression_method][task_name] = compressed_prompts
                all_compression_metadata[compression_method][task_name] = metadata

            # Unload compressor after processing ALL tasks
            del compressor
            clear_memory()
            print(f"   ✅ {compression_method} completed for ALL tasks and unloaded")

        # PHASE 3: Memory cleanup before LLM loading
        print("\n🧹 PHASE 3: Memory Cleanup")
        clear_memory()
        log_memory_usage("after compression cleanup", self.run_info_logger)

        # PHASE 4: Evaluation Pipeline (load LLM once, process ALL cached data)
        print("\n🤖 PHASE 4: Evaluation Pipeline (ALL tasks)")
        results = self._run_multi_task_evaluation_pipeline(
            task_names, all_samples_data, all_compressed_data, all_compression_metadata, benchmark_config
        )

        return results

    def _run_multi_task_evaluation_pipeline(self, task_names: List[str], all_samples_data: Dict[str, Any],
                                          all_compressed_data: Dict[str, Any], all_compression_metadata: Dict[str, Any],
                                          benchmark_config: BenchmarkConfig) -> Dict[str, List[BenchmarkResult]]:
        """Run evaluation pipeline for all tasks with single LLM load."""
        from core.config import settings

        # Load LLM once for ALL tasks
        default_provider = "ollama"
        llm_config = self.config_provider.get_llm_config(default_provider)
        llm = self.llm_factory.create_llm(default_provider, llm_config)
        log_memory_usage("after LLM load for all tasks", self.run_info_logger)

        results = {}
        llm_model_name = get_model_name(default_provider)

        # Process each task
        for task_name in task_names:
            print(f"\n📊 Evaluating {task_name}...")

            task_data = all_samples_data[task_name]
            prompts = task_data['prompts']
            ground_truths = task_data['ground_truths']

            # Create task-specific logger
            task_logger = BenchmarkLogger(
                log_dir=settings.paths.logs_dir,
                results_dir=settings.paths.results_dir,
                task_name=task_name,
                compression_methods=benchmark_config.compression_methods
            )

            # Initialize run info logger for this task
            run_info_logger = RunInfoLogger(log_dir=settings.paths.logs_dir)
            run_config = {
                "task_name": task_name,
                "compression_methods": list(benchmark_config.compression_methods),
                "llm_provider": default_provider,
                "llm_model": llm_config.model_name,
                "num_samples": len(prompts),
                "target_compression_ratio": benchmark_config.target_ratio,
            }
            run_info_logger.update_run_config(run_config)

            # Create evaluator for this task
            evaluator = Evaluator(task=task_name, llm=llm)

            task_results = []
            sample_results = {}

            # Check for cached baseline outputs
            baseline_cached = check_baseline_cache_status(task_name, default_provider, llm_model_name, len(prompts))

            if baseline_cached:
                print(f"   📖 Baseline outputs loaded from cache: {len(prompts)} samples")
                cached_baseline_data = load_baseline_from_cache(task_name, default_provider, llm_model_name, len(prompts))
                
                # Handle different cache formats
                if isinstance(cached_baseline_data, list):
                    # Convert list to dict with sample_id as key
                    baseline_cache_dict = {}
                    for i, item in enumerate(cached_baseline_data):
                        if isinstance(item, dict):
                            baseline_cache_dict[i] = item
                        else:
                            # If item is not a dict, create a basic structure
                            baseline_cache_dict[i] = {
                                'score': 0.0,
                                'latency': 0.0,
                                'llm_response': str(item),
                                'extracted_answer': None
                            }
                else:
                    baseline_cache_dict = cached_baseline_data if isinstance(cached_baseline_data, dict) else {}
            else:
                print(f"   🤖 Generating baseline outputs for {task_name} (not cached)...")
                baseline_cache_dict = {}

            # Process each sample for this task
            for i, (prompt, ground_truth) in enumerate(zip(prompts, ground_truths)):
                sample_results[i] = {
                    "sample_id": i,
                    "task": task_name,
                    "original_prompt": prompt,
                    "ground_truth_answer": ground_truth,
                    "compression_methods": []
                }

                # Evaluate original prompt (baseline) - check cache first
                if baseline_cached and i in baseline_cache_dict:
                    baseline_metrics = baseline_cache_dict[i]
                    print(f"   📋 Sample {i}: Using cached baseline output")
                else:
                    # Format prompt based on task
                    formatted_prompt = self._format_prompt_for_task(prompt, task_name)
                    baseline_metrics = evaluator.evaluate(formatted_prompt, ground_truth)
                    baseline_cache_dict[i] = baseline_metrics

                sample_results[i]["baseline_score"] = baseline_metrics.get('score', 0.0)
                sample_results[i]["baseline_latency"] = baseline_metrics.get('latency', 0.0)
                sample_results[i]["baseline_output"] = baseline_metrics.get('llm_response', '')
                sample_results[i]["baseline_extracted_answer"] = baseline_metrics.get('extracted_answer')

                # Process each compression method for this sample
                for compression_method in benchmark_config.compression_methods:
                    compressed_prompt = all_compressed_data[compression_method][task_name][i]

                    # Format compressed prompt
                    formatted_compressed_prompt = self._format_prompt_for_task(compressed_prompt, task_name)

                    # Evaluate compressed prompt
                    compressed_metrics = evaluator.evaluate(formatted_compressed_prompt, ground_truth)

                    # Create result
                    result = BenchmarkResult(
                        task_name=task_name,
                        compression_method=compression_method,
                        sample_id=i,
                        original_prompt=prompt,
                        compressed_prompt=compressed_prompt,
                        original_score=baseline_metrics.get('score', 0.0),
                        compressed_score=compressed_metrics.get('score', 0.0),
                        latency=compressed_metrics.get('latency', 0.0),
                        memory_usage=0.0
                    )

                    task_results.append(result)

                    # Add to sample results for logging
                    compression_data_entry = {
                        "method": compression_method,
                        "compressed_prompt": compressed_prompt,
                        "compressed_score": compressed_metrics.get('score', 0.0),
                        "compressed_latency": compressed_metrics.get('latency', 0.0),
                        "answers_match": compressed_metrics.get('answers_match', False),
                        "target_compression_ratio": benchmark_config.target_ratio,
                        "actual_compression_ratio": all_compression_metadata[compression_method][task_name]["actual_ratios"][i]
                    }
                    sample_results[i]["compression_methods"].append(compression_data_entry)

                # Log the complete sample result
                task_logger.log_result(sample_results[i])

            # Cache baseline outputs if not already cached
            if not baseline_cached:
                baseline_list = [baseline_cache_dict[i] for i in range(len(prompts))]
                save_baseline_to_cache(task_name, default_provider, llm_model_name, len(prompts), baseline_list)

            # Finalize logging for this task
            task_logger.finalize_and_save()
            summary = task_logger.generate_summary_report()
            analysis_file = task_logger.export_analysis_report()

            results[task_name] = task_results

            print(f"   ✅ {task_name.upper()} evaluation completed!")
            print(f"      📊 Results saved to: {settings.paths.results_dir}")
            print(f"      📈 Analysis report: {analysis_file}")

        # Unload LLM after ALL tasks are complete
        del llm
        clear_memory()
        log_memory_usage("after LLM unload for all tasks", self.run_info_logger)

        print("\n🎉 ALL TASKS COMPLETED!")
        print("=" * 80)

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
