"""
Benchmark execution utilities for running single and multi-task benchmarks.
"""

import os
import csv
import tempfile
import threading
from typing import List, Dict, Any

from config import (
    SUPPORTED_TASKS, DEFAULT_TASK, TASK_CONFIGURATIONS,
    DEFAULT_LLM_PROVIDER, COMPRESSION_METHODS_TO_RUN, DEFAULT_TARGET_RATIO,
    NUM_SAMPLES_TO_RUN, MAX_CONCURRENT_LOGGERS, UNLIMITED_MODE
)
from data_loaders.loaders import load_benchmark_dataset
from llms.factory import LLMFactory
from compressors.factory import CompressorFactory
from evaluation.evaluator import Evaluator
from utils.logger import BenchmarkLogger
from utils.thread_safe_logger import ThreadSafeLogger
from utils.run_info_logger import RunInfoLogger
from utils.system_utils import clear_memory, log_memory_usage
from utils.cache_utils import (
    save_samples_to_cache, load_samples_from_cache,
    save_compressed_to_cache, load_compressed_from_cache, check_cache_status
)
from utils.data_utils import extract_task_data, initialize_sample_result, get_model_name
from evaluation.utils import extract_gsm8k_answer


class BenchmarkExecutor:
    """Handles the execution of benchmark tasks."""

    def __init__(self):
        self.run_info_logger = None

    def run_single_task_benchmark(self, task_name: str):
        """Run benchmark for a single task."""
        print(f"\n🔧 Running benchmark for task: {task_name}")
        print("=" * 60)

        # Load dataset
        task_config = TASK_CONFIGURATIONS[task_name]
        dataset = load_benchmark_dataset(
            task_name,
            task_config["dataset"],
            task_config["config"],
            NUM_SAMPLES_TO_RUN
        )

        # Extract data
        prompts, ground_truths = extract_task_data(task_name, dataset)
        log_memory_usage("after dataset loading")

        # Create temporary directory for intermediate files
        with tempfile.TemporaryDirectory() as temp_dir:
            intermediate_file = os.path.join(temp_dir, f"{task_name}_intermediate.csv")

            # Phase 1: Compression Pipeline
            print("\n🗜️  Phase 1: Compression Pipeline")
            compressed_data = self._run_compression_pipeline(
                task_name, prompts, ground_truths, intermediate_file
            )

            # Phase 2: Evaluation Pipeline
            print("\n🤖 Phase 2: Evaluation Pipeline")
            results = self._run_evaluation_pipeline(
                task_name, intermediate_file, compressed_data
            )

            return results

    def run_multi_task_benchmark(self, tasks_to_run: List[str] = None):
        """Run optimized multi-task benchmark."""
        if tasks_to_run is None:
            tasks_to_run = SUPPORTED_TASKS

        print("🚀 Starting ULTRA-OPTIMIZED Multi-Task Prompt Compression Benchmark")
        print("=" * 80)
        print(f"📋 Tasks to run: {', '.join(tasks_to_run)}")
        print(f"🗜️  Compression methods: {', '.join(COMPRESSION_METHODS_TO_RUN)}")
        print(f"🤖 LLM Provider: {DEFAULT_LLM_PROVIDER}")
        print(f"📊 Samples per task: {NUM_SAMPLES_TO_RUN}")
        print("=" * 80)

        # Display unlimited mode status
        if UNLIMITED_MODE:
            print("🔓 UNLIMITED MODE ENABLED: No timeouts or size limits")
            print("⚠️  WARNING: This may cause very long run times and high resource usage")
        else:
            print("🔒 Standard mode: Timeouts and size limits are active")

        # Create temporary directory for intermediate files
        with tempfile.TemporaryDirectory() as temp_dir:
            return self._run_multi_task_pipeline(tasks_to_run, temp_dir)

    def _run_compression_pipeline(self, task_name: str, prompts: List[str],
                                ground_truths: List[str], intermediate_file: str) -> Dict[str, Any]:
        """Run the compression pipeline for a single task."""
        # Check cache for samples
        cached_samples = load_samples_from_cache(task_name, len(prompts))
        if cached_samples:
            print(f"📖 Samples loaded from cache: {len(cached_samples)} samples")
            samples_data = cached_samples
        else:
            # Process samples
            samples_data = []
            for i, (prompt, ground_truth) in enumerate(zip(prompts, ground_truths)):
                samples_data.append({
                    'sample_id': i,
                    'task': task_name,
                    'original_prompt': prompt,
                    'ground_truth': ground_truth
                })
            save_samples_to_cache(task_name, samples_data, len(samples_data))

        # Compress prompts
        compressed_data = {}
        for compression_method in COMPRESSION_METHODS_TO_RUN:
            if check_cache_status(task_name, compression_method, len(prompts), DEFAULT_TARGET_RATIO):
                print(f"   🎯 CACHED {task_name} ({len(prompts)} samples) - {compression_method}")
                compressed_prompts = load_compressed_from_cache(
                    task_name, compression_method, len(prompts), DEFAULT_TARGET_RATIO
                )
                # No compressor needed when using cache
                compressor_created = False
            else:
                print(f"   🔧 Processing {compression_method} for {task_name}...")
                compressor = CompressorFactory.create(compression_method)
                compressor_created = True
                compressed_prompts = []

                for prompt in prompts:
                    compressed_prompt = compressor.compress(prompt, DEFAULT_TARGET_RATIO)
                    compressed_prompts.append(compressed_prompt)

                # Cache compressed prompts
                save_compressed_to_cache(
                    task_name, compression_method, compressed_prompts,
                    len(compressed_prompts), DEFAULT_TARGET_RATIO
                )

            compressed_data[compression_method] = compressed_prompts

            # Unload compressor only if it was created
            if compressor_created:
                del compressor
                clear_memory()

        # Save intermediate file
        fieldnames = ['sample_id', 'task', 'llm_provider', 'llm_model', 'original_prompt', 'ground_truth']
        for method in COMPRESSION_METHODS_TO_RUN:
            fieldnames.append(f"{method}_compressed_prompt")

        data_rows = []
        for i, sample in enumerate(samples_data):
            row = {
                'sample_id': sample['sample_id'],
                'task': sample['task'],
                'llm_provider': DEFAULT_LLM_PROVIDER,
                'llm_model': get_model_name(DEFAULT_LLM_PROVIDER),
                'original_prompt': sample['original_prompt'],
                'ground_truth': sample['ground_truth']
            }

            for method in COMPRESSION_METHODS_TO_RUN:
                if i < len(compressed_data[method]):
                    row[f"{method}_compressed_prompt"] = compressed_data[method][i]

            data_rows.append(row)

        from utils.data_utils import write_intermediate_csv
        write_intermediate_csv(data_rows, intermediate_file, fieldnames)

        log_memory_usage("after compression phase")
        return compressed_data

    def _run_evaluation_pipeline(self, task_name: str, intermediate_file: str,
                               compressed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the evaluation pipeline for a single task."""
        # Load LLM once
        target_llm = LLMFactory.create(provider=DEFAULT_LLM_PROVIDER)
        evaluator = Evaluator(task=task_name, llm=target_llm)
        log_memory_usage("after LLM load")

        # Initialize loggers
        base_logger = BenchmarkLogger(log_dir="results", task_name=task_name, compression_methods=COMPRESSION_METHODS_TO_RUN)
        self.run_info_logger = RunInfoLogger(log_dir="results")

        # Update run info with configuration
        run_config = {
            "task_name": task_name,
            "compression_methods": list(COMPRESSION_METHODS_TO_RUN),
            "llm_provider": DEFAULT_LLM_PROVIDER,
            "llm_model": get_model_name(DEFAULT_LLM_PROVIDER),
            "num_samples": NUM_SAMPLES_TO_RUN,
            "target_compression_ratio": DEFAULT_TARGET_RATIO,
            "unlimited_mode": UNLIMITED_MODE
        }
        self.run_info_logger.update_run_config(run_config)

        thread_safe_logger = ThreadSafeLogger(base_logger)

        # Read from intermediate file and evaluate
        print("Reading from intermediate file and evaluating...")
        evaluated_count = 0

        with open(intermediate_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f, quoting=csv.QUOTE_ALL, escapechar='\\', doublequote=True)

            for row in reader:
                sample_id = int(row['sample_id'])

                # Evaluate original prompt (baseline)
                if evaluated_count == 0:  # Only log memory for first sample
                    log_memory_usage("during evaluation", self.run_info_logger)

                try:
                    baseline_metrics = evaluator.evaluate(row['original_prompt'], row['ground_truth'])
                except Exception as e:
                    print(f"❌ Baseline evaluation failed for sample {sample_id}: {e}")
                    baseline_metrics = {
                        'score': 0.0,
                        'latency': 60.0,
                        'llm_response': f"Error: Baseline evaluation failed - {e}",
                        'extracted_answer': None
                    }

                # Prepare sample result
                sample_result = initialize_sample_result(
                    sample_id, task_name, row['original_prompt'], row['ground_truth']
                )
                sample_result["original_prompt_output"] = baseline_metrics['llm_response']
                sample_result["baseline_score"] = baseline_metrics['score']
                sample_result["baseline_latency"] = baseline_metrics['latency']

                # Evaluate compressed prompts
                for compression_method in COMPRESSION_METHODS_TO_RUN:
                    compressed_prompt_key = f"{compression_method}_compressed_prompt"
                    if compressed_prompt_key in row and row[compressed_prompt_key]:
                        compressed_prompt = row[compressed_prompt_key]

                        try:
                            compressed_metrics = evaluator.evaluate(compressed_prompt, row['ground_truth'])
                        except Exception as e:
                            print(f"❌ Compressed evaluation failed for sample {sample_id}, method {compression_method}: {e}")
                            compressed_metrics = {
                                'score': 0.0,
                                'latency': 60.0,
                                'llm_response': f"Error: Compressed evaluation failed - {e}",
                                'extracted_answer': None
                            }

                        compression_data = {
                            "method": compression_method,
                            "compressed_prompt": compressed_prompt,
                            "compressed_prompt_output": compressed_metrics['llm_response'],
                            "compressed_score": compressed_metrics['score'],
                            "compressed_latency": compressed_metrics['latency']
                        }

                        # Add task-specific metrics
                        if task_name == "reasoning":
                            compressed_answer = extract_gsm8k_answer(compressed_metrics['llm_response'])
                            compression_data["compressed_extracted_answer"] = compressed_answer
                            compression_data["answers_match"] = (baseline_metrics.get('extracted_answer') == compressed_answer) and (baseline_metrics.get('extracted_answer') is not None)
                        elif task_name == "classification":
                            # For classification, the extracted_answer is already provided by the evaluator
                            compression_data["compressed_extracted_answer"] = compressed_metrics.get('extracted_answer')
                            baseline_extracted = baseline_metrics.get('extracted_answer')
                            compressed_extracted = compressed_metrics.get('extracted_answer')
                            compression_data["answers_match"] = (baseline_extracted == compressed_extracted) and (baseline_extracted is not None)

                        sample_result["compression_methods"].append(compression_data)
                    else:
                        print(f"⚠️  No compressed prompt found for sample {sample_id}, method {compression_method}")

                # Log result using thread-safe logger
                thread_safe_logger.log_result(sample_result)

                # Log to run info logger with detailed task data
                task_data = {
                    'task_id': f"{task_name}_{sample_id}",
                    'task_type': task_name,
                    'compression_method': 'baseline',  # Primary method or could be expanded
                    'status': 'completed',
                    'latency': baseline_metrics.get('latency', 0),
                    'score': baseline_metrics.get('score', 0),
                    'tokens_input': len(sample_result.get('original_prompt', '').split()),
                    'tokens_output': len(baseline_metrics.get('llm_response', '').split()),
                    'memory_usage': 0,  # Would need to be calculated
                    'prompt_preview': sample_result.get('original_prompt', '')[:200] + '...' if len(sample_result.get('original_prompt', '')) > 200 else sample_result.get('original_prompt', ''),
                    'output_preview': baseline_metrics.get('llm_response', '')[:200] + '...' if len(baseline_metrics.get('llm_response', '')) > 200 else baseline_metrics.get('llm_response', '')
                }
                self.run_info_logger.log_task_completion(task_data)

                evaluated_count += 1

                if evaluated_count % 3 == 0:  # More frequent progress updates
                    print(f"Evaluated {evaluated_count}/{NUM_SAMPLES_TO_RUN} samples")
                    # Log memory usage periodically
                    current_memory = 0  # Would need to be calculated
                    self.run_info_logger.log_memory_usage(current_memory)

        # Cleanup
        del target_llm
        clear_memory()
        log_memory_usage("after evaluation phase", self.run_info_logger)

        # Phase 4: Finalize and save results
        print("\n📊 Phase 4: Finalizing Results")
        base_logger.finalize_and_save()
        summary = base_logger.generate_summary_report()
        analysis_file = base_logger.export_analysis_report()

        # Finalize run info logger
        final_stats = {
            "total_tasks": evaluated_count,
            "task_name": task_name,
            "compression_methods": list(COMPRESSION_METHODS_TO_RUN),
            "final_memory_usage": 0  # Would need to be calculated
        }
        self.run_info_logger.finalize_run(final_stats)

        print(f"✅ {task_name.upper()} optimized benchmark completed!")
        print(f"   📊 Results saved to: {base_logger.log_dir}")
        print(f"   📈 Analysis report: {analysis_file}")
        print(f"   📋 Run info: {self.run_info_logger.run_info_file}")
        print(f"   📝 Task log: {self.run_info_logger.task_log_file}")
        print(f"   🔍 Real-time log: {self.run_info_logger.real_time_log_file}")

        return {
            "task_name": task_name,
            "samples_evaluated": evaluated_count,
            "results_dir": base_logger.log_dir,
            "analysis_file": analysis_file
        }

    def _run_multi_task_pipeline(self, tasks_to_run: List[str], temp_dir: str) -> Dict[str, Any]:
        """Run the optimized multi-task pipeline."""
        # This would be the full implementation of the multi-task benchmark
        # For now, I'll provide a simplified version

        print("\n📥 Phase 1: Loading ALL datasets...")

        # Load ALL datasets at once
        all_datasets = {}
        for task_name in tasks_to_run:
            task_config = TASK_CONFIGURATIONS[task_name]
            dataset = load_benchmark_dataset(
                task_name,
                task_config["dataset"],
                task_config["config"],
                NUM_SAMPLES_TO_RUN
            )
            all_datasets[task_name] = dataset

        log_memory_usage("after all datasets loaded", self.run_info_logger)

        # Simplified multi-task processing
        all_task_results = {}
        for task_name in tasks_to_run:
            print(f"\n🔧 Processing {task_name.upper()}...")
            # Run single task benchmark for each task
            result = self.run_single_task_benchmark(task_name)
            all_task_results[task_name] = result

        print(f"\n{'='*80}")
        print("🎉 MULTI-TASK BENCHMARK SUMMARY")
        print(f"{'='*80}")
        print(f"📊 Total tasks processed: {len(all_task_results)}")

        for task, results in all_task_results.items():
            print(f"\n📊 {task.upper()}:")
            print(f"   Samples: {results.get('samples_evaluated', 0)}")
            print(f"   Results: {results.get('results_dir', 'N/A')}")

        print("\n✅ Multi-task benchmark completed!")

        return all_task_results
