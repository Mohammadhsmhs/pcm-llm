"""
Benchmark execution utilities for running single and multi-task benchmarks.
"""

import os
import csv
import tempfile
import threading
from typing import List, Dict, Any, Tuple

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
    save_compressed_to_cache, load_compressed_from_cache, check_cache_status,
    save_baseline_to_cache, load_baseline_from_cache, check_baseline_cache_status
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
            compressed_data, compression_metadata = self._run_compression_pipeline(
                task_name, prompts, ground_truths, intermediate_file
            )

            # Phase 2: Evaluation Pipeline
            print("\n🤖 Phase 2: Evaluation Pipeline")
            results = self._run_evaluation_pipeline(
                task_name, intermediate_file, compressed_data, compression_metadata
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
                                ground_truths: List[str], intermediate_file: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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
        compression_metadata = {}  # Store metadata for each method
        for compression_method in COMPRESSION_METHODS_TO_RUN:
            if check_cache_status(task_name, compression_method, len(prompts), DEFAULT_TARGET_RATIO):
                print(f"   🎯 CACHED {task_name} ({len(prompts)} samples) - {compression_method}")
                compressed_prompts, metadata = load_compressed_from_cache(
                    task_name, compression_method, len(prompts), DEFAULT_TARGET_RATIO
                )
                # Store metadata for later use
                compression_metadata[compression_method] = metadata
                if metadata:
                    print(f"   📊 Using cached compression ratios (avg: {metadata.get('average_actual_ratio', 'N/A'):.2f})")
                # No compressor needed when using cache
                compressor_created = False
            else:
                print(f"   🔧 Processing {compression_method} for {task_name}...")
                compressor = CompressorFactory.create(compression_method)
                compressor_created = True
                compressed_prompts = []
                actual_ratios = []

                for original_prompt in prompts:
                    compressed_prompt = compressor.compress(original_prompt, DEFAULT_TARGET_RATIO)
                    compressed_prompts.append(compressed_prompt)
                    
                    # Calculate actual compression ratio
                    original_tokens = len(original_prompt.split())
                    compressed_tokens = len(compressed_prompt.split())
                    if compressed_tokens > 0:
                        ratio = original_tokens / compressed_tokens
                    else:
                        ratio = 1.0
                    actual_ratios.append(ratio)

                # Cache compressed prompts with actual ratios
                save_compressed_to_cache(
                    task_name, compression_method, compressed_prompts,
                    len(compressed_prompts), DEFAULT_TARGET_RATIO, actual_ratios
                )
                
                # Store metadata for consistency
                compression_metadata[compression_method] = {
                    "average_actual_ratio": sum(actual_ratios) / len(actual_ratios) if actual_ratios else DEFAULT_TARGET_RATIO,
                    "actual_ratios": actual_ratios
                }

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
        return compressed_data, compression_metadata

    def _run_evaluation_pipeline(self, task_name: str, intermediate_file: str,
                               compressed_data: Dict[str, Any], compression_metadata: Dict[str, Any]) -> Dict[str, Any]:
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

        # Check for cached baseline outputs
        llm_model_name = get_model_name(DEFAULT_LLM_PROVIDER)
        baseline_cached = check_baseline_cache_status(task_name, DEFAULT_LLM_PROVIDER, llm_model_name, NUM_SAMPLES_TO_RUN)
        
        if baseline_cached:
            print(f"📖 Baseline outputs loaded from cache: {NUM_SAMPLES_TO_RUN} samples")
            cached_baseline_data = load_baseline_from_cache(task_name, DEFAULT_LLM_PROVIDER, llm_model_name, NUM_SAMPLES_TO_RUN)
            baseline_cache_dict = {item['sample_id']: item for item in cached_baseline_data}
        else:
            print("🤖 Generating baseline outputs (not cached)...")
            baseline_cache_dict = {}

        # Read from intermediate file and evaluate
        print("Reading from intermediate file and evaluating...")
        evaluated_count = 0
        baseline_results = []

        with open(intermediate_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f, quoting=csv.QUOTE_ALL, escapechar='\\', doublequote=True)

            for row in reader:
                sample_id = int(row['sample_id'])

                # Evaluate original prompt (baseline)
                baseline_prompt = row['original_prompt']
                if task_name == "reasoning":
                    baseline_prompt = f"{baseline_prompt}\n\nSolve this and provide the final answer after #### with no extra words or characters."
                elif task_name == "classification":
                    baseline_prompt = f"{baseline_prompt}\n\nAnalyze the sentiment of this movie review. Your response MUST be ONLY the single word 'positive' or 'negative' and nothing else."
                elif task_name == "summarization":
                    baseline_prompt = f"{baseline_prompt}\n\nSummarize the following article in a single, concise paragraph."
                
                # Check if baseline output is cached
                if baseline_cached and sample_id in baseline_cache_dict:
                    baseline_metrics = baseline_cache_dict[sample_id]
                    print(f"📋 Sample {sample_id}: Using cached baseline output")
                else:
                    try:
                        baseline_metrics = evaluator.evaluate(baseline_prompt, row['ground_truth'])
                        # Cache the result
                        baseline_cache_dict[sample_id] = baseline_metrics
                    except Exception as e:
                        print(f"❌ Baseline evaluation failed for sample {sample_id}: {e}")
                        baseline_metrics = {
                            'score': 0.0,
                            'latency': 60.0,
                            'llm_response': f"Error: Baseline evaluation failed - {e}",
                            'extracted_answer': None
                        }
                        baseline_cache_dict[sample_id] = baseline_metrics

                # Enrich cached baseline with missing extracted answers if needed
                baseline_extracted_tmp = baseline_metrics.get('extracted_answer') if isinstance(baseline_metrics, dict) else None
                if not baseline_extracted_tmp:
                    try:
                        if task_name == "reasoning":
                            baseline_extracted_tmp = extract_gsm8k_answer(baseline_metrics.get('llm_response', ''))
                        elif task_name == "classification":
                            _, baseline_extracted_tmp = evaluator._calculate_classification_score(
                                baseline_metrics.get('llm_response', ''), row['ground_truth']
                            )
                        elif task_name == "summarization":
                            baseline_extracted_tmp = baseline_metrics.get('llm_response', '')
                        if baseline_extracted_tmp:
                            baseline_metrics['extracted_answer'] = baseline_extracted_tmp
                            baseline_cache_dict[sample_id] = baseline_metrics
                    except Exception:
                        pass

                # Prepare sample result
                sample_result = initialize_sample_result(
                    sample_id, task_name, row['original_prompt'], row['ground_truth']
                )
                sample_result["original_prompt_output"] = baseline_metrics['llm_response']
                sample_result["baseline_score"] = baseline_metrics['score']
                sample_result["baseline_latency"] = baseline_metrics['latency']
                sample_result["baseline_extracted_answer"] = baseline_metrics.get('extracted_answer')

                baseline_results.append({
                    'sample_id': sample_id,
                    'score': baseline_metrics['score'],
                    'latency': baseline_metrics['latency'],
                    'llm_response': baseline_metrics['llm_response'],
                    'extracted_answer': baseline_metrics.get('extracted_answer')
                })

                # Evaluate compressed prompts
                for compression_method in COMPRESSION_METHODS_TO_RUN:
                    compressed_prompt_key = f"{compression_method}_compressed_prompt"
                    if compressed_prompt_key in row and row[compressed_prompt_key]:
                        compressed_prompt = row[compressed_prompt_key]
                        
                        # Add task-specific format instructions
                        if task_name == "reasoning":
                            compressed_prompt = f"{compressed_prompt}\n\nSolve this and provide the final answer after #### with no extra words or characters."
                        elif task_name == "classification":
                            compressed_prompt = f"{compressed_prompt}\n\nAnalyze the sentiment of this movie review. Your response MUST be ONLY the single word 'positive' or 'negative' and nothing else."
                        elif task_name == "summarization":
                            compressed_prompt = f"{compressed_prompt}\n\nSummarize the following article in a single, concise paragraph."

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

                        # Get actual compression ratio from metadata if available
                        actual_ratio = DEFAULT_TARGET_RATIO  # fallback
                        if compression_method in compression_metadata:
                            metadata = compression_metadata[compression_method]
                            if 'actual_ratios' in metadata and sample_id < len(metadata['actual_ratios']):
                                actual_ratio = metadata['actual_ratios'][sample_id]

                        compression_data = {
                            "method": compression_method,
                            "compressed_prompt": compressed_prompt,
                            "compressed_prompt_output": compressed_metrics['llm_response'],
                            "compressed_score": compressed_metrics['score'],
                            "compressed_latency": compressed_metrics['latency'],
                            "actual_compression_ratio": actual_ratio
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
                        elif task_name == "summarization":
                            # For summarization, compare the full responses (normalized)
                            baseline_response = baseline_metrics.get('llm_response', '').strip().lower()
                            compressed_response = compressed_metrics.get('llm_response', '').strip().lower()
                            compression_data["compressed_extracted_answer"] = compressed_metrics.get('llm_response')
                            compression_data["answers_match"] = baseline_response == compressed_response and baseline_response != ""

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

        # Save baseline outputs to cache if not already cached
        if not baseline_cached:
            save_baseline_to_cache(task_name, DEFAULT_LLM_PROVIDER, llm_model_name, NUM_SAMPLES_TO_RUN, baseline_results)
            print(f"💾 Saved {len(baseline_results)} baseline outputs to cache")
        else:
            # Persist any enrichment we performed on cached baselines
            try:
                enriched_list = [baseline_cache_dict[k] for k in sorted(baseline_cache_dict.keys())]
                save_baseline_to_cache(task_name, DEFAULT_LLM_PROVIDER, llm_model_name, NUM_SAMPLES_TO_RUN, enriched_list)
                print("💾 Updated cached baseline outputs with extracted answers (if any were missing)")
            except Exception:
                pass

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
