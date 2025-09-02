"""
Evaluation pipeline component for the benchmark service.
"""
from typing import List, Dict, Any

from core.config import settings
from core.llm_factory import ILLMFactory
from evaluation.evaluator import Evaluator
from utils.loggers.logger import BenchmarkLogger
from utils.loggers.run_info_logger import RunInfoLogger
from utils.system.system_utils import clear_memory, log_memory_usage
from utils.cache.cache_utils import check_baseline_cache_status, load_baseline_from_cache, save_baseline_to_cache
from utils.data.data_utils import get_model_name


class EvaluationPipeline:
    """Handles the optimized evaluation phase of the benchmark."""

    def __init__(
        self,
        tasks: List[str],
        all_samples_data: Dict[str, Any],
        all_compressed_data: Dict[str, Any],
        all_compression_metadata: Dict[str, Any],
        run_info_logger: RunInfoLogger,
        llm_factory: ILLMFactory,
    ):
        self.tasks = tasks
        self.compression_methods = list(all_compressed_data.keys())
        self.all_samples_data = all_samples_data
        self.all_compressed_data = all_compressed_data
        self.all_compression_metadata = all_compression_metadata
        self.run_info_logger = run_info_logger
        self.llm_factory = llm_factory

    def run(self) -> Dict[str, Any]:
        """
        Runs the evaluation pipeline for all tasks.

        Returns:
            A dictionary containing the evaluation results.
        """
        print("\n🤖 PHASE 4: Evaluation Pipeline (ALL tasks)")
        all_task_results = {}

        for task_name in self.tasks:
            print(f"\n📊 Evaluating {task_name}...")
            task_logger = BenchmarkLogger(
                log_dir=settings.paths.logs_dir,
                results_dir=settings.paths.results_dir,
                task_name=task_name,
                compression_methods=self.compression_methods
            )

            llm_config = settings.get_llm_config(settings.default_llm_provider)
            llm = self.llm_factory.create_llm(provider=settings.default_llm_provider, config=llm_config)
            evaluator = Evaluator(task=task_name, llm=llm)
            log_memory_usage(f"after LLM load for {task_name}", self.run_info_logger)

            task_samples = self.all_samples_data[task_name]
            num_samples = len(task_samples)

            baseline_cache_dict = self._load_baseline_cache(task_name, num_samples)

            for i, sample in enumerate(task_samples):
                # The debug print is now inside _evaluate_sample
                sample_result = self._evaluate_sample(
                    sample, evaluator, baseline_cache_dict
                )
                task_logger.log_result(sample_result)

            self._save_baseline_cache(task_name, num_samples, baseline_cache_dict)

            del llm
            del evaluator
            clear_memory()
            log_memory_usage(f"after LLM unload for {task_name}", self.run_info_logger)

            csv_file_path = task_logger.finalize_and_save()
            analysis_file = task_logger.export_analysis_report(csv_file_path)
            
            all_task_results[task_name] = {
                "samples_evaluated": num_samples,
                "results_dir": settings.paths.results_dir,
                "analysis_file": analysis_file,
            }
            print(f"   ✅ {task_name.upper()} evaluation completed!")
            print(f"      📊 Results saved to: {settings.paths.results_dir}")
            print(f"      📈 Analysis report: {analysis_file}")

        return all_task_results

    def _load_baseline_cache(self, task_name: str, num_samples: int) -> Dict[int, Any]:
        """Loads baseline results from cache if they exist."""
        llm_model_name = get_model_name(settings.default_llm_provider)
        if check_baseline_cache_status(task_name, settings.default_llm_provider, llm_model_name, num_samples):
            print(f"   📖 Baseline outputs loaded from cache: {num_samples} samples")
            cached_data = load_baseline_from_cache(task_name, settings.default_llm_provider, llm_model_name, num_samples)
            return {item['sample_id']: item for item in cached_data}
        print("   🤖 Generating baseline outputs (not cached)...")
        return {}

    def _save_baseline_cache(self, task_name: str, num_samples: int, cache_dict: Dict[int, Any]):
        """Saves baseline results to cache."""
        llm_model_name = get_model_name(settings.default_llm_provider)
        if not check_baseline_cache_status(task_name, settings.default_llm_provider, llm_model_name, num_samples):
            baseline_list = [cache_dict[i] for i in sorted(cache_dict.keys())]
            save_baseline_to_cache(task_name, settings.default_llm_provider, llm_model_name, num_samples, baseline_list)
            print(f"   💾 Saved {len(baseline_list)} baseline outputs to cache for {task_name}")

    def _evaluate_sample(self, sample: Dict[str, Any], evaluator: Evaluator, baseline_cache: Dict) -> Dict[str, Any]:
        """Evaluates a single sample, including baseline and all compression methods."""
        from utils.data.data_utils import initialize_sample_result

        sample_id = sample["sample_id"]
        task_name = sample["task"]
        prompt = sample["original_prompt"]
        ground_truth = sample["ground_truth"]

        sample_result = initialize_sample_result(sample_id, task_name, prompt, ground_truth)

        # Evaluate baseline
        if sample_id in baseline_cache:
            print(f"   📋 Sample {sample_id}: Using cached baseline output")
            baseline_metrics = baseline_cache[sample_id]
        else:
            print(f"   🚀 Sample {sample_id}: Generating baseline output")
            baseline_metrics = evaluator.evaluate(prompt, ground_truth)
            baseline_cache[sample_id] = {
                "sample_id": sample_id, 
                **baseline_metrics
            }
        
        sample_result.update({
            "baseline_output": baseline_metrics.get('llm_response', ''),
            "baseline_score": baseline_metrics.get('score', 0.0),
            "baseline_latency": baseline_metrics.get('latency', 0.0),
            "baseline_extracted_answer": baseline_metrics.get('extracted_answer')
        })

        # Evaluate compressed prompts
        for method in self.compression_methods:
            # Find the corresponding compressed prompt by sample_id
            compressed_item = next((item for item in self.all_compressed_data[method][task_name] if item["sample_id"] == sample_id), None)
            
            if compressed_item:
                compressed_prompt = compressed_item["compressed_prompt"]
                compressed_metrics = evaluator.evaluate(compressed_prompt, ground_truth)
                
                # Find the corresponding metadata by sample_id
                actual_ratio = self.all_compression_metadata[method][task_name]["actual_ratios"].get(str(sample_id))

                method_data = {
                    "method": method,
                    "compressed_prompt": compressed_prompt,
                    "compressed_output": compressed_metrics.get('llm_response', ''),
                    "compressed_score": compressed_metrics.get('score', 0.0),
                    "compressed_latency": compressed_metrics.get('latency', 0.0),
                    "compressed_extracted_answer": compressed_metrics.get('extracted_answer'),
                    "answers_match": compressed_metrics.get('answers_match', False),
                    "actual_compression_ratio": actual_ratio
                }
                sample_result["compression_methods"].append(method_data)
            else:
                print(f"⚠️  Could not find compressed prompt for sample {sample_id} in method {method}")

        return sample_result
