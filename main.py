import torch
import statistics
from config import (
    SUPPORTED_TASKS, DEFAULT_TASK, TASK_CONFIGURATIONS,
    DEFAULT_LLM_PROVIDER, COMPRESSION_METHODS_TO_RUN, DEFAULT_TARGET_RATIO,
    HUGGINGFACE_MODEL, OPENAI_MODEL, LLAMACPP_REPO_ID, LLAMACPP_FILENAME,
    NUM_SAMPLES_TO_RUN
)
from data_loaders.loaders import load_benchmark_dataset
from llms.factory import LLMFactory
from compressors.factory import CompressorFactory

from evaluation.evaluator import Evaluator
from utils.logger import BenchmarkLogger
from evaluation.utils import extract_gsm8k_answer

def clear_memory():
    """Utility function to clear GPU cache if available."""
    if torch.backends.mps.is_available(): torch.mps.empty_cache()
    elif torch.cuda.is_available(): torch.cuda.empty_cache()

def run_benchmark_for_task(task_name: str):
    """Run benchmark for a specific task type."""
    print(f"\n{'='*60}")
    print(f"🧪 Running Benchmark for Task: {task_name.upper()}")
    print(f"{'='*60}")

    # Get task configuration
    task_config = TASK_CONFIGURATIONS[task_name]
    dataset_name = task_config["dataset"]
    dataset_config = task_config["config"]

    # Setup
    logger = BenchmarkLogger()
    dataset = load_benchmark_dataset(task_name, dataset_name, dataset_config, NUM_SAMPLES_TO_RUN)
    if not dataset:
        print(f"❌ Failed to load dataset for task {task_name}")
        return

    # Extract prompts and ground truth based on task type
    prompts, ground_truths = extract_task_data(task_name, dataset)

    # Run benchmark for each compression method
    all_results = []
    sample_results = {}

    for compression_method in COMPRESSION_METHODS_TO_RUN:
        print(f"\n--- Running {compression_method} for {task_name} ---")

        # Compression Phase
        print(f"Compressing {len(prompts)} prompts using {compression_method}...")
        compressed_prompts = []
        try:
            compressor = CompressorFactory.create(compression_method)
            for i, prompt in enumerate(prompts):
                if (i + 1) % 5 == 0:
                    print(f"Compressed {i + 1}/{len(prompts)} prompts")
                compressed_prompts.append(compressor.compress(prompt, DEFAULT_TARGET_RATIO))
            del compressor
            clear_memory()
            print("✅ Compression phase complete")
        except Exception as e:
            print(f"❌ Compression error: {e}")
            continue

        # Evaluation Phase
        print(f"Evaluating {len(dataset)} prompts using {compression_method}...")
        try:
            target_llm = LLMFactory.create(provider=DEFAULT_LLM_PROVIDER)
            evaluator = Evaluator(task=task_name, llm=target_llm)

            for i, (original_prompt, ground_truth) in enumerate(zip(prompts, ground_truths)):
                if (i + 1) % 2 == 0:
                    print(f"Evaluated {i + 1}/{len(prompts)} prompts")

                # Initialize sample result if not exists
                if i not in sample_results:
                    sample_results[i] = initialize_sample_result(i, task_name, original_prompt, ground_truth)

                # Evaluate baseline (original prompt) - only once per sample
                if not sample_results[i].get("baseline_evaluated", False):
                    print(f"  Evaluating baseline for sample {i+1}...")
                    baseline_metrics = evaluator.evaluate(original_prompt, ground_truth)
                    update_baseline_metrics(sample_results[i], baseline_metrics, task_name, ground_truth)

                # Evaluate compressed prompt
                print(f"  Evaluating compressed prompt for sample {i+1}...")
                compressed_metrics = evaluator.evaluate(compressed_prompts[i], ground_truth)

                # Add compression method results
                compression_data = {
                    "method": compression_method,
                    "compressed_prompt": compressed_prompts[i],
                    "compressed_prompt_output": compressed_metrics['llm_response'],
                    "compressed_score": compressed_metrics['score'],
                    "compressed_latency": compressed_metrics['latency']
                }

                # Add task-specific metrics
                if task_name == "reasoning":
                    baseline_answer = sample_results[i]["baseline_extracted_answer"]
                    compressed_answer = extract_gsm8k_answer(compressed_metrics['llm_response'])
                    compression_data["compressed_extracted_answer"] = compressed_answer
                    compression_data["answers_match"] = (baseline_answer == compressed_answer) and (baseline_answer is not None)

                sample_results[i]["compression_methods"].append(compression_data)

                # Legacy logging
                log_data = create_log_data(i, task_name, compression_method, original_prompt,
                                         compressed_prompts[i], ground_truth, sample_results[i],
                                         compressed_metrics)
                all_results.append(log_data)

            del target_llm
            clear_memory()
            print("✅ Evaluation phase complete")
        except Exception as e:
            print(f"❌ Evaluation error: {e}")
            continue

    # Log results
    print("\n--- Logging Results ---")
    for sample_data in sample_results.values():
        logger.log_result(sample_data)

    # Save results
    logger.finalize_and_save()
    summary = logger.generate_summary_report()
    analysis_file = logger.export_analysis_report()

    print(f"✅ {task_name.upper()} benchmark completed!")
    print(f"   📊 Results saved to: {logger.log_dir}")
    print(f"   📈 Analysis report: {analysis_file}")

    return all_results

def extract_task_data(task_name: str, dataset):
    """Extract prompts and ground truth based on task type."""
    if task_name == "reasoning":
        prompts = [sample['question'] for sample in dataset]
        ground_truths = [sample['answer'] for sample in dataset]
    elif task_name == "summarization":
        # For summarization, truncate very long articles to prevent timeout
        prompts = []
        for sample in dataset:
            article = sample['article']
            # Truncate very long articles to ~800 words to fit in context window
            words = article.split()
            if len(words) > 800:
                truncated_article = ' '.join(words[:800]) + "..."
                print(f"⚠️  Truncated article from {len(words)} to 800 words for summarization")
            else:
                truncated_article = article
            prompts.append(f"Summarize the following article:\n\n{truncated_article}")
        ground_truths = [sample['highlights'] for sample in dataset]
    elif task_name == "classification":
        prompts = [f"Analyze the sentiment of this movie review and respond with only 'positive' or 'negative':\n\n{sample['text']}" for sample in dataset]
        ground_truths = [str(sample['label']) for sample in dataset]
    else:
        raise ValueError(f"Unknown task: {task_name}")

    return prompts, ground_truths

def initialize_sample_result(sample_id: int, task_name: str, original_prompt: str, ground_truth: str):
    """Initialize sample result structure."""
    return {
        "sample_id": sample_id + 1,
        "task": task_name,
        "llm_provider": DEFAULT_LLM_PROVIDER,
        "llm_model": get_model_name(DEFAULT_LLM_PROVIDER),
        "original_prompt": original_prompt,
        "ground_truth_answer": ground_truth,
        "compression_methods": [],
        "target_compression_ratio": 1 - DEFAULT_TARGET_RATIO
    }

def get_model_name(provider: str):
    """Get model name based on provider."""
    if provider == "huggingface":
        return HUGGINGFACE_MODEL
    elif provider == "openai":
        return OPENAI_MODEL
    elif provider == "llamacpp":
        return f"{LLAMACPP_REPO_ID}/{LLAMACPP_FILENAME}" if LLAMACPP_REPO_ID else "llama-cpp-local"
    else:
        return "unknown"

def update_baseline_metrics(sample_result: dict, baseline_metrics: dict, task_name: str, ground_truth: str):
    """Update sample result with baseline metrics."""
    sample_result.update({
        "original_prompt_output": baseline_metrics['llm_response'],
        "baseline_score": baseline_metrics['score'],
        "baseline_latency": baseline_metrics['latency'],
        "baseline_evaluated": True
    })

    if task_name == "reasoning":
        baseline_answer = extract_gsm8k_answer(baseline_metrics['llm_response'])
        sample_result["baseline_extracted_answer"] = baseline_answer

def create_log_data(sample_id: int, task_name: str, compression_method: str,
                   original_prompt: str, compressed_prompt: str, ground_truth: str,
                   sample_result: dict, compressed_metrics: dict):
    """Create log data entry."""
    log_data = {
        "sample_id": sample_id + 1,
        "task": task_name,
        "llm_provider": DEFAULT_LLM_PROVIDER,
        "llm_model": sample_result["llm_model"],
        "compression_method": compression_method,
        "target_compression_ratio": 1 - DEFAULT_TARGET_RATIO,
        "original_prompt": original_prompt,
        "compressed_prompt": compressed_prompt,
        "ground_truth_answer": ground_truth,
        "original_prompt_output": sample_result["original_prompt_output"],
        "compressed_prompt_output": compressed_metrics['llm_response'],
        "baseline_score": sample_result["baseline_score"],
        "compressed_score": compressed_metrics['score'],
        "baseline_latency": sample_result["baseline_latency"],
        "compressed_latency": compressed_metrics['latency'],
    }

    if task_name == "reasoning":
        log_data.update({
            "baseline_extracted_answer": sample_result.get("baseline_extracted_answer"),
            "compressed_extracted_answer": compressed_metrics.get('extracted_answer'),
            "answers_match": compressed_metrics.get('extracted_answer') == sample_result.get("baseline_extracted_answer")
        })

    return log_data

def run_multi_task_benchmark(tasks_to_run=None):
    """Run benchmarks for multiple task types."""
    if tasks_to_run is None:
        tasks_to_run = SUPPORTED_TASKS

    print("🚀 Starting Multi-Task Prompt Compression Benchmark")
    print(f"📋 Tasks to run: {', '.join(tasks_to_run)}")
    print(f"🗜️  Compression methods: {', '.join(COMPRESSION_METHODS_TO_RUN)}")
    print(f"🤖 LLM Provider: {DEFAULT_LLM_PROVIDER}")

    all_task_results = {}

    for task in tasks_to_run:
        try:
            task_results = run_benchmark_for_task(task)
            all_task_results[task] = task_results
        except Exception as e:
            print(f"❌ Failed to run benchmark for task {task}: {e}")
            continue

    # Print summary across all tasks
    print(f"\n{'='*80}")
    print("🎉 MULTI-TASK BENCHMARK SUMMARY")
    print(f"{'='*80}")

    for task, results in all_task_results.items():
        if results:
            print(f"\n📊 {task.upper()} Results:")
            print(f"   Samples processed: {len(results)}")
            print(f"   Methods tested: {len(COMPRESSION_METHODS_TO_RUN)}")

    print("\n✅ All benchmarks completed!")
    print(f"📁 Check the results/ directory for detailed reports")

def run_benchmark():
    """Legacy single-task benchmark for backward compatibility."""
    run_multi_task_benchmark([DEFAULT_TASK])

if __name__ == "__main__":
    # Run multi-task benchmark by default
    run_multi_task_benchmark()
