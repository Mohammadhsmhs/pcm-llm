import torch
import statistics
from config import (
    DEFAULT_LLM_PROVIDER, DEFAULT_DATASET, DEFAULT_DATASET_CONFIG, NUM_SAMPLES_TO_RUN,
    DEFAULT_TASK, DEFAULT_COMPRESSION_METHOD, DEFAULT_TARGET_RATIO,
    HUGGINGFACE_MODEL, OPENAI_MODEL, LLAMACPP_REPO_ID, LLAMACPP_FILENAME
)
from data_loaders.loaders import load_benchmark_dataset
from llms.factory import LLMFactory
from compressors.factory import CompressorFactory

from evaluation.evaluator import Evaluator
from utils.logger import BenchmarkLogger
from evaluation.utils import extract_gsm8k_answer
from tqdm import tqdm

def clear_memory():
    """Utility function to clear GPU cache if available."""
    if torch.backends.mps.is_available(): torch.mps.empty_cache()
    elif torch.cuda.is_available(): torch.cuda.empty_cache()

def run_benchmark():
    print("--- Starting Prompt Compression Benchmark ---")
    
    # 1. Setup
    logger = BenchmarkLogger()
    dataset = load_benchmark_dataset(DEFAULT_DATASET, DEFAULT_DATASET_CONFIG, NUM_SAMPLES_TO_RUN)
    if not dataset: return

    # 2. Compression Phase
    print("\n--- Phase 1: Compressing all prompts ---")
    original_prompts = [sample['question'] for sample in dataset]
    compressed_prompts = []
    try:
        compressor = CompressorFactory.create(DEFAULT_COMPRESSION_METHOD)
        # Correctly wrap the iterable with tqdm
        for prompt in tqdm(original_prompts, desc="Compressing prompts"):
             compressed_prompts.append(compressor.compress(prompt, DEFAULT_TARGET_RATIO))
        del compressor
        clear_memory()
        print("--- Compression phase complete ---")
    except Exception as e:
        print(f"An error occurred during the compression phase: {e}")
        return

    # 3. Evaluation Phase
    print("\n--- Phase 2: Evaluating all prompts ---")
    all_results = []
    try:
        target_llm = LLMFactory.create(provider=DEFAULT_LLM_PROVIDER)
        evaluator = Evaluator(task=DEFAULT_TASK, llm=target_llm)
        # Correctly wrap the iterable with tqdm
        for i, sample in enumerate(tqdm(dataset, desc="Evaluating prompts")):
            
            # Evaluate the original prompt (baseline)
            baseline_metrics = evaluator.evaluate(original_prompts[i], sample['answer'])
            
            # Evaluate the compressed prompt
            compressed_metrics = evaluator.evaluate(compressed_prompts[i], sample['answer'])
            
            # Perform the new answer consistency check
            baseline_answer = extract_gsm8k_answer(baseline_metrics['llm_response'])
            compressed_answer = extract_gsm8k_answer(compressed_metrics['llm_response'])
            answers_match = (baseline_answer == compressed_answer) and (baseline_answer is not None and baseline_answer != "")

            # Log all the data for this sample
            log_data = {
                "sample_id": i + 1,
                "llm_provider": DEFAULT_LLM_PROVIDER,
                "llm_model": (
                    HUGGINGFACE_MODEL if DEFAULT_LLM_PROVIDER == "huggingface" 
                    else OPENAI_MODEL if DEFAULT_LLM_PROVIDER == "openai"
                    else f"{LLAMACPP_REPO_ID}/{LLAMACPP_FILENAME}" if DEFAULT_LLM_PROVIDER == "llamacpp" and LLAMACPP_REPO_ID
                    else "llama-cpp-local" if DEFAULT_LLM_PROVIDER == "llamacpp"
                    else "unknown"
                ),
                "compression_method": DEFAULT_COMPRESSION_METHOD,
                "target_compression_ratio": 1 - DEFAULT_TARGET_RATIO,
                "original_prompt": original_prompts[i],
                "compressed_prompt": compressed_prompts[i],
                "ground_truth_answer": sample['answer'],
                "original_prompt_output": baseline_metrics['llm_response'],
                "compressed_prompt_output": compressed_metrics['llm_response'],
                "baseline_score": baseline_metrics['score'],
                "compressed_score": compressed_metrics['score'],
                "answers_match": answers_match,
                "baseline_latency": baseline_metrics['latency'],
                "compressed_latency": compressed_metrics['latency'],
            }
            logger.log_result(log_data)
            all_results.append(log_data)
            
        del target_llm
        clear_memory()
        print("\n--- Evaluation phase complete ---")
    except Exception as e:
        print(f"An error occurred during the evaluation phase: {e}")
        return

    # 4. Report aggregated results
    if all_results:
        avg_baseline_score = statistics.mean([r['baseline_score'] for r in all_results])
        avg_compressed_score = statistics.mean([r['compressed_score'] for r in all_results])
        consistency_rate = statistics.mean([1 if r['answers_match'] else 0 for r in all_results])
        
        print("\n" + "="*50 + "\n--- AGGREGATE BENCHMARK RESULTS ---\n" + "="*50)
        print(f"  Dataset: {DEFAULT_DATASET}, Samples Run: {len(all_results)}")
        print(f"  LLM Provider: {DEFAULT_LLM_PROVIDER}")
        if DEFAULT_LLM_PROVIDER == "huggingface":
            print(f"  Model: {HUGGINGFACE_MODEL}")
        else:
            print(f"  Model: {OPENAI_MODEL}")
        print(f"  Average Baseline Score (vs. Ground Truth): {avg_baseline_score:.2%}")
        print(f"  Average Compressed Score (vs. Ground Truth): {avg_compressed_score:.2%}")
        print(f"  Answer Consistency Rate (Compressed vs. Baseline): {consistency_rate:.2%}")
    
    print("\n--- Benchmark Run Finished ---")

if __name__ == "__main__":
    run_benchmark()
