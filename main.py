import torch
import statistics
from config import *
from data_loaders.loaders import load_benchmark_dataset
from llms.factory import LLMFactory
from compressors.factory import CompressorFactory
from evaluation.evaluator import Evaluator
from utils.logger import BenchmarkLogger
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
        for i, prompt in enumerate(tqdm(original_prompts, desc="Compressing")):
            print(f"Compressing sample {i+1}/{len(dataset)}...")
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
        for i, sample in enumerate(tqdm(dataset, desc="Evaluating")):
            print(f"\n--- Evaluating Sample {i+1}/{len(dataset)} ---")
            
            # Evaluate the original prompt (baseline)
            print("Evaluating original prompt...")
            baseline_metrics = evaluator.evaluate(original_prompts[i], sample['answer'])
            
            # Evaluate the compressed prompt
            print("Evaluating compressed prompt...")
            compressed_metrics = evaluator.evaluate(compressed_prompts[i], sample['answer'])
            
            # Log all the data for this sample
            log_data = {
                "sample_id": i + 1,
                "llm_provider": DEFAULT_LLM_PROVIDER,
                "llm_model": HUGGINGFACE_MODEL if DEFAULT_LLM_PROVIDER == "huggingface" else OPENAI_MODEL,
                "compression_method": DEFAULT_COMPRESSION_METHOD,
                "target_compression_ratio": 1 - DEFAULT_TARGET_RATIO,
                "original_prompt": original_prompts[i],
                "compressed_prompt": compressed_prompts[i],
                "ground_truth_answer": sample['answer'],
                "original_prompt_output": baseline_metrics['llm_response'],
                "compressed_prompt_output": compressed_metrics['llm_response'],
                "baseline_score": baseline_metrics['score'],
                "compressed_score": compressed_metrics['score'],
                "baseline_latency": baseline_metrics['latency'],
                "compressed_latency": compressed_metrics['latency'],
            }
            logger.log_result(log_data)
            all_results.append(log_data)
            
        del target_llm
        clear_memory()
        print("--- Evaluation phase complete ---")
    except Exception as e:
        print(f"An error occurred during the evaluation phase: {e}")
        return

    # 4. Report aggregated results
    if all_results:
        avg_baseline_score = statistics.mean([r['baseline_score'] for r in all_results])
        avg_compressed_score = statistics.mean([r['compressed_score'] for r in all_results])
        
        print("\n" + "="*50 + "\n--- AGGREGATE BENCHMARK RESULTS ---\n" + "="*50)
        print(f"  Dataset: {DEFAULT_DATASET}, Samples Run: {len(all_results)}")
        print(f"  Average Baseline Score: {avg_baseline_score:.2%}")
        print(f"  Average Compressed Score: {avg_compressed_score:.2%}")
    
    print("\n--- Benchmark Run Finished ---")

if __name__ == "__main__":
    run_benchmark()
