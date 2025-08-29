import torch
import statistics
from config import (
    DEFAULT_LLM_PROVIDER, DEFAULT_DATASET, DEFAULT_DATASET_CONFIG, NUM_SAMPLES_TO_RUN,
    DEFAULT_TASK, COMPRESSION_METHODS_TO_RUN, DEFAULT_TARGET_RATIO,
    HUGGINGFACE_MODEL, OPENAI_MODEL, LLAMACPP_REPO_ID, LLAMACPP_FILENAME
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

def run_benchmark():
    print("--- Starting Prompt Compression Benchmark ---")
    
    # 1. Setup
    logger = BenchmarkLogger()
    dataset = load_benchmark_dataset(DEFAULT_DATASET, DEFAULT_DATASET_CONFIG, NUM_SAMPLES_TO_RUN)
    if not dataset: return

    # 2. Get original prompts
    original_prompts = [sample['question'] for sample in dataset]
    
    # 3. Run benchmark for each compression method
    all_results = []
    
    # Initialize results storage for sample-centric logging
    sample_results = {}
    
    for compression_method in COMPRESSION_METHODS_TO_RUN:
        print(f"\n--- Running benchmark for compression method: {compression_method} ---")
        
        # 3a. Compression Phase
        print(f"Compressing {len(original_prompts)} prompts using {compression_method}...")
        compressed_prompts = []
        try:
            compressor = CompressorFactory.create(compression_method)
            for i, prompt in enumerate(original_prompts):
                if (i + 1) % 5 == 0:  # Progress update every 5 prompts
                    print(f"Compressed {i + 1}/{len(original_prompts)} prompts")
                compressed_prompts.append(compressor.compress(prompt, DEFAULT_TARGET_RATIO))
            del compressor
            clear_memory()
            print(f"--- Compression phase complete for {compression_method} ---")
        except Exception as e:
            print(f"An error occurred during the compression phase for {compression_method}: {e}")
            continue

        # 3b. Evaluation Phase
        print(f"Evaluating {len(dataset)} prompts using {compression_method}...")
        try:
            target_llm = LLMFactory.create(provider=DEFAULT_LLM_PROVIDER)
            evaluator = Evaluator(task=DEFAULT_TASK, llm=target_llm)
            
            for i, sample in enumerate(dataset):
                if (i + 1) % 2 == 0:  # Progress update every 2 evaluations
                    print(f"Evaluated {i + 1}/{len(dataset)} prompts")
                
                # Initialize sample result if not exists
                if i not in sample_results:
                    sample_results[i] = {
                        "sample_id": i + 1,
                        "llm_provider": DEFAULT_LLM_PROVIDER,
                        "llm_model": (
                            HUGGINGFACE_MODEL if DEFAULT_LLM_PROVIDER == "huggingface" 
                            else OPENAI_MODEL if DEFAULT_LLM_PROVIDER == "openai"
                            else f"{LLAMACPP_REPO_ID}/{LLAMACPP_FILENAME}" if DEFAULT_LLM_PROVIDER == "llamacpp" and LLAMACPP_REPO_ID
                            else "llama-cpp-local" if DEFAULT_LLM_PROVIDER == "llamacpp"
                            else "unknown"
                        ),
                        "original_prompt": original_prompts[i],
                        "ground_truth_answer": sample['answer'],
                        "compression_methods": [],
                        "target_compression_ratio": 1 - DEFAULT_TARGET_RATIO
                    }
                
                # Evaluate the original prompt (baseline) - only once per sample
                if not sample_results[i].get("baseline_evaluated", False):
                    baseline_metrics = evaluator.evaluate(original_prompts[i], sample['answer'])
                    baseline_answer = extract_gsm8k_answer(baseline_metrics['llm_response'])
                    sample_results[i].update({
                        "original_prompt_output": baseline_metrics['llm_response'],
                        "baseline_extracted_answer": baseline_answer,
                        "baseline_score": baseline_metrics['score'],
                        "baseline_latency": baseline_metrics['latency'],
                        "baseline_evaluated": True
                    })
                
                # Evaluate the compressed prompt
                compressed_metrics = evaluator.evaluate(compressed_prompts[i], sample['answer'])
                
                # Perform the new answer consistency check
                baseline_answer = sample_results[i]["baseline_extracted_answer"]
                compressed_answer = extract_gsm8k_answer(compressed_metrics['llm_response'])
                answers_match = (baseline_answer == compressed_answer) and (baseline_answer is not None and baseline_answer != "")
                
                # Add compression method results to sample
                compression_data = {
                    "method": compression_method,
                    "compressed_prompt": compressed_prompts[i],
                    "compressed_prompt_output": compressed_metrics['llm_response'],
                    "compressed_extracted_answer": compressed_answer,
                    "compressed_score": compressed_metrics['score'],
                    "compressed_latency": compressed_metrics['latency'],
                    "answers_match": answers_match
                }
                sample_results[i]["compression_methods"].append(compression_data)
                
                # Legacy logging for backward compatibility
                log_data = {
                    "sample_id": i + 1,
                    "llm_provider": DEFAULT_LLM_PROVIDER,
                    "llm_model": sample_results[i]["llm_model"],
                    "compression_method": compression_method,
                    "target_compression_ratio": 1 - DEFAULT_TARGET_RATIO,
                    "original_prompt": original_prompts[i],
                    "compressed_prompt": compressed_prompts[i],
                    "ground_truth_answer": sample['answer'],
                    "original_prompt_output": sample_results[i]["original_prompt_output"],
                    "compressed_prompt_output": compressed_metrics['llm_response'],
                    "baseline_extracted_answer": baseline_answer,
                    "compressed_extracted_answer": compressed_answer,
                    "baseline_score": sample_results[i]["baseline_score"],
                    "compressed_score": compressed_metrics['score'],
                    "answers_match": answers_match,
                    "baseline_latency": sample_results[i]["baseline_latency"],
                    "compressed_latency": compressed_metrics['latency'],
                }
                all_results.append(log_data)
                
            del target_llm
            clear_memory()
            print(f"--- Evaluation phase complete for {compression_method} ---")
        except Exception as e:
            print(f"An error occurred during the evaluation phase for {compression_method}: {e}")
            continue

    # Log all sample results using the new sample-centric approach
    print("\n--- Logging Sample-Centric Results ---")
    for sample_data in sample_results.values():
        logger.log_result(sample_data)

    # Finalize and save all logged data to CSV
    print("\n--- Finalizing and Saving Results ---")
    logger.finalize_and_save()

    # Generate comprehensive summary report
    print("\n--- Generating Comprehensive Summary Report ---")
    summary = logger.generate_summary_report()

    # Generate detailed analysis report
    print("\n--- Generating Detailed Analysis Report ---")
    analysis_file = logger.export_analysis_report()

    print(f"\n🎉 Benchmark completed successfully!")
    print(f"   Samples processed: {len(dataset)}")
    print(f"   Compression methods: {len(COMPRESSION_METHODS_TO_RUN)}")
    print(f"   Results saved to: {logger.log_dir}")
    print(f"   Analysis report: {analysis_file}")

    # 4. Legacy aggregate results (for backward compatibility)
    if all_results:
        # Group results by compression method
        method_results = {}
        for result in all_results:
            method = result['compression_method']
            if method not in method_results:
                method_results[method] = []
            method_results[method].append(result)
        
        print("\n" + "="*50 + "\n--- AGGREGATE BENCHMARK RESULTS ---\n" + "="*50)
        print(f"  Dataset: {DEFAULT_DATASET}, Samples Run: {len(dataset)}")
        print(f"  LLM Provider: {DEFAULT_LLM_PROVIDER}")
        if DEFAULT_LLM_PROVIDER == "huggingface":
            print(f"  Model: {HUGGINGFACE_MODEL}")
        elif DEFAULT_LLM_PROVIDER == "openai":
            print(f"  Model: {OPENAI_MODEL}")
        elif DEFAULT_LLM_PROVIDER == "llamacpp":
            if LLAMACPP_REPO_ID:
                print(f"  Model: {LLAMACPP_REPO_ID}/{LLAMACPP_FILENAME}")
            else:
                print(f"  Model: Local llama.cpp model")
        
        for method, results in method_results.items():
            print(f"\n--- Results for {method} ---")
            avg_baseline_score = statistics.mean([r['baseline_score'] for r in results])
            avg_compressed_score = statistics.mean([r['compressed_score'] for r in results])
            consistency_rate = statistics.mean([1 if r['answers_match'] else 0 for r in results])
            
            print(f"  Average Baseline Score (vs. Ground Truth): {avg_baseline_score:.2%}")
            print(f"  Average Compressed Score (vs. Ground Truth): {avg_compressed_score:.2%}")
            print(f"  Answer Consistency Rate (Compressed vs. Baseline): {consistency_rate:.2%}")
    
    print("\n--- Benchmark Run Finished ---")

if __name__ == "__main__":
    run_benchmark()
