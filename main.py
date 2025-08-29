import torch
import statistics
import psutil
import gc
import concurrent.futures
from functools import partial
from config import *
from data_loaders.loaders import load_benchmark_dataset
from llms.factory import LLMFactory
from compressors.factory import CompressorFactory

from evaluation.evaluator import Evaluator
from utils.logger import BenchmarkLogger
from evaluation.utils import extract_gsm8k_answer
from tqdm import tqdm

def get_memory_usage():
    """Get current memory usage statistics."""
    memory_info = {}
    
    # CPU memory
    memory_info['cpu_percent'] = psutil.virtual_memory().percent
    memory_info['cpu_used_gb'] = psutil.virtual_memory().used / (1024**3)
    memory_info['cpu_total_gb'] = psutil.virtual_memory().total / (1024**3)
    
    # GPU memory (if available)
    if torch.cuda.is_available():
        memory_info['gpu_allocated_gb'] = torch.cuda.memory_allocated() / (1024**3)
        memory_info['gpu_reserved_gb'] = torch.cuda.memory_reserved() / (1024**3)
        memory_info['gpu_total_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        memory_info['gpu_percent'] = (memory_info['gpu_allocated_gb'] / memory_info['gpu_total_gb']) * 100
    elif torch.backends.mps.is_available():
        # MPS doesn't have detailed memory stats, but we can track roughly
        memory_info['mps_available'] = True
    
    return memory_info

def clear_memory():
    """Aggressive memory clearing optimized for Colab."""
    # Clear PyTorch cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Ensure all operations are complete
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    # Force garbage collection
    gc.collect()
    
    # Clear any cached variables
    if 'compressor' in globals():
        del globals()['compressor']
    if 'target_llm' in locals():
        del locals()['target_llm']

def check_memory_threshold():
    """Check if memory usage is approaching dangerous levels."""
    if not ENABLE_MEMORY_MONITORING:
        return False
        
    memory_info = get_memory_usage()
    
    # Check GPU memory if available
    if torch.cuda.is_available() and 'gpu_percent' in memory_info:
        if memory_info['gpu_percent'] > MEMORY_WARNING_THRESHOLD * 100:
            print(f"⚠️  WARNING: GPU memory usage at {memory_info['gpu_percent']:.1f}%")
            return True
    
    # Check CPU memory
    if memory_info['cpu_percent'] > 90:
        print(f"⚠️  WARNING: CPU memory usage at {memory_info['cpu_percent']:.1f}%")
        return True
        
    return False

def compress_single_prompt(compressor, prompt, target_ratio):
    """Compress a single prompt - used for parallel processing."""
    return compressor.compress(prompt, target_ratio)

def parallel_compress_prompts(original_prompts, compressor, target_ratio):
    """Compress multiple prompts in parallel for better resource utilization."""
    if not ENABLE_PARALLEL_PROCESSING or len(original_prompts) < 2:
        # Fall back to sequential processing for small batches
        return [compress_single_prompt(compressor, prompt, target_ratio) for prompt in original_prompts]
    
    print(f"🚀 Compressing {len(original_prompts)} prompts in parallel (max {MAX_CONCURRENT_PROCESSES} concurrent)...")
    
    # Use ThreadPoolExecutor for I/O bound compression tasks
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_PROCESSES) as executor:
        # Create partial function with fixed arguments
        compress_func = partial(compress_single_prompt, compressor, target_ratio=target_ratio)
        
        # Submit all compression tasks
        future_to_prompt = {executor.submit(compress_func, prompt): prompt for prompt in original_prompts}
        
        compressed_prompts = []
        for future in tqdm(concurrent.futures.as_completed(future_to_prompt), 
                          total=len(original_prompts), desc="Parallel compression"):
            try:
                compressed_prompts.append(future.result())
            except Exception as exc:
                print(f'❌ Compression failed: {exc}')
                compressed_prompts.append("")  # Add empty string as fallback
        
        return compressed_prompts

def warmup_model(llm):
    """Warm up the model with a simple inference to optimize performance."""
    if not ENABLE_MODEL_WARMUP:
        return
        
    print("🔥 Warming up model for optimal performance...")
    try:
        # Simple warm-up prompt
        warmup_prompt = "What is 2 + 2?"
        _ = llm.get_response(warmup_prompt)
        print("✅ Model warmed up successfully")
    except Exception as e:
        print(f"⚠️  Model warm-up failed (non-critical): {e}")

def batch_evaluate_samples(llm, evaluator, samples, compressed_prompts, logger):
    """Evaluate multiple samples in batches for better GPU utilization."""
    all_results = []
    
    # Process in batches for better GPU utilization
    for i in range(0, len(samples), BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, len(samples))
        batch_samples = samples[i:batch_end]
        batch_compressed = compressed_prompts[i:batch_end]
        
        print(f"📦 Processing batch {i//BATCH_SIZE + 1}/{(len(samples) + BATCH_SIZE - 1)//BATCH_SIZE} (samples {i+1}-{batch_end})")
        
        # Memory check before batch processing
        if check_memory_threshold():
            clear_memory()
            print(f"🧹 Memory cleared before batch {i//BATCH_SIZE + 1}")
        
        # Prepare batch data
        original_prompts_batch = [sample['question'] for sample in batch_samples]
        ground_truths_batch = [sample['answer'] for sample in batch_samples]
        
        try:
            # Batch evaluate original prompts
            baseline_metrics_batch = evaluator.evaluate_batch(original_prompts_batch, ground_truths_batch)
            
            # Batch evaluate compressed prompts
            compressed_metrics_batch = evaluator.evaluate_batch(batch_compressed, ground_truths_batch)
            
            # Process results for each sample in batch
            for j in range(len(baseline_metrics_batch)):
                sample_idx = i + j
                
                # Reconstruct sample dict for this index
                sample = batch_samples[j]
                
                # Perform the new answer consistency check
                baseline_answer = extract_gsm8k_answer(baseline_metrics_batch[j]['llm_response'])
                compressed_answer = extract_gsm8k_answer(compressed_metrics_batch[j]['llm_response'])
                answers_match = (baseline_answer == compressed_answer) and (baseline_answer is not None and baseline_answer != "")

                # Log all the data for this sample
                log_data = {
                    "sample_id": sample_idx + 1,
                    "llm_provider": DEFAULT_LLM_PROVIDER,
                    "llm_model": HUGGINGFACE_MODEL if DEFAULT_LLM_PROVIDER == "huggingface" else OPENAI_MODEL,
                    "compression_method": DEFAULT_COMPRESSION_METHOD,
                    "target_compression_ratio": 1 - DEFAULT_TARGET_RATIO,
                    "original_prompt": sample['question'],
                    "compressed_prompt": batch_compressed[j],
                    "ground_truth_answer": sample['answer'],
                    "original_prompt_output": baseline_metrics_batch[j]['llm_response'],
                    "compressed_prompt_output": compressed_metrics_batch[j]['llm_response'],
                    "baseline_score": baseline_metrics_batch[j]['score'],
                    "compressed_score": compressed_metrics_batch[j]['score'],
                    "answers_match": answers_match,
                    "baseline_latency": baseline_metrics_batch[j]['latency'],
                    "compressed_latency": compressed_metrics_batch[j]['latency'],
                }
                logger.log_result(log_data)
                all_results.append(log_data)
                
        except Exception as e:
            print(f"❌ Error processing batch {i//BATCH_SIZE + 1}: {e}")
            # Fallback to individual processing for this batch
            for j in range(len(batch_samples)):
                sample_idx = i + j
                sample = batch_samples[j]
                try:
                    # Individual evaluation as fallback
                    baseline_metrics = evaluator.evaluate(sample['question'], sample['answer'])
                    compressed_metrics = evaluator.evaluate(batch_compressed[j], sample['answer'])
                    
                    baseline_answer = extract_gsm8k_answer(baseline_metrics['llm_response'])
                    compressed_answer = extract_gsm8k_answer(compressed_metrics['llm_response'])
                    answers_match = (baseline_answer == compressed_answer) and (baseline_answer is not None and baseline_answer != "")

                    log_data = {
                        "sample_id": sample_idx + 1,
                        "llm_provider": DEFAULT_LLM_PROVIDER,
                        "llm_model": HUGGINGFACE_MODEL if DEFAULT_LLM_PROVIDER == "huggingface" else OPENAI_MODEL,
                        "compression_method": DEFAULT_COMPRESSION_METHOD,
                        "target_compression_ratio": 1 - DEFAULT_TARGET_RATIO,
                        "original_prompt": sample['question'],
                        "compressed_prompt": batch_compressed[j],
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
                    
                except Exception as e2:
                    print(f"❌ Error processing sample {sample_idx+1}: {e2}")
                    error_data = {
                        "sample_id": sample_idx + 1,
                        "error": str(e2),
                        "llm_provider": DEFAULT_LLM_PROVIDER,
                        "compression_method": DEFAULT_COMPRESSION_METHOD,
                    }
                    logger.log_result(error_data)
                    continue
        
        # Memory cleanup after each batch (less aggressive than before)
        if (i // BATCH_SIZE + 1) % CLEAR_MEMORY_EVERY_N_SAMPLES == 0:
            clear_memory()
    
    return all_results

def run_benchmark():
    print("--- Starting Prompt Compression Benchmark (Colab Optimized) ---")
    print(f"Memory monitoring: {'Enabled' if ENABLE_MEMORY_MONITORING else 'Disabled'}")
    
    # Initial memory check
    if ENABLE_MEMORY_MONITORING:
        initial_memory = get_memory_usage()
        print(f"Initial memory - CPU: {initial_memory['cpu_used_gb']:.1f}/{initial_memory['cpu_total_gb']:.1f} GB")
        if torch.cuda.is_available():
            print(f"Initial GPU memory: {initial_memory['gpu_allocated_gb']:.1f}/{initial_memory['gpu_total_gb']:.1f} GB")
    
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
        
        # Use parallel compression for better resource utilization
        compressed_prompts = parallel_compress_prompts(original_prompts, compressor, DEFAULT_TARGET_RATIO)
        
        del compressor
        clear_memory()
        print("--- Compression phase complete ---")
        
    except Exception as e:
        print(f"An error occurred during the compression phase: {e}")
        clear_memory()
        return

    # 3. Evaluation Phase
    print("\n--- Phase 2: Evaluating all prompts ---")
    all_results = []
    
    try:
        target_llm = LLMFactory.create(provider=DEFAULT_LLM_PROVIDER)
        
        # Model warm-up for optimal performance
        warmup_model(target_llm)
        
        evaluator = Evaluator(task=DEFAULT_TASK, llm=target_llm)
        
        # Use batch processing for better GPU utilization
        all_results = batch_evaluate_samples(target_llm, evaluator, dataset, compressed_prompts, logger)
        
        del target_llm
        clear_memory()
        print("\n--- Evaluation phase complete ---")
        
    except Exception as e:
        print(f"An error occurred during the evaluation phase: {e}")
        clear_memory()
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
        
        # Final memory report
        if ENABLE_MEMORY_MONITORING:
            final_memory = get_memory_usage()
            print("\n--- FINAL MEMORY USAGE ---")
            print(f"Final CPU memory: {final_memory['cpu_used_gb']:.1f}/{final_memory['cpu_total_gb']:.1f} GB")
            if torch.cuda.is_available():
                print(f"Final GPU memory: {final_memory['gpu_allocated_gb']:.1f}/{final_memory['gpu_total_gb']:.1f} GB")
    
    print("\n--- Benchmark Run Finished ---")

if __name__ == "__main__":
    run_benchmark()
