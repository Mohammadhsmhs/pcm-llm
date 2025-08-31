import torch
import statistics
import threading
import psutil
import os
import csv
import json
import tempfile
import hashlib
from datetime import datetime
from collections import defaultdict
from config import (
    SUPPORTED_TASKS, DEFAULT_TASK, TASK_CONFIGURATIONS,
    DEFAULT_LLM_PROVIDER, COMPRESSION_METHODS_TO_RUN, DEFAULT_TARGET_RATIO,
    HUGGINGFACE_MODEL, LLAMACPP_REPO_ID, LLAMACPP_FILENAME,
    NUM_SAMPLES_TO_RUN,
    # New optimized config
    USE_JSONL, ENABLE_CHECKPOINTING, MAX_CONCURRENT_LOGGERS,
    COMPRESS_INTERMEDIATE, ADAPTIVE_BATCH_SIZE, MEMORY_CHECKPOINT_INTERVAL,
    BATCH_SIZE_BASE, UNLIMITED_MODE
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

# --- Optimized Implementation Utilities ---

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def calculate_adaptive_batch_size(base_batch_size=BATCH_SIZE_BASE):
    """Calculate adaptive batch size based on available memory."""
    if not ADAPTIVE_BATCH_SIZE:
        return base_batch_size

    available_memory = psutil.virtual_memory().available / 1024 / 1024  # MB
    memory_usage = get_memory_usage()

    # Use 50% of available memory as threshold
    memory_threshold = available_memory * 0.5

    if memory_usage > memory_threshold:
        # Reduce batch size if memory usage is high
        adaptive_size = max(1, base_batch_size // 2)
        print(f"⚠️  High memory usage ({memory_usage:.1f}MB), reducing batch size to {adaptive_size}")
        return adaptive_size
    elif memory_usage < memory_threshold * 0.3:
        # Increase batch size if memory usage is low
        adaptive_size = min(base_batch_size * 2, 20)  # Cap at 20
        return adaptive_size

    return base_batch_size

def log_memory_usage(phase_name):
    """Log memory usage summary."""
    memory_mb = get_memory_usage()
    print(f"📊 Memory at {phase_name}: {memory_mb:.1f}MB")

class ThreadSafeLogger:
    """Thread-safe logger using semaphore."""
    def __init__(self, logger_instance):
        self.logger = logger_instance
        self.semaphore = threading.Semaphore(MAX_CONCURRENT_LOGGERS)

    def log_result(self, result_data):
        """Thread-safe logging."""
        with self.semaphore:
            self.logger.log_result(result_data)

def save_checkpoint(checkpoint_data, checkpoint_file):
    """Save checkpoint data to file."""
    if not ENABLE_CHECKPOINTING:
        return

    try:
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        print(f"💾 Checkpoint saved: {checkpoint_file}")
    except Exception as e:
        print(f"⚠️  Failed to save checkpoint: {e}")

def load_checkpoint(checkpoint_file):
    """Load checkpoint data from file."""
    if not ENABLE_CHECKPOINTING or not os.path.exists(checkpoint_file):
        return None

    try:
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Failed to load checkpoint: {e}")
        return None

def write_intermediate_csv(data_rows, csv_file_path, fieldnames):
    """Write data rows to CSV file."""
    try:
        file_exists = os.path.exists(csv_file_path)
        with open(csv_file_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL,
                                  escapechar='\\', doublequote=True)
            if not file_exists:
                writer.writeheader()
            writer.writerows(data_rows)
    except Exception as e:
        print(f"❌ Failed to write to CSV: {e}")
        raise

# --- Compression Caching Functions ---

COMPRESSED_CACHE_DIR = "compressed_cache"
SAMPLES_DIR = os.path.join(COMPRESSED_CACHE_DIR, "samples")
COMPRESSED_DIR = os.path.join(COMPRESSED_CACHE_DIR, "compressed")

def get_cache_key(task_name: str, compression_method: str, num_samples: int, target_ratio: float) -> str:
    """Generate a unique cache key for compressed prompts."""
    # Create a hash based on task, method, sample count, and compression ratio
    cache_string = f"{task_name}_{compression_method}_{num_samples}_{target_ratio}"
    return hashlib.md5(cache_string.encode()).hexdigest()[:16]

def get_samples_cache_path(task_name: str, num_samples: int) -> str:
    """Get path for cached samples file."""
    return os.path.join(SAMPLES_DIR, f"{task_name}_{num_samples}_samples.json")

def get_compressed_cache_path(task_name: str, compression_method: str, num_samples: int, target_ratio: float) -> str:
    """Get path for cached compressed prompts file."""
    cache_key = get_cache_key(task_name, compression_method, num_samples, target_ratio)
    return os.path.join(COMPRESSED_DIR, f"{task_name}_{compression_method}_{cache_key}.json")

def save_samples_to_cache(task_name: str, samples_data: list, num_samples: int):
    """Save original samples to cache."""
    cache_path = get_samples_cache_path(task_name, num_samples)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(samples_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Samples cached: {cache_path}")

def load_samples_from_cache(task_name: str, num_samples: int) -> list:
    """Load original samples from cache if available."""
    cache_path = get_samples_cache_path(task_name, num_samples)
    
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            samples_data = json.load(f)
        print(f"📖 Samples loaded from cache: {cache_path}")
        return samples_data
    
    return None

def save_compressed_to_cache(task_name: str, compression_method: str, compressed_prompts: list, 
                           num_samples: int, target_ratio: float):
    """Save compressed prompts to cache."""
    cache_path = get_compressed_cache_path(task_name, compression_method, num_samples, target_ratio)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    cache_data = {
        "task_name": task_name,
        "compression_method": compression_method,
        "num_samples": num_samples,
        "target_ratio": target_ratio,
        "timestamp": datetime.now().isoformat(),
        "compressed_prompts": compressed_prompts
    }
    
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Compressed prompts cached: {cache_path}")

def load_compressed_from_cache(task_name: str, compression_method: str, num_samples: int, target_ratio: float) -> list:
    """Load compressed prompts from cache if available."""
    cache_path = get_compressed_cache_path(task_name, compression_method, num_samples, target_ratio)
    
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        # Verify cache integrity
        if (cache_data.get("task_name") == task_name and
            cache_data.get("compression_method") == compression_method and
            cache_data.get("num_samples") == num_samples and
            abs(cache_data.get("target_ratio", 0) - target_ratio) < 0.001):
            
            compressed_prompts = cache_data.get("compressed_prompts", [])
            if len(compressed_prompts) == num_samples:
                print(f"📖 Compressed prompts loaded from cache: {cache_path}")
                return compressed_prompts
            else:
                print(f"⚠️  Cache incomplete ({len(compressed_prompts)}/{num_samples} samples), will recompress")
        else:
            print(f"⚠️  Cache mismatch, will recompress")
    
    return None

def check_cache_status(task_name: str, compression_method: str, num_samples: int, target_ratio: float) -> bool:
    """Check if compressed prompts are available in cache."""
    cache_path = get_compressed_cache_path(task_name, compression_method, num_samples, target_ratio)
    return os.path.exists(cache_path)

def clear_compression_cache(task_name: str = None, compression_method: str = None):
    """Clear compression cache. If no parameters provided, clears all cache."""
    import shutil
    
    if task_name and compression_method:
        # Clear specific cache
        cache_path = get_compressed_cache_path(task_name, compression_method, 0, 0)  # Dummy values for path generation
        cache_dir = os.path.dirname(cache_path)
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            print(f"🗑️  Cleared cache for {task_name} - {compression_method}")
    elif task_name:
        # Clear all cache for a task
        samples_path = get_samples_cache_path(task_name, 0)  # Dummy value
        samples_dir = os.path.dirname(samples_path)
        if os.path.exists(samples_dir):
            shutil.rmtree(samples_dir)
            print(f"🗑️  Cleared all cache for {task_name}")
    else:
        # Clear entire cache
        if os.path.exists(COMPRESSED_CACHE_DIR):
            shutil.rmtree(COMPRESSED_CACHE_DIR)
            print("🗑️  Cleared entire compression cache")
        
        # Recreate directories
        os.makedirs(SAMPLES_DIR, exist_ok=True)
        os.makedirs(COMPRESSED_DIR, exist_ok=True)
        print("📁 Cache directories recreated")

def show_cache_info():
    """Show information about cached data."""
    print("📊 Compression Cache Information:")
    print("=" * 50)
    
    if not os.path.exists(COMPRESSED_CACHE_DIR):
        print("❌ No cache directory found")
        return
    
    # Count samples
    samples_count = 0
    if os.path.exists(SAMPLES_DIR):
        for file in os.listdir(SAMPLES_DIR):
            if file.endswith('.json'):
                samples_count += 1
    
    # Count compressed files
    compressed_count = 0
    total_size = 0
    if os.path.exists(COMPRESSED_DIR):
        for file in os.listdir(COMPRESSED_DIR):
            if file.endswith('.json'):
                compressed_count += 1
                file_path = os.path.join(COMPRESSED_DIR, file)
                total_size += os.path.getsize(file_path)
    
    print(f"Samples cached: {samples_count} files")
    print(f"Compressed prompts cached: {compressed_count} files")
    print(f"Total cache size: {total_size / 1024 / 1024:.2f} MB")
    
    if samples_count > 0 or compressed_count > 0:
        print("\n💡 To clear cache, use: clear_compression_cache()")
        print("💡 To clear specific task: clear_compression_cache('reasoning')")
        print("💡 To clear specific method: clear_compression_cache('reasoning', 'llmlingua2')")

def run_benchmark_for_task(task_name: str):
    """Optimized benchmark execution with improved memory management."""
    print(f"\n{'='*60}")
    print(f"🧪 Running Optimized Benchmark for Task: {task_name.upper()}")
    print(f"{'='*60}")
    
    # Display unlimited mode status
    if UNLIMITED_MODE:
        print("🔓 UNLIMITED MODE ENABLED: No timeouts or size limits")
        print("⚠️  WARNING: This may cause very long run times and high resource usage")
    else:
        print("🔒 Standard mode: Timeouts and size limits are active")

    # Get task configuration
    task_config = TASK_CONFIGURATIONS[task_name]
    dataset_name = task_config["dataset"]
    dataset_config = task_config["config"]

    # Create temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        intermediate_file = os.path.join(temp_dir, f"intermediate_{task_name}.csv")
        checkpoint_file = os.path.join(temp_dir, f"checkpoint_{task_name}.json")

        # Phase 1: Load dataset and prepare data
        print("📥 Phase 1: Loading dataset...")
        dataset = load_benchmark_dataset(task_name, dataset_name, dataset_config, NUM_SAMPLES_TO_RUN)
        if not dataset:
            print(f"❌ Failed to load dataset for task {task_name}")
            return

        # Extract prompts and ground truth
        prompts, ground_truths = extract_task_data(task_name, dataset)
        log_memory_usage("after dataset loading")

        # Prepare intermediate data structure
        intermediate_data = []
        for i, (prompt, ground_truth) in enumerate(zip(prompts, ground_truths)):
            intermediate_data.append({
                'sample_id': i + 1,
                'task': task_name,
                'original_prompt': prompt,
                'ground_truth': ground_truth,
                'llm_provider': DEFAULT_LLM_PROVIDER,
                'llm_model': get_model_name(DEFAULT_LLM_PROVIDER)
            })

        # Write original prompts to intermediate CSV
        fieldnames = ['sample_id', 'task', 'original_prompt', 'ground_truth',
                     'llm_provider', 'llm_model'] + \
                    [f"{method}_compressed_prompt" for method in COMPRESSION_METHODS_TO_RUN]

        write_intermediate_csv(intermediate_data, intermediate_file, fieldnames)
        print(f"✅ Original prompts saved to intermediate file")

        # Phase 2: Compression Pipeline
        print("\n🗜️  Phase 2: Compression Pipeline")
        compressed_data = {method: [] for method in COMPRESSION_METHODS_TO_RUN}

        for compression_method in COMPRESSION_METHODS_TO_RUN:
            print(f"\n--- Processing {compression_method} ---")

            # Load compressor
            print(f"Loading {compression_method} compressor...")
            compressor = CompressorFactory.create(compression_method)
            log_memory_usage(f"after {compression_method} load")

            # Compress all prompts
            print(f"Compressing {len(prompts)} prompts...")
            compressed_prompts = []
            for i, prompt in enumerate(prompts):
                if (i + 1) % 10 == 0:
                    print(f"Compressed {i + 1}/{len(prompts)} prompts")

                compressed_prompt = compressor.compress(prompt, DEFAULT_TARGET_RATIO)
                compressed_prompts.append(compressed_prompt)

            # Update intermediate data
            for i, compressed_prompt in enumerate(compressed_prompts):
                intermediate_data[i][f"{compression_method}_compressed_prompt"] = compressed_prompt

            # Save checkpoint
            checkpoint_data = {
                'task': task_name,
                'compression_method': compression_method,
                'completed_samples': len(compressed_prompts),
                'timestamp': datetime.now().isoformat()
            }
            save_checkpoint(checkpoint_data, checkpoint_file)

            # Unload compressor
            del compressor
            clear_memory()
            log_memory_usage(f"after {compression_method} unload")

            print(f"✅ {compression_method} compression completed")

        # Update intermediate file with all compressed prompts
        print("💾 Updating intermediate file with compressed prompts...")
        # Clear file and rewrite with compressed data
        if os.path.exists(intermediate_file):
            os.remove(intermediate_file)
        write_intermediate_csv(intermediate_data, intermediate_file, fieldnames)

        # Clear datasets from memory
        del dataset, prompts, ground_truths, intermediate_data
        clear_memory()
        log_memory_usage("after compression phase")

        # Phase 3: Evaluation Pipeline
        print("\n🤖 Phase 3: Evaluation Pipeline")

        # Load LLM once
        print("Loading LLM for evaluation...")
        target_llm = LLMFactory.create(provider=DEFAULT_LLM_PROVIDER)
        evaluator = Evaluator(task=task_name, llm=target_llm)
        log_memory_usage("after LLM load")

        # Initialize thread-safe logger
        base_logger = BenchmarkLogger(log_dir="results", task_name=task_name, compression_methods=COMPRESSION_METHODS_TO_RUN)
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
                    log_memory_usage("during evaluation")

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
                sample_result = {
                    "sample_id": sample_id,
                    "task": row['task'],
                    "llm_provider": row['llm_provider'],
                    "llm_model": row['llm_model'],
                    "original_prompt": row['original_prompt'],
                    "ground_truth_answer": row['ground_truth'],
                    "compression_methods": [],
                    "target_compression_ratio": 1 - DEFAULT_TARGET_RATIO,
                    "original_prompt_output": baseline_metrics['llm_response'],
                    "baseline_score": baseline_metrics['score'],
                    "baseline_latency": baseline_metrics['latency'],
                    "baseline_evaluated": True
                }

                # Add task-specific baseline data
                if task_name == "reasoning":
                    baseline_answer = extract_gsm8k_answer(baseline_metrics['llm_response'])
                    sample_result["baseline_extracted_answer"] = baseline_answer
                elif task_name == "classification":
                    # For classification, the extracted_answer is already provided by the evaluator
                    sample_result["baseline_extracted_answer"] = baseline_metrics.get('extracted_answer')

                # Evaluate each compressed prompt
                for compression_method in COMPRESSION_METHODS_TO_RUN:
                    compressed_prompt = row.get(f"{compression_method}_compressed_prompt", "")
                    if compressed_prompt:
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
                            compression_data["answers_match"] = (baseline_answer == compressed_answer) and (baseline_answer is not None)
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
                evaluated_count += 1

                if evaluated_count % 3 == 0:  # More frequent progress updates
                    print(f"Evaluated {evaluated_count}/{NUM_SAMPLES_TO_RUN} samples")

        # Cleanup
        del target_llm
        clear_memory()
        log_memory_usage("after evaluation phase")

        # Phase 4: Finalize and save results
        print("\n📊 Phase 4: Finalizing Results")
        base_logger.finalize_and_save()
        summary = base_logger.generate_summary_report()
        analysis_file = base_logger.export_analysis_report()

        print(f"✅ {task_name.upper()} optimized benchmark completed!")
        print(f"   📊 Results saved to: {base_logger.log_dir}")
        print(f"   📈 Analysis report: {analysis_file}")

        return evaluated_count

def extract_task_data(task_name: str, dataset):
    """Extract prompts and ground truth based on task type."""
    if task_name == "reasoning":
        prompts = [sample['question'] for sample in dataset]
        ground_truths = [sample['answer'] for sample in dataset]
    elif task_name == "summarization":
        # For summarization, handle very long articles with better truncation
        prompts = []
        for sample in dataset:
            article = sample['article']
            words = article.split()
            
            # More generous word limit for realistic benchmarking
            if len(words) > 1200:
                # Smart truncation: keep beginning and end, truncate middle
                keep_start = 600
                keep_end = 400
                truncated_words = words[:keep_start] + ['...'] + words[-keep_end:]
                truncated_article = ' '.join(truncated_words)
                print(f"⚠️  Smart truncated article from {len(words)} to ~1000 words for summarization")
            else:
                truncated_article = article
            prompts.append(f"Summarize the following article:\n\n{truncated_article}")
        ground_truths = [sample['highlights'] for sample in dataset]
    elif task_name == "classification":
        prompts = [f"Analyze the sentiment of this movie review and respond with ONLY 'positive' or 'negative' (one word answer):\n\n{sample['text']}" for sample in dataset]
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

def run_optimized_multi_task_benchmark(tasks_to_run=None):
    """Ultra-optimized benchmark: Load each compressor once, compress everything, then load LLM once."""
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
        print("\n📥 Phase 1: Loading ALL datasets...")

        # Load ALL datasets at once
        all_datasets = {}
        all_prompts = {}
        all_ground_truths = {}

        for task_name in tasks_to_run:
            print(f"   Loading {task_name} dataset...")
            task_config = TASK_CONFIGURATIONS[task_name]
            dataset_name = task_config["dataset"]
            dataset_config = task_config["config"]

            dataset = load_benchmark_dataset(task_name, dataset_name, dataset_config, NUM_SAMPLES_TO_RUN)
            if not dataset:
                print(f"❌ Failed to load dataset for task {task_name}")
                continue

            # Extract prompts and ground truth
            prompts, ground_truths = extract_task_data(task_name, dataset)

            all_datasets[task_name] = dataset
            all_prompts[task_name] = prompts
            all_ground_truths[task_name] = ground_truths

            print(f"   ✅ {task_name}: {len(prompts)} samples loaded")

        log_memory_usage("after all datasets loaded")

        # Prepare master intermediate data structure
        master_data = {}
        samples_cache_status = {}
        
        for task_name in tasks_to_run:
            if task_name not in all_prompts:
                continue

            # Check if samples are cached
            cached_samples = load_samples_from_cache(task_name, len(all_prompts[task_name]))
            if cached_samples:
                # Use cached samples
                master_data[task_name] = cached_samples
                samples_cache_status[task_name] = "loaded"
                print(f"   ✅ {task_name}: {len(cached_samples)} samples loaded from cache")
            else:
                # Prepare new samples
                master_data[task_name] = []
                for i, (prompt, ground_truth) in enumerate(zip(all_prompts[task_name], all_ground_truths[task_name])):
                    master_data[task_name].append({
                        'sample_id': i + 1,
                        'task': task_name,
                        'original_prompt': prompt,
                        'ground_truth': ground_truth,
                        'llm_provider': DEFAULT_LLM_PROVIDER,
                        'llm_model': get_model_name(DEFAULT_LLM_PROVIDER)
                    })
                samples_cache_status[task_name] = "new"
                print(f"   ✅ {task_name}: {len(master_data[task_name])} samples prepared")

        # Save new samples to cache
        for task_name in tasks_to_run:
            if task_name in samples_cache_status and samples_cache_status[task_name] == "new":
                save_samples_to_cache(task_name, master_data[task_name], len(master_data[task_name]))

        # Save original prompts to intermediate files
        intermediate_files = {}
        for task_name in tasks_to_run:
            if task_name not in master_data:
                continue

            intermediate_file = os.path.join(temp_dir, f"intermediate_{task_name}.csv")
            intermediate_files[task_name] = intermediate_file

            fieldnames = ['sample_id', 'task', 'original_prompt', 'ground_truth',
                         'llm_provider', 'llm_model'] + \
                        [f"{method}_compressed_prompt" for method in COMPRESSION_METHODS_TO_RUN]

            write_intermediate_csv(master_data[task_name], intermediate_file, fieldnames)
            print(f"   💾 {task_name} original prompts saved to intermediate file")

        # Clear datasets from memory (keep prompts and ground truths)
        del all_datasets
        clear_memory()
        log_memory_usage("after dataset cleanup")

        print("\n🗜️  Phase 2: COMPRESSION PIPELINE (Load each compressor ONCE)")

        # Check cache status for all compression methods
        compression_cache_status = {}
        for compression_method in COMPRESSION_METHODS_TO_RUN:
            compression_cache_status[compression_method] = {}
            for task_name in tasks_to_run:
                if task_name in master_data:
                    num_samples = len(master_data[task_name])
                    is_cached = check_cache_status(task_name, compression_method, num_samples, DEFAULT_TARGET_RATIO)
                    compression_cache_status[compression_method][task_name] = is_cached
                    status = "✅ CACHED" if is_cached else "🔄 TO COMPRESS"
                    print(f"   {status} {task_name} ({num_samples} samples) - {compression_method}")

        # For each compression method, load compressor ONCE and compress ALL prompts for ALL tasks
        for compression_method in COMPRESSION_METHODS_TO_RUN:
            print(f"\n{'─'*60}")
            print(f"🔧 Processing {compression_method.upper()} for ALL TASKS")
            print(f"{'─'*60}")

            total_compressed = 0
            tasks_to_compress = []

            # Check which tasks need compression for this method
            for task_name in tasks_to_run:
                if task_name not in master_data:
                    continue
                
                if not compression_cache_status[compression_method][task_name]:
                    tasks_to_compress.append(task_name)
                else:
                    # Load cached compressed prompts
                    num_samples = len(master_data[task_name])
                    cached_compressed = load_compressed_from_cache(task_name, compression_method, num_samples, DEFAULT_TARGET_RATIO)
                    if cached_compressed:
                        # Update master data with cached compressed prompts
                        for i, compressed_prompt in enumerate(cached_compressed):
                            master_data[task_name][i][f"{compression_method}_compressed_prompt"] = compressed_prompt
                        total_compressed += len(cached_compressed)
                        print(f"   ✅ {task_name}: {len(cached_compressed)} prompts loaded from cache")

            # Only load compressor if we have tasks to compress
            if tasks_to_compress:
                print(f"� Loading {compression_method} compressor...")
                compressor = CompressorFactory.create(compression_method)
                log_memory_usage(f"after {compression_method} load")

                # Compress prompts for tasks that need it
                for task_name in tasks_to_compress:
                    if task_name not in all_prompts:
                        continue

                    print(f"   🗜️  Compressing {len(all_prompts[task_name])} {task_name} prompts...")

                    compressed_prompts = []
                    for i, prompt in enumerate(all_prompts[task_name]):
                        if (i + 1) % 20 == 0:  # Less frequent updates since we're doing more work
                            print(f"      Compressed {i + 1}/{len(all_prompts[task_name])} {task_name} prompts")

                        compressed_prompt = compressor.compress(prompt, DEFAULT_TARGET_RATIO)
                        compressed_prompts.append(compressed_prompt)

                    # Update master data
                    for i, compressed_prompt in enumerate(compressed_prompts):
                        master_data[task_name][i][f"{compression_method}_compressed_prompt"] = compressed_prompt

                    # Save to cache
                    save_compressed_to_cache(task_name, compression_method, compressed_prompts, 
                                           len(compressed_prompts), DEFAULT_TARGET_RATIO)

                    total_compressed += len(compressed_prompts)
                    print(f"   ✅ {task_name}: {len(compressed_prompts)} prompts compressed and cached")

                # Unload compressor
                del compressor
                clear_memory()
                log_memory_usage(f"after {compression_method} unload")
            else:
                print(f"   🎯 All tasks already cached for {compression_method}")

            print(f"✅ {compression_method.upper()} compression completed for ALL tasks ({total_compressed} total prompts)")

        # Clear prompts and ground truths from memory (keep master_data)
        del all_prompts, all_ground_truths
        clear_memory()
        log_memory_usage("after compression phase")

        print("\n🤖 Phase 3: EVALUATION PIPELINE (Load LLM ONCE)")

        # Load LLM ONCE for ALL evaluations
        print("📦 Loading LLM for evaluation...")
        target_llm = LLMFactory.create(provider=DEFAULT_LLM_PROVIDER)
        evaluator = Evaluator(task="multi", llm=target_llm)  # Multi-task evaluator
        log_memory_usage("after LLM load")

        # Initialize loggers for each task
        loggers = {}
        thread_safe_loggers = {}

        for task_name in tasks_to_run:
            if task_name not in master_data:
                continue

            base_logger = BenchmarkLogger(log_dir="results", task_name=task_name, compression_methods=COMPRESSION_METHODS_TO_RUN)
            loggers[task_name] = base_logger
            thread_safe_loggers[task_name] = ThreadSafeLogger(base_logger)

        # Evaluate ALL prompts for ALL tasks
        total_evaluated = 0

        for task_name in tasks_to_run:
            if task_name not in master_data:
                continue

            print(f"\n{'─'*60}")
            print(f"� Evaluating {task_name.upper()} ({len(master_data[task_name])} samples)")
            print(f"{'─'*60}")

            intermediate_file = intermediate_files[task_name]
            evaluated_count = 0

            # Iterate over master_data directly instead of reading from temporary CSV
            for sample_data in master_data[task_name]:
                sample_id = sample_data['sample_id']

                # Create a row-like dictionary from sample_data for compatibility
                row = {
                    'sample_id': str(sample_id),
                    'task': task_name,
                    'llm_provider': DEFAULT_LLM_PROVIDER,
                    'llm_model': get_model_name(DEFAULT_LLM_PROVIDER),
                    'original_prompt': sample_data['original_prompt'],
                    'ground_truth': sample_data['ground_truth']
                }

                # Add compressed prompts to the row
                for compression_method in COMPRESSION_METHODS_TO_RUN:
                    compressed_key = f"{compression_method}_compressed_prompt"
                    if compressed_key in sample_data:
                        row[compressed_key] = sample_data[compressed_key]
                    else:
                        row[compressed_key] = ""

                sample_id = int(row['sample_id'])

                # Evaluate original prompt (baseline)
                if evaluated_count == 0:  # Only log memory for first sample per task
                    log_memory_usage(f"during {task_name} evaluation")

                try:
                    baseline_metrics = evaluator.evaluate(row['original_prompt'], row['ground_truth'])
                except Exception as e:
                    print(f"❌ Baseline evaluation failed for {task_name} sample {sample_id}: {e}")
                    baseline_metrics = {
                        'score': 0.0,
                        'latency': 60.0,
                        'llm_response': f"Error: Baseline evaluation failed - {e}",
                        'extracted_answer': None
                    }

                # Prepare sample result
                sample_result = {
                    "sample_id": sample_id,
                    "task": row['task'],
                    "llm_provider": row['llm_provider'],
                    "llm_model": row['llm_model'],
                    "original_prompt": row['original_prompt'],
                    "ground_truth_answer": row['ground_truth'],
                    "compression_methods": [],
                    "target_compression_ratio": 1 - DEFAULT_TARGET_RATIO,
                    "original_prompt_output": baseline_metrics['llm_response'],
                    "baseline_score": baseline_metrics['score'],
                    "baseline_latency": baseline_metrics['latency'],
                    "baseline_evaluated": True
                }

                # Add task-specific baseline data
                if task_name == "reasoning":
                    baseline_answer = extract_gsm8k_answer(baseline_metrics['llm_response'])
                    sample_result["baseline_extracted_answer"] = baseline_answer
                elif task_name == "classification":
                    # For classification, the extracted_answer is already provided by the evaluator
                    sample_result["baseline_extracted_answer"] = baseline_metrics.get('extracted_answer')

                # Evaluate each compressed prompt
                for compression_method in COMPRESSION_METHODS_TO_RUN:
                    compressed_prompt = row.get(f"{compression_method}_compressed_prompt", "")
                    if compressed_prompt:
                        try:
                            compressed_metrics = evaluator.evaluate(compressed_prompt, row['ground_truth'])
                        except Exception as e:
                            print(f"❌ Compressed evaluation failed for {task_name} sample {sample_id}, method {compression_method}: {e}")
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
                            compression_data["answers_match"] = (baseline_answer == compressed_answer) and (baseline_answer is not None)
                        elif task_name == "classification":
                            # For classification, the extracted_answer is already provided by the evaluator
                            compression_data["compressed_extracted_answer"] = compressed_metrics.get('extracted_answer')
                            baseline_extracted = baseline_metrics.get('extracted_answer')
                            compressed_extracted = compressed_metrics.get('extracted_answer')
                            compression_data["answers_match"] = (baseline_extracted == compressed_extracted) and (baseline_extracted is not None)

                        sample_result["compression_methods"].append(compression_data)
                    else:
                        print(f"⚠️  No compressed prompt found for {task_name} sample {sample_id}, method {compression_method}")

                # Log result using thread-safe logger
                thread_safe_loggers[task_name].log_result(sample_result)
                evaluated_count += 1
                total_evaluated += 1

                if evaluated_count % 5 == 0:  # Progress updates
                    print(f"   Evaluated {evaluated_count}/{len(master_data[task_name])} {task_name} samples")

            print(f"✅ {task_name.upper()}: {evaluated_count} samples evaluated")

        # Cleanup
        del target_llm
        clear_memory()
        log_memory_usage("after evaluation phase")

        # Phase 4: Finalize and save results for ALL tasks
        print("\n📊 Phase 4: Finalizing Results for ALL Tasks")

        all_task_results = {}
        for task_name in tasks_to_run:
            if task_name not in loggers:
                continue

            print(f"   📊 Finalizing {task_name.upper()} results...")
            loggers[task_name].finalize_and_save()
            summary = loggers[task_name].generate_summary_report()
            analysis_file = loggers[task_name].export_analysis_report()

            all_task_results[task_name] = len(master_data[task_name])

            print(f"   ✅ {task_name.upper()} results saved")
            print(f"      📁 Results: {loggers[task_name].log_dir}")
            print(f"      📈 Analysis: {analysis_file}")

        # Print summary across all tasks
        print(f"\n{'='*80}")
        print("🎉 ULTRA-OPTIMIZED MULTI-TASK BENCHMARK SUMMARY")
        print(f"{'='*80}")
        print(f"📊 Total samples processed: {total_evaluated}")
        print(f"🗜️  Compression methods: {len(COMPRESSION_METHODS_TO_RUN)}")
        print(f"📋 Tasks completed: {len(all_task_results)}")

        for task, results in all_task_results.items():
            print(f"\n📊 {task.upper()}:")
            print(f"   Samples: {results}")
            print(f"   Methods tested: {len(COMPRESSION_METHODS_TO_RUN)}")
            print(f"   Results: results/{task}_*")

        print("\n✅ Ultra-optimized benchmark completed!")
        print("🚀 Efficiency improvements:")
        print("   • Each compressor loaded only ONCE")
        print("   • LLM loaded only ONCE")
        print("   • All compressed prompts pre-computed and saved")
        print("   • No redundant loading/unloading cycles")

        return all_task_results

def run_multi_task_benchmark(tasks_to_run=None):
    """Legacy multi-task benchmark (loads/unloads frequently)."""
    if tasks_to_run is None:
        tasks_to_run = SUPPORTED_TASKS

    print("🚀 Starting LEGACY Multi-Task Prompt Compression Benchmark")
    print(f"📋 Tasks to run: {', '.join(tasks_to_run)}")
    print(f"🗜️  Compression methods: {', '.join(COMPRESSION_METHODS_TO_RUN)}")
    print(f"🤖 LLM Provider: {DEFAULT_LLM_PROVIDER}")
    print("⚠️  Note: This loads/unloads compressors and LLM frequently")

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
    print("🎉 LEGACY MULTI-TASK BENCHMARK SUMMARY")
    print(f"{'='*80}")

    for task, results in all_task_results.items():
        if results:
            print(f"\n📊 {task.upper()} Results:")
            print(f"   Samples processed: {results}")
            print(f"   Methods tested: {len(COMPRESSION_METHODS_TO_RUN)}")

    print("\n✅ Legacy benchmark completed!")
    print(f"📁 Check the results/ directory for detailed reports")

def run_benchmark():
    """Legacy single-task benchmark for backward compatibility."""
    run_multi_task_benchmark([DEFAULT_TASK])

if __name__ == "__main__":
    import sys
    
    # Handle command line arguments for cache management
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "cache-info":
            show_cache_info()
        elif command == "clear-cache":
            if len(sys.argv) > 2:
                task = sys.argv[2]
                if len(sys.argv) > 3:
                    method = sys.argv[3]
                    clear_compression_cache(task, method)
                else:
                    clear_compression_cache(task)
            else:
                clear_compression_cache()
        elif command == "rate-limit-info":
            print("📊 OpenRouter Rate Limit Information:")
            print("=" * 50)
            print("Free Tier Limits:")
            print("  • 16 requests per minute")
            print("  • No daily limits mentioned")
            print("")
            print("Current Configuration:")
            from config import OPENROUTER_RATE_LIMIT_RPM
            print(f"  • Configured limit: {OPENROUTER_RATE_LIMIT_RPM} RPM")
            print("")
            print("Tips to avoid rate limits:")
            print("  • Reduce NUM_SAMPLES_TO_RUN in config.py")
            print("  • Add delays between benchmark runs")
            print("  • Consider upgrading to a paid plan")
            print("  • Use a different provider (OpenAI, HuggingFace, etc.)")
            print("")
            print("Alternative Models:")
            print("  • deepseek/deepseek-chat:free (may have different limits)")
            print("  • microsoft/wizardlm-2-8x22b:free")
            print("  • meta-llama/llama-3.1-8b-instruct:free")
        elif command == "help":
            print("PCM-LLM Benchmark Commands:")
            print("  python main.py                    # Run benchmark")
            print("  python main.py cache-info         # Show cache information")
            print("  python main.py clear-cache        # Clear all cache")
            print("  python main.py clear-cache <task> # Clear cache for specific task")
            print("  python main.py clear-cache <task> <method>  # Clear specific cache")
            print("  python main.py rate-limit-info    # Show rate limit information")
            print("  python main.py help               # Show this help message")
        else:
            print(f"Unknown command: {command}")
            print("Use 'python main.py help' for available commands")
    else:
        # Run ultra-optimized benchmark by default
        run_optimized_multi_task_benchmark()
