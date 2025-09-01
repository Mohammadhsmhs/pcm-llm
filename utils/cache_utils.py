"""
Cache management utilities for dataset and compression caching.
"""

import os
import json
import hashlib
from typing import List, Dict, Any
from config import NUM_SAMPLES_TO_RUN, DEFAULT_TARGET_RATIO


def get_cache_key(task_name: str, compression_method: str, num_samples: int, target_ratio: float) -> str:
    """Generate a unique cache key for compression results."""
    key_data = f"{task_name}_{compression_method}_{num_samples}_{target_ratio}"
    return hashlib.md5(key_data.encode()).hexdigest()


def get_samples_cache_path(task_name: str, num_samples: int) -> str:
    """Get the cache file path for dataset samples."""
    return f"compressed_cache/samples/{task_name}_{num_samples}_samples.json"


def get_compressed_cache_path(task_name: str, compression_method: str, num_samples: int, target_ratio: float) -> str:
    """Get the cache file path for compressed prompts."""
    cache_key = get_cache_key(task_name, compression_method, num_samples, target_ratio)
    return f"compressed_cache/compressed/{task_name}_{compression_method}_{cache_key}.json"


def save_samples_to_cache(task_name: str, samples_data: list, num_samples: int):
    """Save dataset samples to cache."""
    cache_path = get_samples_cache_path(task_name, num_samples)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    try:
        with open(cache_path, 'w') as f:
            json.dump(samples_data, f, indent=2)
        print(f"💾 Samples cached: {cache_path}")
    except Exception as e:
        print(f"⚠️  Failed to cache samples: {e}")


def load_samples_from_cache(task_name: str, num_samples: int) -> list:
    """Load dataset samples from cache."""
    cache_path = get_samples_cache_path(task_name, num_samples)

    if not os.path.exists(cache_path):
        return []

    try:
        with open(cache_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Failed to load samples from cache: {e}")
        return []


def save_compressed_to_cache(task_name: str, compression_method: str, compressed_prompts: list,
                           num_samples: int, target_ratio: float, actual_ratios: list = None):
    """Save compressed prompts to cache with metadata."""
    cache_path = get_compressed_cache_path(task_name, compression_method, num_samples, target_ratio)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # Calculate actual compression ratios if not provided
    if actual_ratios is None:
        actual_ratios = []
        for prompt in compressed_prompts:
            if isinstance(prompt, dict) and 'original_length' in prompt and 'compressed_length' in prompt:
                if prompt['compressed_length'] > 0:
                    ratio = prompt['original_length'] / prompt['compressed_length']
                    actual_ratios.append(ratio)
                else:
                    actual_ratios.append(1.0)
            else:
                actual_ratios.append(target_ratio)  # fallback

    # Create cache data with metadata
    cache_data = {
        "metadata": {
            "task_name": task_name,
            "compression_method": compression_method,
            "num_samples": num_samples,
            "target_ratio": target_ratio,
            "average_actual_ratio": sum(actual_ratios) / len(actual_ratios) if actual_ratios else target_ratio,
            "actual_ratios": actual_ratios,
            "timestamp": str(__import__('datetime').datetime.now())
        },
        "compressed_prompts": compressed_prompts
    }

    try:
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f, indent=2)
        avg_ratio = cache_data["metadata"]["average_actual_ratio"]
        print(f"💾 Compressed prompts cached: {cache_path} (avg ratio: {avg_ratio:.2f})")
    except Exception as e:
        print(f"⚠️  Failed to cache compressed prompts: {e}")


def load_compressed_from_cache(task_name: str, compression_method: str, num_samples: int, target_ratio: float) -> tuple:
    """Load compressed prompts from cache with metadata."""
    cache_path = get_compressed_cache_path(task_name, compression_method, num_samples, target_ratio)

    if not os.path.exists(cache_path):
        return [], {}

    try:
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
        
        # Handle both old format (just prompts) and new format (with metadata)
        if isinstance(cache_data, dict) and "compressed_prompts" in cache_data:
            prompts = cache_data["compressed_prompts"]
            metadata = cache_data.get("metadata", {})
        else:
            # Old format - just the prompts list
            prompts = cache_data
            metadata = {}
        
        return prompts, metadata
    except Exception as e:
        print(f"⚠️  Failed to load compressed prompts from cache: {e}")
        return [], {}


def check_cache_status(task_name: str, compression_method: str, num_samples: int, target_ratio: float) -> bool:
    """Check if compressed prompts are cached."""
    cache_path = get_compressed_cache_path(task_name, compression_method, num_samples, target_ratio)
    return os.path.exists(cache_path)


def clear_compression_cache(task_name: str = None, compression_method: str = None):
    """Clear compression cache files."""
    cache_dir = "compressed_cache/compressed"

    if not os.path.exists(cache_dir):
        print("No cache directory found.")
        return

    deleted_count = 0
    for filename in os.listdir(cache_dir):
        if filename.endswith('.json'):
            # Parse filename to check filters
            parts = filename.replace('.json', '').split('_')
            if len(parts) >= 3:
                file_task = parts[0]
                file_method = parts[1]

                # Apply filters
                if task_name and file_task != task_name:
                    continue
                if compression_method and file_method != compression_method:
                    continue

                # Delete file
                file_path = os.path.join(cache_dir, filename)
                try:
                    os.remove(file_path)
                    deleted_count += 1
                    print(f"🗑️  Deleted: {filename}")
                except Exception as e:
                    print(f"⚠️  Failed to delete {filename}: {e}")

    print(f"✅ Cleared {deleted_count} cache files")


def show_cache_info():
    """Display information about cached data."""
    print("📊 Cache Information:")
    print("=" * 50)

    # Samples cache
    samples_dir = "compressed_cache/samples"
    if os.path.exists(samples_dir):
        samples_files = [f for f in os.listdir(samples_dir) if f.endswith('.json')]
        print(f"Samples cache: {len(samples_files)} files")
        for file in samples_files[:5]:  # Show first 5
            print(f"  📄 {file}")
        if len(samples_files) > 5:
            print(f"  ... and {len(samples_files) - 5} more")
    else:
        print("Samples cache: No cache directory")

    # Compressed cache
    compressed_dir = "compressed_cache/compressed"
    if os.path.exists(compressed_dir):
        compressed_files = [f for f in os.listdir(compressed_dir) if f.endswith('.json')]
        print(f"Compressed cache: {len(compressed_files)} files")

        # Group by task and method
        task_methods = {}
        for file in compressed_files:
            parts = file.replace('.json', '').split('_')
            if len(parts) >= 2:
                task = parts[0]
                method = parts[1]
                key = f"{task}_{method}"
                task_methods[key] = task_methods.get(key, 0) + 1

        for key, count in sorted(task_methods.items()):
            print(f"  📂 {key}: {count} files")
    else:
        print("Compressed cache: No cache directory")

    # Baseline cache
    baseline_dir = "compressed_cache/baseline"
    if os.path.exists(baseline_dir):
        baseline_files = [f for f in os.listdir(baseline_dir) if f.endswith('.json')]
        print(f"Baseline cache: {len(baseline_files)} files")
        for file in baseline_files[:3]:  # Show first 3
            print(f"  📋 {file}")
        if len(baseline_files) > 3:
            print(f"  ... and {len(baseline_files) - 3} more")
    else:
        print("Baseline cache: No cache directory")

    print("=" * 50)


def get_baseline_cache_key(task_name: str, llm_provider: str, llm_model: str, num_samples: int) -> str:
    """Generate a unique cache key for baseline LLM outputs."""
    key_data = f"{task_name}_{llm_provider}_{llm_model}_{num_samples}"
    return hashlib.md5(key_data.encode()).hexdigest()


def get_baseline_cache_path(task_name: str, llm_provider: str, llm_model: str, num_samples: int) -> str:
    """Get the cache file path for baseline LLM outputs."""
    cache_key = get_baseline_cache_key(task_name, llm_provider, llm_model, num_samples)
    return f"compressed_cache/baseline/{task_name}_{llm_provider}_{cache_key}.json"


def save_baseline_to_cache(task_name: str, llm_provider: str, llm_model: str, num_samples: int, baseline_data: list):
    """Save baseline LLM outputs to cache."""
    cache_path = get_baseline_cache_path(task_name, llm_provider, llm_model, num_samples)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    try:
        with open(cache_path, 'w') as f:
            json.dump(baseline_data, f, indent=2)
        print(f"💾 Baseline outputs cached: {cache_path}")
    except Exception as e:
        print(f"⚠️  Failed to cache baseline outputs: {e}")


def load_baseline_from_cache(task_name: str, llm_provider: str, llm_model: str, num_samples: int) -> list:
    """Load baseline LLM outputs from cache."""
    cache_path = get_baseline_cache_path(task_name, llm_provider, llm_model, num_samples)

    if not os.path.exists(cache_path):
        return []

    try:
        with open(cache_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Failed to load baseline outputs from cache: {e}")
        return []


def check_baseline_cache_status(task_name: str, llm_provider: str, llm_model: str, num_samples: int) -> bool:
    """Check if baseline LLM outputs are cached."""
    cache_path = get_baseline_cache_path(task_name, llm_provider, llm_model, num_samples)
    return os.path.exists(cache_path)
