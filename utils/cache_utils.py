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
                           num_samples: int, target_ratio: float):
    """Save compressed prompts to cache."""
    cache_path = get_compressed_cache_path(task_name, compression_method, num_samples, target_ratio)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    try:
        with open(cache_path, 'w') as f:
            json.dump(compressed_prompts, f, indent=2)
        print(f"💾 Compressed prompts cached: {cache_path}")
    except Exception as e:
        print(f"⚠️  Failed to cache compressed prompts: {e}")


def load_compressed_from_cache(task_name: str, compression_method: str, num_samples: int, target_ratio: float) -> list:
    """Load compressed prompts from cache."""
    cache_path = get_compressed_cache_path(task_name, compression_method, num_samples, target_ratio)

    if not os.path.exists(cache_path):
        return []

    try:
        with open(cache_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Failed to load compressed prompts from cache: {e}")
        return []


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

    print("=" * 50)
