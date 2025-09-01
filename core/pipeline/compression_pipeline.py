"""
Compression pipeline component for the benchmark service.
"""
from typing import List, Dict, Any

from compressors.factory import CompressorFactory
from utils.cache_utils import check_cache_status, load_compressed_from_cache, save_compressed_to_cache
from utils.system_utils import clear_memory


class CompressionPipeline:
    """Handles the optimized compression phase of the benchmark."""

    def __init__(self, tasks: List[str], compression_methods: List[str], target_ratio: float, all_samples_data: Dict[str, Any]):
        self.tasks = tasks
        self.compression_methods = compression_methods
        self.target_ratio = target_ratio
        self.all_samples_data = all_samples_data

    def run(self) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Runs the compression pipeline for all tasks and methods.

        Returns:
            A tuple containing compressed data and metadata.
        """
        print("\n🗜️  PHASE 2: Compression Pipeline (ALL tasks)")
        all_compressed_data = {}
        all_compression_metadata = {}

        for compression_method in self.compression_methods:
            print(f"   🔧 Processing {compression_method} for ALL tasks...")
            all_compressed_data[compression_method] = {}
            all_compression_metadata[compression_method] = {}

            compressor = CompressorFactory.create(compression_method)

            for task_name in self.tasks:
                task_data = self.all_samples_data[task_name]
                prompts = task_data['prompts']
                num_samples = task_data['num_samples']

                if check_cache_status(task_name, compression_method, num_samples, self.target_ratio):
                    print(f"   🎯 CACHED {task_name} ({num_samples} samples) - {compression_method}")
                    compressed_prompts, metadata = load_compressed_from_cache(
                        task_name, compression_method, num_samples, self.target_ratio
                    )
                else:
                    compressed_prompts = []
                    actual_ratios = []
                    for original_prompt in prompts:
                        compressed_prompt = compressor.compress(original_prompt, self.target_ratio)
                        compressed_prompts.append(compressed_prompt)

                        original_tokens = len(original_prompt.split())
                        compressed_tokens = len(compressed_prompt.split())
                        ratio = original_tokens / compressed_tokens if compressed_tokens > 0 else 1.0
                        actual_ratios.append(ratio)

                    save_compressed_to_cache(
                        task_name, compression_method, compressed_prompts,
                        num_samples, self.target_ratio, actual_ratios
                    )
                    metadata = {
                        "average_actual_ratio": sum(actual_ratios) / len(actual_ratios) if actual_ratios else 0,
                        "actual_ratios": actual_ratios
                    }
                
                all_compressed_data[compression_method][task_name] = compressed_prompts
                all_compression_metadata[compression_method][task_name] = metadata

            del compressor
            clear_memory()
            print(f"   ✅ {compression_method} completed for ALL tasks and unloaded")

        return all_compressed_data, all_compression_metadata
