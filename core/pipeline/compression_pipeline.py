"""
Compression pipeline component for the benchmark service.
"""

from typing import Any, Dict, List

from compressors.factory import CompressorFactory
from utils.cache.cache_utils import (
    check_cache_status,
    load_compressed_from_cache,
    save_compressed_to_cache,
)
from utils.system.system_utils import clear_memory


class CompressionPipeline:
    """Handles the optimized compression phase of the benchmark."""

    def __init__(
        self,
        tasks: List[str],
        compression_methods: List[str],
        target_ratio: float,
        all_samples_data: Dict[str, Any],
    ):
        self.tasks = tasks
        self.compression_methods = compression_methods
        self.target_ratio = target_ratio
        self.all_samples_data = all_samples_data

    def run(self) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Runs the compression pipeline for all tasks and methods.
        The output format is changed to be a list of dicts, each including the sample_id.
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
                samples = self.all_samples_data[task_name]
                num_samples = len(samples)

                if check_cache_status(
                    task_name, compression_method, num_samples, self.target_ratio
                ):
                    print(
                        f"   🎯 CACHED {task_name} ({num_samples} samples) - {compression_method}"
                    )
                    # load_compressed_from_cache now returns a list of dicts
                    compressed_results, metadata = load_compressed_from_cache(
                        task_name, compression_method, num_samples, self.target_ratio
                    )
                else:
                    compressed_results = []
                    actual_ratios = []
                    for sample in samples:
                        original_prompt = sample["original_prompt"]
                        compressed_prompt = compressor.compress(
                            original_prompt, self.target_ratio
                        )

                        original_tokens = len(original_prompt.split())
                        compressed_tokens = len(compressed_prompt.split())
                        ratio = (
                            original_tokens / compressed_tokens
                            if compressed_tokens > 0
                            else 1.0
                        )
                        actual_ratios.append(ratio)

                        # Preserve sample_id and other info
                        compressed_results.append(
                            {
                                "sample_id": sample["sample_id"],
                                "compressed_prompt": compressed_prompt,
                            }
                        )

                    # Ensure actual_ratios are stored in a way that can be looked up by sample_id
                    ratios_by_id = {
                        sample["sample_id"]: ratio
                        for sample, ratio in zip(samples, actual_ratios)
                    }

                    save_compressed_to_cache(
                        task_name,
                        compression_method,
                        compressed_results,
                        num_samples,
                        self.target_ratio,
                        ratios_by_id,
                    )
                    metadata = {
                        "average_actual_ratio": (
                            sum(actual_ratios) / len(actual_ratios)
                            if actual_ratios
                            else 0
                        ),
                        "actual_ratios": ratios_by_id,
                    }

                all_compressed_data[compression_method][task_name] = compressed_results
                all_compression_metadata[compression_method][task_name] = metadata

                # Add a print statement to verify the output of the compression pipeline
                if compressed_results:
                    print(
                        f"DEBUG: Compression output for '{task_name}', method '{compression_method}', sample 0: {str(compressed_results[0])[:200]}..."
                    )

            del compressor
            clear_memory()
            print(f"   ✅ {compression_method} completed for ALL tasks and unloaded")

        return all_compressed_data, all_compression_metadata
