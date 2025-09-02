"""
Data loading pipeline component for the benchmark service.
"""
from typing import List, Dict, Any

from core.config import TaskConfig
from data_loaders.loaders import load_benchmark_dataset
from utils.cache.cache_utils import load_samples_from_cache, save_samples_to_cache
from utils.loggers.run_info_logger import RunInfoLogger
from utils.system.system_utils import log_memory_usage


class DataLoaderPipeline:
    """Handles the data loading and caching phase of the benchmark."""

    def __init__(self, tasks: List[str], num_samples: int, run_info_logger: RunInfoLogger):
        self.tasks = tasks
        self.num_samples = num_samples
        self.run_info_logger = run_info_logger

    def run(self) -> Dict[str, Dict[str, Any]]:
        """
        Loads all datasets for the specified tasks.

        Returns:
            A dictionary containing the loaded data for each task.
        """
        print("\n📥 PHASE 1: Loading ALL Sample Data")
        all_samples_data = {}
        for task_name in self.tasks:
            # Load samples as a list of dictionaries, each containing the full sample info
            samples = self._load_and_cache_samples(task_name, self.num_samples)
            all_samples_data[task_name] = samples
            # Add a print statement to verify the output of the data loader
            if samples:
                print(f"DEBUG: DataLoader output for '{task_name}' sample 0: {str(samples[0])[:200]}...")

        log_memory_usage("after loading all datasets", self.run_info_logger)
        return all_samples_data

    def _load_and_cache_samples(self, task_name: str, num_samples: int) -> List[Dict[str, Any]]:
        """
        Loads a single dataset, from cache if available, and returns a list of sample dictionaries.
        This method now has a consistent return type.
        """
        cached_samples = load_samples_from_cache(task_name, num_samples)
        if cached_samples:
            print(f"📖 Samples loaded from cache for '{task_name}': {len(cached_samples)} samples")
            return cached_samples

        # If not cached, load from source
        from core.config import settings
        task_config = settings.get_task_config(task_name)
        dataset = load_benchmark_dataset(task_name, task_config.dataset, task_config.config, num_samples)
        
        from utils.data.data_utils import extract_task_data
        prompts, ground_truths = extract_task_data(task_name, dataset)

        # Structure the data into the standard list of dictionaries format
        samples_to_cache = []
        for i, (prompt, ground_truth) in enumerate(zip(prompts, ground_truths)):
            samples_to_cache.append({
                'sample_id': f"{task_name}-{i}",  # Ensure unique sample_id across tasks
                'task': task_name,
                'original_prompt': prompt,
                'ground_truth': ground_truth
            })
            
        # Cache the newly loaded samples and return them
        save_samples_to_cache(task_name, samples_to_cache, self.num_samples)
        print(f"💾 Samples for '{task_name}' cached: {len(samples_to_cache)} samples")
        
        return samples_to_cache
