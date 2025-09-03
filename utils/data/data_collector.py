from collections import defaultdict
from datetime import datetime


class DataCollector:
    """
    Responsible for collecting and storing benchmark data.
    Single Responsibility: Data collection and storage.
    """

    def __init__(self):
        self.sample_data = defaultdict(dict)  # sample_id -> data
        self.compression_methods = set()

    def log_result(self, result_data):
        """Store benchmark result data."""
        sample_id = result_data["sample_id"]

        # Track compression methods
        for compression_data in result_data.get("compression_methods", []):
            self.compression_methods.add(compression_data["method"])

        # Initialize sample data if new
        if sample_id not in self.sample_data:
            self.sample_data[sample_id] = {
                "task": result_data.get("task", ""),
                "llm_provider": result_data.get("llm_provider", ""),
                "llm_model": result_data.get("llm_model", ""),
                "original_prompt": result_data.get("original_prompt", ""),
                "ground_truth_answer": result_data.get("ground_truth_answer", ""),
                "baseline_output": result_data.get(
                    "baseline_output", ""
                ),  # Fix: use baseline_output
                "baseline_extracted_answer": result_data.get(
                    "baseline_extracted_answer", ""
                ),
                "baseline_score": result_data.get("baseline_score", 0),
                "baseline_latency": result_data.get("baseline_latency", 0),
                "methods": {},
            }

        # Store method-specific data
        for compression_data in result_data.get("compression_methods", []):
            method = compression_data["method"]
            self.sample_data[sample_id]["methods"][method] = {
                "compressed_prompt": compression_data.get("compressed_prompt", ""),
                "compressed_output": compression_data.get(
                    "compressed_prompt_output", ""
                ),
                "compressed_extracted_answer": compression_data.get(
                    "compressed_extracted_answer", ""
                ),
                "compressed_score": compression_data.get("compressed_score", 0),
                "compressed_latency": compression_data.get("compressed_latency", 0),
                "answers_match": compression_data.get("answers_match", False),
                "target_compression_ratio": result_data.get(
                    "target_compression_ratio", 0.8
                ),
            }

    def get_sample_data(self):
        """Get all collected sample data."""
        return self.sample_data

    def get_compression_methods(self):
        """Get all compression methods encountered."""
        return self.compression_methods
