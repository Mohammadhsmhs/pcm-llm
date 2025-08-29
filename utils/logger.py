import csv
import os
from datetime import datetime

class BenchmarkLogger:
    """
    A class to handle logging of benchmark results to a timestamped CSV file.
    """
    def __init__(self, log_dir="results"):
        self.log_dir = log_dir
        # Create the results directory if it doesn't exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        # Create a unique filename based on the current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"benchmark_{timestamp}.csv")
        
        # Define the headers for the CSV file
        self.fieldnames = [
            "sample_id", "llm_provider", "llm_model", "compression_method", 
            "target_compression_ratio", "original_prompt", "compressed_prompt", 
            "ground_truth_answer", "original_prompt_output", "compressed_prompt_output", 
           "baseline_score", "compressed_score", 
           "baseline_score", "compressed_score", "answers_match", 
            "baseline_latency", "compressed_latency"
        ]
        
        # Write the header row to the new CSV file
        with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
        
        print(f"Logging results to {self.log_file}")

    def log_result(self, result_data):
        """
        Appends a single result dictionary as a new row to the CSV file.
        """
        with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(result_data)