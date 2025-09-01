import pandas as pd
import os
import json
from datetime import datetime


class FileWriter:
    """
    Responsible for writing data to various file formats using pandas for robustness.
    Single Responsibility: File I/O operations.
    """

    def __init__(self, log_dir, base_name):
        self.log_dir = log_dir
        self.base_name = base_name
        self.log_file = os.path.join(log_dir, f"{base_name}.csv")
        self.summary_file = os.path.join(log_dir, f"{base_name}_summary.json")

    def write_csv(self, sample_data, compression_methods):
        """Write benchmark data to a CSV file using pandas."""
        if not sample_data:
            return

        # Create dynamic fieldnames
        fieldnames = [
            "sample_id", "task", "llm_provider", "llm_model",
            "original_prompt", "ground_truth_answer", "baseline_output",
            "baseline_extracted_answer", "baseline_score", "baseline_latency"
        ]

        # Add columns for each compression method
        sorted_methods = sorted(compression_methods)
        for method in sorted_methods:
            fieldnames.extend([
                f"{method}_compressed_prompt", f"{method}_compressed_output",
                f"{method}_compressed_extracted_answer", f"{method}_compressed_score",
                f"{method}_compressed_latency", f"{method}_answers_match",
                f"{method}_target_ratio", f"{method}_actual_ratio",
                f"{method}_compression_efficiency", f"{method}_tokens_saved",
                f"{method}_score_preservation", f"{method}_latency_overhead",
                f"{method}_latency_overhead_seconds", f"{method}_latency_overhead_percent",
                f"{method}_quality_degradation", f"{method}_quality_degradation_percent"
            ])

        records = []
        for sample_id, sample in sample_data.items():
            record = {
                'sample_id': sample_id,
                'task': sample.get('task', ''),
                'llm_provider': sample.get('llm_provider', ''),
                'llm_model': sample.get('llm_model', ''),
                'original_prompt': sample.get('original_prompt', ''),
                'ground_truth_answer': sample.get('ground_truth_answer', ''),
                'baseline_output': sample.get('baseline_output', ''),
                'baseline_extracted_answer': sample.get('baseline_extracted_answer', ''),
                'baseline_score': sample.get('baseline_score', 0.0),
                'baseline_latency': sample.get('baseline_latency', 0.0)
            }

            for method in sorted_methods:
                method_data = sample.get('methods', {}).get(method, {})
                if method_data:
                    record.update({
                        f"{method}_compressed_prompt": method_data.get('compressed_prompt'),
                        f"{method}_compressed_output": method_data.get('compressed_output'),
                        f"{method}_compressed_extracted_answer": method_data.get('compressed_extracted_answer'),
                        f"{method}_compressed_score": method_data.get('compressed_score'),
                        f"{method}_compressed_latency": method_data.get('compressed_latency'),
                        f"{method}_answers_match": method_data.get('answers_match'),
                        f"{method}_target_ratio": method_data.get('target_compression_ratio'),
                        f"{method}_actual_ratio": method_data.get('actual_compression_ratio'),
                        f"{method}_compression_efficiency": method_data.get('compression_efficiency'),
                        f"{method}_tokens_saved": method_data.get('tokens_saved'),
                        f"{method}_score_preservation": method_data.get('score_preservation'),
                        f"{method}_latency_overhead": method_data.get('latency_overhead'),
                        f"{method}_latency_overhead_seconds": method_data.get('latency_overhead_seconds'),
                        f"{method}_latency_overhead_percent": method_data.get('latency_overhead_percent'),
                        f"{method}_quality_degradation": method_data.get('quality_degradation'),
                        f"{method}_quality_degradation_percent": method_data.get('quality_degradation_percent')
                    })
            records.append(record)

        if not records:
            return

        df = pd.DataFrame(records)
        df = df.reindex(columns=fieldnames)

        df.to_csv(self.log_file, index=False, encoding='utf-8')
        
        print(f"✅ CSV file saved: {self.log_file}")
        return self.log_file

    def write_summary(self, summary_data):
        """Write summary data to JSON file."""
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=4)
        print(f"✅ JSON summary saved: {self.summary_file}")
