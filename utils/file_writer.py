import csv
import os
import json
from datetime import datetime


class FileWriter:
    """
    Responsible for writing data to various file formats.
    Single Responsibility: File I/O operations.
    """

    def __init__(self, log_dir, base_name):
        self.log_dir = log_dir
        self.base_name = base_name
        self.log_file = os.path.join(log_dir, f"{base_name}.csv")
        self.summary_file = os.path.join(log_dir, f"{base_name}_summary.json")

    def write_csv(self, sample_data, compression_methods):
        """Write benchmark data to CSV file."""
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
                f"{method}_quality_degradation"
            ])

        # Write CSV file
        with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames,
                                  quoting=csv.QUOTE_ALL,
                                  escapechar='\\',
                                  doublequote=True)
            writer.writeheader()

            for sample_id in sorted(sample_data.keys()):
                sample = sample_data[sample_id]
                row = {
                    'sample_id': sample_id,
                    'task': sample['task'],
                    'llm_provider': sample['llm_provider'],
                    'llm_model': sample['llm_model'],
                    'original_prompt': sample['original_prompt'],
                    'ground_truth_answer': sample['ground_truth_answer'],
                    'baseline_output': sample['baseline_output'],
                    'baseline_extracted_answer': sample.get('baseline_extracted_answer', ''),
                    'baseline_score': sample['baseline_score'],
                    'baseline_latency': sample['baseline_latency']
                }

                # Add method-specific columns
                for method in sorted_methods:
                    if method in sample['methods']:
                        method_data = sample['methods'][method]
                        row.update({
                            f"{method}_compressed_prompt": method_data['compressed_prompt'],
                            f"{method}_compressed_output": method_data['compressed_output'],
                            f"{method}_compressed_extracted_answer": method_data['compressed_extracted_answer'],
                            f"{method}_compressed_score": method_data['compressed_score'],
                            f"{method}_compressed_latency": method_data['compressed_latency'],
                            f"{method}_answers_match": method_data['answers_match'],
                            f"{method}_target_ratio": method_data['target_compression_ratio'],
                            f"{method}_actual_ratio": method_data['actual_compression_ratio'],
                            f"{method}_compression_efficiency": method_data['compression_efficiency'],
                            f"{method}_tokens_saved": method_data['tokens_saved'],
                            f"{method}_score_preservation": method_data['score_preservation'],
                            f"{method}_latency_overhead": method_data['latency_overhead'],
                            f"{method}_quality_degradation": method_data['quality_degradation']
                        })
                    else:
                        # Fill missing method data with empty values
                        for col in [f"{method}_compressed_prompt", f"{method}_compressed_output",
                                  f"{method}_compressed_extracted_answer", f"{method}_compressed_score",
                                  f"{method}_compressed_latency", f"{method}_answers_match",
                                  f"{method}_target_ratio", f"{method}_actual_ratio",
                                  f"{method}_compression_efficiency", f"{method}_tokens_saved",
                                  f"{method}_score_preservation", f"{method}_latency_overhead",
                                  f"{method}_quality_degradation"]:
                            row[col] = ''

                writer.writerow(row)

        return self.log_file

    def write_json_summary(self, summary):
        """Write summary data to JSON file."""
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return self.summary_file
