"""
Adapter to integrate the new BenchmarkAnalyzer with the existing logger framework.
Provides backward compatibility while using the enhanced analysis capabilities.
"""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from benchmark_analyzer import BenchmarkAnalyzer


class AnalyzerAdapter:
    """
    Adapter class that bridges the new BenchmarkAnalyzer with the existing logger interface.
    Provides the same API as the old DataAnalyzer while using the new enhanced analysis.
    """

    def __init__(self, sample_data: Dict, compression_methods: set, results_dir: str = "results"):
        self.sample_data = sample_data
        self.compression_methods = compression_methods
        self.results_dir = Path(results_dir)

        # Create a temporary CSV file for the new analyzer to work with
        self.temp_csv_path = self._create_temp_csv()

        # Initialize the new analyzer
        self.new_analyzer = BenchmarkAnalyzer(results_dir=str(self.results_dir))

    def _create_temp_csv(self) -> str:
        """Create a temporary CSV file from the sample data for the new analyzer."""
        # Convert sample data to the format expected by the new analyzer
        csv_data = []

        for sample_id, sample in self.sample_data.items():
            base_row = {
                'sample_id': sample_id,
                'task': sample.get('task', ''),
                'llm_provider': sample.get('llm_provider', ''),
                'llm_model': sample.get('llm_model', ''),
                'original_prompt': sample.get('original_prompt', ''),
                'ground_truth_answer': sample.get('ground_truth_answer', ''),
                'baseline_output': sample.get('baseline_output', ''),
                'baseline_extracted_answer': sample.get('baseline_extracted_answer', ''),
                'baseline_score': sample.get('baseline_score', 0),
                'baseline_latency': sample.get('baseline_latency', 0),
            }

            # Add data for each compression method
            for method_name, method_data in sample.get('methods', {}).items():
                row = base_row.copy()
                row.update({
                    f'{method_name}_compressed_prompt': method_data.get('compressed_prompt', ''),
                    f'{method_name}_compressed_output': method_data.get('compressed_output', ''),
                    f'{method_name}_compressed_extracted_answer': method_data.get('compressed_extracted_answer', ''),
                    f'{method_name}_compressed_score': method_data.get('compressed_score', 0),
                    f'{method_name}_compressed_latency': method_data.get('compressed_latency', 0),
                    f'{method_name}_answers_match': method_data.get('answers_match', False),
                    f'{method_name}_target_ratio': method_data.get('target_compression_ratio', 0.8),
                    f'{method_name}_actual_ratio': method_data.get('actual_compression_ratio', 1.0),
                    f'{method_name}_compression_efficiency': method_data.get('compression_efficiency', 0.0),
                    f'{method_name}_tokens_saved': method_data.get('tokens_saved', 0),
                    f'{method_name}_score_preservation': method_data.get('score_preservation', 0.0),
                    f'{method_name}_latency_overhead': method_data.get('latency_overhead', 0.0),
                    f'{method_name}_latency_overhead_seconds': method_data.get('latency_overhead_seconds', 0.0),
                    f'{method_name}_latency_overhead_percent': method_data.get('latency_overhead_percent', 0.0),
                    f'{method_name}_quality_degradation': method_data.get('quality_degradation', 0.0),
                    f'{method_name}_quality_degradation_percent': method_data.get('quality_degradation_percent', 0.0),
                })
                csv_data.append(row)

        # Create DataFrame and save to temporary CSV
        if csv_data:
            df = pd.DataFrame(csv_data)
            temp_path = self.results_dir / f"temp_analysis_{datetime.now().strftime('%H%M%S')}.csv"
            df.to_csv(temp_path, index=False)
            return str(temp_path)

        return ""

    def calculate_method_summary(self, method: str) -> Dict[str, Any]:
        """Calculate summary statistics for a single compression method (old API)."""
        if not self.temp_csv_path or not os.path.exists(self.temp_csv_path):
            return {}

        try:
            # Use the new analyzer to get comprehensive metrics
            results_df = self.new_analyzer.analyze_single_file(self.temp_csv_path)

            if results_df.empty:
                return {}

            # Find the method in the results
            method_data = results_df[results_df['method'].str.lower() == method.lower()]
            if method_data.empty:
                return {}

            row = method_data.iloc[0]

            # Convert to old API format
            return {
                "sample_count": len(results_df),  # Use total rows as sample count
                "baseline_performance": {
                    "mean_score": row['baseline_accuracy'] / 100,
                    "std_score": 0.0,  # New analyzer doesn't calculate std
                    "accuracy_rate": row['baseline_accuracy'] / 100
                },
                "compressed_performance": {
                    "mean_score": row['compressed_accuracy'] / 100,
                    "std_score": 0.0,
                    "accuracy_rate": row['compressed_accuracy'] / 100
                },
                "compression_metrics": {
                    "mean_compression_ratio": row['actual_compression_ratio'] / 100,
                    "std_compression_ratio": 0.0,
                    "mean_compression_efficiency": row['compression_efficiency'] / 100 if pd.notna(row['compression_efficiency']) else 0.0,
                    "mean_tokens_saved": row['avg_tokens_saved'] if pd.notna(row['avg_tokens_saved']) else 0
                },
                "quality_metrics": {
                    "mean_score_preservation": row['score_preservation'] / 100,
                    "mean_quality_degradation": row['quality_degradation'] / 100 if pd.notna(row['quality_degradation']) else 0.0,
                    "answer_consistency_rate": row['answers_match_rate'] / 100 if pd.notna(row['answers_match_rate']) else 0
                },
                "latency_metrics": {
                    "mean_baseline_latency": row['baseline_latency'],
                    "mean_compressed_latency": row['compressed_latency'],
                    "mean_latency_overhead": row['latency_overhead_percent'] / 100
                }
            }

        except Exception as e:
            print(f"Error in calculate_method_summary: {e}")
            return {}

    def calculate_comparative_analysis(self) -> Dict[str, Any]:
        """Calculate comparative analysis across all compression methods (old API)."""
        if not self.temp_csv_path or not os.path.exists(self.temp_csv_path):
            return {"note": "No data available for comparative analysis"}

        try:
            # Use the new analyzer for comprehensive analysis
            results_df = self.new_analyzer.analyze_single_file(self.temp_csv_path)

            if results_df.empty:
                return {"note": "No data available for comparative analysis"}

            # Find best performers using new analyzer's metrics
            best_preservation = results_df.loc[results_df['score_preservation'].idxmax()]
            best_compression = results_df.loc[results_df['actual_compression_ratio'].idxmax()]

            return {
                "best_compression_ratio": (best_compression['method'], best_compression['actual_compression_ratio'] / 100),
                "best_compression_efficiency": (best_preservation['method'], best_preservation['score_preservation'] / 100),  # Using preservation as proxy
                "best_score_preservation": (best_preservation['method'], best_preservation['score_preservation'] / 100),
                "compression_ratio_ranking": [(row['method'], row['actual_compression_ratio'] / 100) for _, row in results_df.sort_values('actual_compression_ratio').iterrows()],
                "compression_efficiency_ranking": [(row['method'], row['score_preservation'] / 100) for _, row in results_df.sort_values('score_preservation', ascending=False).iterrows()],
                "score_preservation_ranking": [(row['method'], row['score_preservation'] / 100) for _, row in results_df.sort_values('score_preservation', ascending=False).iterrows()]
            }

        except Exception as e:
            print(f"Error in calculate_comparative_analysis: {e}")
            return {"note": "Error during comparative analysis"}

    def calculate_task_summary(self, task: str) -> Dict[str, Any]:
        """Calculate summary statistics for a single task type (old API)."""
        if not self.temp_csv_path or not os.path.exists(self.temp_csv_path):
            return {}

        try:
            # Filter data for the specific task
            df = pd.read_csv(self.temp_csv_path)
            task_df = df[df['task'].str.lower() == task.lower()]

            if task_df.empty:
                return {}

            # Calculate basic statistics
            baseline_scores = []
            compressed_scores = []
            compression_ratios = []
            latencies = []

            # Extract data from all method columns
            for _, row in task_df.iterrows():
                baseline_scores.append(row['baseline_score'])
                latencies.append(row['baseline_latency'])

                for method in self.compression_methods:
                    if f'{method}_compressed_score' in row:
                        compressed_scores.append(row[f'{method}_compressed_score'])
                        if f'{method}_actual_ratio' in row:
                            compression_ratios.append(row[f'{method}_actual_ratio'])
                        latencies.append(row[f'{method}_compressed_latency'])

            if not baseline_scores:
                return {}

            return {
                "sample_count": len(task_df),
                "methods_tested": len(self.compression_methods),
                "baseline_performance": {
                    "mean_score": sum(baseline_scores) / len(baseline_scores),
                    "std_score": 0.0,
                    "accuracy_rate": sum(baseline_scores) / len(baseline_scores)
                },
                "overall_compressed_performance": {
                    "mean_score": sum(compressed_scores) / len(compressed_scores) if compressed_scores else 0,
                    "std_score": 0.0,
                    "mean_compression_ratio": sum(compression_ratios) / len(compression_ratios) if compression_ratios else 0
                },
                "latency_analysis": {
                    "mean_latency": sum(latencies) / len(latencies) if latencies else 0,
                    "std_latency": 0.0
                }
            }

        except Exception as e:
            print(f"Error in calculate_task_summary: {e}")
            return {}

    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_csv_path and os.path.exists(self.temp_csv_path):
            try:
                os.remove(self.temp_csv_path)
            except Exception as e:
                print(f"Warning: Could not remove temporary file {self.temp_csv_path}: {e}")


# Factory function to create analyzer (maintains compatibility)
def DataAnalyzer(sample_data, compression_methods):
    """Factory function that returns the new analyzer adapter."""
    return AnalyzerAdapter(sample_data, compression_methods)
