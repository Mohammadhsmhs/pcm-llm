"""
Adapter to integrate the new BenchmarkAnalyzer with the existing logger framework.
Provides backward compatibility while using the enhanced analysis capabilities.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any

from utils.data.data_enhancer import DataEnhancer


class DataAnalyzer:
    """
    An analyzer that uses a pandas DataFrame for in-memory analysis
    and mimics the old file-based analyzer's interface.
    """

    def __init__(self, sample_data: Dict, compression_methods: set, results_dir: str = "results"):
        self.sample_data = sample_data
        self.compression_methods = compression_methods
        self.results_dir = Path(results_dir)
        self.df = self._create_dataframe()
        self.temp_csv_path = None  # No longer using temp files

    def _create_dataframe(self):
        """Create a pandas DataFrame from the collected sample data."""
        records = []
        if not self.sample_data:
            return pd.DataFrame()

        for sample_id, sample in self.sample_data.items():
            for method in self.compression_methods:
                method_data = sample.get('methods', {}).get(method, {})
                
                # Prepare data for enhancer
                enhancer_input = {
                    'baseline_score': sample.get('baseline_score', 0.0),
                    'compressed_score': method_data.get('compressed_score', 0.0),
                    'original_prompt': sample.get('original_prompt', ''),
                    'compressed_prompt': method_data.get('compressed_prompt', ''),
                    'baseline_latency': sample.get('baseline_latency', 0.0),
                    'compressed_latency': method_data.get('compressed_latency', 0.0),
                    'target_compression_ratio': method_data.get('target_compression_ratio', 0.8)
                }
                
                enhanced_data = DataEnhancer.enhance_result_data(enhancer_input)

                record = {
                    'sample_id': sample_id,
                    'task': sample.get('task', 'unknown'),
                    'method': method,
                    'baseline_score': sample.get('baseline_score', 0.0),
                    'compressed_score': method_data.get('compressed_score', 0.0),
                    'answers_match': method_data.get('answers_match', False),
                    'actual_compression_ratio': enhanced_data.get('actual_compression_ratio', 0.0),
                    'score_preservation': enhanced_data.get('score_preservation', 0.0),
                    'compression_efficiency': enhanced_data.get('compression_efficiency', 0.0),
                    'quality_degradation': enhanced_data.get('quality_degradation', 0.0),
                    'latency_overhead': enhanced_data.get('latency_overhead_seconds', 0.0),
                    'tokens_saved': enhanced_data.get('tokens_saved', 0)
                }
                records.append(record)
        
        return pd.DataFrame(records) if records else pd.DataFrame()

    def calculate_method_summary(self, method: str) -> Dict[str, Any]:
        """Calculate summary statistics for a single compression method."""
        method_df = self.df[self.df['method'].str.lower() == method.lower()]
        if method_df.empty:
            return {}

        try:
            summary = {
                "answers_match_rate": f"{method_df['answers_match'].mean():.2%}",
                "avg_score_preservation": f"{method_df['score_preservation'].mean():.2%}",
                "avg_compression_ratio": f"{method_df['actual_compression_ratio'].mean():.2%}",
                "avg_compression_efficiency": f"{method_df['compression_efficiency'].mean():.2%}",
                "avg_tokens_saved": f"{method_df['tokens_saved'].mean():.2f}",
            }
            return summary
        except KeyError as e:
            print(f"Error in calculate_method_summary: {e}")
            return {}

    def calculate_task_summary(self, task: str) -> Dict[str, Any]:
        """Calculate summary statistics for a single task."""
        task_df = self.df[self.df['task'].str.lower() == task.lower()]
        if task_df.empty:
            return {}
        
        return {
            "sample_count": len(task_df['sample_id'].unique()),
            "avg_baseline_score": task_df['baseline_score'].mean(),
            "avg_score_preservation": task_df['score_preservation'].mean(),
        }

    def calculate_comparative_analysis(self) -> Dict[str, Any]:
        """Perform a comparative analysis across all methods."""
        if self.df.empty:
            return {}

        # Find best method for score preservation and compression ratio
        avg_metrics = self.df.groupby('method').mean(numeric_only=True)
        
        best_preservation = avg_metrics['score_preservation'].idxmax()
        best_preservation_val = avg_metrics['score_preservation'].max()
        
        best_compression = avg_metrics['actual_compression_ratio'].idxmax()
        best_compression_val = avg_metrics['actual_compression_ratio'].max()

        return {
            "best_score_preservation": (best_preservation, best_preservation_val),
            "best_compression_ratio": (best_compression, best_compression_val),
        }

    def get_full_dataframe(self) -> pd.DataFrame:
        """Return the full DataFrame for external use."""
        return self.df
