"""
Quick Benchmark Analysis Script
Simple, focused analysis for individual CSV files or comparative analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import sys


class QuickAnalyzer:
    """Quick analyzer for benchmark CSV files with essential metrics."""

    def __init__(self):
        self.compression_methods = ["llmlingua2", "naive_truncation", "selective_context"]

    def analyze_single_file(self, csv_path: str, task_name: str = None) -> pd.DataFrame:
        """Analyze a single benchmark CSV file."""
        if not Path(csv_path).exists():
            print(f"❌ File not found: {csv_path}")
            return pd.DataFrame()

        print(f"📊 Analyzing: {Path(csv_path).name}")

        try:
            df = pd.read_csv(csv_path)
            if task_name is None:
                # Try to extract task name from filename
                filename = Path(csv_path).name
                if "classification" in filename:
                    task_name = "classification"
                elif "reasoning" in filename:
                    task_name = "reasoning"
                elif "summarization" in filename:
                    task_name = "summarization"
                else:
                    task_name = "unknown"

            results = self._calculate_metrics(df, task_name)
            return results

        except Exception as e:
            print(f"❌ Error analyzing {csv_path}: {e}")
            return pd.DataFrame()

    def analyze_multiple_files(self, file_dict: Dict[str, str]) -> pd.DataFrame:
        """Analyze multiple files and combine results."""
        all_results = []

        for task, filepath in file_dict.items():
            results = self.analyze_single_file(filepath, task)
            if not results.empty:
                all_results.append(results)

        if all_results:
            return pd.concat(all_results, ignore_index=True)
        return pd.DataFrame()

    def _calculate_metrics(self, df: pd.DataFrame, task_name: str) -> pd.DataFrame:
        """Calculate key metrics for a dataframe."""
        results = []

        # Overall baseline metrics
        baseline_accuracy = df['baseline_score'].mean() * 100
        baseline_latency = df['baseline_latency'].mean()

        for method in self.compression_methods:
            if f'{method}_compressed_score' not in df.columns:
                continue

            result = {
                'Task': task_name.title(),
                'Method': method.replace('_', ' ').title(),
                'Samples': len(df),
                'Baseline_Accuracy': baseline_accuracy,
                'Compressed_Accuracy': df[f'{method}_compressed_score'].mean() * 100,
                'Score_Preservation': (df[f'{method}_compressed_score'].mean() / df['baseline_score'].mean()) * 100,
                'Compression_Ratio': df[f'{method}_actual_ratio'].mean() * 100 if f'{method}_actual_ratio' in df.columns else np.nan,
                'Baseline_Latency': baseline_latency,
                'Compressed_Latency': df[f'{method}_compressed_latency'].mean(),
                'Latency_Overhead': ((df[f'{method}_compressed_latency'].mean() - baseline_latency) / baseline_latency) * 100 if baseline_latency > 0 else 0,
                'Tokens_Saved': df[f'{method}_tokens_saved'].mean() if f'{method}_tokens_saved' in df.columns else np.nan,
                'Answers_Match_Rate': df[f'{method}_answers_match'].mean() * 100 if f'{method}_answers_match' in df.columns else np.nan
            }

            results.append(result)

        return pd.DataFrame(results)

    def print_summary_table(self, results_df: pd.DataFrame):
        """Print a formatted summary table."""
        if results_df.empty:
            print("❌ No results to display")
            return

        print("\n" + "="*100)
        print("📊 BENCHMARK ANALYSIS SUMMARY")
        print("="*100)

        # Group by task for better organization
        for task in results_df['Task'].unique():
            task_data = results_df[results_df['Task'] == task]
            print(f"\n🎯 {task.upper()} TASK")
            print("-"*50)

            # Print header
            print("<25")
            print("-" * 85)

            # Print each method
            for _, row in task_data.iterrows():
                print("<25"
                      "<10.1f"
                      "<15.1f"
                      "<10.1f"
                      "<10.1f")

        print("\n" + "="*100)

    def export_to_csv(self, results_df: pd.DataFrame, output_path: str):
        """Export results to CSV."""
        if not results_df.empty:
            results_df.to_csv(output_path, index=False)
            print(f"💾 Results exported to: {output_path}")


def main():
    """Main function for command-line usage."""
    analyzer = QuickAnalyzer()

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python quick_analyzer.py <csv_file> [task_name]")
        print("  python quick_analyzer.py --multi")
        return

    if sys.argv[1] == "--multi":
        # Analyze all recent files
        results_dir = Path("results")
        if not results_dir.exists():
            print("❌ Results directory not found")
            return

        files = {
            "classification": None,
            "reasoning": None,
            "summarization": None
        }

        # Find latest files for each task
        for task in files.keys():
            csv_files = list(results_dir.glob(f"benchmark_{task}_*.csv"))
            if csv_files:
                latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
                files[task] = str(latest_file)

        # Remove tasks with no files
        files = {k: v for k, v in files.items() if v is not None}

        if not files:
            print("❌ No benchmark CSV files found")
            return

        print(f"📊 Analyzing {len(files)} tasks: {', '.join(files.keys())}")

        results = analyzer.analyze_multiple_files(files)
        analyzer.print_summary_table(results)

        # Export results
        output_file = f"quick_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        analyzer.export_to_csv(results, output_file)

    else:
        # Analyze single file
        csv_file = sys.argv[1]
        task_name = sys.argv[2] if len(sys.argv) > 2 else None

        results = analyzer.analyze_single_file(csv_file, task_name)
        if not results.empty:
            analyzer.print_summary_table(results)


if __name__ == "__main__":
    main()
