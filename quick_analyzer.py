"""
Quick Benchmark Analysis Script
Simple, focused analysis for individual CSV files or comparative analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import sys
from datetime import datetime


class QuickAnalyzer:
    """Quick analyzer for benchmark CSV files with essential metrics."""

    def __init__(self):
        self.compression_methods = ["llmlingua2", "naive_truncation", "selective_context"]

    def analyze_single_file(self, csv_path: str, task_name: str = None) -> pd.DataFrame:
        """Analyze a single benchmark CSV file."""
        try:
            df = pd.read_csv(csv_path)

            # Extract task name from filename if not provided
            if task_name is None:
                filename = Path(csv_path).stem
                if 'cla' in filename.lower():
                    task_name = 'Classification'
                elif 'sum' in filename.lower():
                    task_name = 'Summarization'
                elif 'rea' in filename.lower():
                    task_name = 'Reading Comprehension'
                else:
                    task_name = 'Unknown'

            # Parse the actual CSV structure with separate columns for each method
            results = []

            for method in self.compression_methods:
                # FIXED: Handle new column structure with backward compatibility
                score_col = None
                compression_col = None
                latency_col = f"{method}_latency_overhead_percent"
                tokens_col = f"{method}_tokens_saved"
                
                # Try new column names first (preferred)
                if f"{method}_score_preservation" in df.columns:
                    score_col = f"{method}_score_preservation"
                elif f"{method}_quality_degradation" in df.columns:
                    # Calculate score preservation from quality degradation
                    score_col = f"{method}_quality_degradation"
                
                # For compression ratio, try preservation ratio first, then compression ratio
                if f"{method}_actual_preservation_ratio" in df.columns:
                    compression_col = f"{method}_actual_preservation_ratio"
                elif f"{method}_actual_compression_ratio" in df.columns:
                    compression_col = f"{method}_actual_compression_ratio"
                # Backward compatibility: try old column name
                elif f"{method}_actual_ratio" in df.columns:
                    compression_col = f"{method}_actual_ratio"

                if score_col and compression_col:
                    # Calculate averages for this method
                    if score_col.endswith('_quality_degradation'):
                        # Convert quality degradation to score preservation
                        quality_deg = df[score_col].mean()
                        score_pres = max(0, 100 - (quality_deg * 100))  # Convert to percentage
                    else:
                        score_pres = df[score_col].mean()  # Already in percentage format (0-100)

                    # Handle compression ratio based on column type
                    if compression_col.endswith('_actual_preservation_ratio'):
                        # This is preservation ratio (fraction kept) - convert to percentage
                        preservation_ratio = df[compression_col].mean()
                        compression = preservation_ratio * 100  # Convert to percentage of original size retained
                    elif compression_col.endswith('_actual_compression_ratio'):
                        # This is compression ratio (fraction removed) - convert to preservation percentage
                        compression_ratio = df[compression_col].mean()
                        compression = (1.0 - compression_ratio) * 100  # Convert to percentage of original size retained
                    else:
                        # Old format - assume it's already in the right format
                        compression = df[compression_col].mean()

                    latency = df[latency_col].mean() if latency_col in df.columns else 0  # Already in percentage
                    tokens = df[tokens_col].mean() if tokens_col in df.columns else 0  # Already in correct format

                    results.append({
                        'Task': task_name,
                        'Method': method,
                        'Score_Preservation': score_pres,
                        'Compression_Ratio': compression,
                        'Latency_Overhead': latency,
                        'Tokens_Saved': tokens
                    })

            return pd.DataFrame(results)

        except Exception as e:
            print(f"❌ Error analyzing file {csv_path}: {e}")
            return pd.DataFrame()

    def analyze_multiple_files(self, csv_paths: List[str]) -> pd.DataFrame:
        """Analyze multiple benchmark CSV files."""
        all_results = []

        for csv_path in csv_paths:
            results = self.analyze_single_file(csv_path)
            if not results.empty:
                all_results.append(results)

        if all_results:
            return pd.concat(all_results, ignore_index=True)
        return pd.DataFrame()

    def print_summary_table(self, results_df: pd.DataFrame):
        """Print a formatted summary table."""
        if results_df.empty:
            print("❌ No results to display")
            return

        print("\n" + "="*85)
        print("🎯 QUICK BENCHMARK ANALYSIS RESULTS")
        print("="*85)

        # Group by task for better organization
        for task in results_df['Task'].unique():
            task_data = results_df[results_df['Task'] == task]
            print(f"\n📊 {task.upper()} TASK ANALYSIS")
            print("-" * 60)

            # Create a clean table format
            print("┌─────────────────────────────┬─────────────┬─────────────┬────────────┬────────────┐")
            print("│ Method                      │ Score Pres. │ Compression │ Latency    │ Tokens     │")
            print("├─────────────────────────────┼─────────────┼─────────────┼────────────┼────────────┤")

            # Print each method with proper formatting
            for _, row in task_data.iterrows():
                method = str(row.get('Method', 'N/A'))[:25]  # Truncate long method names
                score_pres = row.get('Score_Preservation', 0)
                compression = row.get('Compression_Ratio', 0)
                latency = row.get('Latency_Overhead', 0)
                tokens = row.get('Tokens_Saved', 0)

                # Format with proper alignment and signs
                latency_str = f"{latency:+.1f}%" if latency != 0 else "0.0%"
                tokens_str = f"{tokens:+.0f}" if tokens != 0 else "0"

                print(f"│ {method:<27} │ {score_pres:>9.1f}% │ {compression:>9.1f}% │ {latency_str:>8} │ {tokens_str:>8} │")

            print("└─────────────────────────────┴─────────────┴─────────────┴────────────┴────────────┘")

            # Add summary insights
            if not task_data.empty:
                best_score = task_data.loc[task_data['Score_Preservation'].idxmax()]
                best_compression = task_data.loc[task_data['Compression_Ratio'].idxmax()]
                best_latency = task_data.loc[task_data['Latency_Overhead'].idxmin()]

                print(f"\n💡 Key Insights:")
                print(f"   🏆 Best Score Preservation: {best_score['Method']} ({best_score['Score_Preservation']:.1f}%)")
                print(f"   🗜️  Best Compression: {best_compression['Method']} ({best_compression['Compression_Ratio']:.1f}%)")
                print(f"   ⚡ Best Latency: {best_latency['Method']} ({best_latency['Latency_Overhead']:+.1f}%)")

        print("\n" + "="*85)

    def export_to_md(self, results_df: pd.DataFrame, output_path: str, filename: str = None):
        """Export results to a formatted Markdown file."""
        if results_df.empty:
            print("❌ No results to export")
            return

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"quick_analysis_report_{timestamp}.md"

        full_path = output_path / filename

        with open(full_path, 'w', encoding='utf-8') as f:
            f.write("# 🎯 Quick Benchmark Analysis Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")

            # Summary statistics
            f.write("## 📊 Summary Statistics\n\n")
            total_methods = len(results_df)
            avg_score_pres = results_df['Score_Preservation'].mean()
            avg_compression = results_df['Compression_Ratio'].mean()
            avg_latency = results_df['Latency_Overhead'].mean()

            f.write(f"- **Total Methods Analyzed:** {total_methods}\n")
            f.write(f"- **Average Score Preservation:** {avg_score_pres:.1f}%\n")
            f.write(f"- **Average Compression Ratio:** {avg_compression:.1f}%\n")
            f.write(f"- **Average Latency Overhead:** {avg_latency:+.1f}%\n\n")

            # Group by task
            for task in results_df['Task'].unique():
                task_data = results_df[results_df['Task'] == task]
                f.write(f"## 🎯 {task.upper()} Task Results\n\n")

                # Create markdown table
                f.write("| Method | Score Preservation | Compression Ratio | Latency Overhead | Tokens Saved |\n")
                f.write("|--------|-------------------|------------------|------------------|--------------|\n")

                for _, row in task_data.iterrows():
                    method = str(row.get('Method', 'N/A'))
                    score_pres = row.get('Score_Preservation', 0)
                    compression = row.get('Compression_Ratio', 0)
                    latency = row.get('Latency_Overhead', 0)
                    tokens = row.get('Tokens_Saved', 0)

                    latency_str = f"{latency:+.1f}%" if latency != 0 else "0.0%"
                    tokens_str = f"{tokens:+.0f}" if tokens != 0 else "0"

                    f.write(f"| {method} | {score_pres:.1f}% | {compression:.1f}% | {latency_str} | {tokens_str} |\n")

                f.write("\n")

                # Add insights for each task
                if not task_data.empty:
                    best_score = task_data.loc[task_data['Score_Preservation'].idxmax()]
                    best_compression = task_data.loc[task_data['Compression_Ratio'].idxmax()]
                    best_latency = task_data.loc[task_data['Latency_Overhead'].idxmin()]

                    f.write("### 💡 Key Insights\n\n")
                    f.write(f"- **🏆 Best Score Preservation:** {best_score['Method']} ({best_score['Score_Preservation']:.1f}%)\n")
                    f.write(f"- **🗜️ Best Compression:** {best_compression['Method']} ({best_compression['Compression_Ratio']:.1f}%)\n")
                    f.write(f"- **⚡ Best Latency:** {best_latency['Method']} ({best_latency['Latency_Overhead']:+.1f}%)\n\n")

            f.write("---\n\n")
            f.write("*Report generated by Quick Analyzer*\n")

        print(f"✅ Report exported to: {full_path}")

    def run_analysis(self, csv_path: str, output_dir: str = "results"):
        """Run complete analysis on a CSV file."""
        # Analyze the file
        results_df = self.analyze_single_file(csv_path)

        if results_df.empty:
            print("❌ No results to analyze")
            return

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Print formatted table
        self.print_summary_table(results_df)

        # Generate filename matching CSV pattern
        csv_filename = Path(csv_path).stem  # e.g., "bench_cla_001"
        md_filename = f"quick_analysis_{csv_filename}.md"

        # Export to markdown
        self.export_to_md(results_df, output_path, md_filename)


def main():
    """Main function for command-line usage."""
    analyzer = QuickAnalyzer()

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python quick_analyzer.py <csv_file> [task_name]")
        print("  python quick_analyzer.py <csv_file1> <csv_file2> ... (multiple files)")
        print("  python quick_analyzer.py --multi")
        return

    if sys.argv[1] == "--multi":
        # Multi-file analysis with hardcoded paths
        csv_files = [
            "results/bench_cla_001.csv",
            "results/bench_rea_001.csv",
            "results/bench_sum_001.csv"
        ]

        print("🔍 Analyzing multiple benchmark files...")
        results_df = analyzer.analyze_multiple_files(csv_files)

        if not results_df.empty:
            output_path = Path("results")
            output_path.mkdir(exist_ok=True)

            analyzer.print_summary_table(results_df)
            analyzer.export_to_md(results_df, output_path)  # Uses default timestamp filename
        else:
            print("❌ No results found in the specified files")
    elif len(sys.argv) > 2 and not sys.argv[2].endswith('.csv'):
        # Single file analysis with task name
        csv_path = sys.argv[1]
        task_name = sys.argv[2]

        print(f"🔍 Analyzing: {csv_path}")
        analyzer.run_analysis(csv_path, "results")
    elif len(sys.argv) > 2:
        # Multiple file analysis with command line arguments
        csv_files = sys.argv[1:]
        
        print(f"🔍 Analyzing {len(csv_files)} benchmark files...")
        results_df = analyzer.analyze_multiple_files(csv_files)

        if not results_df.empty:
            output_path = Path("results")
            output_path.mkdir(exist_ok=True)

            analyzer.print_summary_table(results_df)
            analyzer.export_to_md(results_df, output_path)  # Uses default timestamp filename
        else:
            print("❌ No results found in the specified files")
    else:
        # Single file analysis
        csv_path = sys.argv[1]

        print(f"🔍 Analyzing: {csv_path}")
        analyzer.run_analysis(csv_path, "results")


if __name__ == "__main__":
    main()
