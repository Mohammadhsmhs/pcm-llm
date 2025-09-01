"""
Comprehensive Benchmark Analysis Tool
Generates reliable analysis from benchmark CSV results with enhanced metrics and visualizations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Optional imports for visualizations
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("⚠️  Visualization libraries (matplotlib, seaborn) not available. Visualizations will be skipped.")


class BenchmarkAnalyzer:
    """Comprehensive analyzer for benchmark results with advanced metrics and visualizations."""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.compression_methods = ["llmlingua2", "naive_truncation", "selective_context"]
        self.tasks = ["classification", "reasoning", "summarization"]

    def load_latest_results(self) -> Dict[str, pd.DataFrame]:
        """Load the most recent CSV files for each task."""
        results = {}

        for task in self.tasks:
            # Find the latest CSV file for this task
            csv_files = list(self.results_dir.glob(f"benchmark_{task}_*.csv"))
            if not csv_files:
                print(f"⚠️  No CSV files found for {task}")
                continue

            # Get the most recent file
            latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
            print(f"📊 Loading {task}: {latest_file.name}")

            try:
                df = pd.read_csv(latest_file)
                results[task] = df
                print(f"   ✅ Loaded {len(df)} samples")
            except Exception as e:
                print(f"   ❌ Error loading {latest_file}: {e}")

        return results

    def analyze_single_file(self, csv_path: str) -> pd.DataFrame:
        """Analyze a single CSV file and return comprehensive metrics."""
        try:
            # Load the CSV file
            df = pd.read_csv(csv_path)
            print(f"📊 Analyzing file: {csv_path}")

            # Extract task name from filename or use default
            task_name = "unknown"
            if "reasoning" in csv_path.lower():
                task_name = "reasoning"
            elif "summarization" in csv_path.lower():
                task_name = "summarization"
            elif "classification" in csv_path.lower():
                task_name = "classification"

            # Calculate comprehensive metrics
            analysis_df = self.calculate_comprehensive_metrics(df, task_name)

            return analysis_df

        except Exception as e:
            print(f"❌ Error analyzing file {csv_path}: {e}")
            return pd.DataFrame()

    def calculate_comprehensive_metrics(self, df: pd.DataFrame, task: str) -> pd.DataFrame:
        """Calculate comprehensive metrics for a task's results."""
        metrics_data = []

        # Overall baseline metrics
        has_baseline_score = 'baseline_score' in df.columns
        if not has_baseline_score:
            print("   ⚠️  Warning: 'baseline_score' column not found. Some metrics will be skipped.")
            baseline_accuracy = np.nan
        else:
            baseline_accuracy = df['baseline_score'].mean() * 100

        has_baseline_latency = 'baseline_latency' in df.columns
        if not has_baseline_latency:
            baseline_latency = np.nan
        else:
            baseline_latency = df['baseline_latency'].mean()

        # Find which compression methods actually have data in this CSV
        available_methods = []
        for method in self.compression_methods:
            if f'{method}_compressed_score' in df.columns:
                available_methods.append(method)

        print(f"📊 Found data for methods: {available_methods}")

        for method in available_methods:
            method_data = {}

            # Basic metrics
            method_data['method'] = method.replace('_', ' ').title()
            method_data['task'] = task.title()

            # Accuracy metrics
            compressed_accuracy = df[f'{method}_compressed_score'].mean() * 100
            method_data['baseline_accuracy'] = baseline_accuracy
            method_data['compressed_accuracy'] = compressed_accuracy
            method_data['accuracy_drop'] = baseline_accuracy - compressed_accuracy if has_baseline_score else np.nan
            method_data['score_preservation'] = (compressed_accuracy / baseline_accuracy * 100) if has_baseline_score and baseline_accuracy > 0 else np.nan

            # Compression metrics
            actual_ratio = df[f'{method}_actual_ratio'].mean()
            method_data['target_ratio'] = df[f'{method}_target_ratio'].iloc[0] if f'{method}_target_ratio' in df.columns else 0.8
            method_data['actual_compression_ratio'] = actual_ratio
            method_data['compression_efficiency'] = df[f'{method}_compression_efficiency'].mean() * 100 if f'{method}_compression_efficiency' in df.columns else np.nan

            # Token savings
            method_data['avg_tokens_saved'] = df[f'{method}_tokens_saved'].mean() if f'{method}_tokens_saved' in df.columns else np.nan

            # Latency metrics
            compressed_latency = df[f'{method}_compressed_latency'].mean()
            method_data['baseline_latency'] = baseline_latency
            method_data['compressed_latency'] = compressed_latency
            method_data['latency_overhead_seconds'] = compressed_latency - baseline_latency if has_baseline_latency else np.nan
            method_data['latency_overhead_percent'] = ((compressed_latency - baseline_latency) / baseline_latency * 100) if has_baseline_latency and baseline_latency > 0 else np.nan

            # Quality degradation
            method_data['quality_degradation'] = df[f'{method}_quality_degradation'].mean() * 100 if f'{method}_quality_degradation' in df.columns else 0
            method_data['answers_match_rate'] = df[f'{method}_answers_match'].mean() * 100 if f'{method}_answers_match' in df.columns else np.nan

            # Advanced metrics
            method_data['compression_effectiveness'] = self._calculate_compression_effectiveness(
                compressed_accuracy, baseline_accuracy, actual_ratio
            ) if has_baseline_score else np.nan
            method_data['efficiency_score'] = self._calculate_efficiency_score(
                method_data['score_preservation'], method_data['actual_compression_ratio'],
                method_data['latency_overhead_percent']
            ) if has_baseline_score and has_baseline_latency else np.nan

            metrics_data.append(method_data)

        return pd.DataFrame(metrics_data)

    def _calculate_compression_effectiveness(self, compressed_acc: float, baseline_acc: float,
                                           compression_ratio: float) -> float:
        """Calculate compression effectiveness score (0-100)."""
        if baseline_acc == 0:
            return 0

        # Score preservation factor (higher is better)
        preservation_factor = compressed_acc / baseline_acc

        # Compression factor (higher compression ratio is better, but penalize extreme compression)
        compression_factor = min(compression_ratio / 0.8, 1.0)  # Normalize to target ratio

        # Combined effectiveness (weighted average)
        effectiveness = (preservation_factor * 0.7 + compression_factor * 0.3) * 100

        return max(0, min(100, effectiveness))

    def _calculate_efficiency_score(self, score_preservation: float, compression_ratio: float,
                                  latency_overhead: float) -> float:
        """Calculate overall efficiency score considering multiple factors."""
        # Normalize latency overhead (negative overhead is good, positive is bad)
        latency_factor = max(0, 100 - abs(latency_overhead))

        # Combined efficiency score
        efficiency = (score_preservation * 0.5 + compression_ratio * 0.3 + latency_factor * 0.2)

        return max(0, min(100, efficiency))

    def generate_comparative_analysis(self, results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate comparative analysis across all tasks and methods."""
        all_metrics = []

        for task, df in results.items():
            task_metrics = self.calculate_comprehensive_metrics(df, task)
            all_metrics.append(task_metrics)

        if not all_metrics:
            return pd.DataFrame()

        combined_df = pd.concat(all_metrics, ignore_index=True)
        return combined_df

    def create_visualizations(self, analysis_df: pd.DataFrame, output_dir: str = "analysis_output"):
        """Create comprehensive visualizations for the analysis."""
        if not VISUALIZATION_AVAILABLE:
            print("⚠️  Skipping visualizations - matplotlib/seaborn not available")
            return

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        if analysis_df.empty:
            print("⚠️  No data available for visualizations")
            return

        # Set up the plotting area
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Prompt Compression Benchmark Analysis', fontsize=16, fontweight='bold')

        # 1. Score Preservation by Task and Method
        ax1 = axes[0, 0]
        preservation_pivot = analysis_df.pivot_table(
            values='score_preservation', index='task', columns='method', aggfunc='mean'
        )
        preservation_pivot.plot(kind='bar', ax=ax1, width=0.8)
        ax1.set_title('Score Preservation by Task')
        ax1.set_ylabel('Score Preservation (%)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.tick_params(axis='x', rotation=45)

        # 2. Compression Ratio Comparison
        ax2 = axes[0, 1]
        compression_pivot = analysis_df.pivot_table(
            values='actual_compression_ratio', index='task', columns='method', aggfunc='mean'
        )
        compression_pivot.plot(kind='bar', ax=ax2, width=0.8)
        ax2.set_title('Compression Ratio by Task')
        ax2.set_ylabel('Compression Ratio (%)')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.tick_params(axis='x', rotation=45)

        # 3. Latency Overhead Analysis
        ax3 = axes[0, 2]
        latency_pivot = analysis_df.pivot_table(
            values='latency_overhead_percent', index='task', columns='method', aggfunc='mean'
        )
        latency_pivot.plot(kind='bar', ax=ax3, width=0.8)
        ax3.set_title('Latency Overhead by Task')
        ax3.set_ylabel('Latency Overhead (%)')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.tick_params(axis='x', rotation=45)

        # 4. Efficiency Score Heatmap
        ax4 = axes[1, 0]
        efficiency_pivot = analysis_df.pivot_table(
            values='efficiency_score', index='task', columns='method', aggfunc='mean'
        )
        sns.heatmap(efficiency_pivot, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax4, cbar_kws={'label': 'Efficiency Score'})
        ax4.set_title('Efficiency Score Heatmap')

        # 5. Method Comparison Across Tasks
        ax5 = axes[1, 1]
        method_comparison = analysis_df.groupby('method')[['score_preservation', 'actual_compression_ratio']].mean()
        method_comparison.plot(kind='bar', ax=ax5, width=0.8)
        ax5.set_title('Average Performance by Method')
        ax5.set_ylabel('Score (%)')
        ax5.legend(['Score Preservation', 'Compression Ratio'])
        ax5.tick_params(axis='x', rotation=45)

        # 6. Compression Effectiveness Scatter
        ax6 = axes[1, 2]
        for method in analysis_df['method'].unique():
            method_data = analysis_df[analysis_df['method'] == method]
            ax6.scatter(method_data['actual_compression_ratio'], method_data['score_preservation'],
                       label=method, alpha=0.7, s=100)

        ax6.set_xlabel('Compression Ratio (%)')
        ax6.set_ylabel('Score Preservation (%)')
        ax6.set_title('Compression vs Preservation Trade-off')
        ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / 'comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"📊 Visualizations saved to: {output_path / 'comprehensive_analysis.png'}")

    def generate_detailed_report(self, analysis_df: pd.DataFrame, results: Dict[str, pd.DataFrame],
                               output_dir: str = "analysis_output") -> str:
        """Generate a detailed markdown report."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_path / f"benchmark_analysis_report_{timestamp}.md"

        with open(report_file, 'w') as f:
            f.write("# 🚀 Prompt Compression Benchmark Analysis Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")

            if analysis_df.empty:
                f.write("⚠️  **No data available for analysis**\n\n")
                return str(report_file)

            # Executive Summary
            f.write("## 📊 Executive Summary\n\n")
            f.write(self._generate_executive_summary(analysis_df))
            f.write("\n---\n\n")

            # Task-by-Task Analysis
            for task in self.tasks:
                task_data = analysis_df[analysis_df['task'] == task.title()]
                if not task_data.empty:
                    f.write(f"## 🎯 {task.title()} Task Analysis\n\n")
                    f.write(self._generate_task_analysis(task_data, task, results.get(task)))
                    f.write("\n---\n\n")

            # Method Comparison
            f.write("## 🔄 Method Comparison\n\n")
            f.write(self._generate_method_comparison(analysis_df))
            f.write("\n---\n\n")

            # Recommendations
            f.write("## 💡 Recommendations\n\n")
            f.write(self._generate_recommendations(analysis_df))
            f.write("\n---\n\n")

            # Raw Data Summary
            f.write("## 📋 Raw Data Summary\n\n")
            f.write("### Sample Sizes:\n")
            for task, df in results.items():
                f.write(f"- **{task.title()}:** {len(df)} samples\n")
            f.write("\n")

            f.write("### Detailed Metrics Table:\n\n")
            # Format the dataframe for markdown
            try:
                f.write(analysis_df.to_markdown(index=False))
            except ImportError:
                # Fallback to CSV-style format if tabulate is not available
                f.write("Task,Method,Baseline Accuracy,Compressed Accuracy,Score Preservation,Compression Ratio,Efficiency Score\n")
                for _, row in analysis_df.iterrows():
                    f.write(f"{row['task']},{row['method']},{row['baseline_accuracy']:.2f},{row['compressed_accuracy']:.2f},{row['score_preservation']:.2f},{row['actual_compression_ratio']:.2f},{row['efficiency_score']:.2f}\n")
            f.write("\n\n")

        print(f"📄 Detailed report saved to: {report_file}")
        return str(report_file)

    def _generate_executive_summary(self, df: pd.DataFrame) -> str:
        """Generate executive summary section."""
        summary = []

        # Overall statistics
        avg_preservation = df['score_preservation'].mean()
        avg_compression = df['actual_compression_ratio'].mean()
        avg_efficiency = df['efficiency_score'].mean()

        summary.append(f"- **Average Score Preservation:** {avg_preservation:.1f}%")
        summary.append(f"- **Average Compression Ratio:** {avg_compression:.1f}%")
        summary.append(f"- **Average Efficiency Score:** {avg_efficiency:.1f}")
        # Best performing method
        best_method = df.groupby('method')['efficiency_score'].mean().idxmax()
        summary.append(f"- **Best Overall Method:** {best_method}")

        # Task with highest preservation
        best_task = df.groupby('task')['score_preservation'].mean().idxmax()
        summary.append(f"- **Best Task Performance:** {best_task}")

        return "\n".join(summary)

    def _generate_task_analysis(self, task_df: pd.DataFrame, task_name: str,
                              raw_df: Optional[pd.DataFrame]) -> str:
        """Generate detailed analysis for a specific task."""
        analysis = []

        # Method rankings for this task
        rankings = task_df.sort_values('efficiency_score', ascending=False)

        analysis.append("### Method Rankings:\n")
        for i, (_, row) in enumerate(rankings.iterrows(), 1):
            analysis.append(f"{i}. **{row['method']}**: Efficiency={row['efficiency_score']:.1f}, Preservation={row['score_preservation']:.1f}%, Ratio={row['actual_compression_ratio']:.1f}%")

        analysis.append("\n### Key Metrics:\n")
        for _, row in task_df.iterrows():
            analysis.append(f"**{row['method']}:**")
            analysis.append(f"      - Score Preservation: {row['score_preservation']:.1f}%")
            analysis.append(f"      - Compression Ratio: {row['actual_compression_ratio']:.1f}%")
            analysis.append(f"      - Latency Overhead: {row['latency_overhead_percent']:.1f}%")
            analysis.append(f"      - Efficiency Score: {row['efficiency_score']:.1f}")

        if raw_df is not None:
            analysis.append(f"### Dataset Info:\n")
            analysis.append(f"- **Samples:** {len(raw_df)}")
            analysis.append(f"- **Baseline Accuracy:** {raw_df['baseline_score'].mean() * 100:.2f}%")
            analysis.append(f"- **Average Baseline Latency:** {raw_df['baseline_latency'].mean():.2f}s")
        return "\n".join(analysis)

    def _generate_method_comparison(self, df: pd.DataFrame) -> str:
        """Generate method comparison section."""
        comparison = []

        # Average performance by method
        method_avg = df.groupby('method')[['score_preservation', 'actual_compression_ratio',
                                         'latency_overhead_percent', 'efficiency_score']].mean()

        comparison.append("### Average Performance Across All Tasks:\n\n")
        try:
            comparison.append(method_avg.to_markdown(floatfmt=".2f"))
        except ImportError:
            # Fallback to simple text format if tabulate is not available
            comparison.append("Method | Score Preservation | Compression Ratio | Latency Overhead | Efficiency Score")
            comparison.append("-------|-------------------|------------------|------------------|------------------")
            for method, row in method_avg.iterrows():
                comparison.append(f"{method} | {row['score_preservation']:.2f} | {row['actual_compression_ratio']:.2f} | {row['latency_overhead_percent']:.2f} | {row['efficiency_score']:.2f}")
        comparison.append("\n")

        # Best method for each metric
        metrics = {
            'Score Preservation': 'score_preservation',
            'Compression Ratio': 'actual_compression_ratio',
            'Efficiency Score': 'efficiency_score'
        }

        comparison.append("### Best Method by Metric:\n")
        for metric_name, column in metrics.items():
            best_method = df.groupby('method')[column].mean().idxmax()
            best_value = df.groupby('method')[column].mean().max()
            comparison.append(f"- **{metric_name}:** {best_method} ({best_value:.2f})")

        return "\n".join(comparison)

    def _generate_recommendations(self, df: pd.DataFrame) -> str:
        """Generate recommendations based on analysis."""
        recommendations = []

        # Overall best method
        best_overall = df.groupby('method')['efficiency_score'].mean().idxmax()
        recommendations.append(f"1. **Primary Recommendation:** Use **{best_overall}** for general-purpose compression "
                              "as it provides the best balance of score preservation, compression ratio, and efficiency.")

        # Task-specific recommendations
        task_best = df.groupby(['task', 'method'])['efficiency_score'].max().groupby('task').idxmax()
        recommendations.append("\n2. **Task-Specific Recommendations:**")
        for task, (task_name, method) in task_best.items():
            recommendations.append(f"   - **{task_name}:** {method}")

        # Performance thresholds
        high_preservation = df[df['score_preservation'] > 90]
        if not high_preservation.empty:
            recommendations.append("\n3. **High-Performance Combinations:**")
            for _, row in high_preservation.iterrows():
                recommendations.append(f"   - {row['method']} on {row['task']} ({row['score_preservation']:.1f}% preservation)")

        return "\n".join(recommendations)

    def run_complete_analysis(self, output_dir: str = "analysis_output") -> Dict[str, Any]:
        """Run complete analysis pipeline."""
        print("🚀 Starting Comprehensive Benchmark Analysis")
        print("=" * 60)

        # Load data
        print("\n📥 Loading benchmark results...")
        results = self.load_latest_results()

        if not results:
            print("❌ No results found to analyze")
            return {}

        # Generate analysis
        print("\n🔍 Generating comprehensive analysis...")
        analysis_df = self.generate_comparative_analysis(results)

        # Create visualizations
        print("\n📊 Creating visualizations...")
        self.create_visualizations(analysis_df, output_dir)

        # Generate report
        print("\n📄 Generating detailed report...")
        report_path = self.generate_detailed_report(analysis_df, results, output_dir)

        print("\n✅ Analysis Complete!")
        print(f"   📊 Visualizations: {output_dir}/comprehensive_analysis.png")
        print(f"   📄 Report: {report_path}")

        return {
            'analysis_dataframe': analysis_df,
            'results_dataframes': results,
            'report_path': report_path,
            'visualization_path': f"{output_dir}/comprehensive_analysis.png"
        }


def main():
    """Main function to run the analysis."""
    analyzer = BenchmarkAnalyzer()

    # Run complete analysis
    results = analyzer.run_complete_analysis()

    if results:
        print("\n🎉 Analysis Summary:")
        print(f"   📊 Analyzed {len(results['results_dataframes'])} tasks")
        print(f"   📈 Generated {len(results['analysis_dataframe'])} method-task combinations")
        print(f"   📄 Report available at: {results['report_path']}")


if __name__ == "__main__":
    main()
