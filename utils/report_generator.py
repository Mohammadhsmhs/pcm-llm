import os
from datetime import datetime


class ReportGenerator:
    """
    Responsible for generating various types of reports.
    Single Responsibility: Report generation and formatting.
    """

    def __init__(self, log_dir, base_name):
        self.log_dir = log_dir
        self.base_name = base_name

    def generate_summary_report(self, sample_data, compression_methods, analyzer):
        """Generate comprehensive summary statistics."""
        summary = {
            "benchmark_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_samples": len(sample_data),
                "compression_methods": sorted(list(compression_methods)),
                "tasks": list(set(sample['task'] for sample in sample_data.values() if sample['task']))
            },
            "task_summaries": {},
            "method_summaries": {},
            "comparative_analysis": {}
        }

        # Generate per-task summaries
        for task in summary["benchmark_metadata"]["tasks"]:
            summary["task_summaries"][task] = analyzer.calculate_task_summary(task)

        # Generate per-method summaries
        for method in compression_methods:
            summary["method_summaries"][method] = analyzer.calculate_method_summary(method)

        # Generate comparative analysis
        summary["comparative_analysis"] = analyzer.calculate_comparative_analysis()

        return summary

    def print_summary_report(self, summary):
        """Print formatted summary report to console."""
        meta = summary["benchmark_metadata"]
        print(f"Timestamp: {meta['timestamp']}")
        print(f"Total Samples: {meta['total_samples']}")
        print(f"Tasks: {', '.join(meta['tasks'])}")
        print(f"Compression Methods: {', '.join(meta['compression_methods'])}")

        print(f"\n{'='*40}")
        print("PER-TASK PERFORMANCE SUMMARY")
        print(f"{'='*40}")

        for task, stats in summary["task_summaries"].items():
            print(f"\n🎯 {task.upper()}")
            print(f"  Samples: {stats['sample_count']}")
            print(f"  Methods Tested: {stats['methods_tested']}")

            perf = stats['baseline_performance']
            print(f"  Baseline Accuracy: {perf['mean_score']:.1%} (±{perf['std_score']:.3f})")

            perf = stats['overall_compressed_performance']
            print(f"  Avg Compressed Accuracy: {perf['mean_score']:.1%} (±{perf['std_score']:.3f})")
            print(f"  Avg Compression Ratio: {perf['mean_compression_ratio']:.1%}")

            lat = stats['latency_analysis']
            print(f"  Avg Latency: {lat['mean_latency']:.2f}s (±{lat['std_latency']:.2f})")

        print(f"\n{'='*40}")
        print("PER-METHOD PERFORMANCE SUMMARY")
        print(f"{'='*40}")

        for method, stats in summary["method_summaries"].items():
            print(f"\n🔧 {method.upper()}")
            print(f"  Samples: {stats['sample_count']}")

            perf = stats['baseline_performance']
            print(f"  Baseline Accuracy: {perf['mean_score']:.1%} (±{perf['std_score']:.3f})")

            perf = stats['compressed_performance']
            print(f"  Compressed Accuracy: {perf['mean_score']:.1%} (±{perf['std_score']:.3f})")

            comp = stats['compression_metrics']
            print(f"  Compression Ratio: {comp['mean_compression_ratio']:.1%} (±{comp['std_compression_ratio']:.3f})")
            print(f"  Compression Efficiency: {comp['mean_compression_efficiency']:.1%}")
            print(f"  Avg Tokens Saved: {comp['mean_tokens_saved']:.1f}")

            qual = stats['quality_metrics']
            print(f"  Score Preservation: {qual['mean_score_preservation']:.1%}")
            print(f"  Answer Consistency: {qual['answer_consistency_rate']:.1%}")

            lat = stats['latency_metrics']
            print(f"  Latency Overhead: {lat['mean_latency_overhead']:.1%}")

        if "comparative_analysis" in summary and "best_compression_ratio" in summary["comparative_analysis"]:
            comp = summary["comparative_analysis"]
            print(f"\n{'='*40}")
            print("COMPARATIVE ANALYSIS")
            print(f"{'='*40}")

            print(f"🏆 Best Compression: {comp['best_compression_ratio'][0]} ({comp['best_compression_ratio'][1]:.1%})")
            print(f"🏆 Best Efficiency: {comp['best_compression_efficiency'][0]} ({comp['best_compression_efficiency'][1]:.1%})")
            print(f"🏆 Best Preservation: {comp['best_score_preservation'][0]} ({comp['best_score_preservation'][1]:.1%})")

            print(f"\n📊 Rankings:")
            print("  Compression Ratio (lowest first):")
            for i, (method, ratio) in enumerate(comp["compression_ratio_ranking"], 1):
                print(f"    {i}. {method}: {ratio:.1%}")

            print("  Score Preservation (highest first):")
            for i, (method, preservation) in enumerate(comp["score_preservation_ranking"], 1):
                print(f"    {i}. {method}: {preservation:.1%}")

    def export_analysis_report(self, summary, output_file=None):
        """Export detailed analysis report in markdown format."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            compression_methods = summary["benchmark_metadata"]["compression_methods"]
            if compression_methods:
                methods_str = "_".join(compression_methods)
                base_name = f"analysis_{methods_str}_{timestamp}"
            else:
                base_name = f"analysis_report_{timestamp}"
            output_file = os.path.join(self.log_dir, f"{base_name}.md")

        meta = summary["benchmark_metadata"]

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Prompt Compression Benchmark Analysis Report\n\n")
            f.write(f"**Generated:** {meta['timestamp']}\n")
            f.write(f"**Total Samples:** {meta['total_samples']}\n")
            f.write(f"**Tasks:** {', '.join(meta['tasks'])}\n")
            f.write(f"**Compression Methods:** {', '.join(meta['compression_methods'])}\n\n")

            f.write("## Executive Summary\n\n")
            f.write("This report provides a comprehensive analysis of multiple prompt compression methods ")
            f.write("tested on the GSM8K reasoning dataset.\n\n")

            f.write("## Task Performance Overview\n\n")

            for task, stats in summary["task_summaries"].items():
                f.write(f"### {task.upper()}\n\n")
                f.write("#### Performance Metrics\n")
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                f.write(f"| Samples | {stats['sample_count']} |\n")
                f.write(f"| Methods Tested | {stats['methods_tested']} |\n")
                f.write(f"| Baseline Accuracy | {stats['baseline_performance']['mean_score']:.1%} |\n")
                f.write(f"| Avg Compressed Accuracy | {stats['overall_compressed_performance']['mean_score']:.1%} |\n")
                f.write(f"| Avg Compression Ratio | {stats['overall_compressed_performance']['mean_compression_ratio']:.1%} |\n")
                f.write(f"| Avg Latency | {stats['latency_analysis']['mean_latency']:.2f}s |\n\n")

            f.write("## Detailed Method Analysis\n\n")

            for method, stats in summary["method_summaries"].items():
                f.write(f"### {method.upper()}\n\n")
                f.write("#### Performance Metrics\n")
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                f.write(f"| Samples | {stats['sample_count']} |\n")
                f.write(f"| Baseline Accuracy | {stats['baseline_performance']['mean_score']:.1%} |\n")
                f.write(f"| Compressed Accuracy | {stats['compressed_performance']['mean_score']:.1%} |\n")
                f.write(f"| Score Preservation | {stats['quality_metrics']['mean_score_preservation']:.1%} |\n")
                f.write(f"| Answer Consistency | {stats['quality_metrics']['answer_consistency_rate']:.1%} |\n")

                f.write("\n#### Compression Metrics\n")
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                f.write(f"| Mean Compression Ratio | {stats['compression_metrics']['mean_compression_ratio']:.1%} |\n")
                f.write(f"| Compression Efficiency | {stats['compression_metrics']['mean_compression_efficiency']:.1%} |\n")
                f.write(f"| Average Tokens Saved | {stats['compression_metrics']['mean_tokens_saved']:.1f} |\n")

                f.write("\n#### Latency Analysis\n")
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                f.write(f"| Baseline Latency | {stats['latency_metrics']['mean_baseline_latency']:.2f}s |\n")
                f.write(f"| Compressed Latency | {stats['latency_metrics']['mean_compressed_latency']:.2f}s |\n")
                f.write(f"| Latency Overhead | {stats['latency_metrics']['mean_latency_overhead']:.1%} |\n")

                f.write("\n#### Quality Assessment\n")
                quality_degradation = stats['quality_metrics']['mean_quality_degradation']
                if quality_degradation < 0.05:
                    assessment = "Excellent - Minimal quality loss"
                elif quality_degradation < 0.15:
                    assessment = "Good - Acceptable quality trade-off"
                elif quality_degradation < 0.30:
                    assessment = "Fair - Moderate quality impact"
                else:
                    assessment = "Poor - Significant quality degradation"

                f.write(f"**Quality Degradation:** {quality_degradation:.1%}\n")
                f.write(f"**Assessment:** {assessment}\n\n")

            if len(meta['compression_methods']) > 1:
                comp = summary.get("comparative_analysis", {})
                if "best_compression_ratio" in comp:
                    f.write("## Comparative Analysis\n\n")

                    f.write("### Rankings\n\n")
                    f.write("#### Best Compression Ratio\n")
                    f.write(f"🏆 **{comp['best_compression_ratio'][0]}**: {comp['best_compression_ratio'][1]:.1%}\n\n")

                    f.write("#### Best Compression Efficiency\n")
                    f.write(f"🏆 **{comp['best_compression_efficiency'][0]}**: {comp['best_compression_efficiency'][1]:.1%}\n\n")

                    f.write("#### Best Score Preservation\n")
                    f.write(f"🏆 **{comp['best_score_preservation'][0]}**: {comp['best_score_preservation'][1]:.1%}\n\n")

                    f.write("### Detailed Rankings\n\n")
                    f.write("#### Compression Ratio (Lowest First)\n")
                    for i, (method, ratio) in enumerate(comp["compression_ratio_ranking"], 1):
                        f.write(f"{i}. **{method}**: {ratio:.1%}\n")
                    f.write("\n")

                    f.write("#### Score Preservation (Highest First)\n")
                    for i, (method, preservation) in enumerate(comp["score_preservation_ranking"], 1):
                        f.write(f"{i}. **{method}**: {preservation:.1%}\n")
                    f.write("\n")

            f.write("## Recommendations\n\n")

            if len(meta['compression_methods']) > 1:
                comp = summary.get("comparative_analysis", {})
                if "best_score_preservation" in comp:
                    best_method = comp['best_score_preservation'][0]
                    f.write(f"1. **For Quality Preservation:** Use {best_method} - it maintains the highest ")
                    f.write("answer accuracy while compressing prompts.\n")

                if "best_compression_ratio" in comp:
                    best_method = comp['best_compression_ratio'][0]
                    f.write(f"2. **For Maximum Compression:** Use {best_method} - it achieves the ")
                    f.write("highest compression ratio.\n")

                if "best_compression_efficiency" in comp:
                    best_method = comp['best_compression_efficiency'][0]
                    f.write(f"3. **For Efficiency:** Use {best_method} - it comes closest to ")
                    f.write("the target compression ratio.\n")
            else:
                method = meta['compression_methods'][0]
                stats = summary["method_summaries"][method]
                quality_degradation = stats['quality_metrics']['mean_quality_degradation']

                if quality_degradation < 0.10:
                    f.write(f"✅ **{method.upper()}** shows excellent performance with minimal quality degradation. ")
                    f.write("Recommended for production use.\n")
                elif quality_degradation < 0.25:
                    f.write(f"⚠️ **{method.upper()}** shows acceptable performance. ")
                    f.write("Consider fine-tuning or testing with different parameters.\n")
                else:
                    f.write(f"❌ **{method.upper()}** shows significant quality degradation. ")
                    f.write("Not recommended without further optimization.\n")

            f.write("\n## Files Generated\n\n")
            f.write("- **CSV Results:** `*.csv`\n")
            f.write("- **JSON Summary:** `*_summary.json`\n")
            f.write("- **Analysis Report:** `*_analysis.md`\n")

        print(f"📊 Detailed analysis report exported to {output_file}")
        return output_file
