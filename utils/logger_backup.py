import csv
import os
import json
from datetime import datetime
from collections import defaultdict
import statistics

class BenchmarkLogger:
    """
    An advanced logging system for multi-compressor benchmark results.
    Handles CSV logging, summary statistics, and comparative analysis.
    Properly escapes CSV fields and supports flexible compressor configurations.
    """

    def __init__(self, log_dir="results"):
        self.log_dir = log_dir
        # Create the results directory if it doesn't exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Create a unique filename based on the current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"benchmark_{timestamp}.csv")
        self.summary_file = os.path.join(self.log_dir, f"summary_{timestamp}.json")

        # Enhanced fieldnames for more detailed logging
        self.fieldnames = [
            # Basic identifiers
            "sample_id", "llm_provider", "llm_model",
            "compression_method", "target_compression_ratio",

            # Prompt data (will be properly escaped)
            "original_prompt", "compressed_prompt",
            "original_prompt_length", "compressed_prompt_length",
            "actual_compression_ratio", "tokens_saved",

            # Ground truth and responses (will be properly escaped)
            "ground_truth_answer", "original_prompt_output", "compressed_prompt_output",
            "baseline_extracted_answer", "compressed_extracted_answer",

            # Performance metrics
            "baseline_score", "compressed_score", "score_preservation",
            "answers_match", "baseline_latency", "compressed_latency",

            # Advanced metrics
            "compression_efficiency", "latency_overhead", "quality_degradation"
        ]

        # Initialize results storage for summary statistics
        self.results_data = defaultdict(list)

        # Write the header row to the new CSV file with proper quoting
        with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames,
                                  quoting=csv.QUOTE_ALL,  # Quote all fields to handle commas and newlines
                                  escapechar='\\',
                                  doublequote=True)
            writer.writeheader()

        print(f"Logging detailed results to {self.log_file}")
        print(f"Summary statistics will be saved to {self.summary_file}")

    def log_result(self, result_data):
        """
        Appends a single result dictionary as a new row to the CSV file.
        Enhanced to calculate additional metrics automatically and properly escape CSV fields.
        """
        # Calculate additional metrics
        enhanced_data = self._enhance_result_data(result_data)

        # Store for summary statistics
        method = enhanced_data['compression_method']
        self.results_data[method].append(enhanced_data)

        # Write to CSV with proper escaping
        with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames,
                                  quoting=csv.QUOTE_ALL,  # Quote all fields to handle commas and newlines
                                  escapechar='\\',
                                  doublequote=True)
            writer.writerow(enhanced_data)

    def _enhance_result_data(self, result_data):
        """
        Calculate additional metrics from the basic result data.
        """
        enhanced = result_data.copy()

        # Calculate prompt lengths and compression metrics
        original_length = len(result_data.get('original_prompt', ''))
        compressed_length = len(result_data.get('compressed_prompt', ''))

        enhanced['original_prompt_length'] = original_length
        enhanced['compressed_prompt_length'] = compressed_length

        if original_length > 0:
            actual_ratio = compressed_length / original_length
            enhanced['actual_compression_ratio'] = round(actual_ratio, 4)
            enhanced['tokens_saved'] = original_length - compressed_length

            # Compression efficiency (how close we got to target)
            target_ratio = result_data.get('target_compression_ratio', 0.8)
            efficiency = 1 - abs(actual_ratio - target_ratio) / target_ratio
            enhanced['compression_efficiency'] = round(max(0, efficiency), 4)
        else:
            enhanced['actual_compression_ratio'] = 1.0
            enhanced['tokens_saved'] = 0
            enhanced['compression_efficiency'] = 0.0

        # Calculate score preservation
        baseline_score = result_data.get('baseline_score', 0)
        compressed_score = result_data.get('compressed_score', 0)
        if baseline_score > 0:
            enhanced['score_preservation'] = round(compressed_score / baseline_score, 4)
        else:
            enhanced['score_preservation'] = 0.0

        # Calculate latency overhead
        baseline_latency = result_data.get('baseline_latency', 0)
        compressed_latency = result_data.get('compressed_latency', 0)
        if baseline_latency > 0:
            overhead = (compressed_latency - baseline_latency) / baseline_latency
            enhanced['latency_overhead'] = round(overhead, 4)
        else:
            enhanced['latency_overhead'] = 0.0

        # Calculate quality degradation
        if baseline_score > 0:
            degradation = (baseline_score - compressed_score) / baseline_score
            enhanced['quality_degradation'] = round(max(0, degradation), 4)
        else:
            enhanced['quality_degradation'] = 0.0

        return enhanced

    def generate_summary_report(self):
        """
        Generate comprehensive summary statistics and save to JSON file.
        """
        summary = {
            "benchmark_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_samples": sum(len(results) for results in self.results_data.values()),
                "compression_methods": list(self.results_data.keys())
            },
            "method_summaries": {},
            "comparative_analysis": {}
        }

        # Generate per-method summaries
        for method, results in self.results_data.items():
            summary["method_summaries"][method] = self._calculate_method_summary(results)

        # Generate comparative analysis
        summary["comparative_analysis"] = self._calculate_comparative_analysis()

        # Save summary to JSON file
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*60}")
        print("COMPREHENSIVE BENCHMARK SUMMARY")
        print(f"{'='*60}")
        self._print_summary_report(summary)
        print(f"{'='*60}")

        return summary

    def _calculate_method_summary(self, results):
        """
        Calculate summary statistics for a single compression method.
        """
        if not results:
            return {}

        # Extract metrics
        baseline_scores = [r['baseline_score'] for r in results]
        compressed_scores = [r['compressed_score'] for r in results]
        compression_ratios = [r['actual_compression_ratio'] for r in results]
        compression_efficiencies = [r['compression_efficiency'] for r in results]
        score_preservations = [r['score_preservation'] for r in results]
        latency_overheads = [r['latency_overhead'] for r in results]
        quality_degradations = [r['quality_degradation'] for r in results]
        answers_match = [1 if r['answers_match'] else 0 for r in results]

        return {
            "sample_count": len(results),
            "baseline_performance": {
                "mean_score": round(statistics.mean(baseline_scores), 4),
                "std_score": round(statistics.stdev(baseline_scores) if len(baseline_scores) > 1 else 0, 4),
                "accuracy_rate": round(statistics.mean(baseline_scores), 4)
            },
            "compressed_performance": {
                "mean_score": round(statistics.mean(compressed_scores), 4),
                "std_score": round(statistics.stdev(compressed_scores) if len(compressed_scores) > 1 else 0, 4),
                "accuracy_rate": round(statistics.mean(compressed_scores), 4)
            },
            "compression_metrics": {
                "mean_compression_ratio": round(statistics.mean(compression_ratios), 4),
                "std_compression_ratio": round(statistics.stdev(compression_ratios) if len(compression_ratios) > 1 else 0, 4),
                "mean_compression_efficiency": round(statistics.mean(compression_efficiencies), 4),
                "mean_tokens_saved": round(statistics.mean([r['tokens_saved'] for r in results]), 2)
            },
            "quality_metrics": {
                "mean_score_preservation": round(statistics.mean(score_preservations), 4),
                "mean_quality_degradation": round(statistics.mean(quality_degradations), 4),
                "answer_consistency_rate": round(statistics.mean(answers_match), 4)
            },
            "latency_metrics": {
                "mean_baseline_latency": round(statistics.mean([r['baseline_latency'] for r in results]), 2),
                "mean_compressed_latency": round(statistics.mean([r['compressed_latency'] for r in results]), 2),
                "mean_latency_overhead": round(statistics.mean(latency_overheads), 4)
            }
        }

    def _calculate_comparative_analysis(self):
        """
        Calculate comparative analysis across all compression methods.
        """
        if len(self.results_data) < 2:
            return {"note": "Need at least 2 compression methods for comparative analysis"}

        methods = list(self.results_data.keys())
        comparison = {}

        # Compare compression effectiveness
        compression_ratios = {}
        compression_efficiencies = {}
        score_preservations = {}

        for method in methods:
            results = self.results_data[method]
            compression_ratios[method] = statistics.mean([r['actual_compression_ratio'] for r in results])
            compression_efficiencies[method] = statistics.mean([r['compression_efficiency'] for r in results])
            score_preservations[method] = statistics.mean([r['score_preservation'] for r in results])

        # Find best performers
        comparison["best_compression_ratio"] = min(compression_ratios.items(), key=lambda x: x[1])
        comparison["best_compression_efficiency"] = max(compression_efficiencies.items(), key=lambda x: x[1])
        comparison["best_score_preservation"] = max(score_preservations.items(), key=lambda x: x[1])

        # Calculate rankings
        comparison["compression_ratio_ranking"] = sorted(compression_ratios.items(), key=lambda x: x[1])
        comparison["compression_efficiency_ranking"] = sorted(compression_efficiencies.items(), key=lambda x: x[1], reverse=True)
        comparison["score_preservation_ranking"] = sorted(score_preservations.items(), key=lambda x: x[1], reverse=True)

        return comparison

    def export_analysis_report(self, output_file=None):
        """
        Export a detailed analysis report in markdown format.
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.log_dir, f"analysis_report_{timestamp}.md")

        summary = self.generate_summary_report()
        meta = summary["benchmark_metadata"]

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Prompt Compression Benchmark Analysis Report\n\n")
            f.write(f"**Generated:** {meta['timestamp']}\n")
            f.write(f"**Total Samples:** {meta['total_samples']}\n")
            f.write(f"**Compression Methods:** {', '.join(meta['compression_methods'])}\n\n")

            f.write("## Executive Summary\n\n")
            f.write("This report provides a comprehensive analysis of multiple prompt compression methods ")
            f.write("tested on the GSM8K reasoning dataset.\n\n")

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
            f.write(f"- **CSV Results:** `{self.log_file}`\n")
            f.write(f"- **JSON Summary:** `{self.summary_file}`\n")
            f.write(f"- **Analysis Report:** `{output_file}`\n")

        print(f"📊 Detailed analysis report exported to {output_file}")
        return output_file

    def _print_summary_report(self, summary):
        """
        Print a formatted summary report to console.
        """
        meta = summary["benchmark_metadata"]
        print(f"Timestamp: {meta['timestamp']}")
        print(f"Total Samples: {meta['total_samples']}")
        print(f"Compression Methods: {', '.join(meta['compression_methods'])}")

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

    def validate_csv_integrity(self):
        """
        Validate that the CSV file has the correct number of columns and no corruption.
        Returns True if valid, False if corrupted.
        """
        try:
            with open(self.log_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f, quoting=csv.QUOTE_ALL, escapechar='\\', doublequote=True)
                expected_columns = len(self.fieldnames)

                for i, row in enumerate(reader, 1):
                    if len(row) != expected_columns:
                        print(f"❌ CSV integrity check failed at row {i}: expected {expected_columns} columns, got {len(row)}")
                        return False

            print(f"✅ CSV integrity check passed: {i} rows validated")
            return True
        except Exception as e:
            print(f"❌ CSV integrity check failed: {e}")
            return False

    def get_compression_methods_summary(self):
        """
        Get a summary of all compression methods used in the benchmark.
        """
        methods = list(self.results_data.keys())
        method_counts = {method: len(results) for method, results in self.results_data.items()}

        print(f"📊 Compression Methods Summary:")
        for method, count in method_counts.items():
            print(f"  • {method}: {count} samples")

        return methods, method_counts