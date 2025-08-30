import csv
import os
import json
from datetime import datetime
from collections import defaultdict
import statistics

class BenchmarkLogger:
    """
    A sample-centric multi-compressor benchmark logger.
    Each row represents one sample with separate columns for each compression method.
    Properly escapes CSV fields and supports any number of compressor configurations.
    """

    def __init__(self, log_dir="results", task_name=None, compression_methods=None):
        self.log_dir = log_dir
        # Create the results directory if it doesn't exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Create meaningful filename based on task and methods
        if task_name and compression_methods:
            methods_str = "_".join(sorted(compression_methods))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"benchmark_{task_name}_{methods_str}_{timestamp}"
        else:
            # Fallback to timestamp-based naming
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"benchmark_{timestamp}"

        self.log_file = os.path.join(self.log_dir, f"{base_name}.csv")
        self.summary_file = os.path.join(self.log_dir, f"{base_name}_summary.json")

        # Track compression methods and samples
        self.compression_methods = set()
        self.task_name = task_name
        self.sample_data = defaultdict(dict)  # sample_id -> method -> data

        # Base fieldnames (will be extended dynamically)
        self.base_fieldnames = [
            # Basic identifiers
            "sample_id", "task", "llm_provider", "llm_model",

            # Original data (shared across all methods)
            "original_prompt", "ground_truth_answer", "baseline_output",
            "baseline_extracted_answer", "baseline_score", "baseline_latency"
        ]

        print(f"Sample-centric logging initialized")
        print(f"Results will be saved to {self.log_dir}")
        print(f"📁 Output files: {base_name}.*")

    def log_result(self, result_data):
        """
        Log a result for a sample with multiple compression methods.
        Results are organized by sample_id with method-specific columns.
        """
        sample_id = result_data['sample_id']
        
        # Track compression methods from this sample
        for compression_data in result_data.get('compression_methods', []):
            method = compression_data['method']
            self.compression_methods.add(method)

        # Store the result data for this sample
        if sample_id not in self.sample_data:
            # Initialize sample data with base information
            self.sample_data[sample_id] = {
                'task': result_data.get('task', ''),
                'llm_provider': result_data.get('llm_provider', ''),
                'llm_model': result_data.get('llm_model', ''),
                'original_prompt': result_data.get('original_prompt', ''),
                'ground_truth_answer': result_data.get('ground_truth_answer', ''),
                'baseline_output': result_data.get('original_prompt_output', ''),
                'baseline_extracted_answer': result_data.get('baseline_extracted_answer', ''),
                'baseline_score': result_data.get('baseline_score', 0),
                'baseline_latency': result_data.get('baseline_latency', 0),
                'methods': {}
            }

        # Process each compression method for this sample
        for compression_data in result_data.get('compression_methods', []):
            method = compression_data['method']
            
            # Create enhanced data for this method
            method_result_data = {
                'compression_method': method,
                'original_prompt': result_data.get('original_prompt', ''),
                'compressed_prompt': compression_data.get('compressed_prompt', ''),
                'original_prompt_output': result_data.get('original_prompt_output', ''),
                'compressed_prompt_output': compression_data.get('compressed_prompt_output', ''),
                'baseline_score': result_data.get('baseline_score', 0),
                'compressed_score': compression_data.get('compressed_score', 0),
                'baseline_latency': result_data.get('baseline_latency', 0),
                'compressed_latency': compression_data.get('compressed_latency', 0),
                'answers_match': compression_data.get('answers_match', False),
                'target_compression_ratio': result_data.get('target_compression_ratio', 0.8)
            }
            
            # Calculate enhanced metrics
            enhanced_data = self._enhance_result_data(method_result_data)

            # Store method-specific data
            method_data = {
                'compressed_prompt': enhanced_data.get('compressed_prompt', ''),
                'compressed_output': enhanced_data.get('compressed_prompt_output', ''),
                'compressed_extracted_answer': compression_data.get('compressed_extracted_answer', ''),
                'compressed_score': enhanced_data.get('compressed_score', 0),
                'compressed_latency': enhanced_data.get('compressed_latency', 0),
                'answers_match': enhanced_data.get('answers_match', False),
                'target_compression_ratio': enhanced_data.get('target_compression_ratio', 0.8),
                'actual_compression_ratio': enhanced_data.get('actual_compression_ratio', 1.0),
                'compression_efficiency': enhanced_data.get('compression_efficiency', 0.0),
                'tokens_saved': enhanced_data.get('tokens_saved', 0),
                'score_preservation': enhanced_data.get('score_preservation', 0.0),
                'latency_overhead': enhanced_data.get('latency_overhead', 0.0),
                'quality_degradation': enhanced_data.get('quality_degradation', 0.0)
            }

            self.sample_data[sample_id]['methods'][method] = method_data

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

    def finalize_and_save(self):
        """
        Finalize the logging by creating the complete CSV with all method columns.
        This should be called after all results are logged.
        """
        if not self.sample_data:
            print("No data to save.")
            return

        # Create dynamic fieldnames based on compression methods
        self.fieldnames = self.base_fieldnames.copy()

        # Sort methods for consistent column ordering
        sorted_methods = sorted(self.compression_methods)

        # Add columns for each compression method
        for method in sorted_methods:
            self.fieldnames.extend([
                f"{method}_compressed_prompt",
                f"{method}_compressed_output",
                f"{method}_compressed_extracted_answer",
                f"{method}_compressed_score",
                f"{method}_compressed_latency",
                f"{method}_answers_match",
                f"{method}_target_ratio",
                f"{method}_actual_ratio",
                f"{method}_compression_efficiency",
                f"{method}_tokens_saved",
                f"{method}_score_preservation",
                f"{method}_latency_overhead",
                f"{method}_quality_degradation"
            ])

        # Write the CSV file with proper escaping
        with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames,
                                  quoting=csv.QUOTE_ALL,
                                  escapechar='\\',
                                  doublequote=True)
            writer.writeheader()

            # Write each sample as a row
            for sample_id in sorted(self.sample_data.keys()):
                sample = self.sample_data[sample_id]
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
                        row.update({
                            f"{method}_compressed_prompt": '',
                            f"{method}_compressed_output": '',
                            f"{method}_compressed_extracted_answer": '',
                            f"{method}_compressed_score": '',
                            f"{method}_compressed_latency": '',
                            f"{method}_answers_match": '',
                            f"{method}_target_ratio": '',
                            f"{method}_actual_ratio": '',
                            f"{method}_compression_efficiency": '',
                            f"{method}_tokens_saved": '',
                            f"{method}_score_preservation": '',
                            f"{method}_latency_overhead": '',
                            f"{method}_quality_degradation": ''
                        })

                writer.writerow(row)

        print(f"✅ CSV file saved with {len(self.sample_data)} samples and {len(sorted_methods)} compression methods")
        print(f"   Methods: {', '.join(sorted_methods)}")
        print(f"   Total columns: {len(self.fieldnames)}")
        print(f"   File: {self.log_file}")

    def generate_summary_report(self):
        """
        Generate comprehensive summary statistics and save to JSON file.
        """
        if not self.sample_data:
            print("No data available for summary generation.")
            return {}

        # Ensure CSV is finalized before generating summary
        if not os.path.exists(self.log_file):
            self.finalize_and_save()

        summary = {
            "benchmark_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_samples": len(self.sample_data),
                "compression_methods": sorted(list(self.compression_methods)),
                "tasks": list(set(sample['task'] for sample in self.sample_data.values() if sample['task']))
            },
            "task_summaries": {},
            "method_summaries": {},
            "comparative_analysis": {}
        }

        # Generate per-task summaries
        for task in summary["benchmark_metadata"]["tasks"]:
            summary["task_summaries"][task] = self._calculate_task_summary(task)

        # Generate per-method summaries
        for method in self.compression_methods:
            summary["method_summaries"][method] = self._calculate_method_summary(method)

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

    def _calculate_method_summary(self, method):
        """
        Calculate summary statistics for a single compression method.
        """
        method_results = []
        for sample in self.sample_data.values():
            if method in sample['methods']:
                method_results.append(sample['methods'][method])

        if not method_results:
            return {}

        # Extract metrics
        compressed_scores = [r['compressed_score'] for r in method_results]
        compression_ratios = [r['actual_compression_ratio'] for r in method_results]
        compression_efficiencies = [r['compression_efficiency'] for r in method_results]
        score_preservations = [r['score_preservation'] for r in method_results]
        latency_overheads = [r['latency_overhead'] for r in method_results]
        quality_degradations = [r['quality_degradation'] for r in method_results]
        answers_match = [1 if r['answers_match'] else 0 for r in method_results]

        # Get baseline scores for this method
        baseline_scores = []
        baseline_latencies = []
        for sample in self.sample_data.values():
            if method in sample['methods']:
                baseline_scores.append(sample['baseline_score'])
                baseline_latencies.append(sample['baseline_latency'])

        return {
            "sample_count": len(method_results),
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
                "mean_tokens_saved": round(statistics.mean([r['tokens_saved'] for r in method_results]), 2)
            },
            "quality_metrics": {
                "mean_score_preservation": round(statistics.mean(score_preservations), 4),
                "mean_quality_degradation": round(statistics.mean(quality_degradations), 4),
                "answer_consistency_rate": round(statistics.mean(answers_match), 4)
            },
            "latency_metrics": {
                "mean_baseline_latency": round(statistics.mean(baseline_latencies), 2),
                "mean_compressed_latency": round(statistics.mean([r['compressed_latency'] for r in method_results]), 2),
                "mean_latency_overhead": round(statistics.mean(latency_overheads), 4)
            }
        }

    def _calculate_comparative_analysis(self):
        """
        Calculate comparative analysis across all compression methods.
        """
        if len(self.compression_methods) < 2:
            return {"note": "Need at least 2 compression methods for comparative analysis"}

        methods = list(self.compression_methods)
        comparison = {}

        # Compare compression effectiveness
        compression_ratios = {}
        compression_efficiencies = {}
        score_preservations = {}

        for method in methods:
            stats = self._calculate_method_summary(method)
            compression_ratios[method] = stats['compression_metrics']['mean_compression_ratio']
            compression_efficiencies[method] = stats['compression_metrics']['mean_compression_efficiency']
            score_preservations[method] = stats['quality_metrics']['mean_score_preservation']

        # Find best performers
        comparison["best_compression_ratio"] = min(compression_ratios.items(), key=lambda x: x[1])
        comparison["best_compression_efficiency"] = max(compression_efficiencies.items(), key=lambda x: x[1])
        comparison["best_score_preservation"] = max(score_preservations.items(), key=lambda x: x[1])

        # Calculate rankings
        comparison["compression_ratio_ranking"] = sorted(compression_ratios.items(), key=lambda x: x[1])
        comparison["compression_efficiency_ranking"] = sorted(compression_efficiencies.items(), key=lambda x: x[1], reverse=True)
        comparison["score_preservation_ranking"] = sorted(score_preservations.items(), key=lambda x: x[1], reverse=True)

        return comparison

    def _calculate_task_summary(self, task):
        """
        Calculate summary statistics for a single task type.
        """
        task_samples = [sample for sample in self.sample_data.values() if sample['task'] == task]

        if not task_samples:
            return {}

        # Extract metrics across all methods for this task
        all_baseline_scores = []
        all_compressed_scores = []
        all_compression_ratios = []
        all_latencies = []

        for sample in task_samples:
            all_baseline_scores.append(sample['baseline_score'])
            all_latencies.append(sample['baseline_latency'])

            for method_data in sample['methods'].values():
                all_compressed_scores.append(method_data['compressed_score'])
                all_compression_ratios.append(method_data['actual_compression_ratio'])
                all_latencies.append(method_data['compressed_latency'])

        return {
            "sample_count": len(task_samples),
            "methods_tested": len(self.compression_methods),
            "baseline_performance": {
                "mean_score": round(statistics.mean(all_baseline_scores), 4),
                "std_score": round(statistics.stdev(all_baseline_scores) if len(all_baseline_scores) > 1 else 0, 4),
                "accuracy_rate": round(statistics.mean(all_baseline_scores), 4)
            },
            "overall_compressed_performance": {
                "mean_score": round(statistics.mean(all_compressed_scores), 4),
                "std_score": round(statistics.stdev(all_compressed_scores) if len(all_compressed_scores) > 1 else 0, 4),
                "mean_compression_ratio": round(statistics.mean(all_compression_ratios), 4)
            },
            "latency_analysis": {
                "mean_latency": round(statistics.mean(all_latencies), 2),
                "std_latency": round(statistics.stdev(all_latencies) if len(all_latencies) > 1 else 0, 2)
            }
        }

    def _print_summary_report(self, summary):
        """
        Print a formatted summary report to console.
        """
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

    def export_analysis_report(self, output_file=None):
        """
        Export a detailed analysis report in markdown format.
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if self.task_name and self.compression_methods:
                methods_str = "_".join(sorted(self.compression_methods))
                base_name = f"analysis_{self.task_name}_{methods_str}_{timestamp}"
            else:
                base_name = f"analysis_report_{timestamp}"
            output_file = os.path.join(self.log_dir, f"{base_name}.md")

        summary = self.generate_summary_report()
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
            f.write(f"- **CSV Results:** `{self.log_file}`\n")
            f.write(f"- **JSON Summary:** `{self.summary_file}`\n")
            f.write(f"- **Analysis Report:** `{output_file}`\n")

        print(f"📊 Detailed analysis report exported to {output_file}")
        return output_file
