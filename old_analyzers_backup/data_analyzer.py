import statistics


class DataAnalyzer:
    """
    Responsible for analyzing benchmark data and generating statistics.
    Single Responsibility: Data analysis and statistical calculations.
    """

    def __init__(self, sample_data, compression_methods):
        self.sample_data = sample_data
        self.compression_methods = compression_methods

    def calculate_method_summary(self, method):
        """Calculate summary statistics for a single compression method."""
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

    def calculate_comparative_analysis(self):
        """Calculate comparative analysis across all compression methods."""
        if len(self.compression_methods) < 2:
            return {"note": "Need at least 2 compression methods for comparative analysis"}

        methods = list(self.compression_methods)
        comparison = {}

        # Compare compression effectiveness
        compression_ratios = {}
        compression_efficiencies = {}
        score_preservations = {}

        for method in methods:
            stats = self.calculate_method_summary(method)
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

    def calculate_task_summary(self, task):
        """Calculate summary statistics for a single task type."""
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
