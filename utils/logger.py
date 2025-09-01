import os
from datetime import datetime
import pandas as pd

from .data_collector import DataCollector
from .data_enhancer import DataEnhancer
from .file_writer import FileWriter
from .analyzer_adapter import DataAnalyzer  # Updated to use new analyzer


class SimpleReportGenerator:
    """
    Simple report generator that works with the new analyzer.
    Provides the same interface as the old ReportGenerator.
    """

    def __init__(self, log_dir="logs", base_name="benchmark"):
        self.log_dir = log_dir
        self.base_name = base_name

    def generate_summary_report(self, sample_data, compression_methods, analyzer):
        """Generate comprehensive summary statistics using the new analyzer."""
        try:
            # Use the analyzer's comparative analysis method
            comparative_analysis = analyzer.calculate_comparative_analysis()

            # Get tasks from the analyzer's CSV data instead of sample_data
            tasks = []
            if hasattr(analyzer, 'temp_csv_path') and analyzer.temp_csv_path:
                try:
                    import pandas as pd
                    df = pd.read_csv(analyzer.temp_csv_path)
                    tasks = list(df['task'].unique())
                except Exception:
                    # Fallback to sample_data if CSV reading fails
                    tasks = list(set(sample['task'] for sample in sample_data.values() if sample.get('task')))

            summary = {
                "benchmark_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "total_samples": len(sample_data),
                    "compression_methods": sorted(list(compression_methods)),
                    "tasks": tasks
                },
                "task_summaries": {},
                "method_summaries": {},
                "comparative_analysis": comparative_analysis
            }

            # Generate per-task summaries
            for task in tasks:
                summary["task_summaries"][task] = analyzer.calculate_task_summary(task)

            # Generate per-method summaries
            for method in compression_methods:
                summary["method_summaries"][method] = analyzer.calculate_method_summary(method)

            return summary

        except Exception as e:
            print(f"Error generating summary report: {e}")
            import traceback
            traceback.print_exc()
            return {
                "benchmark_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "total_samples": len(sample_data),
                    "compression_methods": sorted(list(compression_methods)),
                    "tasks": []
                },
                "task_summaries": {},
                "method_summaries": {},
                "comparative_analysis": {"note": f"Error during analysis: {e}"}
            }

    def print_summary_report(self, summary):
        """Print formatted summary report to console."""
        try:
            meta = summary.get("benchmark_metadata", {})
            print(f"\n{'='*60}")
            print("BENCHMARK SUMMARY REPORT")
            print(f"{'='*60}")
            print(f"Timestamp: {meta.get('timestamp', 'Unknown')}")
            print(f"Total Samples: {meta.get('total_samples', 0)}")
            print(f"Tasks: {', '.join(meta.get('tasks', []))}")
            print(f"Compression Methods: {', '.join(meta.get('compression_methods', []))}")

            # Print comparative analysis if available
            comp = summary.get("comparative_analysis", {})
            if isinstance(comp, dict) and "best_score_preservation" in comp:
                print(f"\n🏆 Best Score Preservation: {comp['best_score_preservation'][0]} ({comp['best_score_preservation'][1]:.1%})")
                if "best_compression_ratio" in comp:
                    print(f"🏆 Best Compression Ratio: {comp['best_compression_ratio'][0]} ({comp['best_compression_ratio'][1]:.1%})")

            print(f"{'='*60}")

        except Exception as e:
            print(f"Error printing summary report: {e}")
            print(f"Raw summary data: {summary}")


class BenchmarkLogger:
    """
    Refactored benchmark logger using composition and focused classes.
    Maintains the same public API while following SOLID principles.
    """

    def __init__(self, log_dir="logs", results_dir="results", task_name=None, compression_methods=None):
        self.log_dir = log_dir
        self.results_dir = results_dir
        
        # Create the directories if they don't exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        # Create meaningful filename based on task and methods
        if task_name and compression_methods:
            methods_str = "_".join(sorted(compression_methods))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"benchmark_{task_name}_{methods_str}_{timestamp}"
        else:
            # Fallback to timestamp-based naming
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"benchmark_{timestamp}"

        # Initialize focused components using composition
        # FileWriter uses results_dir for CSV and JSON files
        self.data_collector = DataCollector()
        self.data_enhancer = DataEnhancer()
        self.file_writer = FileWriter(results_dir, base_name)
        self.report_generator = SimpleReportGenerator(log_dir, base_name)  # Add report generator back

        # Store configuration for compatibility
        self.task_name = task_name
        self.log_file = self.file_writer.log_file
        self.summary_file = self.file_writer.summary_file

        print(f"Sample-centric logging initialized")
        print(f"Results will be saved to {self.results_dir}")
        print(f"Analysis reports will be saved to {self.results_dir}")
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
            self.data_collector.compression_methods.add(method)

        # Store the result data for this sample
        if sample_id not in self.data_collector.sample_data:
            # Initialize sample data with base information
            self.data_collector.sample_data[sample_id] = {
                'task': result_data.get('task', ''),
                'llm_provider': result_data.get('llm_provider', ''),
                'llm_model': result_data.get('llm_model', ''),
                'original_prompt': result_data.get('original_prompt', ''),
                'ground_truth_answer': result_data.get('ground_truth_answer', ''),
                'baseline_output': result_data.get('baseline_output', ''),  # Fix: use baseline_output
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
                'original_prompt_output': result_data.get('baseline_output', ''),  # Fix: use baseline_output
                'compressed_prompt_output': compression_data.get('compressed_prompt_output', ''),
                'baseline_score': result_data.get('baseline_score', 0),
                'compressed_score': compression_data.get('compressed_score', 0),
                'baseline_latency': result_data.get('baseline_latency', 0),
                'compressed_latency': compression_data.get('compressed_latency', 0),
                'answers_match': compression_data.get('answers_match', False),
                'target_compression_ratio': result_data.get('target_compression_ratio', 0.8)
            }

            # Calculate enhanced metrics
            enhanced_data = self.data_enhancer.enhance_result_data(method_result_data)

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
                'latency_overhead_seconds': enhanced_data.get('latency_overhead_seconds', 0.0),
                'latency_overhead_percent': enhanced_data.get('latency_overhead_percent', 0.0),
                'quality_degradation': enhanced_data.get('quality_degradation', 0.0),
                'quality_degradation_percent': enhanced_data.get('quality_degradation_percent', 0.0)
            }

            self.data_collector.sample_data[sample_id]['methods'][method] = method_data

    def _enhance_result_data(self, result_data):
        """
        Calculate additional metrics from the basic result data.
        """
        return self.data_enhancer.enhance_result_data(result_data)

    def finalize_and_save(self):
        """
        Finalize the logging by creating the complete CSV with all method columns.
        This should be called after all results are logged.
        """
        sample_data = self.data_collector.get_sample_data()
        compression_methods = self.data_collector.get_compression_methods()

        if not sample_data:
            print("No data to save.")
            return

        # Delegate to FileWriter
        csv_file = self.file_writer.write_csv(sample_data, compression_methods)
        print(f"✅ CSV file saved: {csv_file}")

        # Also save JSON summary
        analyzer = DataAnalyzer(sample_data, compression_methods)
        summary = self.report_generator.generate_summary_report(sample_data, compression_methods, analyzer)

        # Clean up analyzer resources
        if hasattr(analyzer, 'cleanup'):
            analyzer.cleanup()

        json_file = self.file_writer.write_json_summary(summary)
        print(f"✅ JSON summary saved: {json_file}")

    def generate_summary_report(self):
        """
        Generate comprehensive summary statistics and save to JSON file.
        """
        if not self.data_collector.sample_data:
            print("No data available for summary generation.")
            return {}

        # Ensure CSV is finalized before generating summary
        if not os.path.exists(self.log_file):
            self.finalize_and_save()

        # Create analyzer with current data
        analyzer = DataAnalyzer(self.data_collector.sample_data, self.data_collector.compression_methods)

        # Generate summary using analyzer (new analyzer handles reporting internally)
        summary = analyzer.calculate_comparative_analysis()  # Use new analyzer's comprehensive analysis

        # Clean up analyzer resources
        if hasattr(analyzer, 'cleanup'):
            analyzer.cleanup()

        # Save summary to JSON file
        self.file_writer.write_json_summary(summary)

        print(f"\n{'='*60}")
        print("COMPREHENSIVE BENCHMARK SUMMARY")
        print(f"{'='*60}")
        self.report_generator.print_summary_report(summary)
        print(f"{'='*60}")

        return summary

    def _calculate_method_summary(self, method):
        """
        Calculate summary statistics for a single compression method.
        """
        analyzer = DataAnalyzer(self.data_collector.sample_data, self.data_collector.compression_methods)
        result = analyzer.calculate_method_summary(method)

        # Clean up analyzer resources
        if hasattr(analyzer, 'cleanup'):
            analyzer.cleanup()

        return result

    def _calculate_comparative_analysis(self):
        """
        Calculate comparative analysis across all compression methods.
        """
        analyzer = DataAnalyzer(self.data_collector.sample_data, self.data_collector.compression_methods)
        result = analyzer.calculate_comparative_analysis()

        # Clean up analyzer resources
        if hasattr(analyzer, 'cleanup'):
            analyzer.cleanup()

        return result

    def _calculate_task_summary(self, task):
        """
        Calculate summary statistics for a single task type.
        """
        analyzer = DataAnalyzer(self.data_collector.sample_data, self.data_collector.compression_methods)
        result = analyzer.calculate_task_summary(task)

        # Clean up analyzer resources
        if hasattr(analyzer, 'cleanup'):
            analyzer.cleanup()

        return result

    def _print_summary_report(self, summary):
        """
        Print a formatted summary report to console.
        """
        print(f"\n{'='*60}")
        print("COMPREHENSIVE BENCHMARK SUMMARY")
        print(f"{'='*60}")

        if "best_compression_ratio" in summary:
            comp = summary
            print(f"🏆 Best Compression: {comp['best_compression_ratio'][0]} ({comp['best_compression_ratio'][1]:.1%})")
            print(f"🏆 Best Efficiency: {comp['best_compression_efficiency'][0]} ({comp['best_compression_efficiency'][1]:.1%})")
            print(f"🏆 Best Preservation: {comp['best_score_preservation'][0]} ({comp['best_score_preservation'][1]:.1%})")

        print(f"{'='*60}")

    def export_analysis_report(self, output_file=None):
        """
        Export a detailed analysis report using the new analyzer.
        """
        # Use the new analyzer's comprehensive reporting
        from benchmark_analyzer import BenchmarkAnalyzer

        analyzer = BenchmarkAnalyzer(results_dir=self.results_dir)

        # Find the latest CSV file for analysis
        import os
        from pathlib import Path

        results_path = Path(self.results_dir)
        csv_files = list(results_path.glob("*.csv"))

        if not csv_files:
            print("❌ No CSV files found for analysis")
            return None

        # Get the most recent CSV file
        latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)

        # Generate comprehensive analysis
        results_df = analyzer.analyze_single_file(str(latest_csv))
        if results_df.empty:
            print("❌ Analysis failed - no results generated")
            return None

        # Load raw CSV data for detailed report generation
        raw_df = pd.read_csv(str(latest_csv))

        # Generate detailed report
        summary = {
            "benchmark_metadata": {
                "timestamp": "2025-09-01T00:00:00",
                "total_samples": len(results_df),
                "compression_methods": list(results_df['method'].str.lower()),
                "tasks": ["classification", "reasoning", "summarization"]  # Default tasks
            },
            "method_summaries": {},
            "comparative_analysis": analyzer.generate_comparative_analysis({self.task_name or "unknown": results_df})
        }

        # Convert results to old format for compatibility
        for _, row in results_df.iterrows():
            method = row['method'].lower()
            summary["method_summaries"][method] = {
                "sample_count": len(results_df),  # Use total rows as sample count
                "baseline_performance": {
                    "mean_score": row['baseline_accuracy'] / 100,
                    "std_score": 0.0,
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

        # Generate the report using new analyzer with raw data
        report_path = analyzer.generate_detailed_report(
            results_df,  # Analysis results
            {self.task_name or "unknown": raw_df},  # Raw CSV data for detailed metrics
            self.results_dir
        )

        return report_path
