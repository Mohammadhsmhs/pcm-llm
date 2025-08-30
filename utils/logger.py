import os
from datetime import datetime

# Import focused components
from .data_collector import DataCollector
from .data_enhancer import DataEnhancer
from .file_writer import FileWriter
from .data_analyzer import DataAnalyzer
from .report_generator import ReportGenerator


class BenchmarkLogger:
    """
    Refactored benchmark logger using composition and focused classes.
    Maintains the same public API while following SOLID principles.
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

        # Initialize focused components using composition
        self.data_collector = DataCollector()
        self.data_enhancer = DataEnhancer()
        self.file_writer = FileWriter(log_dir, base_name)
        self.report_generator = ReportGenerator(log_dir, base_name)

        # Store configuration for compatibility
        self.task_name = task_name
        self.log_file = self.file_writer.log_file
        self.summary_file = self.file_writer.summary_file

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
                'quality_degradation': enhanced_data.get('quality_degradation', 0.0)
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

        # Generate summary using report generator
        summary = self.report_generator.generate_summary_report(
            self.data_collector.sample_data,
            self.data_collector.compression_methods,
            analyzer
        )

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
        return analyzer.calculate_method_summary(method)

    def _calculate_comparative_analysis(self):
        """
        Calculate comparative analysis across all compression methods.
        """
        analyzer = DataAnalyzer(self.data_collector.sample_data, self.data_collector.compression_methods)
        return analyzer.calculate_comparative_analysis()

    def _calculate_task_summary(self, task):
        """
        Calculate summary statistics for a single task type.
        """
        analyzer = DataAnalyzer(self.data_collector.sample_data, self.data_collector.compression_methods)
        return analyzer.calculate_task_summary(task)

    def _print_summary_report(self, summary):
        """
        Print a formatted summary report to console.
        """
        self.report_generator.print_summary_report(summary)

    def export_analysis_report(self, output_file=None):
        """
        Export a detailed analysis report in markdown format.
        """
        summary = self.generate_summary_report()
        return self.report_generator.export_analysis_report(summary, output_file)
