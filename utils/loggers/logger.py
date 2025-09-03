import os
from datetime import datetime
import pandas as pd
import glob
import re

from utils.data.data_collector import DataCollector
from utils.data.data_enhancer import DataEnhancer
from utils.data.file_writer import FileWriter
from utils.analysis.analyzer_adapter import DataAnalyzer


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

            # Get tasks from the analyzer's DataFrame
            tasks = []
            if hasattr(analyzer, 'df') and not analyzer.df.empty:
                tasks = list(analyzer.df['task'].unique())
            else:
                # Fallback to sample_data if DataFrame is not available or empty
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

            comp = summary.get("comparative_analysis", {})
            if isinstance(comp, dict) and comp.get("best_score_preservation"):
                print(f"\n🏆 Best Score Preservation: {comp['best_score_preservation'][0]} ({comp['best_score_preservation'][1]:.1%})")
                if comp.get("best_compression_ratio"):
                    print(f"🏆 Best Compression Ratio: {comp['best_compression_ratio'][0]} ({comp['best_compression_ratio'][1]:.1%})")

            print(f"{'='*60}")

        except Exception as e:
            print(f"Error printing summary report: {e}")
            print(f"Raw summary data: {summary}")


class BenchmarkLogger:
    """
    Refactored benchmark logger using composition and focused classes.
    """

    def _get_next_file_number(self, task_prefix):
        """Get the next available file number for the given task prefix."""
        pattern = os.path.join(self.results_dir, f"bench_{task_prefix}_*.csv")
        existing_files = glob.glob(pattern)
        
        if not existing_files:
            return 0
        
        # Extract numbers from existing filenames
        numbers = []
        for file_path in existing_files:
            filename = os.path.basename(file_path)
            # Match pattern: bench_{task}_NNN.csv
            match = re.match(rf"bench_{task_prefix}_(\d+)\.csv", filename)
            if match:
                numbers.append(int(match.group(1)))
        
        # Return the next number (highest + 1, or 0 if no valid numbers found)
        return max(numbers) + 1 if numbers else 0

    def __init__(self, log_dir="logs", results_dir="results", task_name=None, compression_methods=None):
        self.log_dir = log_dir
        self.results_dir = results_dir
        
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        if task_name and compression_methods:
            # Use sequential numbering: bench_{task}_000.csv, bench_{task}_001.csv, etc.
            task_prefix = task_name[:3]
            next_number = self._get_next_file_number(task_prefix)
            base_name = f"bench_{task_prefix}_{next_number:03d}"
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
            base_name = f"benchmark_{timestamp}"

        self.data_collector = DataCollector()
        self.data_enhancer = DataEnhancer()
        self.file_writer = FileWriter(results_dir, base_name)
        self.report_generator = SimpleReportGenerator(log_dir, base_name)

        self.task_name = task_name
        self.log_file = self.file_writer.log_file
        self.summary_file = self.file_writer.summary_file

        print(f"Sample-centric logging initialized")
        print(f"Results will be saved to {self.results_dir}")
        print(f"Analysis reports will be saved to {self.results_dir}")
        print(f"📁 Output files: {base_name}.*")

    def log_result(self, result_data):
        """Log a result for a sample."""
        sample_id = result_data['sample_id']

        for compression_data in result_data.get('compression_methods', []):
            self.data_collector.compression_methods.add(compression_data['method'])

        if sample_id not in self.data_collector.sample_data:
            self.data_collector.sample_data[sample_id] = {
                'task': result_data.get('task', ''),
                'llm_provider': result_data.get('llm_provider', ''),
                'llm_model': result_data.get('llm_model', ''),
                'original_prompt': result_data.get('original_prompt', ''),
                'ground_truth_answer': result_data.get('ground_truth_answer', ''),
                'baseline_output': result_data.get('baseline_output', ''),
                'baseline_extracted_answer': result_data.get('baseline_extracted_answer', ''),
                'baseline_score': result_data.get('baseline_score', 0),
                'baseline_latency': result_data.get('baseline_latency', 0),
                'methods': {}
            }

        for compression_data in result_data.get('compression_methods', []):
            method = compression_data['method']
            method_result_data = {
                'compression_method': method,
                'original_prompt': result_data.get('original_prompt', ''),
                'compressed_prompt': compression_data.get('compressed_prompt', ''),
                'original_prompt_output': result_data.get('baseline_output', ''),
                'compressed_prompt_output': compression_data.get('compressed_prompt_output', ''),
                'baseline_score': result_data.get('baseline_score', 0),
                'compressed_score': compression_data.get('compressed_score', 0),
                'baseline_latency': result_data.get('baseline_latency', 0),
                'compressed_latency': compression_data.get('compressed_latency', 0),
                'answers_match': compression_data.get('answers_match', False),
                'target_compression_ratio': result_data.get('target_compression_ratio', 0.8)
            }
            enhanced_data = self.data_enhancer.enhance_result_data(method_result_data)
            method_data = {
                'compressed_prompt': enhanced_data.get('compressed_prompt', ''),
                'compressed_output': enhanced_data.get('compressed_prompt_output', ''),
                'compressed_extracted_answer': compression_data.get('compressed_extracted_answer', ''),
                'compressed_score': enhanced_data.get('compressed_score', 0),
                'compressed_latency': enhanced_data.get('compressed_latency', 0),
                'answers_match': compression_data.get('answers_match', False),
                'target_compression_ratio': method_result_data.get('target_compression_ratio', 0.8),
                **enhanced_data
            }
            self.data_collector.sample_data[sample_id]['methods'][method] = method_data

    def finalize_and_save(self):
        """Finalize logging and save CSV."""
        try:
            csv_file_path = self.file_writer.write_csv(
                self.data_collector.sample_data,
                self.data_collector.compression_methods
            )
            print(f"✅ CSV file saved: {csv_file_path}")
            return csv_file_path
        except Exception as e:
            print(f"Error writing CSV file: {e}")
            return None

    def generate_summary_report(self):
        """Generate and save a summary report."""
        try:
            analyzer = DataAnalyzer(
                self.data_collector.sample_data,
                self.data_collector.compression_methods
            )
            summary_report = self.report_generator.generate_summary_report(
                self.data_collector.sample_data,
                self.data_collector.compression_methods,
                analyzer
            )
            self.file_writer.write_summary(summary_report)
            print(f"✅ JSON summary saved: {self.file_writer.summary_file}")
            return summary_report
        except Exception as e:
            print(f"Error generating summary report: {e}")
            return None

    def export_analysis_report(self, csv_file_path: str):
        """Export a detailed analysis report."""
        try:
            # Import with proper path handling
            import sys
            import os
            current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)

            from benchmark_analyzer import BenchmarkAnalyzer
            analyzer = BenchmarkAnalyzer(results_dir=self.results_dir)
            if not os.path.exists(csv_file_path):
                print(f"❌ CSV file not found for analysis: {csv_file_path}")
                return None
            results_df = analyzer.analyze_single_file(csv_file_path)
            if results_df.empty:
                print("❌ Analysis failed - no results generated")
                return None
            raw_df = pd.read_csv(csv_file_path)
            report_path = analyzer.generate_detailed_report(results_df, {self.task_name or "unknown": raw_df}, self.results_dir)
            print(f"📄 Detailed report saved to: {report_path}")
            return report_path
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("💡 Make sure benchmark_analyzer.py is in the project root directory")
            return None
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            return None
