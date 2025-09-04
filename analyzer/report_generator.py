#!/usr/bin/env python3
"""
Report Generation Module for PCM-LLM Analysis
Handles generation of comprehensive analysis reports.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Generates comprehensive analysis reports."""
    
    def __init__(self, output_dir: str = "analysis_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def generate_comprehensive_report(self, df: pd.DataFrame, analysis_results: Dict, timestamp: str) -> str:
        """Generate a comprehensive analysis report."""
        
        report_file = self.output_dir / f"ultra_comprehensive_analysis_{timestamp}.md"
        
        with open(report_file, 'w') as f:
            self._write_header(f, df, timestamp)
            self._write_executive_summary(f, df)
            
            if 'ratio_analysis' in analysis_results:
                self._write_ratio_analysis(f, analysis_results['ratio_analysis'], df)
            
            if 'method_analysis' in analysis_results:
                self._write_method_analysis(f, analysis_results['method_analysis'], df)
            
            if 'task_analysis' in analysis_results:
                self._write_task_analysis(f, analysis_results['task_analysis'], df)
            
            if 'model_analysis' in analysis_results:
                self._write_model_analysis(f, analysis_results['model_analysis'], df)
            
            if 'correlations' in analysis_results:
                self._write_correlation_analysis(f, analysis_results['correlations'])
            
            if 'outliers_trends' in analysis_results:
                self._write_outlier_trends_analysis(f, analysis_results['outliers_trends'])
            
            self._write_recommendations(f, df, analysis_results)
            self._write_data_summary(f, df)
        
        return str(report_file)

    def _write_header(self, f, df: pd.DataFrame, timestamp: str):
        """Write report header."""
        f.write(f"# Ultra-Comprehensive PCM-LLM Analysis Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Analysis ID:** {timestamp}\n")
        f.write(f"**Total Samples:** {len(df)}\n")
        f.write(f"**Unique Ratios:** {df['actual_compression_ratio'].nunique() if 'actual_compression_ratio' in df.columns else 'N/A'}\n")
        f.write(f"**Methods:** {', '.join(df['method'].unique()) if 'method' in df.columns else 'N/A'}\n")
        f.write(f"**Tasks:** {', '.join(df['task'].unique()) if 'task' in df.columns else 'N/A'}\n")
        f.write(f"**Models:** {', '.join(df['model'].unique()) if 'model' in df.columns else 'N/A'}\n\n")
        f.write("---\n\n")

    def _write_executive_summary(self, f, df: pd.DataFrame):
        """Write executive summary section."""
        f.write("## 📋 Executive Summary\n\n")
        
        # Overall performance metrics
        avg_preservation = df['score_preservation'].mean()
        avg_accuracy = df['compressed_accuracy'].mean()
        
        f.write(f"### Key Performance Indicators\n\n")
        f.write(f"- **Average Score Preservation:** {avg_preservation:.2f}%\n")
        f.write(f"- **Average Compressed Accuracy:** {avg_accuracy:.2f}%\n")
        f.write(f"- **Performance Standard Deviation:** {df['score_preservation'].std():.2f}%\n")
        
        if 'cost_benefit_ratio' in df.columns:
            avg_cost_benefit = df['cost_benefit_ratio'].mean()
            f.write(f"- **Average Cost-Benefit Ratio:** {avg_cost_benefit:.2f}\n")
        
        f.write(f"\n### Data Overview\n\n")
        f.write(f"- **Total Samples Analyzed:** {len(df):,}\n")
        
        if 'actual_compression_ratio' in df.columns:
            f.write(f"- **Compression Ratio Range:** {df['actual_compression_ratio'].min():.3f} - {df['actual_compression_ratio'].max():.3f}\n")
            f.write(f"- **Unique Compression Ratios:** {df['actual_compression_ratio'].nunique():,}\n")
        
        if 'method' in df.columns:
            f.write(f"- **Methods Tested:** {df['method'].nunique()} ({', '.join(df['method'].unique())})\n")
        
        if 'task' in df.columns:
            f.write(f"- **Tasks Evaluated:** {df['task'].nunique()} ({', '.join(df['task'].unique())})\n")
        
        if 'model' in df.columns:
            f.write(f"- **Models Used:** {df['model'].nunique()}\n")
        
        # Performance highlights
        f.write(f"\n### Performance Highlights\n\n")
        
        if 'score_preservation' in df.columns:
            best_sample = df.loc[df['score_preservation'].idxmax()]
            worst_sample = df.loc[df['score_preservation'].idxmin()]
            
            f.write(f"- **Best Performance:** {best_sample['score_preservation']:.2f}% preservation")
            if 'method' in df.columns:
                f.write(f" (Method: {best_sample['method']}")
            if 'task' in df.columns:
                f.write(f", Task: {best_sample['task']}")
            f.write(")\n")
            
            f.write(f"- **Lowest Performance:** {worst_sample['score_preservation']:.2f}% preservation")
            if 'method' in df.columns:
                f.write(f" (Method: {worst_sample['method']}")
            if 'task' in df.columns:
                f.write(f", Task: {worst_sample['task']}")
            f.write(")\n")
        
        f.write("\n---\n\n")

    def _write_ratio_analysis(self, f, ratio_analysis: Dict, df: pd.DataFrame):
        """Write compression ratio analysis section."""
        f.write("## 📊 Compression Ratio Analysis\n\n")
        
        if 'overall_stats' in ratio_analysis and not ratio_analysis['overall_stats'].empty:
            f.write("### Performance by Compression Ratio\n\n")
            stats_df = ratio_analysis['overall_stats']
            
            f.write("| Compression Ratio | Avg Preservation | Std Dev | Sample Count |\n")
            f.write("|------------------|-----------------|---------|-------------|\n")
            
            for ratio_idx, row in stats_df.iterrows():
                preservation_mean = row[('score_preservation', 'mean')] if ('score_preservation', 'mean') in row else 0
                preservation_std = row[('score_preservation', 'std')] if ('score_preservation', 'std') in row else 0
                count = row[('score_preservation', 'count')] if ('score_preservation', 'count') in row else 0
                
                f.write(f"| {ratio_idx:.3f} | {preservation_mean:.2f}% | {preservation_std:.2f}% | {int(count)} |\n")
        
        if 'optimal_ratios' in ratio_analysis:
            f.write(f"\n### Optimal Compression Ratios\n\n")
            optimal = ratio_analysis['optimal_ratios']
            
            if 'max_cost_benefit' in optimal:
                f.write(f"- **Best Cost-Benefit Ratio:** {optimal['max_cost_benefit']:.3f}\n")
            
            if 'high_preservation' in optimal:
                f.write(f"- **Best High-Preservation Ratio:** {optimal['high_preservation']:.3f}\n")
        
        f.write("\n---\n\n")

    def _write_method_analysis(self, f, method_analysis: Dict, df: pd.DataFrame):
        """Write method comparison analysis section."""
        f.write("## 🔧 Method Analysis\n\n")
        
        if 'overall_performance' in method_analysis and not method_analysis['overall_performance'].empty:
            f.write("### Method Performance Comparison\n\n")
            perf_df = method_analysis['overall_performance']
            
            f.write("| Method | Avg Preservation | Std Dev | Sample Count | Avg Accuracy |\n")
            f.write("|--------|-----------------|---------|-------------|-------------|\n")
            
            for method, row in perf_df.iterrows():
                preservation_mean = row[('score_preservation', 'mean')] if ('score_preservation', 'mean') in row else 0
                preservation_std = row[('score_preservation', 'std')] if ('score_preservation', 'std') in row else 0
                count = row[('score_preservation', 'count')] if ('score_preservation', 'count') in row else 0
                accuracy_mean = row[('compressed_accuracy', 'mean')] if ('compressed_accuracy', 'mean') in row else 0
                
                f.write(f"| {method} | {preservation_mean:.2f}% | {preservation_std:.2f}% | {int(count)} | {accuracy_mean:.2f}% |\n")
        
        if 'consistency' in method_analysis and not method_analysis['consistency'].empty:
            f.write(f"\n### Method Consistency Analysis\n\n")
            consistency_df = method_analysis['consistency']
            
            f.write("| Method | Std Deviation | Coefficient of Variation |\n")
            f.write("|--------|--------------|-------------------------|\n")
            
            for method, row in consistency_df.iterrows():
                std_dev = row['std_dev'] if 'std_dev' in row else 0
                coeff_var = row['coeff_variation'] if 'coeff_variation' in row else 0
                f.write(f"| {method} | {std_dev:.2f}% | {coeff_var:.2f}% |\n")
        
        if 'rankings' in method_analysis:
            f.write(f"\n### Method Rankings\n\n")
            rankings = method_analysis['rankings']
            
            if 'by_preservation' in rankings:
                f.write("**By Score Preservation:**\n")
                for i, (method, score) in enumerate(rankings['by_preservation'].items(), 1):
                    f.write(f"{i}. {method}: {score:.2f}%\n")
                f.write("\n")
            
            if 'by_consistency' in rankings:
                f.write("**By Consistency (Lower is Better):**\n")
                for i, (method, std) in enumerate(rankings['by_consistency'].items(), 1):
                    f.write(f"{i}. {method}: {std:.2f}% std dev\n")
                f.write("\n")
        
        f.write("---\n\n")

    def _write_task_analysis(self, f, task_analysis: Dict, df: pd.DataFrame):
        """Write task analysis section."""
        f.write("## 📝 Task Analysis\n\n")
        
        for task, analysis in task_analysis.items():
            f.write(f"### {task.title()} Task\n\n")
            
            f.write(f"- **Sample Count:** {analysis.get('sample_count', 0)}\n")
            f.write(f"- **Unique Ratios:** {analysis.get('unique_ratios', 0)}\n")
            
            if analysis.get('best_method'):
                f.write(f"- **Best Method:** {analysis['best_method']}\n")
            
            if analysis.get('optimal_ratio'):
                f.write(f"- **Optimal Ratio:** {analysis['optimal_ratio']:.3f}\n")
            
            # Method comparison table for this task
            if 'method_comparison' in analysis and not analysis['method_comparison'].empty:
                f.write(f"\n**Method Performance for {task.title()}:**\n\n")
                method_df = analysis['method_comparison']
                
                f.write("| Method | Avg Preservation | Avg Accuracy | Sample Count |\n")
                f.write("|--------|-----------------|-------------|-------------|\n")
                
                for method, row in method_df.iterrows():
                    preservation = row[('score_preservation', 'mean')] if ('score_preservation', 'mean') in row else 0
                    accuracy = row[('compressed_accuracy', 'mean')] if ('compressed_accuracy', 'mean') in row else 0
                    count = row[('score_preservation', 'count')] if ('score_preservation', 'count') in row else 0
                    
                    f.write(f"| {method} | {preservation:.2f}% | {accuracy:.2f}% | {int(count)} |\n")
            
            f.write("\n")
        
        f.write("---\n\n")

    def _write_model_analysis(self, f, model_analysis: Dict, df: pd.DataFrame):
        """Write model analysis section."""
        f.write("## 🤖 Model Analysis\n\n")
        
        for model, analysis in model_analysis.items():
            f.write(f"### {model}\n\n")
            
            stats = analysis.get('statistics', {})
            f.write(f"- **Total Samples:** {stats.get('total_samples', 0)}\n")
            f.write(f"- **Average Score Preservation:** {stats.get('avg_score_preservation', 0):.2f}%\n")
            f.write(f"- **Average Accuracy:** {stats.get('avg_accuracy', 0):.2f}%\n")
            f.write(f"- **Performance Std Dev:** {stats.get('performance_std', 0):.2f}%\n")
            f.write(f"- **Methods Tested:** {stats.get('methods_tested', 0)}\n")
            f.write(f"- **Tasks Tested:** {stats.get('tasks_tested', 0)}\n")
            
            ratio_range = stats.get('ratio_range', (0, 0))
            f.write(f"- **Ratio Range:** {ratio_range[0]:.3f} - {ratio_range[1]:.3f}\n\n")
        
        f.write("---\n\n")

    def _write_correlation_analysis(self, f, correlations: Dict):
        """Write correlation analysis section."""
        f.write("## 🔗 Correlation Analysis\n\n")
        
        f.write("### Key Metric Correlations\n\n")
        
        for correlation_name, value in correlations.items():
            correlation_desc = correlation_name.replace('_', ' ').title()
            strength = self._interpret_correlation(value)
            f.write(f"- **{correlation_desc}:** {value:.3f} ({strength})\n")
        
        f.write("\n---\n\n")

    def _write_outlier_trends_analysis(self, f, outlier_trends: Dict):
        """Write outlier and trends analysis section."""
        f.write("## 📈 Outliers and Trends Analysis\n\n")
        
        f.write("### Outlier Detection\n\n")
        
        outlier_metrics = [key for key in outlier_trends.keys() if key.endswith('_outliers')]
        
        for metric in outlier_metrics:
            outlier_data = outlier_trends[metric]
            metric_name = metric.replace('_outliers', '').replace('_', ' ').title()
            
            f.write(f"**{metric_name} Outliers:**\n")
            f.write(f"- Count: {outlier_data.get('count', 0)}\n")
            f.write(f"- Percentage: {outlier_data.get('percentage', 0):.2f}%\n\n")
        
        if 'trends' in outlier_trends:
            trends = outlier_trends['trends']
            f.write("### Performance Trends\n\n")
            
            if 'correlation' in trends:
                f.write(f"- **Ratio vs Performance Correlation:** {trends['correlation']:.3f}\n")
            
            if 'trend_slope' in trends:
                trend_direction = "Positive" if trends['trend_slope'] > 0 else "Negative"
                f.write(f"- **Trend Direction:** {trend_direction} (slope: {trends['trend_slope']:.3f})\n")
            
            if 'sample_count' in trends:
                f.write(f"- **Trend Sample Count:** {trends['sample_count']}\n")
        
        f.write("\n---\n\n")

    def _write_recommendations(self, f, df: pd.DataFrame, analysis_results: Dict):
        """Write recommendations section."""
        f.write("## 💡 Recommendations\n\n")
        
        # Method recommendations
        if 'method_analysis' in analysis_results:
            method_analysis = analysis_results['method_analysis']
            
            if 'rankings' in method_analysis and 'by_preservation' in method_analysis['rankings']:
                best_method = list(method_analysis['rankings']['by_preservation'].keys())[0]
                f.write(f"### Recommended Method: {best_method}\n\n")
                f.write(f"Based on overall score preservation analysis, **{best_method}** shows the best performance.\n\n")
        
        # Ratio recommendations
        if 'ratio_analysis' in analysis_results:
            ratio_analysis = analysis_results['ratio_analysis']
            
            if 'optimal_ratios' in ratio_analysis:
                optimal = ratio_analysis['optimal_ratios']
                if 'max_cost_benefit' in optimal:
                    f.write(f"### Recommended Compression Ratio: {optimal['max_cost_benefit']:.3f}\n\n")
                    f.write(f"This ratio provides the best balance between compression and performance preservation.\n\n")
        
        # Task-specific recommendations
        if 'task_analysis' in analysis_results:
            f.write("### Task-Specific Recommendations\n\n")
            
            for task, analysis in analysis_results['task_analysis'].items():
                if analysis.get('best_method') and analysis.get('optimal_ratio'):
                    f.write(f"- **{task.title()}:** Use {analysis['best_method']} with ratio {analysis['optimal_ratio']:.3f}\n")
        
        f.write("\n---\n\n")

    def _write_data_summary(self, f, df: pd.DataFrame):
        """Write data summary section."""
        f.write("## 📊 Data Summary\n\n")
        
        f.write("### Dataset Characteristics\n\n")
        f.write(f"- **Total Records:** {len(df):,}\n")
        f.write(f"- **Memory Usage:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")
        
        if 'actual_compression_ratio' in df.columns:
            f.write(f"- **Compression Ratio Statistics:**\n")
            f.write(f"  - Mean: {df['actual_compression_ratio'].mean():.3f}\n")
            f.write(f"  - Median: {df['actual_compression_ratio'].median():.3f}\n")
            f.write(f"  - Std Dev: {df['actual_compression_ratio'].std():.3f}\n")
        
        if 'score_preservation' in df.columns:
            f.write(f"- **Score Preservation Statistics:**\n")
            f.write(f"  - Mean: {df['score_preservation'].mean():.2f}%\n")
            f.write(f"  - Median: {df['score_preservation'].median():.2f}%\n")
            f.write(f"  - Std Dev: {df['score_preservation'].std():.2f}%\n")
        
        f.write(f"\n### Data Quality\n\n")
        f.write(f"- **Missing Values:**\n")
        
        missing_data = df.isnull().sum()
        for column, missing_count in missing_data.items():
            if missing_count > 0:
                percentage = (missing_count / len(df)) * 100
                f.write(f"  - {column}: {missing_count} ({percentage:.1f}%)\n")
        
        if missing_data.sum() == 0:
            f.write("  - No missing values detected\n")
        
        f.write(f"\n---\n\n")
        f.write("*Report generated by PCM-LLM Ultra-Comprehensive Analyzer*\n")

    def _interpret_correlation(self, correlation: float) -> str:
        """Interpret correlation strength."""
        abs_corr = abs(correlation)
        if abs_corr >= 0.8:
            return "Very Strong"
        elif abs_corr >= 0.6:
            return "Strong" 
        elif abs_corr >= 0.4:
            return "Moderate"
        elif abs_corr >= 0.2:
            return "Weak"
        else:
            return "Very Weak"

    def save_data_csv(self, df: pd.DataFrame, timestamp: str) -> str:
        """Save the analysis data as CSV."""
        csv_file = self.output_dir / f"ultra_comprehensive_data_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        return str(csv_file)
