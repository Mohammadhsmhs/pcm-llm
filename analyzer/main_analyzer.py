#!/usr/bin/env python3
"""
Main Ultra-Comprehensive Benchmark Analyzer Module
Coordinates all analysis components and provides main interface.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime
import warnings
import logging

# Import analysis modules
from .data_loader import BenchmarkDataLoader
from .metrics_calculator import MetricsCalculator
from .analysis_engine import AnalysisEngine
from .report_generator import ReportGenerator
from .visualization_generator import VisualizationGenerator

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UltraComprehensiveBenchmarkAnalyzer:
    """
    Ultra-comprehensive analyzer for multi-dimensional benchmark comparison.
    
    This is the main orchestrator class that coordinates all analysis components:
    - Data loading and preprocessing
    - Metrics calculation
    - Multi-dimensional analysis
    - Report generation
    - Visualization creation
    """

    def __init__(self, results_dir: str = "results"):
        """Initialize the analyzer with all its components."""
        self.results_dir = Path(results_dir)
        
        # Initialize analysis components
        self.data_loader = BenchmarkDataLoader(results_dir)
        self.metrics_calculator = MetricsCalculator()
        self.analysis_engine = AnalysisEngine()
        self.report_generator = ReportGenerator()
        self.visualization_generator = VisualizationGenerator()
        
        logger.info(f"Initialized Ultra-Comprehensive Analyzer for {results_dir}")

    def run_ultra_comprehensive_analysis(self) -> Dict:
        """Run the complete ultra-comprehensive analysis pipeline."""
        
        print("🚀 Starting ultra-comprehensive multi-dimensional analysis...")
        
        try:
            # Step 1: Load all data
            all_results = self.data_loader.load_all_ratio_results()
            if not all_results:
                raise ValueError("No data files found to analyze")
            
            # Step 2: Calculate advanced metrics for all samples
            analysis_df = self._process_all_data(all_results)
            if analysis_df.empty:
                raise ValueError("No valid metrics could be calculated")
            
            # Step 3: Perform multi-dimensional analysis
            analysis_results = self._perform_comprehensive_analysis(analysis_df)
            
            # Step 4: Generate timestamp for outputs
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Step 5: Generate report
            report_file = self.report_generator.generate_comprehensive_report(
                analysis_df, analysis_results, timestamp
            )
            
            # Step 6: Save data
            csv_file = self.report_generator.save_data_csv(analysis_df, timestamp)
            
            # Step 7: Create visualizations
            visualization_files = self.visualization_generator.create_all_visualizations(
                analysis_df, timestamp
            )
            
            # Step 8: Log results
            unique_ratios = analysis_df['actual_compression_ratio'].nunique() if 'actual_compression_ratio' in analysis_df.columns else 0
            logger.info(f"Processed {len(analysis_df)} individual samples across {unique_ratios} unique compression ratios")
            
            print(f"✅ Ultra-comprehensive analysis complete!")
            print(f"   📄 Main Report: {report_file}")
            print(f"   📊 Complete Data: {csv_file}")
            print(f"   📈 Analyzed {len(analysis_df)} total combinations")
            print(f"   📈 Across {unique_ratios} ratios")
            print(f"   📈 With {analysis_df['method'].nunique() if 'method' in analysis_df.columns else 0} methods")
            print(f"   📈 On {analysis_df['task'].nunique() if 'task' in analysis_df.columns else 0} tasks")
            print(f"   📈 Using {analysis_df['model'].nunique() if 'model' in analysis_df.columns else 0} models")
            
            if visualization_files:
                print(f"   🎨 Visualizations: {', '.join([Path(f).name for f in visualization_files])}")
            
            return {
                'analysis_df': analysis_df,
                'analysis_results': analysis_results,
                'report_file': report_file,
                'csv_file': csv_file,
                'visualization_files': visualization_files,
                'timestamp': timestamp,
                'summary': {
                    'total_samples': len(analysis_df),
                    'unique_ratios': unique_ratios,
                    'methods': analysis_df['method'].nunique() if 'method' in analysis_df.columns else 0,
                    'tasks': analysis_df['task'].nunique() if 'task' in analysis_df.columns else 0,
                    'models': analysis_df['model'].nunique() if 'model' in analysis_df.columns else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise

    def _process_all_data(self, all_results: Dict[str, List[pd.DataFrame]]) -> pd.DataFrame:
        """Process all data files and calculate advanced metrics."""
        
        all_sample_metrics = []
        
        for task, dataframes in all_results.items():
            for i, df in enumerate(dataframes):
                # Clean and validate data
                df = self.data_loader.clean_and_normalize_data(df)
                
                if not self.data_loader.validate_dataframe(df, f"{task}_{i}"):
                    logger.warning(f"Skipping invalid dataframe for {task}_{i}")
                    continue
                
                file_name = df['source_file'].iloc[0]
                model_info = self.data_loader.extract_model_info(df)
                
                print(f"\n🎯 Analyzing {task} - {file_name} (Model: {model_info})")
                
                # Process ALL methods available in the data
                available_methods = self._get_available_methods(df)
                
                for method in available_methods:
                    print(f"   📊 Processing method: {method}")
                    
                    # Calculate metrics for individual samples for this method
                    sample_metrics = self.metrics_calculator.calculate_sample_metrics(df, method, task)
                    all_sample_metrics.extend(sample_metrics)

        # Convert to DataFrame
        if not all_sample_metrics:
            return pd.DataFrame()
            
        analysis_df = pd.DataFrame(all_sample_metrics)
        
        # Additional data validation and cleaning
        analysis_df = self._clean_analysis_dataframe(analysis_df)
        
        return analysis_df

    def _get_available_methods(self, df: pd.DataFrame) -> List[str]:
        """Get all available compression methods from the dataframe columns."""
        
        available_methods = []
        
        # Check for method-specific columns to determine available methods
        method_patterns = {
            'llmlingua2': 'llmlingua2_compressed_score',
            'naive_truncation': 'naive_truncation_compressed_score', 
            'selective_context': 'selective_context_compressed_score'
        }
        
        for method_name, pattern in method_patterns.items():
            if pattern in df.columns:
                # Check if there are valid values for this method
                if df[pattern].notna().sum() > 0:
                    available_methods.append(method_name)
        
        if not available_methods:
            # Fallback to filename-based detection if no method-specific columns found
            available_methods = [self._determine_method(df, df['source_file'].iloc[0])]
        
        return available_methods
        
    def _determine_method(self, df: pd.DataFrame, file_name: str) -> str:
        """Determine the compression method from dataframe or filename."""
        
        # Try to get method from dataframe
        if 'method' in df.columns and not df['method'].isna().all():
            method = df['method'].iloc[0]
            if isinstance(method, str) and method.strip():
                return method.strip()
        
        # Try to determine from filename
        file_lower = file_name.lower()
        
        if 'llmlingua2' in file_lower or 'llm_lingua' in file_lower:
            return 'llmlingua2'
        elif 'selective' in file_lower:
            return 'selective_context'
        elif 'naive' in file_lower or 'truncation' in file_lower:
            return 'naive_truncation'
        else:
            # Default fallback
            return 'unknown_method'

    def _clean_analysis_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the final analysis dataframe."""
        
        # Remove rows with invalid core metrics
        initial_count = len(df)
        
        # Filter out rows with missing essential data - use the actual column names
        essential_columns = []
        if 'baseline_accuracy' in df.columns:
            essential_columns.append('baseline_accuracy')
        if 'compressed_accuracy' in df.columns:
            essential_columns.append('compressed_accuracy')
        
        if essential_columns:
            df = df.dropna(subset=essential_columns)
        
        # Filter out unrealistic values if the columns exist
        if 'baseline_accuracy' in df.columns and 'compressed_accuracy' in df.columns:
            df = df[
                (df['baseline_accuracy'] >= 0) & (df['baseline_accuracy'] <= 100) &
                (df['compressed_accuracy'] >= 0) & (df['compressed_accuracy'] <= 100)
            ]
        
        # Filter score preservation if it exists
        if 'score_preservation' in df.columns:
            # Allow negative preservation (performance degradation) and up to 200% for improvements
            df = df[
                (df['score_preservation'] >= -50) & (df['score_preservation'] <= 200)
            ]
        
        # Handle infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        final_count = len(df)
        
        if final_count < initial_count:
            logger.info(f"Cleaned dataframe: {initial_count} -> {final_count} rows ({initial_count - final_count} removed)")
            # Log details about what was removed
            if final_count != initial_count - 2:
                logger.warning(f"Expected 2 samples to be removed, but {initial_count - final_count} were removed")
        
        return df

    def _perform_comprehensive_analysis(self, df: pd.DataFrame) -> Dict:
        """Perform all types of analysis on the processed data."""
        
        analysis_results = {}
        
        try:
            # Ratio-based analysis
            analysis_results['ratio_analysis'] = self.analysis_engine.analyze_by_ratio(df)
            
            # Method-based analysis
            analysis_results['method_analysis'] = self.analysis_engine.analyze_by_method(df)
            
            # Task-based analysis
            analysis_results['task_analysis'] = self.analysis_engine.analyze_by_task(df)
            
            # Model-based analysis
            analysis_results['model_analysis'] = self.analysis_engine.analyze_by_model(df)
            
            # Correlation analysis
            analysis_results['correlations'] = self.analysis_engine.analyze_correlations(df)
            
            # Outlier and trend analysis
            analysis_results['outliers_trends'] = self.analysis_engine.detect_outliers_and_trends(df)
            
            logger.info("Completed all analysis dimensions")
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            # Continue with partial results
        
        return analysis_results

    def get_data_summary(self) -> Dict:
        """Get a summary of available data without running full analysis."""
        try:
            all_results = self.data_loader.load_all_ratio_results()
            return self.data_loader.get_data_summary(all_results)
        except Exception as e:
            logger.error(f"Error getting data summary: {e}")
            return {}

    def analyze_specific_task(self, task_name: str) -> Optional[Dict]:
        """Run analysis for a specific task only."""
        try:
            all_results = self.data_loader.load_all_ratio_results()
            
            if task_name not in all_results:
                logger.error(f"Task {task_name} not found in data")
                return None
            
            # Process only the specific task
            task_results = {task_name: all_results[task_name]}
            analysis_df = self._process_all_data(task_results)
            
            if analysis_df.empty:
                logger.error(f"No valid data for task {task_name}")
                return None
            
            # Run analysis
            analysis_results = self._perform_comprehensive_analysis(analysis_df)
            
            return {
                'task': task_name,
                'data': analysis_df,
                'analysis': analysis_results,
                'summary': {
                    'samples': len(analysis_df),
                    'methods': analysis_df['method'].nunique() if 'method' in analysis_df.columns else 0,
                    'ratios': analysis_df['actual_compression_ratio'].nunique() if 'actual_compression_ratio' in analysis_df.columns else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing task {task_name}: {e}")
            return None

    def compare_methods(self, methods: List[str]) -> Optional[Dict]:
        """Run analysis comparing specific methods only."""
        try:
            all_results = self.data_loader.load_all_ratio_results()
            analysis_df = self._process_all_data(all_results)
            
            if analysis_df.empty or 'method' not in analysis_df.columns:
                return None
            
            # Filter to specific methods
            method_df = analysis_df[analysis_df['method'].isin(methods)]
            
            if method_df.empty:
                logger.error(f"No data found for methods: {methods}")
                return None
            
            # Focus on method comparison
            method_analysis = self.analysis_engine.analyze_by_method(method_df)
            
            return {
                'methods': methods,
                'data': method_df,
                'analysis': method_analysis,
                'summary': {
                    'samples_per_method': method_df['method'].value_counts().to_dict(),
                    'total_samples': len(method_df),
                    'tasks': method_df['task'].unique().tolist() if 'task' in method_df.columns else []
                }
            }
            
        except Exception as e:
            logger.error(f"Error comparing methods {methods}: {e}")
            return None
