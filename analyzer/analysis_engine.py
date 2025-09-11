#!/usr/bin/env python3
"""
Analysis Engine Module for PCM-LLM Analysis
Handles different types of analysis (by ratio, method, task, etc.).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class AnalysisEngine:
    """Performs various types of analysis on benchmark data."""
    
    def __init__(self):
        self.analysis_types = [
            'by_ratio', 'by_method', 'by_task', 'by_model',
            'correlation', 'trends', 'outliers'
        ]

    def analyze_by_ratio(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Analyze performance by compression ratio."""
        if 'actual_compression_ratio' not in df.columns:
            logger.warning("No compression ratio data available")
            return {}
        
        ratio_analysis = {}
        
        # Group by actual compression ratio
        ratio_groups = df.groupby('actual_compression_ratio')
        
        # Overall statistics by ratio
        ratio_stats = ratio_groups.agg({
            'score_preservation': ['mean', 'std', 'count'],
            'compressed_accuracy': ['mean', 'std'],
            'cost_benefit_ratio': ['mean', 'std'] if 'cost_benefit_ratio' in df.columns else []
        }).round(3)
        
        ratio_analysis['overall_stats'] = ratio_stats
        
        # Method performance by ratio
        if 'method' in df.columns:
            method_by_ratio = df.pivot_table(
                values=['score_preservation', 'compressed_accuracy'],
                index='actual_compression_ratio',
                columns='method',
                aggfunc='mean'
            ).round(3)
            ratio_analysis['method_performance'] = method_by_ratio
        
        # Task performance by ratio  
        if 'task' in df.columns:
            task_by_ratio = df.pivot_table(
                values=['score_preservation', 'compressed_accuracy'],
                index='actual_compression_ratio',
                columns='task',
                aggfunc='mean'
            ).round(3)
            ratio_analysis['task_performance'] = task_by_ratio
        
        # Find optimal ratios
        ratio_analysis['optimal_ratios'] = self._find_optimal_ratios(df)
        
        return ratio_analysis

    def analyze_by_method(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Analyze performance by compression method."""
        if 'method' not in df.columns:
            logger.warning("No method data available")
            return {}
        
        method_analysis = {}
        
        # Overall method performance
        method_stats = df.groupby('method').agg({
            'score_preservation': ['mean', 'std', 'min', 'max', 'count'],
            'compressed_accuracy': ['mean', 'std', 'min', 'max'],
            'actual_compression_ratio': ['mean', 'std', 'min', 'max'],
            'cost_benefit_ratio': ['mean', 'std'] if 'cost_benefit_ratio' in df.columns else []
        }).round(3)
        
        method_analysis['overall_performance'] = method_stats
        
        # Method performance by task
        if 'task' in df.columns:
            task_method_matrix = df.pivot_table(
                values=['score_preservation', 'compressed_accuracy', 'cost_benefit_ratio'],
                index='method',
                columns='task',
                aggfunc='mean'
            ).round(3)
            method_analysis['task_breakdown'] = task_method_matrix
        
        # Consistency analysis (std deviation as measure of consistency)
        consistency_analysis = df.groupby('method')['score_preservation'].agg([
            'std', 'var', lambda x: x.std() / x.mean() * 100  # Coefficient of variation
        ]).round(3)
        consistency_analysis.columns = ['std_dev', 'variance', 'coeff_variation']
        method_analysis['consistency'] = consistency_analysis
        
        # Method rankings
        method_analysis['rankings'] = self._rank_methods(df)
        
        return method_analysis

    def analyze_by_task(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Analyze performance by task type."""
        if 'task' not in df.columns:
            logger.warning("No task data available")
            return {}
        
        task_analysis = {}
        
        for task in df['task'].unique():
            task_data = df[df['task'] == task]
            
            if len(task_data) == 0:
                continue
            
            # Method comparison for this task
            if 'method' in task_data.columns:
                method_comparison = task_data.groupby('method').agg({
                    'compressed_accuracy': ['mean', 'std', 'count'],
                    'score_preservation': ['mean', 'std', 'count'],
                    'actual_compression_ratio': ['mean', 'std'],
                    'cost_benefit_ratio': ['mean', 'std'] if 'cost_benefit_ratio' in task_data.columns else []
                }).round(3)
            else:
                method_comparison = pd.DataFrame()
            
            # Compression ratio sensitivity for this task
            if 'actual_compression_ratio' in task_data.columns:
                ratio_sensitivity = task_data.groupby('actual_compression_ratio').agg({
                    'score_preservation': ['mean', 'std', 'count'],
                    'compressed_accuracy': ['mean', 'std'],
                    'cost_benefit_ratio': 'mean' if 'cost_benefit_ratio' in task_data.columns else []
                }).round(3)
            else:
                ratio_sensitivity = pd.DataFrame()
            
            # Best performers for this task
            best_sample_idx = task_data['score_preservation'].idxmax() if not task_data['score_preservation'].empty else None
            optimal_ratio = None
            best_method = None
            
            if best_sample_idx is not None:
                if 'method' in task_data.columns:
                    best_method = task_data.loc[best_sample_idx, 'method']
                if 'cost_benefit_ratio' in task_data.columns and not task_data['cost_benefit_ratio'].empty:
                    optimal_idx = task_data['cost_benefit_ratio'].idxmax()
                    optimal_ratio = task_data.loc[optimal_idx, 'actual_compression_ratio'] if 'actual_compression_ratio' in task_data.columns else None
            
            task_analysis[task] = {
                'method_comparison': method_comparison,
                'ratio_sensitivity': ratio_sensitivity,
                'best_method': best_method,
                'optimal_ratio': optimal_ratio,
                'sample_count': len(task_data),
                'unique_ratios': len(task_data['actual_compression_ratio'].unique()) if 'actual_compression_ratio' in task_data.columns else 0
            }
        
        return task_analysis

    def analyze_by_model(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Analyze performance by model."""
        if 'model' not in df.columns:
            logger.warning("No model data available")
            return {}
        
        model_analysis = {}
        
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            
            if len(model_data) == 0:
                continue
            
            # Performance across methods and tasks
            available_columns = ['score_preservation', 'compressed_accuracy']
            if 'cost_benefit_ratio' in model_data.columns:
                available_columns.append('cost_benefit_ratio')
            
            if 'method' in model_data.columns and 'task' in model_data.columns:
                performance_matrix = model_data.pivot_table(
                    values=available_columns,
                    index='method',
                    columns='task',
                    aggfunc='mean'
                ).round(3)
            else:
                performance_matrix = pd.DataFrame()
            
            # Overall model statistics
            model_stats = {
                'total_samples': len(model_data),
                'avg_score_preservation': model_data['score_preservation'].mean(),
                'avg_accuracy': model_data['compressed_accuracy'].mean(),
                'performance_std': model_data['score_preservation'].std(),
                'methods_tested': model_data['method'].nunique() if 'method' in model_data.columns else 0,
                'tasks_tested': model_data['task'].nunique() if 'task' in model_data.columns else 0,
                'ratio_range': (
                    model_data['actual_compression_ratio'].min(),
                    model_data['actual_compression_ratio'].max()
                ) if 'actual_compression_ratio' in model_data.columns else (0, 0)
            }
            
            model_analysis[model] = {
                'performance_matrix': performance_matrix,
                'statistics': model_stats
            }
        
        return model_analysis

    def analyze_correlations(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze correlations between different metrics."""
        numeric_columns = [
            'actual_compression_ratio', 'score_preservation', 'compressed_accuracy',
            'accuracy_drop', 'cost_benefit_ratio', 'compression_efficiency'
        ]
        
        available_columns = [col for col in numeric_columns if col in df.columns]
        
        if len(available_columns) < 2:
            logger.warning("Not enough numeric columns for correlation analysis")
            return {}
        
        correlation_matrix = df[available_columns].corr()
        
        # Extract key correlations
        correlations = {}
        
        if 'actual_compression_ratio' in correlation_matrix.columns and 'score_preservation' in correlation_matrix.columns:
            correlations['ratio_vs_preservation'] = correlation_matrix.loc['actual_compression_ratio', 'score_preservation']
        
        if 'compressed_accuracy' in correlation_matrix.columns and 'score_preservation' in correlation_matrix.columns:
            correlations['accuracy_vs_preservation'] = correlation_matrix.loc['compressed_accuracy', 'score_preservation']
        
        if 'cost_benefit_ratio' in correlation_matrix.columns and 'score_preservation' in correlation_matrix.columns:
            correlations['cost_benefit_vs_preservation'] = correlation_matrix.loc['cost_benefit_ratio', 'score_preservation']
        
        return correlations

    def detect_outliers_and_trends(self, df: pd.DataFrame) -> Dict:
        """Detect outliers and analyze trends."""
        outlier_analysis = {}
        
        # Outlier detection for key metrics
        key_metrics = ['score_preservation', 'compressed_accuracy', 'actual_compression_ratio']
        available_metrics = [col for col in key_metrics if col in df.columns]
        
        for metric in available_metrics:
            outliers = self._detect_outliers_iqr(df[metric])
            outlier_analysis[f'{metric}_outliers'] = {
                'count': outliers.sum(),
                'percentage': outliers.mean() * 100,
                'outlier_indices': outliers[outliers].index.tolist()
            }
        
        # Trend analysis
        if 'actual_compression_ratio' in df.columns and 'score_preservation' in df.columns:
            trend_analysis = self._analyze_performance_trends(df)
            outlier_analysis['trends'] = trend_analysis
        
        return outlier_analysis

    def _find_optimal_ratios(self, df: pd.DataFrame) -> Dict:
        """Find optimal compression ratios based on different criteria."""
        optimal_ratios = {}
        
        if 'cost_benefit_ratio' in df.columns:
            max_cb_idx = df['cost_benefit_ratio'].idxmax()
            optimal_ratios['max_cost_benefit'] = df.loc[max_cb_idx, 'actual_compression_ratio']
        
        if 'score_preservation' in df.columns:
            # Find ratio with best preservation above 90%
            high_preservation = df[df['score_preservation'] >= 90.0]
            if not high_preservation.empty and 'actual_compression_ratio' in high_preservation.columns:
                optimal_ratios['high_preservation'] = high_preservation['actual_compression_ratio'].min()
        
        return optimal_ratios

    def _rank_methods(self, df: pd.DataFrame) -> Dict:
        """Rank methods by various criteria."""
        if 'method' not in df.columns:
            return {}
        
        rankings = {}
        
        # By average score preservation
        if 'score_preservation' in df.columns:
            preservation_ranking = df.groupby('method')['score_preservation'].mean().sort_values(ascending=False)
            rankings['by_preservation'] = preservation_ranking.to_dict()
        
        # By consistency (lower std deviation is better)
        if 'score_preservation' in df.columns:
            consistency_ranking = df.groupby('method')['score_preservation'].std().sort_values(ascending=True)
            rankings['by_consistency'] = consistency_ranking.to_dict()
        
        # By cost-benefit ratio
        if 'cost_benefit_ratio' in df.columns:
            cb_ranking = df.groupby('method')['cost_benefit_ratio'].mean().sort_values(ascending=False)
            rankings['by_cost_benefit'] = cb_ranking.to_dict()
        
        return rankings

    def _detect_outliers_iqr(self, data: pd.Series) -> pd.Series:
        """Detect outliers using IQR method."""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (data < lower_bound) | (data > upper_bound)

    def _analyze_performance_trends(self, df: pd.DataFrame) -> Dict:
        """Analyze performance trends across compression ratios."""
        df_clean = df.dropna(subset=['actual_compression_ratio', 'score_preservation'])
        
        if len(df_clean) < 3:
            return {}
        
        # Calculate correlation
        correlation = df_clean['actual_compression_ratio'].corr(df_clean['score_preservation'])
        
        # Fit trend line
        try:
            coeffs = np.polyfit(df_clean['actual_compression_ratio'], df_clean['score_preservation'], 1)
            slope = coeffs[0]
            intercept = coeffs[1]
        except:
            slope = 0
            intercept = 0
        
        return {
            'correlation': correlation,
            'trend_slope': slope,
            'trend_intercept': intercept,
            'sample_count': len(df_clean),
            'ratio_range': (df_clean['actual_compression_ratio'].min(), df_clean['actual_compression_ratio'].max())
        }
