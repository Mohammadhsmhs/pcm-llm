#!/usr/bin/env python3
"""
Metrics Calculation Module for PCM-LLM Analysis
Handles calculation of performance metrics for individual samples.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class MetricsCalculator:
    """Calculates performance metrics for benchmark samples."""
    
    def __init__(self):
        self.required_columns = ['baseline_accuracy', 'compressed_accuracy']
        self.optional_columns = [
            'target_ratio', 'actual_compression_ratio', 
            'baseline_latency', 'compressed_latency'
        ]

    def calculate_sample_metrics(self, df: pd.DataFrame, method: str, task: str) -> List[Dict]:
        """Calculate metrics for individual samples in a DataFrame - handles method-specific columns."""
        sample_metrics = []
        model_info = self._extract_model_info(df)
        
        # Get method-specific column names
        method_cols = self._get_method_columns(df, method)
        
        if not method_cols:
            logger.warning(f"No method-specific columns found for method: {method}")
            return sample_metrics
        
        for idx, row in df.iterrows():
            try:
                metrics = self._calculate_single_sample_metrics(row, method, task, model_info, df, idx, method_cols)
                if metrics:  # Only add if metrics were successfully calculated
                    sample_metrics.append(metrics)
            except Exception as e:
                logger.warning(f"Error calculating metrics for sample {idx}: {e}")
                continue
                
        return sample_metrics

    def _get_method_columns(self, df: pd.DataFrame, method: str) -> Dict[str, str]:
        """Get the actual column names for a specific method."""
        
        # Map method names to column prefixes
        method_prefixes = {
            'llmlingua2': 'llmlingua2',
            'naive_truncation': 'naive_truncation', 
            'selective_context': 'selective_context',
            'unknown_method': None
        }
        
        prefix = method_prefixes.get(method)
        if not prefix:
            # Try to detect method from available columns
            for method_name, method_prefix in method_prefixes.items():
                if method_prefix and f'{method_prefix}_compressed_score' in df.columns:
                    prefix = method_prefix
                    break
        
        if not prefix:
            return {}
        
        # Return mapping of standard names to actual column names
        return {
            'compressed_score': f'{prefix}_compressed_score',
            'compressed_latency': f'{prefix}_compressed_latency', 
            'target_ratio': f'{prefix}_target_ratio',
            'actual_ratio': f'{prefix}_actual_ratio',
            'tokens_saved': f'{prefix}_tokens_saved',
            'score_preservation': f'{prefix}_score_preservation',
            'latency_overhead': f'{prefix}_latency_overhead',
            'latency_overhead_percent': f'{prefix}_latency_overhead_percent'
        }

    def _calculate_single_sample_metrics(self, row: pd.Series, method: str, task: str, 
                                       model_info: str, df: pd.DataFrame, idx: int, method_cols: Dict[str, str]) -> Dict:
        """Calculate metrics for a single sample - handles method-specific columns."""
        
        # Extract base scores - use baseline_score as primary
        if 'baseline_score' in row:
            baseline_acc = pd.to_numeric(row['baseline_score'], errors='coerce')
        elif 'baseline_accuracy' in row:
            baseline_acc = pd.to_numeric(row['baseline_accuracy'], errors='coerce')
        else:
            baseline_acc = 0.0
            
        # Extract compressed score using method-specific column
        compressed_acc = 0.0
        if method_cols.get('compressed_score') and method_cols['compressed_score'] in row:
            compressed_acc = pd.to_numeric(row[method_cols['compressed_score']], errors='coerce')
        elif 'compressed_accuracy' in row:
            compressed_acc = pd.to_numeric(row['compressed_accuracy'], errors='coerce')
        
        # Skip if we don't have valid scores
        if pd.isna(baseline_acc) or pd.isna(compressed_acc):
            baseline_acc = 0.0 if pd.isna(baseline_acc) else baseline_acc
            compressed_acc = 0.0 if pd.isna(compressed_acc) else compressed_acc
        
        # Score preservation (main metric)
        score_preservation = (compressed_acc / baseline_acc * 100) if baseline_acc > 0 else 0
        accuracy_drop = baseline_acc - compressed_acc
        
        # Compression ratios - try method-specific columns first
        target_ratio = np.nan
        actual_ratio = np.nan
        
        if method_cols.get('target_ratio') and method_cols['target_ratio'] in row:
            target_ratio = pd.to_numeric(row[method_cols['target_ratio']], errors='coerce')
        elif 'target_ratio' in row:
            target_ratio = pd.to_numeric(row['target_ratio'], errors='coerce')
        
        if method_cols.get('actual_ratio') and method_cols['actual_ratio'] in row:
            actual_ratio = pd.to_numeric(row[method_cols['actual_ratio']], errors='coerce')
            # Convert percentage format to decimal if needed (values > 1 are likely percentages)
            if not pd.isna(actual_ratio) and actual_ratio > 1:
                actual_ratio = actual_ratio / 100.0
        elif 'actual_compression_ratio' in row:
            actual_ratio = pd.to_numeric(row['actual_compression_ratio'], errors='coerce')
            # Convert percentage format to decimal if needed
            if not pd.isna(actual_ratio) and actual_ratio > 1:
                actual_ratio = actual_ratio / 100.0
        elif not pd.isna(target_ratio):
            actual_ratio = target_ratio
        
        # Latency metrics - try method-specific columns first
        baseline_lat = pd.to_numeric(row.get('baseline_latency', np.nan), errors='coerce')
        compressed_lat = np.nan
        
        if method_cols.get('compressed_latency') and method_cols['compressed_latency'] in row:
            compressed_lat = pd.to_numeric(row[method_cols['compressed_latency']], errors='coerce')
        elif 'compressed_latency' in row:
            compressed_lat = pd.to_numeric(row['compressed_latency'], errors='coerce')
        
        latency_overhead = compressed_lat - baseline_lat if not pd.isna(baseline_lat) and not pd.isna(compressed_lat) else np.nan
        latency_overhead_percent = (latency_overhead / baseline_lat * 100) if not pd.isna(latency_overhead) and baseline_lat > 0 else np.nan
        
        # Token savings - try method-specific columns first
        tokens_saved = np.nan
        if method_cols.get('tokens_saved') and method_cols['tokens_saved'] in row:
            tokens_saved = pd.to_numeric(row[method_cols['tokens_saved']], errors='coerce')
        elif 'tokens_saved' in row:
            tokens_saved = pd.to_numeric(row['tokens_saved'], errors='coerce')
        
        # Compression efficiency (how well the actual ratio matches target)
        compression_efficiency = ((1 - actual_ratio) / (1 - target_ratio) * 100) if not pd.isna(target_ratio) and not pd.isna(actual_ratio) and target_ratio < 1 else np.nan
        
        # Cost-benefit ratio (performance per compression)
        cost_benefit = compressed_acc / (1 - actual_ratio) if not pd.isna(actual_ratio) and actual_ratio < 1 else compressed_acc
        
        return {
            'task': task.title(),
            'method': method.replace('_', ' ').title(),
            'model': model_info,
            'sample_id': row.get('sample_id', f'{task}_{idx}'),
            
            # Ratio information (fixed terminology)
            'target_ratio': target_ratio,
            'actual_preservation_ratio': actual_ratio,  # This is preservation ratio, not compression ratio!
            'actual_compression_ratio': (1 - actual_ratio) if not pd.isna(actual_ratio) else np.nan,  # True compression ratio
            'compression_efficiency': compression_efficiency,
            
            # Performance metrics
            'baseline_accuracy': baseline_acc,
            'compressed_accuracy': compressed_acc,
            'score_preservation': score_preservation,
            'accuracy_drop': accuracy_drop,
            
            # Latency metrics
            'baseline_latency': baseline_lat,
            'compressed_latency': compressed_lat,
            'latency_overhead': latency_overhead,
            'latency_overhead_percent': latency_overhead_percent,
            
            # Token metrics
            'tokens_saved': tokens_saved,
            
            # Quality metrics
            'cost_benefit_ratio': cost_benefit,
            
            # Meta information
            'file_name': df.get('source_file', df.index.name).iloc[0] if hasattr(df.get('source_file', df.index.name), 'iloc') else 'unknown',
            'run_timestamp': df.get('file_timestamp', 'unknown').iloc[0] if hasattr(df.get('file_timestamp', 'unknown'), 'iloc') else 'unknown'
        }

    def _extract_model_info(self, df: pd.DataFrame) -> str:
        """Extract model information from DataFrame."""
        try:
            # Check for llm_model column first (most common in our data)
            if 'llm_model' in df.columns:
                model_info = df['llm_model'].iloc[0]
                if isinstance(model_info, str) and model_info.strip():
                    # Clean up model string for better readability
                    # Extract model name from paths like "hf.co/microsoft/Phi-3-mini-4k-instruct-gguf:latest"
                    if '/' in model_info:
                        parts = model_info.split('/')
                        if len(parts) >= 2:
                            # Get the last part and remove version tags
                            model_name = parts[-1].split(':')[0]  # Remove :latest or similar tags
                            return model_name
                    return model_info.strip()
            
            # Fallback to standard 'model' column
            if 'model' in df.columns:
                model_info = df['model'].iloc[0]
                if isinstance(model_info, str) and model_info.strip():
                    if '/' in model_info:
                        parts = model_info.split('/')
                        if len(parts) >= 2:
                            return f"{parts[-2]}/{parts[-1]}"
                    return model_info.strip()
            return "Unknown Model"
        except Exception:
            return "Unknown Model"

    def aggregate_metrics_by_group(self, df: pd.DataFrame, group_by: str) -> pd.DataFrame:
        """Aggregate metrics by a grouping column."""
        numeric_columns = [
            'score_preservation', 'compressed_accuracy', 'actual_compression_ratio',
            'compression_efficiency', 'cost_benefit_ratio', 'accuracy_drop'
        ]
        
        # Filter to only numeric columns that exist
        available_numeric = [col for col in numeric_columns if col in df.columns]
        
        if not available_numeric:
            logger.warning("No numeric columns available for aggregation")
            return pd.DataFrame()
        
        aggregations = {}
        for col in available_numeric:
            aggregations[col] = ['mean', 'std', 'min', 'max', 'count']
        
        return df.groupby(group_by).agg(aggregations).round(3)

    def calculate_correlation_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix for numeric metrics."""
        numeric_columns = [
            'actual_compression_ratio', 'score_preservation', 'compressed_accuracy', 
            'accuracy_drop', 'cost_benefit_ratio', 'compression_efficiency',
            'latency_overhead_percent'
        ]
        
        available_columns = [col for col in numeric_columns if col in df.columns]
        correlation_data = df[available_columns].select_dtypes(include=[np.number])
        
        return correlation_data.corr()

    def detect_outliers(self, df: pd.DataFrame, column: str, method: str = 'iqr') -> pd.Series:
        """Detect outliers in a specific column."""
        if column not in df.columns:
            return pd.Series(dtype=bool)
        
        data = df[column].dropna()
        
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return (data < lower_bound) | (data > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs((data - data.mean()) / data.std())
            return z_scores > 3
        
        return pd.Series(dtype=bool)

    def calculate_performance_trends(self, df: pd.DataFrame) -> Dict:
        """Calculate performance trends across compression ratios."""
        if 'actual_compression_ratio' not in df.columns or 'score_preservation' not in df.columns:
            return {}
        
        # Group by compression ratio bins for trend analysis
        df_clean = df.dropna(subset=['actual_compression_ratio', 'score_preservation'])
        if len(df_clean) < 3:
            return {}
        
        df_clean = df_clean.sort_values('actual_compression_ratio')
        
        # Calculate trend metrics
        corr_ratio_performance = df_clean['actual_compression_ratio'].corr(df_clean['score_preservation'])
        
        # Find optimal ratio range (highest cost-benefit)
        if 'cost_benefit_ratio' in df_clean.columns:
            optimal_idx = df_clean['cost_benefit_ratio'].idxmax()
            optimal_ratio = df_clean.loc[optimal_idx, 'actual_compression_ratio']
        else:
            optimal_ratio = df_clean['actual_compression_ratio'].median()
        
        return {
            'ratio_performance_correlation': corr_ratio_performance,
            'optimal_compression_ratio': optimal_ratio,
            'performance_variance': df_clean['score_preservation'].var(),
            'ratio_range': (df_clean['actual_compression_ratio'].min(), 
                           df_clean['actual_compression_ratio'].max()),
            'sample_count': len(df_clean)
        }
