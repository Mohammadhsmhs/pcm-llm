#!/usr/bin/env python3
"""
Data Loading Module for PCM-LLM Analysis
Handles loading and preprocessing of benchmark data files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import re
import logging

logger = logging.getLogger(__name__)

class BenchmarkDataLoader:
    """Handles loading and preprocessing of benchmark data files."""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.compression_methods = ["llmlingua2", "naive_truncation", "selective_context"]
        self.tasks = ["classification", "reasoning", "summarization"]
        self.task_abbrevs = {
            "classification": "cla",
            "reasoning": "rea", 
            "summarization": "sum"
        }

    def load_all_ratio_results(self) -> Dict[str, List[pd.DataFrame]]:
        """Load all benchmark results organized by task."""
        all_results = {}
        
        for task in self.tasks:
            print(f"📊 Loading {task} files:")
            task_results = []
            task_pattern = f"bench_{self.task_abbrevs[task]}_*.csv"
            
            for file_path in sorted(self.results_dir.glob(task_pattern)):
                try:
                    df = pd.read_csv(file_path)
                    df['source_file'] = file_path.name
                    df['file_timestamp'] = file_path.stat().st_mtime
                    task_results.append(df)
                    print(f"   ✅ {file_path.name}: {len(df)} samples")
                except Exception as e:
                    print(f"   ❌ Error loading {file_path.name}: {e}")
                    continue
            
            if task_results:
                all_results[task] = task_results
            else:
                print(f"   ⚠️  No files found for {task}")
        
        return all_results

    def extract_model_info(self, df: pd.DataFrame) -> str:
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
                    # Extract just the model name part if it's a long path
                    if '/' in model_info:
                        parts = model_info.split('/')
                        if len(parts) >= 2:
                            return f"{parts[-2]}/{parts[-1]}"
                    return model_info.strip()

            # Try to extract from baseline_model if model column doesn't exist
            if 'baseline_model' in df.columns:
                model_info = df['baseline_model'].iloc[0]
                if isinstance(model_info, str) and model_info.strip():
                    return model_info.strip()
                
            return "Unknown Model"
        except Exception as e:
            logger.warning(f"Error extracting model info: {e}")
            return "Unknown Model"

    def validate_dataframe(self, df: pd.DataFrame, file_name: str) -> bool:
        """Validate that DataFrame has required columns."""
        # Check for baseline score (should be present)
        baseline_cols = ['baseline_score']
        has_baseline = any(col in df.columns for col in baseline_cols)
        
        # Check for method-specific compressed scores
        method_patterns = ['llmlingua2_compressed_score', 'naive_truncation_compressed_score', 'selective_context_compressed_score']
        has_compressed = any(col in df.columns for col in method_patterns)
        
        if not has_baseline:
            logger.warning(f"Missing baseline score columns in {file_name}")
            return False
            
        if not has_compressed:
            logger.warning(f"Missing compressed score columns in {file_name}")
            return False
        
        return True

    def clean_and_normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize data for consistency."""
        df = df.copy()
        
        # Ensure numeric columns are properly typed
        numeric_columns = [
            'baseline_score', 'baseline_latency'
        ]
        
        # Add method-specific numeric columns
        method_prefixes = ['llmlingua2', 'naive_truncation', 'selective_context']
        for prefix in method_prefixes:
            numeric_columns.extend([
                f'{prefix}_compressed_score',
                f'{prefix}_compressed_latency',
                f'{prefix}_target_ratio',
                f'{prefix}_actual_ratio',
                f'{prefix}_tokens_saved'
            ])
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df

    def get_data_summary(self, all_results: Dict[str, List[pd.DataFrame]]) -> Dict:
        """Generate a summary of loaded data."""
        summary = {
            'total_files': 0,
            'total_samples': 0,
            'tasks': {},
            'methods': set(),
            'models': set(),
            'ratio_ranges': {}
        }
        
        for task, dataframes in all_results.items():
            task_samples = 0
            task_files = len(dataframes)
            task_ratios = []
            
            for df in dataframes:
                task_samples += len(df)
                if 'method' in df.columns:
                    summary['methods'].update(df['method'].unique())
                if 'model' in df.columns:
                    summary['models'].add(self.extract_model_info(df))
                if 'actual_compression_ratio' in df.columns:
                    task_ratios.extend(df['actual_compression_ratio'].dropna().tolist())
            
            summary['tasks'][task] = {
                'files': task_files,
                'samples': task_samples,
                'ratio_range': (min(task_ratios), max(task_ratios)) if task_ratios else (0, 0)
            }
            
            summary['total_files'] += task_files
            summary['total_samples'] += task_samples
        
        return summary
