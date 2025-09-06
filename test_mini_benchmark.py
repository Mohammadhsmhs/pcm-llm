#!/usr/bin/env python3
"""
Small test benchmark to validate the complete pipeline with fixes.
This will run just a few samples to ensure everything works before the massive run.
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.benchmark_service import BenchmarkService
from utils.data.file_writer import FileWriter

def create_test_samples():
    """Create a few test samples to run through the pipeline."""
    test_samples = {
        "test_001": {
            "task": "summarization",
            "original_prompt": "Please provide a comprehensive summary of machine learning techniques including supervised learning methods like linear regression and decision trees, unsupervised learning approaches such as clustering and dimensionality reduction, and reinforcement learning concepts.",
            "ground_truth_answer": "Machine learning includes supervised (regression, classification), unsupervised (clustering, dimensionality reduction), and reinforcement learning approaches."
        },
        "test_002": {
            "task": "qa", 
            "original_prompt": "What are the key differences between artificial intelligence, machine learning, and deep learning? Please explain how they relate to each other and provide examples of applications for each field.",
            "ground_truth_answer": "AI is the broad field, ML is a subset using data to learn patterns, and deep learning is a subset of ML using neural networks. Examples: AI (chatbots), ML (recommendation systems), deep learning (image recognition)."
        },
        "test_003": {
            "task": "classification",
            "original_prompt": "Given the following text about a restaurant review, classify the sentiment as positive, negative, or neutral: 'The food was decent and the service was okay, but nothing really stood out as exceptional. The prices were reasonable though.'",
            "ground_truth_answer": "neutral"
        }
    }
    return test_samples

def run_mini_benchmark():
    """Run a small benchmark test."""
    print("🧪 MINI BENCHMARK TEST")
    print("=" * 60)
    print("Running a small test to validate the complete pipeline...")
    
    # Create test directory
    test_dir = Path("test_results/mini_benchmark")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration
    config = {
        "llm": {
            "provider": "llamacpp",
            "model_path": "models/qwen2.5-0.5b-instruct-q4_k_m.gguf",  # Small model for testing
            "context_length": 2048,
            "max_tokens": 512,
            "temperature": 0.1
        },
        "compression": {
            "methods": ["naive_truncation"],  # Just one method for testing
            "target_ratios": [0.7],
            "naive_truncation": {
                "strategy": "middle"
            }
        },
        "output": {
            "log_dir": str(test_dir),
            "base_name": "mini_test"
        }
    }
    
    try:
        # Initialize benchmark service
        benchmark_service = BenchmarkService(config)
        
        # Create test samples
        samples = create_test_samples()
        print(f"Created {len(samples)} test samples")
        
        # Run benchmark
        print("Running mini benchmark...")
        results = benchmark_service.run_benchmark(samples)
        
        # Analyze results
        print("\n📊 ANALYZING RESULTS...")
        print("=" * 40)
        
        if not results:
            print("❌ No results returned!")
            return False
            
        # Check a sample result
        sample_id = list(results.keys())[0]
        sample_result = results[sample_id]
        
        print(f"Sample ID: {sample_id}")
        print(f"Baseline score: {sample_result.get('baseline_score', 'N/A')}")
        
        # Check compression method results
        methods_data = sample_result.get('methods', {})
        if 'naive_truncation' in methods_data:
            method_data = methods_data['naive_truncation']
            print(f"Compressed score: {method_data.get('compressed_score', 'N/A')}")
            
            # Check for the new ratio fields
            preservation = method_data.get('actual_preservation_ratio', 'N/A')
            compression = method_data.get('actual_compression_ratio', 'N/A')
            tokens_saved = method_data.get('tokens_saved', 'N/A')
            
            print(f"Preservation ratio: {preservation}")
            print(f"Compression ratio: {compression}")
            print(f"Tokens saved: {tokens_saved}")
            
            # Validate ratios
            if isinstance(preservation, (int, float)) and isinstance(compression, (int, float)):
                ratio_sum = preservation + compression
                print(f"Ratio sum: {ratio_sum:.4f}")
                
                if abs(ratio_sum - 1.0) < 0.01:
                    print("✅ Ratios sum to ~1.0 (logical consistency)")
                else:
                    print(f"❌ Ratios don't sum to 1.0: {ratio_sum}")
                    return False
                    
                if 0 <= preservation <= 2.0 and 0 <= compression <= 1.0:
                    print("✅ Ratios in valid ranges")
                else:
                    print(f"❌ Ratios out of valid ranges: P={preservation}, C={compression}")
                    return False
            else:
                print(f"❌ Invalid ratio types: P={type(preservation)}, C={type(compression)}")
                return False
        else:
            print("❌ No naive_truncation results found!")
            return False
            
        # Check CSV output
        csv_file = test_dir / "mini_test.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            print(f"\n📄 CSV OUTPUT:")
            print(f"   Rows: {len(df)}")
            print(f"   Columns: {len(df.columns)}")
            
            # Check for new column names
            preservation_cols = [col for col in df.columns if 'actual_preservation_ratio' in col]
            compression_cols = [col for col in df.columns if 'actual_compression_ratio' in col]
            old_cols = [col for col in df.columns if col.endswith('_actual_ratio') and 'preservation' not in col and 'compression' not in col]
            
            print(f"   Preservation ratio columns: {preservation_cols}")
            print(f"   Compression ratio columns: {compression_cols}")
            print(f"   Old ambiguous columns: {old_cols}")
            
            if len(preservation_cols) > 0 and len(compression_cols) > 0 and len(old_cols) == 0:
                print("✅ CSV columns correctly named")
            else:
                print("❌ CSV column naming issues")
                return False
                
            # Check actual values in CSV
            if len(preservation_cols) > 0:
                preservation_col = preservation_cols[0]
                preservation_values = df[preservation_col].dropna()
                if len(preservation_values) > 0:
                    max_preservation = preservation_values.max()
                    min_preservation = preservation_values.min()
                    print(f"   Preservation ratios: {min_preservation:.4f} to {max_preservation:.4f}")
                    
                    if max_preservation <= 2.0 and min_preservation >= 0.0:  # Allow up to 200% for edge cases
                        print("✅ All preservation ratios in valid range")
                    else:
                        print(f"❌ Some preservation ratios out of range!")
                        return False
        else:
            print("❌ CSV file not created!")
            return False
            
        print("\n🎉 MINI BENCHMARK TEST PASSED!")
        print("All fixes are working correctly in the full pipeline.")
        return True
        
    except Exception as e:
        print(f"❌ MINI BENCHMARK TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 STARTING MINI BENCHMARK VALIDATION")
    print("=" * 60)
    print("This will test the complete pipeline with all fixes...")
    
    success = run_mini_benchmark()
    
    print(f"\n{'='*60}")
    if success:
        print("✅ MINI BENCHMARK VALIDATION SUCCESSFUL!")
        print("   Ready for the full massive benchmark run.")
        print("   All ratio calculations and field names are working correctly.")
    else:
        print("❌ MINI BENCHMARK VALIDATION FAILED!")
        print("   Please fix issues before running the full benchmark.")
        
    print(f"{'='*60}")
