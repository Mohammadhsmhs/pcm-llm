#!/usr/bin/env python3
"""
Simplified test that validates the data processing pipeline without needing an LLM.
This tests just the data enhancement and CSV generation parts.
"""

import os
import sys
import pandas as pd
from pathlib import Path
import tempfile

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.data.data_enhancer import DataEnhancer
from utils.data.file_writer import FileWriter

def create_mock_benchmark_results():
    """Create mock benchmark results that simulate what would come from a real benchmark."""
    mock_results = {
        "test_001": {
            "task": "summarization",
            "llm_provider": "test_provider",
            "llm_model": "test_model",
            "original_prompt": "Please provide a comprehensive summary of machine learning techniques including supervised learning methods like linear regression and decision trees, unsupervised learning approaches such as clustering and dimensionality reduction, and reinforcement learning concepts.",
            "ground_truth_answer": "Machine learning includes supervised, unsupervised, and reinforcement learning approaches.",
            "baseline_output": "Machine learning encompasses three main categories: supervised learning (like linear regression and decision trees), unsupervised learning (clustering and dimensionality reduction), and reinforcement learning for decision-making tasks.",
            "baseline_extracted_answer": "Machine learning includes supervised, unsupervised, and reinforcement learning approaches.",
            "baseline_score": 0.95,
            "baseline_latency": 1.2,
            "methods": {
                "naive_truncation": {
                    "compressed_prompt": "Please provide a comprehensive summary of machine learning techniques including supervised learning methods",
                    "compressed_output": "Machine learning encompasses supervised learning methods like linear regression and decision trees.",
                    "compressed_extracted_answer": "Machine learning includes supervised learning methods.",
                    "compressed_score": 0.78,
                    "compressed_latency": 0.8,
                    "answers_match": False,
                    "target_compression_ratio": 0.6
                    # Note: actual_preservation_ratio and actual_compression_ratio will be added by DataEnhancer
                }
            }
        },
        "test_002": {
            "task": "qa", 
            "llm_provider": "test_provider",
            "llm_model": "test_model",
            "original_prompt": "What are the key differences between artificial intelligence, machine learning, and deep learning? Please explain how they relate to each other and provide examples of applications for each field.",
            "ground_truth_answer": "AI is the broad field, ML is a subset, deep learning is a subset of ML using neural networks.",
            "baseline_output": "AI is the broad field that encompasses all intelligent systems. Machine learning is a subset of AI that learns from data. Deep learning is a subset of ML that uses neural networks with multiple layers.",
            "baseline_extracted_answer": "AI is the broad field, ML is a subset, deep learning is a subset of ML using neural networks.",
            "baseline_score": 0.92,
            "baseline_latency": 1.5,
            "methods": {
                "naive_truncation": {
                    "compressed_prompt": "What are the key differences between artificial intelligence, machine learning, and deep learning?",
                    "compressed_output": "AI is broad, ML learns from data, deep learning uses neural networks.",
                    "compressed_extracted_answer": "AI is broad, ML is subset, deep learning uses neural networks.",
                    "compressed_score": 0.85,
                    "compressed_latency": 1.0,
                    "answers_match": True,
                    "target_compression_ratio": 0.5
                }
            }
        }
    }
    return mock_results

def test_data_processing_pipeline():
    """Test the complete data processing pipeline."""
    print("🔧 TESTING DATA PROCESSING PIPELINE")
    print("=" * 60)
    
    # Create mock results
    mock_results = create_mock_benchmark_results()
    print(f"Created {len(mock_results)} mock benchmark results")
    
    # Enhance the data using DataEnhancer
    print("\n📊 ENHANCING DATA...")
    enhanced_results = {}
    
    for sample_id, sample_data in mock_results.items():
        enhanced_sample = dict(sample_data)  # Copy sample data
        
        # Enhance each method's data
        if 'methods' in enhanced_sample:
            for method_name, method_data in enhanced_sample['methods'].items():
                # Combine baseline and method data for enhancement
                combined_data = {
                    'baseline_score': enhanced_sample['baseline_score'],
                    'compressed_score': method_data['compressed_score'],
                    'baseline_latency': enhanced_sample['baseline_latency'], 
                    'compressed_latency': method_data['compressed_latency'],
                    'original_prompt': enhanced_sample['original_prompt'],
                    'compressed_prompt': method_data['compressed_prompt'],
                    'target_compression_ratio': method_data['target_compression_ratio']
                }
                
                # Enhance the data
                enhanced_method_data = DataEnhancer.enhance_result_data(combined_data)
                
                # Update method data with enhancements
                enhanced_sample['methods'][method_name].update(enhanced_method_data)
                
        enhanced_results[sample_id] = enhanced_sample
    
    print("✅ Data enhancement complete")
    
    # Test CSV generation
    print("\n📄 TESTING CSV GENERATION...")
    with tempfile.TemporaryDirectory() as temp_dir:
        file_writer = FileWriter(temp_dir, "pipeline_test")
        
        # Write CSV
        compression_methods = ["naive_truncation"]
        file_writer.write_csv(enhanced_results, compression_methods)
        
        # Read and analyze CSV
        csv_file = Path(temp_dir) / "pipeline_test.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            print(f"✅ CSV created with {len(df)} rows and {len(df.columns)} columns")
            
            # Check specific columns
            preservation_cols = [col for col in df.columns if 'actual_preservation_ratio' in col]
            compression_cols = [col for col in df.columns if 'actual_compression_ratio' in col]
            old_ambiguous_cols = [col for col in df.columns if col.endswith('_actual_ratio') and 'preservation' not in col and 'compression' not in col]
            
            print(f"   Preservation ratio columns: {preservation_cols}")
            print(f"   Compression ratio columns: {compression_cols}")
            print(f"   Old ambiguous columns: {old_ambiguous_cols}")
            
            # Validate column naming
            if len(preservation_cols) > 0 and len(compression_cols) > 0 and len(old_ambiguous_cols) == 0:
                print("✅ CSV column naming is correct")
            else:
                print("❌ CSV column naming issues detected")
                return False
                
            # Check data values
            print("\n📊 VALIDATING DATA VALUES...")
            for preservation_col in preservation_cols:
                values = df[preservation_col].dropna()
                if len(values) > 0:
                    min_val, max_val = values.min(), values.max()
                    print(f"   {preservation_col}: {min_val:.4f} to {max_val:.4f}")
                    
                    if max_val > 2.0 or min_val < 0:
                        print(f"❌ Invalid preservation ratios detected!")
                        return False
                        
            for compression_col in compression_cols:
                values = df[compression_col].dropna()
                if len(values) > 0:
                    min_val, max_val = values.min(), values.max()
                    print(f"   {compression_col}: {min_val:.4f} to {max_val:.4f}")
                    
                    if max_val > 1.0 or min_val < 0:
                        print(f"❌ Invalid compression ratios detected!")
                        return False
                        
            # Check ratio consistency
            print("\n🧮 CHECKING RATIO CONSISTENCY...")
            for i, row in df.iterrows():
                sample_id = row['sample_id']
                for preservation_col in preservation_cols:
                    # Find corresponding compression column
                    method = preservation_col.replace('_actual_preservation_ratio', '')
                    compression_col = f"{method}_actual_compression_ratio"
                    
                    if compression_col in df.columns:
                        preservation = row[preservation_col]
                        compression = row[compression_col]
                        
                        if pd.notna(preservation) and pd.notna(compression):
                            ratio_sum = preservation + compression
                            print(f"   Sample {sample_id}: P={preservation:.4f} + C={compression:.4f} = {ratio_sum:.4f}")
                            
                            if abs(ratio_sum - 1.0) > 0.01:
                                print(f"❌ Ratios don't sum to 1.0 for sample {sample_id}!")
                                return False
                                
            print("✅ All ratio consistency checks passed")
            
            print("\n📋 SAMPLE DATA PREVIEW:")
            print(df[['sample_id'] + preservation_cols + compression_cols].head())
            
        else:
            print("❌ CSV file was not created!")
            return False
            
    print("\n🎉 DATA PROCESSING PIPELINE TEST PASSED!")
    return True

if __name__ == "__main__":
    print("🚀 STARTING DATA PROCESSING PIPELINE TEST")
    print("=" * 60)
    print("This tests the data enhancement and CSV generation without needing an LLM...")
    
    try:
        success = test_data_processing_pipeline()
        
        print(f"\n{'='*60}")
        if success:
            print("✅ PIPELINE TEST SUCCESSFUL!")
            print("   All data processing, enhancement, and CSV generation is working correctly.")
            print("   Ready for the full benchmark run with actual LLMs.")
        else:
            print("❌ PIPELINE TEST FAILED!")
            print("   Please fix issues before running the full benchmark.")
            
    except Exception as e:
        print(f"❌ PIPELINE TEST FAILED WITH EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        success = False
        
    print(f"{'='*60}")
    
    if success:
        exit(0)
    else:
        exit(1)
