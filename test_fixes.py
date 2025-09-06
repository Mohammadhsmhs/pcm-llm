#!/usr/bin/env python3
"""
Test script to validate the fixed tokenization and ratio calculations
before running the full benchmark suite.
"""

import os
import pandas as pd
from pathlib import Path
from utils.data.data_enhancer import DataEnhancer

def test_data_enhancer_fixes():
    """Test the fixed data enhancer with various scenarios."""
    print("🧪 TESTING DATA ENHANCER FIXES")
    print("=" * 50)
    
    test_cases = [
        {
            "name": "Normal Compression",
            "data": {
                "baseline_score": 0.8,
                "compressed_score": 0.75,
                "baseline_latency": 1.0,
                "compressed_latency": 0.9,
                "original_prompt": "This is a test prompt with multiple tokens and subword tokenization.",
                "compressed_prompt": "This is a test prompt with multiple tokens",
                "target_compression_ratio": 0.7
            }
        },
        {
            "name": "Minimal Compression", 
            "data": {
                "baseline_score": 0.9,
                "compressed_score": 0.85,
                "baseline_latency": 1.2,
                "compressed_latency": 1.1,
                "original_prompt": "Short prompt for testing purposes.",
                "compressed_prompt": "Short prompt for testing",
                "target_compression_ratio": 0.9
            }
        },
        {
            "name": "Heavy Compression",
            "data": {
                "baseline_score": 0.7,
                "compressed_score": 0.6,
                "baseline_latency": 1.5,
                "compressed_latency": 0.8,
                "original_prompt": "This is a very long prompt with many words, tokens, and complex subword tokenization that should be heavily compressed for testing the edge cases of our ratio calculations.",
                "compressed_prompt": "This is a very long prompt",
                "target_compression_ratio": 0.3
            }
        }
    ]
    
    all_passed = True
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {test_case['name']} ---")
        
        enhanced = DataEnhancer.enhance_result_data(test_case['data'])
        
        # Extract key metrics
        preservation = enhanced['actual_preservation_ratio']
        compression = enhanced['actual_compression_ratio']
        tokens_saved = enhanced['tokens_saved']
        
        print(f"Original: '{test_case['data']['original_prompt'][:50]}...'")
        print(f"Compressed: '{test_case['data']['compressed_prompt'][:50]}...'")
        print(f"Preservation ratio: {preservation:.4f}")
        print(f"Compression ratio: {compression:.4f}")
        print(f"Tokens saved: {tokens_saved}")
        print(f"Sum (preservation + compression): {preservation + compression:.4f}")
        
        # Validation checks
        checks_passed = 0
        total_checks = 5
        
        if 0 <= preservation <= 2.0:  # Allow up to 200% for edge cases
            print("✅ Preservation ratio in valid range")
            checks_passed += 1
        else:
            print(f"❌ Preservation ratio out of range: {preservation}")
            
        if 0 <= compression <= 1.0:
            print("✅ Compression ratio in valid range")
            checks_passed += 1
        else:
            print(f"❌ Compression ratio out of range: {compression}")
            
        if abs((preservation + compression) - 1.0) < 0.01:
            print("✅ Ratios sum to ~1.0 (logical consistency)")
            checks_passed += 1
        else:
            print(f"❌ Ratios don't sum to 1.0: {preservation + compression}")
            
        if tokens_saved >= 0 or preservation > 1.0:  # Allow negative if expansion
            print("✅ Token count makes sense")
            checks_passed += 1
        else:
            print(f"❌ Token count inconsistent: {tokens_saved}")
            
        if 'actual_preservation_ratio' in enhanced and 'actual_compression_ratio' in enhanced:
            print("✅ Both ratio fields present")
            checks_passed += 1
        else:
            print("❌ Missing ratio fields")
            
        if checks_passed == total_checks:
            print(f"✅ Test case {i} PASSED ({checks_passed}/{total_checks})")
        else:
            print(f"❌ Test case {i} FAILED ({checks_passed}/{total_checks})")
            all_passed = False
    
    print(f"\n{'='*50}")
    if all_passed:
        print("🎉 ALL TESTS PASSED! Ready for full benchmark run.")
    else:
        print("❌ Some tests failed. Need to investigate before full run.")
    
    return all_passed

def test_csv_field_names():
    """Test that CSV field names are correctly updated."""
    print("\n🗂️  TESTING CSV FIELD NAME UPDATES")
    print("=" * 50)
    
    # Test that our enhanced data structure contains the new field names
    test_data = {
        "baseline_score": 0.8,
        "compressed_score": 0.75,
        "baseline_latency": 1.0,
        "compressed_latency": 0.9,
        "original_prompt": "This is a test prompt with multiple tokens.",
        "compressed_prompt": "This is a test prompt",
        "target_compression_ratio": 0.7
    }
    
    enhanced = DataEnhancer.enhance_result_data(test_data)
    
    # Check that the new field names exist
    required_fields = [
        'actual_preservation_ratio',
        'actual_compression_ratio',
        'tokens_saved'
    ]
    
    print("Checking for new field names in enhanced data...")
    all_present = True
    
    for field in required_fields:
        if field in enhanced:
            print(f"✅ Field '{field}' present: {enhanced[field]}")
        else:
            print(f"❌ Field '{field}' missing!")
            all_present = False
    
    # Check that old ambiguous fields are not present
    old_fields = ['actual_ratio']
    for field in old_fields:
        if field in enhanced:
            print(f"❌ Old ambiguous field '{field}' still present!")
            all_present = False
        else:
            print(f"✅ Old ambiguous field '{field}' correctly removed")
    
    if all_present:
        print("✅ CSV field names correctly updated")
        return True
    else:
        print("❌ CSV field names not properly updated") 
        return False

if __name__ == "__main__":
    print("🔧 PRE-BENCHMARK VALIDATION TEST")
    print("=" * 60)
    print("Testing all fixes before running full benchmark suite...")
    
    # Run tests
    enhancer_ok = test_data_enhancer_fixes()
    csv_ok = test_csv_field_names()
    
    print(f"\n{'='*60}")
    print("🎯 FINAL VALIDATION RESULTS:")
    print(f"   Data Enhancer: {'✅ PASS' if enhancer_ok else '❌ FAIL'}")
    print(f"   CSV Field Names: {'✅ PASS' if csv_ok else '❌ FAIL'}")
    
    if enhancer_ok and csv_ok:
        print("\n🚀 ALL VALIDATIONS PASSED!")
        print("   Ready to proceed with full benchmark run.")
        print("   The fixes should resolve all ratio calculation issues.")
    else:
        print("\n❌ VALIDATION FAILURES DETECTED!")
        print("   Please review and fix issues before full benchmark run.")
