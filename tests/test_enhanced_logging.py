#!/usr/bin/env python3
"""
Test script for the enhanced logging system.
Tests the RunInfoLogger and improved BenchmarkLogger functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.run_info_logger import RunInfoLogger
from utils.logger import BenchmarkLogger
import time

def test_enhanced_logging():
    """Test the enhanced logging system."""
    print("🧪 Testing Enhanced Logging System")
    print("=" * 50)

    # Initialize RunInfoLogger
    run_logger = RunInfoLogger(run_id="test_run_001")
    print("✅ RunInfoLogger initialized")

    # Update configuration
    config = {
        "test_mode": True,
        "llm_provider": "huggingface",
        "llm_model": "test-model",
        "compression_methods": ["llmlingua2"]
    }
    run_logger.update_run_config(config)
    print("✅ Configuration updated")

    # Initialize BenchmarkLogger
    benchmark_logger = BenchmarkLogger(
        log_dir="test_results",
        results_dir="test_results",
        task_name="test_task",
        compression_methods=["llmlingua2"]
    )
    print("✅ BenchmarkLogger initialized")

    # Simulate task completion
    for i in range(3):
        task_data = {
            'task_id': f'test_task_{i}',
            'task_type': 'reasoning',
            'compression_method': 'llmlingua2',
            'status': 'completed',
            'latency': 2.5 + i * 0.5,
            'score': 0.85 - i * 0.05,
            'tokens_input': 150 + i * 20,
            'tokens_output': 50 + i * 10,
            'memory_usage': 45.0 + i * 5.0,
            'prompt_preview': f'This is test prompt {i} for reasoning task...',
            'output_preview': f'This is the model output {i} with reasoning...'
        }

        # Log to RunInfoLogger
        run_logger.log_task_completion(task_data)

        # Log to BenchmarkLogger (simulated data)
        sample_result = {
            "sample_id": i,
            "task": "reasoning",
            "llm_provider": "huggingface",
            "llm_model": "test-model",
            "original_prompt": f"Test prompt {i}",
            "ground_truth_answer": f"Answer {i}",
            "compression_methods": [{
                "method": "llmlingua2",
                "compressed_prompt": f"Compressed prompt {i}",
                "compressed_prompt_output": f"Compressed output {i}",
                "compressed_score": 0.8 - i * 0.05,
                "compressed_latency": 1.5 + i * 0.3
            }]
        }
        benchmark_logger.log_result(sample_result)

        print(f"✅ Logged task {i+1}/3")
        time.sleep(0.1)  # Small delay

    # Log memory usage
    run_logger.log_memory_usage(150.5)
    print("✅ Memory usage logged")

    # Finalize both loggers
    benchmark_logger.finalize_and_save()
    run_logger.finalize_run({"total_tasks": 3, "test_completed": True})

    print("\n📊 Test Results:")
    print(f"   📋 Run Info: {run_logger.run_info_file}")
    print(f"   📝 Task Log: {run_logger.task_log_file}")
    print(f"   🔍 Real-time Log: {run_logger.real_time_log_file}")
    print(f"   📊 Benchmark CSV: {benchmark_logger.log_file}")
    print(f"   📈 Benchmark JSON: {benchmark_logger.summary_file}")

    print("\n✅ Enhanced Logging System Test Completed!")
    return True

if __name__ == "__main__":
    success = test_enhanced_logging()
    sys.exit(0 if success else 1)
