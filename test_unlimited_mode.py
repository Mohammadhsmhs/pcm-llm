#!/usr/bin/env python3
"""
Test script to demonstrate unlimited mode functionality.
This script shows how to enable unlimited mode for full-speed benchmarking
without timeouts or size restrictions.
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Temporarily override the UNLIMITED_MODE setting for testing
import config
original_unlimited_mode = config.UNLIMITED_MODE
config.UNLIMITED_MODE = True

print("🔓 Testing Unlimited Mode Functionality")
print("=" * 50)

# Import after setting the config
from evaluation.evaluator import Evaluator
from llms.mock_llm import MockLLM

def test_unlimited_mode():
    """Test that unlimited mode removes restrictions."""

    print("1. Testing Evaluator with unlimited mode...")
    llm = MockLLM()
    evaluator = Evaluator(task="reasoning", llm=llm)

    # Test with a very long prompt (normally would timeout)
    long_prompt = "What is 2 + 2? " * 1000  # Very long prompt
    ground_truth = "#### 4"

    print(f"   Prompt length: {len(long_prompt.split())} words")
    print("   Unlimited mode: No timeout restrictions should apply")

    try:
        result = evaluator.evaluate(long_prompt, ground_truth)
        print("   ✅ Evaluation completed successfully")
        print(f"   📊 Score: {result['score']}")
        print(f"   ⏱️  Latency: {result['latency']:.2f}s")
    except Exception as e:
        print(f"   ❌ Evaluation failed: {e}")

    print("\n2. Testing LLM response generation...")
    try:
        response = llm.get_response(long_prompt)
        print("   ✅ LLM response generated successfully")
        print(f"   📝 Response length: {len(response.split())} words")
    except Exception as e:
        print(f"   ❌ LLM response failed: {e}")

def test_standard_mode():
    """Test standard mode with restrictions."""
    print("\n" + "=" * 50)
    print("🔒 Testing Standard Mode (with restrictions)")
    print("=" * 50)

    # Restore original setting
    config.UNLIMITED_MODE = False

    print("1. Testing Evaluator with standard mode...")
    llm = MockLLM()
    evaluator = Evaluator(task="reasoning", llm=llm)

    # Test with a moderately long prompt
    medium_prompt = "What is 2 + 2? " * 100  # Medium prompt
    ground_truth = "#### 4"

    print(f"   Prompt length: {len(medium_prompt.split())} words")
    print("   Standard mode: Timeout restrictions apply")

    try:
        result = evaluator.evaluate(medium_prompt, ground_truth)
        print("   ✅ Evaluation completed successfully")
        print(f"   📊 Score: {result['score']}")
        print(f"   ⏱️  Latency: {result['latency']:.2f}s")
    except Exception as e:
        print(f"   ❌ Evaluation failed: {e}")

if __name__ == "__main__":
    print("PCM-LLM Unlimited Mode Test")
    print("This script demonstrates the difference between unlimited and standard modes.\n")

    test_unlimited_mode()
    test_standard_mode()

    print("\n" + "=" * 50)
    print("🎯 How to Enable Unlimited Mode:")
    print("1. Open config.py")
    print("2. Change: UNLIMITED_MODE = False")
    print("3. To:     UNLIMITED_MODE = True")
    print("4. Run your benchmark as usual")
    print("=" * 50)

    # Restore original setting
    config.UNLIMITED_MODE = original_unlimited_mode
