#!/usr/bin/env python3
"""
Test script for Ollama LLM integration.
This script tests the Ollama LLM implementation to ensure it works correctly
before running the full benchmark.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llms.factory import LLMFactory
from config import DEFAULT_LLM_PROVIDER

def test_ollama_integration():
    """Test the Ollama LLM integration with a simple prompt."""
    print("Testing Ollama LLM integration...")

    try:
        # Create Ollama LLM instance
        llm = LLMFactory.create("ollama")
        print(f"✓ Created Ollama LLM instance with model: {llm.model_name}")

        # Test simple prompt
        test_prompt = "What is 2 + 2? Please respond with just the answer."
        print(f"Test prompt: {test_prompt}")

        response = llm.get_response(test_prompt)
        print(f"Response: {response}")

        # Check for repetition (common issue with Qwen models)
        words = response.split()
        if len(words) > 10:
            # Check if the same phrase is repeated
            first_half = ' '.join(words[:len(words)//2])
            second_half = ' '.join(words[len(words)//2:])
            if first_half in second_half:
                print("⚠️  WARNING: Detected potential repetition in response")
            else:
                print("✓ No obvious repetition detected")

        print("✓ Ollama integration test completed successfully")
        return True

    except Exception as e:
        print(f"✗ Ollama integration test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_ollama_integration()
    sys.exit(0 if success else 1)
