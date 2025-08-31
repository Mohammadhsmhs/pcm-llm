#!/usr/bin/env python3
"""
Simple test script to see raw LlamaCPP model behavior without restrictions.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llms.llamacpp_simple import LlamaCpp_Simple_LLM

def test_simple_llamacpp():
    # Initialize the simple model
    llm = LlamaCpp_Simple_LLM(
        repo_id="Qwen/Qwen3-14B-GGUF",
        filename="Qwen3-14B-Q4_K_M.gguf"
    )

    # Test prompt
    prompt = "Darrell and Allen's ages are in the ratio of 7:11. If their total age now is 162, calculate Allen's age 10 years from now."

    print("=" * 80)
    print("TESTING SIMPLE LLAMACPP - NO RESTRICTIONS")
    print("=" * 80)
    print(f"Prompt: {prompt}")
    print("-" * 80)

    # Get response
    response = llm.get_response(prompt)

    print("-" * 80)
    print("FULL RAW RESPONSE:")
    print("-" * 80)
    print(response)
    print("-" * 80)
    print(f"Response length: {len(response)} characters")
    print(f"Response word count: {len(response.split())} words")

if __name__ == "__main__":
    test_simple_llamacpp()
