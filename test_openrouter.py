#!/usr/bin/env python3
"""
Test script for OpenRouter DeepSeek R1:free integration
"""

import os
from llms.factory import LLMFactory

def test_openrouter():
    """Test the OpenRouter DeepSeek R1:free model"""

    # Check if API key is set
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ OPENROUTER_API_KEY environment variable not set!")
        print("Please set it with: export OPENROUTER_API_KEY='your_api_key_here'")
        print("Get your API key from: https://openrouter.ai/keys")
        return

    print("🚀 Testing OpenRouter DeepSeek R1:free integration...")

    try:
        # Create the LLM instance
        llm = LLMFactory.create("openrouter")
        print("✅ OpenRouter LLM initialized successfully!")

        # Test with a simple prompt
        test_prompt = "What is 2 + 2? Please show your reasoning step by step."

        print(f"\n📝 Test Prompt: {test_prompt}")
        print("\n🤖 Getting response...")

        response = llm.get_response(test_prompt)

        print("\n📄 Response:")
        print(response)

        if "Error" not in response:
            print("\n✅ Test successful! OpenRouter integration is working.")
        else:
            print(f"\n❌ Test failed with error: {response}")

    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        print("Make sure you have the correct API key and internet connection.")

if __name__ == "__main__":
    test_openrouter()
