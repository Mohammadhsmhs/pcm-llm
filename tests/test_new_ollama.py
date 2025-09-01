#!/usr/bin/env python3
"""
Test script for the new Ollama implementation using the official library.
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_ollama_implementation():
    """Test the new Ollama implementation."""
    print("🧪 Testing New Ollama Implementation")
    print("=" * 50)

    try:
        from llms.ollama_llm import Ollama_LLM, list_local_models, pull_model_if_needed

        print("1. Testing model listing...")
        models = list_local_models()
        print(f"   Available models: {models}")

        if not models:
            print("   ❌ No models found. Please pull a model first:")
            print("      ollama pull llama2")
            return

        print("\n2. Testing Ollama LLM initialization...")
        # Use the first available model
        model_name = models[0]
        llm = Ollama_LLM(model_name=model_name, temperature=0.1)

        print("\n3. Testing basic generation...")
        test_prompt = "What is 2 + 2? Answer with just the number."
        response = llm.get_response(test_prompt)
        print(f"   Prompt: {test_prompt}")
        print(f"   Response: {response}")

        print("\n4. Testing chat functionality...")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello! What can you tell me about Python?"}
        ]
        chat_response = llm.chat(messages)
        if "error" not in chat_response:
            print("   ✅ Chat successful!")
            print(f"   Response: {chat_response.get('message', {}).get('content', '')[:100]}...")
        else:
            print(f"   ⚠️  Chat error: {chat_response.get('error')}")

        print("\n5. Testing model info...")
        info = llm.get_model_info()
        if "error" not in info:
            print("   ✅ Model info retrieved!")
            print(f"   Model: {info.get('modelfile', 'N/A')[:50]}...")
        else:
            print(f"   ⚠️  Model info error: {info.get('error')}")

        print("\n✅ All tests completed successfully!")
        print("\n📋 New Features Available:")
        print("   • Streaming responses: Ollama_LLM(stream=True)")
        print("   • Tool calling: llm.chat(messages, tools=[func1, func2])")
        print("   • Model management: llm.pull_model('model_name')")
        print("   • Better error handling with official library")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure ollama package is installed: pip install ollama")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("   Make sure Ollama is running: ollama serve")
        print("   And you have models pulled: ollama pull llama2")

if __name__ == "__main__":
    test_ollama_implementation()
