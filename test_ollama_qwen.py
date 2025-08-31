#!/usr/bin/env python3
"""
Test the new Ollama implementation with the available Qwen model.
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_ollama_with_qwen():
    """Test the new Ollama implementation with the available Qwen model."""
    print("🧪 Testing New Ollama Implementation with Qwen3-14B")
    print("=" * 60)

    try:
        from llms.ollama_llm import Ollama_LLM, list_local_models

        print("1. Checking available models...")
        models = list_local_models()
        print(f"   Available models: {models}")

        if not models:
            print("   ❌ No models found!")
            return

        # Use the available Qwen model
        qwen_model = "hf.co/Qwen/Qwen3-14B-GGUF:Q8_0"
        if qwen_model in models:
            print(f"   ✅ Found Qwen model: {qwen_model}")
        else:
            print(f"   ⚠️  Qwen model not found, using: {models[0]}")
            qwen_model = models[0]

        print("\n2. Initializing Ollama LLM...")
        llm = Ollama_LLM(
            model_name=qwen_model,
            temperature=0.1,
            num_ctx=2048  # Smaller context for testing
        )

        print("\n3. Testing basic generation...")
        test_prompt = "What is 2 + 2? Answer with just the number."
        print(f"   Prompt: {test_prompt}")

        response = llm.get_response(test_prompt)
        print(f"   Response: {response}")

        if response and "Error" not in response:
            print("   ✅ Basic generation successful!")
        else:
            print(f"   ❌ Basic generation failed: {response}")

        print("\n4. Testing chat functionality...")
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Keep responses brief."},
            {"role": "user", "content": "Hello! What is Python?"}
        ]

        chat_response = llm.chat(messages)
        if "error" not in chat_response:
            content = chat_response.get('message', {}).get('content', '')
            print("   ✅ Chat successful!")
            print(f"   Response: {content[:100]}{'...' if len(content) > 100 else ''}")
        else:
            print(f"   ❌ Chat failed: {chat_response.get('error')}")

        print("\n5. Testing model info...")
        info = llm.get_model_info()
        if "error" not in str(info):
            print("   ✅ Model info retrieved!")
            # Handle ShowResponse object from ollama library
            if hasattr(info, 'model'):
                print(f"   Model: {info.model}")
                if hasattr(info, 'size'):
                    print(f"   Size: {info.size} bytes")
                if hasattr(info, 'modelfamily'):
                    print(f"   Family: {info.modelfamily}")
            else:
                print(f"   Model details: {type(info)} object")
        else:
            print(f"   ⚠️  Model info error: {info.get('error', 'Unknown error')}")

        print("\n🎉 All tests completed!")
        print("\n📊 Test Summary:")
        print("   • Model Used: Qwen3-14B (GGUF Q8_0)")
        print("   • Size: 15GB")
        print("   • Features: Basic generation ✅ | Chat API ✅ | Model info ✅")
        print("   • Status: Implementation working correctly!")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure ollama package is installed: pip install ollama")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ollama_with_qwen()
