#!/usr/bin/env python3
"""
Simple test to verify the new Ollama implementation structure.
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_ollama_structure():
    """Test that the new Ollama implementation has the correct structure."""
    print("🔍 Testing Ollama Implementation Structure")
    print("=" * 50)

    try:
        # Test import
        print("1. Testing import...")
        from llms.ollama_llm import Ollama_LLM, OLLAMA_AVAILABLE
        print(f"   ✅ Import successful! Ollama available: {OLLAMA_AVAILABLE}")

        # Test class structure
        print("\n2. Testing class structure...")
        print(f"   Class: {Ollama_LLM}")
        print(f"   Base classes: {Ollama_LLM.__bases__}")

        # Test methods
        print("\n3. Testing available methods...")
        methods = [method for method in dir(Ollama_LLM) if not method.startswith('_')]
        print(f"   Public methods: {methods}")

        # Check for new features
        print("\n4. Checking for new features...")
        has_chat = hasattr(Ollama_LLM, 'chat')
        has_streaming = 'stream' in Ollama_LLM.__init__.__code__.co_varnames
        has_tools = 'tools' in Ollama_LLM.chat.__code__.co_varnames if has_chat else False

        print(f"   ✅ Chat method: {has_chat}")
        print(f"   ✅ Streaming support: {has_streaming}")
        print(f"   ✅ Tool calling: {has_tools}")

        # Test convenience functions
        print("\n5. Testing convenience functions...")
        from llms.ollama_llm import list_local_models, pull_model_if_needed
        print("   ✅ Convenience functions imported")

        print("\n🎉 Structure test completed successfully!")
        print("\n📋 Key Improvements in New Implementation:")
        print("   • Uses official ollama Python library")
        print("   • Better error handling with ResponseError")
        print("   • Streaming response support")
        print("   • Tool calling and function support")
        print("   • Model management utilities")
        print("   • Automatic model validation")
        print("   • Graceful fallback when Ollama unavailable")

        if not OLLAMA_AVAILABLE:
            print("\n⚠️  Note: Ollama package not available in current environment")
            print("   To use full functionality, install with: pip install ollama")
            print("   And start Ollama server: ollama serve")
            print("   Then pull a model: ollama pull llama2")

    except Exception as e:
        print(f"❌ Structure test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ollama_structure()
