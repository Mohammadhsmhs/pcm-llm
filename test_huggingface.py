#!/usr/bin/env python3
"""
Test script to verify the new HuggingFace LLM implementation.
"""

import torch
from llms.huggingface_llm import HuggingFace_LLM
from config import HUGGINGFACE_MODEL, HUGGINGFACE_QUANTIZATION

def test_huggingface_llm():
    """Test the new HuggingFace LLM implementation."""
    
    print("=== Testing HuggingFace LLM Implementation ===\n")
    
    # Show system information
    print(f"Model: {HUGGINGFACE_MODEL}")
    print(f"Quantization: {HUGGINGFACE_QUANTIZATION}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    elif torch.backends.mps.is_available():
        print("Using Apple Silicon GPU (MPS)")
    else:
        print("Using CPU")
    
    try:
        print(f"\nInitializing HuggingFace LLM...")
        
        # Create LLM instance with quantization
        llm = HuggingFace_LLM(model_name=HUGGINGFACE_MODEL, quantization=HUGGINGFACE_QUANTIZATION)
        
        # Test a simple prompt
        test_prompt = "What is 2 + 2? Please provide the numerical answer."
        print(f"\nTesting with prompt: '{test_prompt}'")
        print("Generating response (tokens will stream in real-time):")
        
        response = llm.get_response(test_prompt)
        print(f"\nFinal response: {response}")
        
        print("\n✅ HuggingFace LLM test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during HuggingFace LLM test: {e}")
        print("This might be due to:")
        print("- Insufficient memory (try a smaller model)")
        print("- Network issues downloading the model")
        print("- Incompatible model format")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_huggingface_llm()
