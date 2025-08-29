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

def test_quantization_memory_comparison():
    """Test to compare memory usage between quantized and non-quantized models."""
    print("\n=== Testing Quantization Memory Comparison ===")

    if not torch.cuda.is_available():
        print("⚠️  CUDA not available - skipping memory comparison test")
        return

    try:
        import gc

        # Test full precision model
        print("Loading full precision model...")
        llm_full = HuggingFace_LLM(model_name=HUGGINGFACE_MODEL, quantization="none")
        full_memory = sum(param.numel() * param.element_size() for param in llm_full.model.parameters())
        print(f"Full precision memory: {full_memory / 1024**3:.2f} GB")

        # Clear memory
        del llm_full
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Test quantized model
        print("Loading quantized model...")
        llm_quantized = HuggingFace_LLM(model_name=HUGGINGFACE_MODEL, quantization=HUGGINGFACE_QUANTIZATION)
        quantized_memory = sum(param.numel() * param.element_size() for param in llm_quantized.model.parameters())
        print(f"Quantized memory: {quantized_memory / 1024**3:.2f} GB")

        # Calculate memory savings
        memory_savings = full_memory - quantized_memory
        savings_percentage = (memory_savings / full_memory) * 100

        print("\nMemory comparison results:")
        print(f"Memory saved: {memory_savings / 1024**3:.2f} GB")
        print(f"Savings percentage: {savings_percentage:.1f}%")

        if savings_percentage > 10:
            print("✅ Quantization is working - significant memory savings detected!")
        else:
            print("⚠️  Warning: Memory savings are lower than expected")

        del llm_quantized
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"❌ Error during memory comparison: {e}")
        import traceback
        traceback.print_exc()

def test_multi_task_setup():
    """Test the new multi-task configuration setup."""
    print("\n=== Testing Multi-Task Configuration ===")

    from config import SUPPORTED_TASKS, TASK_CONFIGURATIONS, DEFAULT_TASK
    from data_loaders.loaders import load_benchmark_dataset

    print(f"Supported tasks: {SUPPORTED_TASKS}")
    print(f"Default task: {DEFAULT_TASK}")

    print("\nTask configurations:")
    for task, config in TASK_CONFIGURATIONS.items():
        print(f"  {task}: {config['description']}")
        print(f"    Dataset: {config['dataset']}")
        print(f"    Config: {config['config']}")

    # Test loading mock data for each task
    print("\nTesting mock data loading for each task:")
    for task in SUPPORTED_TASKS:
        try:
            config = TASK_CONFIGURATIONS[task]
            dataset = load_benchmark_dataset(task, config['dataset'], config['config'], 2)
            if dataset:
                print(f"  ✅ {task}: Loaded {len(dataset)} samples")
            else:
                print(f"  ❌ {task}: Failed to load data")
        except Exception as e:
            print(f"  ❌ {task}: Error - {e}")

if __name__ == "__main__":
    test_huggingface_llm()
    test_quantization_memory_comparison()
    test_multi_task_setup()
