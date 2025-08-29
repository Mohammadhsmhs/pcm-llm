
def _get_mock_dataset(dataset_name: str, num_samples: int):
    """Provides mock dataset samples for testing without external dependencies."""
    if dataset_name == "gsm8k":
        mock_samples = [
            {
                "question": "The store sells apples for $2 each and oranges for $3 each. Sarah buys 5 apples and 3 oranges for her family. She pays the cashier with a $20 bill. How much change should Sarah receive from the cashier?",
                "answer": "#### 7"
            },
            {
                "question": "A train travels 120 miles in 2 hours. How many miles will it travel in 5 hours?",
                "answer": "#### 300"
            },
            {
                "question": "If a rectangle has a length of 8 units and a width of 6 units, what is its area?",
                "answer": "#### 48"
            }
        ]
        return mock_samples[:num_samples]
    else:
        print(f"Mock data not available for dataset '{dataset_name}'")
        return None

def load_benchmark_dataset(dataset_name: str, config: str, num_samples: int):
    """
    Loads a specified number of samples from a dataset on the Hugging Face Hub.
    Falls back to mock data if the datasets library is not available or fails.
    """
    print(f"\nLoading {num_samples} samples from '{dataset_name}' dataset...")
    
    try:
        from datasets import load_dataset
        # Load the test split and select a random subset of samples
        dataset = load_dataset(dataset_name, config, split='test').shuffle(seed=42).select(range(num_samples))
        return dataset
    except ImportError:
        print("Hugging Face datasets library not available. Using mock data instead.")
        return _get_mock_dataset(dataset_name, num_samples)
    except Exception as e:
        print(f"Failed to load dataset '{dataset_name}'. Error: {e}")
        print("Falling back to mock data.")
        return _get_mock_dataset(dataset_name, num_samples)
