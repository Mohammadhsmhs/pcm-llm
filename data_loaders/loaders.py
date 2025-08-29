
def _get_mock_dataset(task: str, dataset_name: str, num_samples: int):
    """Provides mock dataset samples for testing without external dependencies."""
    if task == "reasoning" and dataset_name == "gsm8k":
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

    elif task == "summarization" and dataset_name == "cnn_dailymail":
        mock_samples = [
            {
                "article": "The United Nations climate summit in Paris ended with a historic agreement to combat global warming. World leaders pledged to limit temperature rise to well below 2 degrees Celsius above pre-industrial levels. The accord requires all nations to submit plans for reducing greenhouse gas emissions. Environmental groups called it a major step forward, though some critics said the targets were not ambitious enough.",
                "highlights": "UN climate summit reaches historic agreement to limit global warming below 2°C. All nations must submit emission reduction plans. Environmental groups praise accord but critics say targets insufficient."
            },
            {
                "article": "Apple Inc. reported record quarterly earnings of $18.4 billion, driven by strong iPhone sales and services growth. The company's revenue grew 21% year-over-year, with China showing particularly strong performance. Apple also announced plans to invest $350 billion in the US economy over the next five years, including new manufacturing facilities and data centers.",
                "highlights": "Apple reports record $18.4B quarterly earnings with 21% revenue growth. Strong iPhone and services performance drives results. Company announces $350B US investment plan over next 5 years."
            },
            {
                "article": "NASA's Perseverance rover successfully landed on Mars after a seven-month journey. The rover will search for signs of ancient microbial life and collect rock samples for future return to Earth. The mission represents a major step in humanity's exploration of the red planet and could provide insights into whether life ever existed on Mars.",
                "highlights": "NASA's Perseverance rover successfully lands on Mars after 7-month journey. Rover will search for ancient microbial life and collect samples. Mission advances Mars exploration and search for extraterrestrial life."
            }
        ]
        return mock_samples[:num_samples]

    elif task == "classification" and dataset_name == "imdb":
        mock_samples = [
            {
                "text": "This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout. I highly recommend it to anyone who enjoys good cinema.",
                "label": 1  # positive
            },
            {
                "text": "I was really disappointed with this film. The story was confusing and the acting felt forced. I wouldn't recommend it to anyone.",
                "label": 0  # negative
            },
            {
                "text": "An average movie with some good moments but overall nothing special. It was entertaining enough but I've seen much better films.",
                "label": 1  # positive (borderline)
            }
        ]
        return mock_samples[:num_samples]

    else:
        print(f"Mock data not available for task '{task}' with dataset '{dataset_name}'")
        return None

def load_benchmark_dataset(task: str, dataset_name: str, config: str, num_samples: int):
    """
    Loads a specified number of samples from a dataset on the Hugging Face Hub.
    Falls back to mock data if the datasets library is not available or fails.
    Supports multiple task types with appropriate data extraction.
    """
    print(f"\nLoading {num_samples} samples from '{dataset_name}' dataset for task '{task}'...")

    try:
        from datasets import load_dataset

        # Load dataset based on task type
        if task == "reasoning":
            dataset = load_dataset(dataset_name, config, split='test').shuffle(seed=42).select(range(num_samples))
            # Extract question and answer fields
            processed_dataset = []
            for sample in dataset:
                processed_dataset.append({
                    "question": sample.get("question", ""),
                    "answer": sample.get("answer", "")
                })
            return processed_dataset

        elif task == "summarization":
            dataset = load_dataset(dataset_name, config, split='test').shuffle(seed=42).select(range(num_samples))
            # Extract article and highlights/summary fields
            processed_dataset = []
            for sample in dataset:
                processed_dataset.append({
                    "article": sample.get("article", ""),
                    "highlights": sample.get("highlights", "")
                })
            return processed_dataset

        elif task == "classification":
            dataset = load_dataset(dataset_name, config, split='test').shuffle(seed=42).select(range(num_samples))
            # Extract text and label fields
            processed_dataset = []
            for sample in dataset:
                processed_dataset.append({
                    "text": sample.get("text", ""),
                    "label": sample.get("label", 0)
                })
            return processed_dataset

        else:
            print(f"Unknown task type: {task}")
            return None

    except ImportError:
        print("Hugging Face datasets library not available. Using mock data instead.")
        return _get_mock_dataset(task, dataset_name, num_samples)
    except Exception as e:
        print(f"Failed to load dataset '{dataset_name}' for task '{task}'. Error: {e}")
        print("Falling back to mock data.")
        return _get_mock_dataset(task, dataset_name, num_samples)
