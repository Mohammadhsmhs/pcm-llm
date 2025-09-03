"""
Test fixtures and utilities for testing.
"""

import tempfile
import os
import json
from typing import Dict, Any
from tests.fixtures.base_fixture import TestFixture


class ConfigFixture(TestFixture):
    """Fixture for configuration testing."""

    def setup(self):
        """Set up configuration test fixture."""
        super().setup()

        # Create temporary config file
        self.config_file = os.path.join(self.temp_dir, "test_config.json")
        self.test_config = {
            "llm_providers": {
                "huggingface": {
                    "model_name": "test-model",
                    "quantization": "none"
                },
                "openai": {
                    "model_name": "gpt-3.5-turbo",
                    "api_key": "test-key"
                }
            },
            "benchmark": {
                "default_task": "reasoning",
                "num_samples": 5,
                "compression_methods": ["llmlingua2"]
            }
        }

        with open(self.config_file, 'w') as f:
            json.dump(self.test_config, f)

        self.add_resource("config_file", self.config_file)
        self.add_resource("test_config", self.test_config)

    def get_test_config(self) -> Dict[str, Any]:
        """Get test configuration."""
        return self.test_config


class DatasetFixture(TestFixture):
    """Fixture for dataset testing."""

    def setup(self):
        """Set up dataset test fixture."""
        super().setup()

        # Create sample datasets
        self.datasets = {
            "reasoning": [
                {"question": "What is 2+2?", "answer": "4"},
                {"question": "What is 3+3?", "answer": "6"},
                {"question": "What is 4+4?", "answer": "8"}
            ],
            "summarization": [
                {"article": "Test article 1", "highlights": "Summary 1"},
                {"article": "Test article 2", "highlights": "Summary 2"}
            ],
            "classification": [
                {"text": "Great product!", "label": "positive"},
                {"text": "Terrible service", "label": "negative"}
            ]
        }

        # Save datasets to files
        for dataset_name, data in self.datasets.items():
            dataset_file = os.path.join(self.temp_dir, f"{dataset_name}_dataset.json")
            with open(dataset_file, 'w') as f:
                json.dump(data, f)
            self.add_resource(f"{dataset_name}_dataset", dataset_file)

        self.add_resource("datasets", self.datasets)

    def get_dataset(self, name: str) -> list:
        """Get test dataset by name."""
        return self.datasets.get(name, [])


class LLMFixture(TestFixture):
    """Fixture for LLM testing."""

    def setup(self):
        """Set up LLM test fixture."""
        super().setup()

        # Create mock LLM responses
        self.responses = {
            "reasoning": {
                "prompt": "What is 2+2?",
                "response": "4",
                "expected_score": 1.0
            },
            "summarization": {
                "prompt": "Summarize this article...",
                "response": "This is a summary.",
                "expected_score": 0.8
            },
            "classification": {
                "prompt": "Classify this sentiment...",
                "response": "positive",
                "expected_score": 0.9
            }
        }

        self.add_resource("responses", self.responses)

    def get_response(self, task_type: str) -> Dict[str, Any]:
        """Get mock response for task type."""
        return self.responses.get(task_type, {})


class BenchmarkFixture(TestFixture):
    """Fixture for benchmark testing."""

    def setup(self):
        """Set up benchmark test fixture."""
        super().setup()

        # Create test results directory
        self.results_dir = os.path.join(self.temp_dir, "results")
        os.makedirs(self.results_dir)

        # Create sample benchmark results
        self.sample_results = {
            "task_name": "reasoning",
            "compression_method": "llmlingua2",
            "sample_id": 0,
            "original_prompt": "What is 2+2?",
            "compressed_prompt": "What is 2+2?",
            "original_score": 1.0,
            "compressed_score": 0.9,
            "latency": 1.5,
            "memory_usage": 1024
        }

        results_file = os.path.join(self.results_dir, "benchmark_results.json")
        with open(results_file, 'w') as f:
            json.dump([self.sample_results], f)

        self.add_resource("results_dir", self.results_dir)
        self.add_resource("sample_results", self.sample_results)
        self.add_resource("results_file", results_file)

    def get_sample_results(self) -> Dict[str, Any]:
        """Get sample benchmark results."""
        return self.sample_results


# Test data generators
def generate_test_prompts(task_type: str, count: int = 3) -> list:
    """Generate test prompts for a task type."""
    templates = {
        "reasoning": "Solve this math problem: What is {} + {}?",
        "summarization": "Summarize this article: {}",
        "classification": "Classify the sentiment of: {}"
    }

    template = templates.get(task_type, "Test prompt: {}")
    return [template.format(f"test_{i}") for i in range(count)]


def generate_test_ground_truths(task_type: str, count: int = 3) -> list:
    """Generate test ground truths for a task type."""
    templates = {
        "reasoning": "{}",
        "summarization": "Summary of {}",
        "classification": "neutral"
    }

    template = templates.get(task_type, "answer_{}")
    return [template.format(f"test_{i}") for i in range(count)]


def create_mock_evaluator():
    """Create a mock evaluator for testing."""
    from unittest.mock import Mock

    mock_evaluator = Mock()
    mock_evaluator.evaluate.side_effect = lambda prompt, gt: {
        "score": 0.9 if "compressed" in prompt else 1.0,
        "latency": 1.0,
        "tokens": len(prompt.split())
    }

    return mock_evaluator


def create_mock_llm():
    """Create a mock LLM for testing."""
    from unittest.mock import Mock

    mock_llm = Mock()
    mock_llm.get_response.side_effect = lambda prompt: f"Response to: {prompt[:50]}..."

    return mock_llm
