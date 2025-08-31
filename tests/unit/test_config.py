"""
Unit tests for configuration management system.
"""

import unittest
from unittest.mock import Mock, patch
from tests import UnitTestCase
from core.config.config_manager import (
    EnvironmentConfigProvider,
    TaskConfig,
    LLMConfig,
    BenchmarkConfig
)


class TestTaskConfig(UnitTestCase):
    """Test TaskConfig data class."""

    def test_task_config_creation(self):
        """Test creating a TaskConfig instance."""
        config = TaskConfig(
            name="reasoning",
            dataset="gsm8k",
            config="main",
            description="Mathematical reasoning"
        )

        self.assertEqual(config.name, "reasoning")
        self.assertEqual(config.dataset, "gsm8k")
        self.assertEqual(config.config, "main")
        self.assertEqual(config.description, "Mathematical reasoning")


class TestLLMConfig(UnitTestCase):
    """Test LLMConfig data class."""

    def test_llm_config_creation(self):
        """Test creating an LLMConfig instance."""
        config = LLMConfig(
            provider="huggingface",
            model_name="test-model",
            api_key="test-key",
            quantization="none",
            temperature=0.5,
            max_tokens=1000,
            stream_tokens=True,
            unlimited_mode=False
        )

        self.assertEqual(config.provider, "huggingface")
        self.assertEqual(config.model_name, "test-model")
        self.assertEqual(config.api_key, "test-key")
        self.assertEqual(config.quantization, "none")
        self.assertEqual(config.temperature, 0.5)
        self.assertEqual(config.max_tokens, 1000)
        self.assertTrue(config.stream_tokens)
        self.assertFalse(config.unlimited_mode)


class TestBenchmarkConfig(UnitTestCase):
    """Test BenchmarkConfig data class."""

    def test_benchmark_config_creation(self):
        """Test creating a BenchmarkConfig instance."""
        tasks = {
            "reasoning": TaskConfig("reasoning", "gsm8k", "main", "Math reasoning")
        }

        config = BenchmarkConfig(
            tasks=tasks,
            default_task="reasoning",
            num_samples=5,
            compression_methods=["llmlingua2"],
            target_ratio=0.8,
            unlimited_mode=True,
            stream_tokens=False
        )

        self.assertEqual(config.default_task, "reasoning")
        self.assertEqual(config.num_samples, 5)
        self.assertEqual(config.compression_methods, ["llmlingua2"])
        self.assertEqual(config.target_ratio, 0.8)
        self.assertTrue(config.unlimited_mode)
        self.assertFalse(config.stream_tokens)


class TestEnvironmentConfigProvider(UnitTestCase):
    """Test EnvironmentConfigProvider implementation."""

    def setUp(self):
        super().setUp()
        self.provider = EnvironmentConfigProvider()

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_get_llm_config_openai(self):
        """Test getting OpenAI LLM configuration."""
        config = self.provider.get_llm_config("openai")

        self.assertEqual(config.provider, "openai")
        self.assertEqual(config.model_name, "gpt-3.5-turbo")
        self.assertEqual(config.api_key, "test-key")
        self.assertEqual(config.temperature, 0.0)
        self.assertEqual(config.max_tokens, 4096)

    def test_get_llm_config_huggingface(self):
        """Test getting HuggingFace LLM configuration."""
        config = self.provider.get_llm_config("huggingface")

        self.assertEqual(config.provider, "huggingface")
        self.assertEqual(config.model_name, "microsoft/Phi-3.5-mini-instruct")
        self.assertEqual(config.quantization, "none")
        self.assertEqual(config.temperature, 0.0)

    def test_get_llm_config_invalid_provider(self):
        """Test getting config for invalid provider raises error."""
        with self.assertRaises(ValueError):
            self.provider.get_llm_config("invalid_provider")

    def test_get_task_config_valid(self):
        """Test getting valid task configuration."""
        config = self.provider.get_task_config("reasoning")

        self.assertEqual(config.name, "reasoning")
        self.assertEqual(config.dataset, "gsm8k")
        self.assertEqual(config.config, "main")

    def test_get_task_config_invalid(self):
        """Test getting invalid task configuration raises error."""
        with self.assertRaises(ValueError):
            self.provider.get_task_config("invalid_task")

    @patch.dict('os.environ', {'NUM_SAMPLES': '10', 'UNLIMITED_MODE': 'false'})
    def test_get_benchmark_config_with_env(self):
        """Test getting benchmark config with environment variables."""
        config = self.provider.get_benchmark_config()

        self.assertEqual(config.num_samples, 10)
        self.assertFalse(config.unlimited_mode)
        self.assertEqual(config.default_task, "reasoning")
        self.assertEqual(config.compression_methods, ["llmlingua2"])


if __name__ == '__main__':
    unittest.main()
