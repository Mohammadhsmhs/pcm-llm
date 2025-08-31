"""
Integration tests for benchmark service.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from tests import IntegrationTestCase
from core.benchmark_service import BenchmarkService, DataLoaderAdapter
from core.config import EnvironmentConfigProvider
from core.llm_factory import LLMFactory
from utils.logger import BenchmarkLogger
from utils.run_info_logger import RunInfoLogger


class TestDataLoaderAdapter(IntegrationTestCase):
    """Test DataLoaderAdapter integration."""

    def setUp(self):
        super().setUp()
        self.adapter = DataLoaderAdapter()

    @patch('core.benchmark_service.load_benchmark_dataset')
    def test_load_dataset_reasoning(self, mock_load_dataset):
        """Test loading reasoning dataset."""
        # Mock dataset
        mock_dataset = [
            {"question": "What is 2+2?", "answer": "4"},
            {"question": "What is 3+3?", "answer": "6"}
        ]
        mock_load_dataset.return_value = mock_dataset

        from core.config import TaskConfig
        task_config = TaskConfig(
            name="reasoning",
            dataset="gsm8k",
            config="main",
            description="Math reasoning"
        )

        prompts, ground_truths = self.adapter.load_dataset(task_config, 2)

        self.assertEqual(len(prompts), 2)
        self.assertEqual(len(ground_truths), 2)
        self.assertEqual(prompts[0], "What is 2+2?")
        self.assertEqual(ground_truths[0], "4")

    @patch('core.benchmark_service.load_benchmark_dataset')
    def test_load_dataset_summarization(self, mock_load_dataset):
        """Test loading summarization dataset."""
        mock_dataset = [
            {"article": "Test article", "highlights": "Test summary"},
            {"article": "Another article", "highlights": "Another summary"}
        ]
        mock_load_dataset.return_value = mock_dataset

        from core.config import TaskConfig
        task_config = TaskConfig(
            name="summarization",
            dataset="cnn_dailymail",
            config="3.0.0",
            description="News summarization"
        )

        prompts, ground_truths = self.adapter.load_dataset(task_config, 2)

        self.assertEqual(len(prompts), 2)
        self.assertEqual(len(ground_truths), 2)
        self.assertEqual(prompts[0], "Test article")
        self.assertEqual(ground_truths[0], "Test summary")


class TestBenchmarkServiceIntegration(IntegrationTestCase):
    """Integration tests for BenchmarkService."""

    def setUp(self):
        super().setUp()

        # Create mocks
        self.config_provider = Mock(spec=EnvironmentConfigProvider)
        self.llm_factory = Mock(spec=LLMFactory)
        self.logger = Mock(spec=BenchmarkLogger)
        self.run_info_logger = Mock(spec=RunInfoLogger)

        # Setup mock logger attributes
        self.logger.log_dir = "/tmp/test_logs"

        # Setup mock config
        mock_benchmark_config = Mock()
        mock_benchmark_config.num_samples = 1
        mock_benchmark_config.compression_methods = ["llmlingua2"]
        mock_benchmark_config.target_ratio = 0.8

        mock_task_config = Mock()
        mock_task_config.name = "reasoning"

        self.config_provider.get_benchmark_config.return_value = mock_benchmark_config
        self.config_provider.get_task_config.return_value = mock_task_config

        # Create service
        self.service = BenchmarkService(
            self.config_provider,
            self.llm_factory,
            DataLoaderAdapter(),
            self.logger,
            self.run_info_logger
        )

    @patch('core.benchmark_service.CompressorFactory')
    @patch('core.benchmark_service.Evaluator')
    def test_run_single_task_benchmark_integration(self, mock_evaluator_class, mock_compressor_factory):
        """Test running single task benchmark with mocked dependencies."""
        # Setup mocks
        mock_llm = Mock()
        mock_llm_factory = Mock()
        mock_llm_factory.create_llm.return_value = mock_llm
        self.llm_factory = mock_llm_factory

        mock_evaluator = Mock()
        mock_evaluator.evaluate.side_effect = [
            {"score": 1.0, "latency": 1.0},  # baseline
            {"score": 0.9, "latency": 1.2}   # compressed
        ]
        mock_evaluator_class.return_value = mock_evaluator

        mock_compressor = Mock()
        mock_compressor.compress.return_value = "compressed prompt"
        mock_compressor_factory.create.return_value = mock_compressor

        # Mock dataset loading
        with patch.object(DataLoaderAdapter, 'load_dataset', return_value=(["test prompt"], ["test answer"])):
            results = self.service.run_single_task_benchmark("reasoning")

        # Verify results
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.task_name, "reasoning")
        self.assertEqual(result.compression_method, "llmlingua2")
        self.assertEqual(result.original_prompt, "test prompt")
        self.assertEqual(result.compressed_prompt, "compressed prompt")
        self.assertEqual(result.original_score, 1.0)
        self.assertEqual(result.compressed_score, 0.9)

        # Verify mocks were called
        self.logger.log_result.assert_called()
        self.logger.finalize_and_save.assert_called()

    def test_run_multi_task_benchmark(self):
        """Test running multiple task benchmarks."""
        with patch.object(self.service, 'run_single_task_benchmark') as mock_run_single:
            mock_run_single.side_effect = [
                [Mock(task_name="reasoning")],
                [Mock(task_name="summarization")]
            ]

            results = self.service.run_multi_task_benchmark(["reasoning", "summarization"])

            self.assertIn("reasoning", results)
            self.assertIn("summarization", results)
            self.assertEqual(len(results["reasoning"]), 1)
            self.assertEqual(len(results["summarization"]), 1)


if __name__ == '__main__':
    unittest.main()
