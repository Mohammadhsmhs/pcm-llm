"""
Functional tests for CLI commands.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from tests import FunctionalTestCase
import sys
import os


class TestCLICommands(FunctionalTestCase):
    """Functional tests for CLI commands."""

    def setUp(self):
        super().setUp()
        # Save original argv
        self.original_argv = sys.argv.copy()

    def tearDown(self):
        super().tearDown()
        # Restore original argv
        sys.argv = self.original_argv

    @patch('core.cli.BenchmarkExecutor')
    def test_run_default_benchmark_command(self, mock_executor_class):
        """Test running default benchmark command."""
        mock_executor = Mock()
        mock_executor.run_single_task_benchmark.return_value = "success"
        mock_executor_class.return_value = mock_executor

        # Test the CLI command pattern
        from core.cli import BenchmarkCommand

        command = BenchmarkCommand(None)  # benchmark_service would be mocked in real test
        self.assertIsNotNone(command.get_description())

    @patch('core.cli.BenchmarkExecutor')
    def test_run_task_benchmark_valid_task(self, mock_executor_class):
        """Test running benchmark for valid task."""
        mock_executor = Mock()
        mock_executor.run_single_task_benchmark.return_value = "success"
        mock_executor_class.return_value = mock_executor

        from core.cli import run_task_benchmark

        result = run_task_benchmark("reasoning")

        mock_executor.run_single_task_benchmark.assert_called_once_with("reasoning")

    def test_run_task_benchmark_invalid_task(self):
        """Test running benchmark for invalid task."""
        from core.cli import run_task_benchmark
        from core.config import settings

        # Mock the tasks in settings
        with patch.object(settings.benchmark, 'tasks', {'reasoning': Mock(), 'summarization': Mock()}):
            result = run_task_benchmark("invalid_task")

            self.assertIsNone(result)

    @patch('core.cli.BenchmarkExecutor')
    def test_run_all_benchmarks_command(self, mock_executor_class):
        """Test running all benchmarks command."""
        mock_executor = Mock()
        mock_executor.run_multi_task_benchmark.return_value = {"results": "success"}
        mock_executor_class.return_value = mock_executor

        from core.config import settings
        with patch.object(settings.benchmark, 'tasks', {'reasoning': Mock(), 'summarization': Mock()}):
            from core.cli import run_all_benchmarks

            result = run_all_benchmarks()

            mock_executor.run_multi_task_benchmark.assert_called_once_with(['reasoning', 'summarization'])

    @patch('utils.cache.cache_utils.clear_compression_cache')
    def test_handle_clear_cache_command_no_args(self, mock_clear_cache):
        """Test clear cache command with no arguments."""
        sys.argv = ['main.py', 'clear-cache']

        from core.cli import handle_clear_cache_command

        handle_clear_cache_command()

        mock_clear_cache.assert_called_once()

    @patch('utils.cache.cache_utils.clear_compression_cache')
    def test_handle_clear_cache_command_with_task(self, mock_clear_cache):
        """Test clear cache command with task argument."""
        sys.argv = ['main.py', 'clear-cache', 'reasoning']

        from core.cli import handle_clear_cache_command

        handle_clear_cache_command()

        mock_clear_cache.assert_called_once_with('reasoning')

    @patch('utils.cache.cache_utils.clear_compression_cache')
    def test_handle_clear_cache_command_with_task_and_method(self, mock_clear_cache):
        """Test clear cache command with task and method arguments."""
        sys.argv = ['main.py', 'clear-cache', 'reasoning', 'llmlingua2']

        from core.cli import handle_clear_cache_command

        handle_clear_cache_command()

        mock_clear_cache.assert_called_once_with('reasoning', 'llmlingua2')

    @patch('utils.cache.cache_utils.show_cache_info')
    def test_handle_cache_info_command(self, mock_show_cache):
        """Test cache info command."""
        from core.cli import handle_cache_info_command

        handle_cache_info_command()

        mock_show_cache.assert_called_once()

    @patch('builtins.print')
    def test_handle_rate_limit_info_command(self, mock_print):
        """Test rate limit info command."""
        from core.cli import handle_rate_limit_info_command

        handle_rate_limit_info_command()

        # Verify that print was called multiple times with rate limit info
        self.assertTrue(mock_print.called)
        call_args_list = mock_print.call_args_list
        self.assertTrue(any("OpenRouter Rate Limit Information" in str(call) for call in call_args_list))


class TestCLIMainFunction(FunctionalTestCase):
    """Functional tests for main CLI function."""

    def setUp(self):
        super().setUp()
        self.original_argv = sys.argv.copy()

    def tearDown(self):
        super().tearDown()
        sys.argv = self.original_argv

    @patch('core.cli.run_default_benchmark')
    def test_main_no_arguments(self, mock_run_default):
        """Test main function with no arguments."""
        sys.argv = ['main.py']

        from core.cli import main

        main()

        mock_run_default.assert_called_once()

    @patch('core.cli.show_help')
    def test_main_help_command(self, mock_show_help):
        """Test main function with help command."""
        sys.argv = ['main.py', 'help']

        from core.cli import main

        main()

        mock_show_help.assert_called_once()

    @patch('core.cli.run_task_benchmark')
    def test_main_reasoning_command(self, mock_run_task):
        """Test main function with reasoning command."""
        sys.argv = ['main.py', 'reasoning']

        from core.cli import main

        main()

        mock_run_task.assert_called_once_with('reasoning')

    @patch('core.cli.run_all_benchmarks')
    def test_main_all_command(self, mock_run_all):
        """Test main function with all command."""
        sys.argv = ['main.py', 'all']

        from core.cli import main

        main()

        mock_run_all.assert_called_once()

    @patch('core.cli.handle_clear_cache_command')
    def test_main_clear_cache_command(self, mock_clear_cache):
        """Test main function with clear-cache command."""
        sys.argv = ['main.py', 'clear-cache']

        from core.cli import main

        main()

        mock_clear_cache.assert_called_once()

    @patch('core.cli.show_help')
    def test_main_unknown_command(self, mock_show_help):
        """Test main function with unknown command."""
        sys.argv = ['main.py', 'unknown_command']

        from core.cli import main

        main()

        mock_show_help.assert_called_once()


if __name__ == '__main__':
    unittest.main()
