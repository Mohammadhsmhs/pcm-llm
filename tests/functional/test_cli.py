import unittest
from unittest.mock import Mock, patch
from tests import FunctionalTestCase
import sys

class TestCLICommands(FunctionalTestCase):
    """Functional tests for the refactored CLI commands."""

    def setUp(self):
        super().setUp()
        self.original_argv = sys.argv.copy()

    def tearDown(self):
        super().tearDown()
        sys.argv = self.original_argv

    @patch('core.cli.RunBenchmarkCommand')
    def test_run_default_benchmark_command(self, mock_run_benchmark_command):
        """Test that the default command (no args) runs all benchmarks."""
        sys.argv = ['main.py']
        from core.cli import main
        main()
        # Check that RunBenchmarkCommand was instantiated and executed
        mock_run_benchmark_command.return_value.execute.assert_called_once()

    @patch('core.cli.RunBenchmarkCommand')
    def test_run_task_benchmark_valid_task(self, mock_run_benchmark_command):
        """Test running a benchmark for a single valid task."""
        sys.argv = ['main.py', 'reasoning']
        from core.cli import main
        main()
        # Check that RunBenchmarkCommand was instantiated for 'reasoning' and executed
        mock_run_benchmark_command.assert_called_with(unittest.mock.ANY, ['reasoning'], None)
        mock_run_benchmark_command.return_value.execute.assert_called_once()

    @patch('core.cli.HelpCommand')
    def test_run_task_benchmark_invalid_task(self, mock_help_command):
        """Test that an invalid task shows the help command."""
        sys.argv = ['main.py', 'invalid_task']
        from core.cli import main
        main()
        # Check that HelpCommand was instantiated and executed
        mock_help_command.return_value.execute.assert_called_once()

    @patch('core.cli.RunBenchmarkCommand')
    def test_run_all_benchmarks_command(self, mock_run_benchmark_command):
        """Test the 'all' command."""
        sys.argv = ['main.py', 'all']
        from core.cli import main
        main()
        # Check that RunBenchmarkCommand was instantiated and executed
        mock_run_benchmark_command.return_value.execute.assert_called_once()

    @patch('core.cli.CacheCommand')
    def test_handle_clear_cache_command_no_args(self, mock_cache_command):
        """Test clear-cache command with no arguments."""
        sys.argv = ['main.py', 'clear-cache']
        from core.cli import main
        main()
        mock_cache_command.assert_called_with("clear", None, None)
        mock_cache_command.return_value.execute.assert_called_once()

    @patch('core.cli.CacheCommand')
    def test_handle_clear_cache_command_with_task(self, mock_cache_command):
        """Test clear-cache command with a task argument."""
        sys.argv = ['main.py', 'clear-cache', 'reasoning']
        from core.cli import main
        main()
        mock_cache_command.assert_called_with("clear", "reasoning", None)
        mock_cache_command.return_value.execute.assert_called_once()

    @patch('core.cli.CacheCommand')
    def test_handle_clear_cache_command_with_task_and_method(self, mock_cache_command):
        """Test clear-cache command with task and method arguments."""
        sys.argv = ['main.py', 'clear-cache', 'reasoning', 'llmlingua2']
        from core.cli import main
        main()
        mock_cache_command.assert_called_with("clear", "reasoning", "llmlingua2")
        mock_cache_command.return_value.execute.assert_called_once()

    @patch('core.cli.CacheCommand')
    def test_handle_cache_info_command(self, mock_cache_command):
        """Test the cache-info command."""
        sys.argv = ['main.py', 'cache-info']
        from core.cli import main
        main()
        mock_cache_command.assert_called_with("info")
        mock_cache_command.return_value.execute.assert_called_once()

class TestCLIMainFunction(FunctionalTestCase):
    """Functional tests for the main CLI function and command routing."""

    def setUp(self):
        super().setUp()
        self.original_argv = sys.argv.copy()

    def tearDown(self):
        super().tearDown()
        sys.argv = self.original_argv

    @patch('core.cli.CLIApplication.run')
    def test_main_entry_point(self, mock_cli_run):
        """Test that the main function constructs a CLIApplication and runs it."""
        sys.argv = ['main.py', 'some-command']
        from core.cli import main
        main()
        mock_cli_run.assert_called_once_with(['some-command'])

    @patch('core.cli.HelpCommand')
    def test_main_help_command(self, mock_help_command):
        """Test that 'help' command instantiates HelpCommand."""
        sys.argv = ['main.py', 'help']
        from core.cli import main
        main()
        mock_help_command.return_value.execute.assert_called_once()

    @patch('core.cli.RunBenchmarkCommand')
    def test_main_task_command_with_sample(self, mock_run_benchmark_command):
        """Test a task command with a --sample option."""
        sys.argv = ['main.py', 'reasoning', '--sample', '50']
        from core.cli import main
        main()
        mock_run_benchmark_command.assert_called_with(unittest.mock.ANY, ['reasoning'], 50)
        mock_run_benchmark_command.return_value.execute.assert_called_once()

    @patch('core.cli.HelpCommand')
    def test_main_unknown_command(self, mock_help_command):
        """Test that an unknown command shows help."""
        sys.argv = ['main.py', 'nonexistent-command']
        from core.cli import main
        main()
        mock_help_command.return_value.execute.assert_called_once()

if __name__ == '__main__':
    unittest.main()
