"""
Command Line Interface following SOLID principles.
Uses dependency injection and proper abstraction layers.
"""

import sys
from typing import Optional, List
from abc import ABC, abstractmethod

from core.bootstrap import get_app
from core.config import IConfigProvider
from core.llm_factory import ILLMFactory
from core.benchmark_service import IBenchmarkService
from utils.logger import BenchmarkLogger
from utils.run_info_logger import RunInfoLogger
from utils.cache_utils import clear_compression_cache, show_cache_info


class ICommand(ABC):
    """Interface for CLI commands following Interface Segregation Principle."""

    @abstractmethod
    def execute(self) -> Optional[int]:
        """Execute the command."""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Get command description."""
        pass


class BenchmarkCommand(ICommand):
    """Command for running benchmarks."""

    def __init__(self, benchmark_service: IBenchmarkService, task_name: Optional[str] = None, num_samples: Optional[int] = None):
        self.benchmark_service = benchmark_service
        self.task_name = task_name
        self.num_samples = num_samples

    def execute(self) -> Optional[int]:
        """Execute benchmark command."""
        try:
            if self.task_name:
                self.benchmark_service.run_single_task_benchmark(self.task_name, self.num_samples)
            else:
                # Run default task
                config = self.benchmark_service.config_provider.get_benchmark_config()
                self.benchmark_service.run_single_task_benchmark(config.default_task, self.num_samples)
            return 0
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            return 1

    def get_description(self) -> str:
        """Get command description."""
        if self.task_name:
            return f"Run {self.task_name} benchmark"
        return "Run default benchmark"


class MultiBenchmarkCommand(ICommand):
    """Command for running multiple benchmarks."""

    def __init__(self, benchmark_service: IBenchmarkService, task_names: List[str], num_samples: Optional[int] = None):
        self.benchmark_service = benchmark_service
        self.task_names = task_names
        self.num_samples = num_samples

    def execute(self) -> Optional[int]:
        """Execute multi-benchmark command."""
        try:
            self.benchmark_service.run_multi_task_benchmark(self.task_names, self.num_samples)
            return 0
        except Exception as e:
            print(f"❌ Multi-benchmark failed: {e}")
            return 1

    def get_description(self) -> str:
        """Get command description."""
        return f"Run benchmarks for: {', '.join(self.task_names)}"


class CacheCommand(ICommand):
    """Command for cache operations."""

    def __init__(self, operation: str, task: Optional[str] = None, method: Optional[str] = None):
        self.operation = operation
        self.task = task
        self.method = method

    def execute(self) -> Optional[int]:
        """Execute cache command."""
        try:
            if self.operation == "clear":
                if self.task and self.method:
                    clear_compression_cache(self.task, self.method)
                    print(f"✅ Cleared cache for {self.task}/{self.method}")
                elif self.task:
                    clear_compression_cache(self.task)
                    print(f"✅ Cleared cache for {self.task}")
                else:
                    clear_compression_cache()
                    print("✅ Cleared entire cache")
            elif self.operation == "info":
                show_cache_info()
            return 0
        except Exception as e:
            print(f"❌ Cache operation failed: {e}")
            return 1

    def get_description(self) -> str:
        """Get command description."""
        if self.operation == "clear":
            if self.task and self.method:
                return f"Clear cache for {self.task}/{self.method}"
            elif self.task:
                return f"Clear cache for {self.task}"
            else:
                return "Clear entire cache"
        elif self.operation == "info":
            return "Show cache information"
        return f"Cache operation: {self.operation}"


class HelpCommand(ICommand):
    """Command for showing help information."""

    def execute(self) -> Optional[int]:
        """Execute help command."""
        self.show_help()
        return 0

    def get_description(self) -> str:
        """Get command description."""
        return "Show help information"

    def show_help(self):
        """Display help information."""
        print("🚀 PCM-LLM: Prompt Compression Benchmark Tool")
        print("=" * 50)
        print()
        print("Usage:")
        print("  python main.py [command] [options]")
        print()
        print("Commands:")
        print("  (no args)           Run default benchmark (reasoning)")
        print("  reasoning           Run reasoning benchmark")
        print("  summarization       Run summarization benchmark")
        print("  classification      Run classification benchmark")
        print("  all                 Run all benchmarks")
        print("  help                Show this help message")
        print()
        print("Options:")
        print("  --samples N         Number of samples to run (overrides config)")
        print()
        print("Cache Management:")
        print("  cache-info          Show cache status")
        print("  clear-cache         Clear entire cache")
        print("  clear-cache reasoning  # Clear specific task cache")
        print("  clear-cache reasoning llmlingua2  # Clear specific method for a task")
        print()
        print("Examples:")
        print("  python main.py reasoning")
        print("  python main.py all --samples 10")
        print("  python main.py reasoning --samples 5")
        print("  python main.py clear-cache reasoning")


class CLIApplication:
    """Main CLI application following SOLID principles."""

    def __init__(self):
        # Initialize application
        self.app = get_app()
        
        # Get dependencies from bootstrap
        self.config_provider = self.app.get_config_provider()
        self.benchmark_config = self.config_provider.get_benchmark_config()
        self.llm_factory = self.app.get_llm_factory()
        self.benchmark_service = self.app.get_benchmark_service()

    def parse_args(self, args: List[str]) -> tuple:
        """Parse command line arguments to extract command, task, and options."""
        command = None
        task = None
        num_samples = None
        
        i = 0
        while i < len(args):
            arg = args[i]
            if arg == "--samples" and i + 1 < len(args):
                try:
                    num_samples = int(args[i + 1])
                    i += 2
                except ValueError:
                    print(f"❌ Invalid value for --samples: {args[i + 1]}")
                    i += 2
            elif not command:
                command = arg
                i += 1
            elif not task and command in self.benchmark_config.tasks:
                task = arg
                i += 1
            else:
                i += 1
        
        return command, task, num_samples

    def create_command(self, args: List[str]) -> ICommand:
        """Create command from arguments following Factory Pattern."""
        if len(args) == 0:
            return BenchmarkCommand(self.benchmark_service)

        command, task, num_samples = self.parse_args(args)

        if not command:
            return BenchmarkCommand(self.benchmark_service, num_samples=num_samples)

        command = command.lower()

        if command in ["help", "-h", "--help"]:
            return HelpCommand()
        elif command == "all":
            return MultiBenchmarkCommand(self.benchmark_service, list(self.benchmark_config.tasks.keys()), num_samples)
        elif command == "clear-cache":
            # Handle cache commands with remaining args
            remaining_args = args[1:]  # Skip the 'clear-cache' command
            cache_task = remaining_args[0] if len(remaining_args) > 0 else None
            cache_method = remaining_args[1] if len(remaining_args) > 1 else None
            return CacheCommand("clear", cache_task, cache_method)
        elif command == "cache-info":
            return CacheCommand("info")
        elif command in self.benchmark_config.tasks:
            return BenchmarkCommand(self.benchmark_service, command, num_samples)
        else:
            print(f"❌ Unknown command: {command}")
            return HelpCommand()

    def run(self, args: List[str]) -> int:
        """Run the CLI application."""
        try:
            command = self.create_command(args)
            return command.execute() or 0
        except Exception as e:
            print(f"❌ CLI execution failed: {e}")
            return 1


def main():
    """Main CLI entry point."""
    try:
        # Initialize application
        app = get_app()
        
        # Create and run CLI
        cli = CLIApplication()
        return cli.run(sys.argv[1:])
    except Exception as e:
        print(f"❌ Application initialization failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
