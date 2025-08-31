"""
Refactored Command Line Interface following SOLID principles.
"""

import sys
from typing import Optional
from abc import ABC, abstractmethod

from core.config import IConfigProvider, config_provider
from core.llm_factory import create_llm_factory
from core.benchmark_service import (
    IBenchmarkService,
    BenchmarkService,
    DataLoaderAdapter
)
from utils.logger import BenchmarkLogger
from utils.run_info_logger import RunInfoLogger
from utils.cache_utils import clear_compression_cache, show_cache_info


class ICommand(ABC):
    """Interface for CLI commands."""

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

    def __init__(self, benchmark_service: IBenchmarkService, task_name: Optional[str] = None):
        self.benchmark_service = benchmark_service
        self.task_name = task_name

    def execute(self) -> Optional[int]:
        """Execute benchmark command."""
        try:
            if self.task_name:
                self.benchmark_service.run_single_task_benchmark(self.task_name)
            else:
                # Run default task
                config = self.benchmark_service.config_provider.get_benchmark_config()
                self.benchmark_service.run_single_task_benchmark(config.default_task)
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

    def __init__(self, benchmark_service: IBenchmarkService, task_names: list[str]):
        self.benchmark_service = benchmark_service
        self.task_names = task_names

    def execute(self) -> Optional[int]:
        """Execute multi-benchmark command."""
        try:
            self.benchmark_service.run_multi_task_benchmark(self.task_names)
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
                elif self.task:
                    clear_compression_cache(self.task)
                else:
                    clear_compression_cache()
                print("✅ Cache cleared successfully")
            elif self.operation == "info":
                show_cache_info()
            return 0
        except Exception as e:
            print(f"❌ Cache operation failed: {e}")
            return 1

    def get_description(self) -> str:
        """Get command description."""
        return f"Cache {self.operation} operation"


class HelpCommand(ICommand):
    """Command for displaying help."""

    def execute(self) -> Optional[int]:
        """Execute help command."""
        self.show_help()
        return 0

    def get_description(self) -> str:
        """Get command description."""
        return "Show help information"

    def show_help(self):
        """Display help information."""
        print("PCM-LLM Benchmark Tool")
        print("=" * 50)
        print("Usage: python main.py [command] [options]")
        print("")
        print("Commands:")
        print("  (no command)     Run default benchmark")
        print("  reasoning        Run reasoning benchmark only")
        print("  summarization    Run summarization benchmark only")
        print("  classification   Run classification benchmark only")
        print("  all              Run all benchmarks")
        print("  clear-cache      Clear compression cache")
        print("  cache-info       Show cache information")
        print("  help             Show this help message")
        print("")
        print("Examples:")
        print("  python main.py                    # Run default task")
        print("  python main.py reasoning          # Run reasoning only")
        print("  python main.py all                # Run all tasks")
        print("  python main.py clear-cache        # Clear all cache")
        print("  python main.py clear-cache reasoning llmlingua2  # Clear specific cache")


class CLIApplication:
    """Main CLI application following SOLID principles."""

    def __init__(self, config_provider: IConfigProvider):
        self.config_provider = config_provider
        self.benchmark_config = config_provider.get_benchmark_config()

        # Initialize dependencies
        self.llm_factory = create_llm_factory(config_provider)
        self.logger = BenchmarkLogger(log_dir="results")
        self.run_info_logger = RunInfoLogger(log_dir="results")
        self.data_loader = DataLoaderAdapter()

        # Create benchmark service
        self.benchmark_service = BenchmarkService(
            config_provider=config_provider,
            llm_factory=self.llm_factory,
            data_loader=self.data_loader,
            logger=self.logger,
            run_info_logger=self.run_info_logger
        )

    def create_command(self, args: list[str]) -> ICommand:
        """Create command from arguments."""
        if len(args) == 0:
            return BenchmarkCommand(self.benchmark_service)

        command = args[0].lower()

        if command == "help" or command == "-h" or command == "--help":
            return HelpCommand()
        elif command == "all":
            return MultiBenchmarkCommand(self.benchmark_service, list(self.benchmark_config.tasks.keys()))
        elif command == "clear-cache":
            task = args[1] if len(args) > 1 else None
            method = args[2] if len(args) > 2 else None
            return CacheCommand("clear", task, method)
        elif command == "cache-info":
            return CacheCommand("info")
        elif command in self.benchmark_config.tasks:
            return BenchmarkCommand(self.benchmark_service, command)
        else:
            print(f"❌ Unknown command: {command}")
            return HelpCommand()

    def run(self, args: list[str]) -> int:
        """Run the CLI application."""
        command = self.create_command(args)
        return command.execute() or 0


def main():
    """Main CLI entry point."""
    app = CLIApplication(config_provider)
    return app.run(sys.argv[1:] if len(sys.argv) > 1 else [])


if __name__ == "__main__":
    main()

import sys
import os
from config import OPENROUTER_RATE_LIMIT_RPM
from utils.cache_utils import clear_compression_cache, show_cache_info
from core.benchmark_executor import BenchmarkExecutor


def show_help():
    """Display help information."""
    print("PCM-LLM Benchmark Tool")
    print("=" * 50)
    print("Usage: python main.py [command] [options]")
    print("")
    print("Commands:")
    print("  (no command)     Run default benchmark")
    print("  reasoning        Run reasoning benchmark only")
    print("  summarization    Run summarization benchmark only")
    print("  classification   Run classification benchmark only")
    print("  all              Run all benchmarks")
    print("  clear-cache      Clear compression cache")
    print("  cache-info       Show cache information")
    print("  rate-limit-info  Show OpenRouter rate limit info")
    print("  help             Show this help message")
    print("")
    print("Examples:")
    print("  python main.py                    # Run default task")
    print("  python main.py reasoning          # Run reasoning only")
    print("  python main.py all                # Run all tasks")
    print("  python main.py clear-cache        # Clear all cache")
    print("  python main.py clear-cache reasoning llmlingua2  # Clear specific cache")


def handle_clear_cache_command():
    """Handle clear-cache command."""
    if len(sys.argv) > 2:
        task = sys.argv[2]
        if len(sys.argv) > 3:
            method = sys.argv[3]
            clear_compression_cache(task, method)
        else:
            clear_compression_cache(task)
    else:
        clear_compression_cache()


def handle_cache_info_command():
    """Handle cache-info command."""
    show_cache_info()


def handle_rate_limit_info_command():
    """Handle rate-limit-info command."""
    print("📊 OpenRouter Rate Limit Information:")
    print("=" * 50)
    print("Free Tier Limits:")
    print("  • 16 requests per minute")
    print("  • No daily limits mentioned")
    print("")
    print("Current Configuration:")
    print(f"  • Configured limit: {OPENROUTER_RATE_LIMIT_RPM} RPM")
    print("")
    print("Tips to avoid rate limits:")
    print("  • Reduce NUM_SAMPLES_TO_RUN in config.py")
    print("  • Add delays between benchmark runs")
    print("  • Consider upgrading to a paid plan")
    print("  • Use a different provider (OpenAI, HuggingFace, etc.)")
    print("")
    print("Alternative Models:")
    print("  • deepseek/deepseek-chat:free (may have different limits)")
    print("  • microsoft/wizardlm-2-8x22b:free")
    print("  • meta-llama/llama-3.1-8b-instruct:free")


def run_default_benchmark():
    """Run the default benchmark."""
    from config import DEFAULT_TASK
    executor = BenchmarkExecutor()
    return executor.run_single_task_benchmark(DEFAULT_TASK)


def run_task_benchmark(task_name: str):
    """Run benchmark for a specific task."""
    from config import SUPPORTED_TASKS
    if task_name not in SUPPORTED_TASKS:
        print(f"❌ Unknown task: {task_name}")
        print(f"Supported tasks: {', '.join(SUPPORTED_TASKS)}")
        return None

    executor = BenchmarkExecutor()
    return executor.run_single_task_benchmark(task_name)


def run_all_benchmarks():
    """Run all benchmarks."""
    from config import SUPPORTED_TASKS
    executor = BenchmarkExecutor()
    return executor.run_multi_task_benchmark(SUPPORTED_TASKS)


def main():
    """Main CLI entry point."""
    if len(sys.argv) == 1:
        # No arguments - run default
        return run_default_benchmark()

    command = sys.argv[1].lower()

    if command == "help" or command == "-h" or command == "--help":
        show_help()
    elif command == "clear-cache":
        handle_clear_cache_command()
    elif command == "cache-info":
        handle_cache_info_command()
    elif command == "rate-limit-info":
        handle_rate_limit_info_command()
    elif command == "all":
        return run_all_benchmarks()
    elif command in ["reasoning", "summarization", "classification"]:
        return run_task_benchmark(command)
    else:
        print(f"❌ Unknown command: {command}")
        show_help()
        return None


if __name__ == "__main__":
    main()
