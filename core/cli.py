"""
Command Line Interface for PCM-LLM.
"""

import sys
from abc import ABC, abstractmethod
from typing import List, Optional

from core.benchmark_service import IBenchmarkService
from core.bootstrap import get_app
from utils.cache.cache_utils import clear_compression_cache, show_cache_info


class ICommand(ABC):
    """Interface for CLI commands."""

    @abstractmethod
    def execute(self) -> Optional[int]:
        """Execute the command."""
        pass


class RunBenchmarkCommand(ICommand):
    """Command for running single or multiple benchmarks."""

    def __init__(
        self,
        benchmark_service: IBenchmarkService,
        task_names: List[str],
        num_samples: Optional[int] = None,
        unlimited_mode: Optional[bool] = None,
    ):
        self.benchmark_service = benchmark_service
        self.task_names = task_names
        self.num_samples = num_samples
        self.unlimited_mode = unlimited_mode

    def execute(self) -> Optional[int]:
        """Execute benchmark command."""
        try:
            # Override unlimited mode if specified
            if self.unlimited_mode is not None:
                from core.config import settings

                original_unlimited = settings.evaluation.unlimited_mode
                settings.evaluation.unlimited_mode = self.unlimited_mode
                try:
                    self.benchmark_service.run_multi_task_benchmark(
                        self.task_names, self.num_samples
                    )
                finally:
                    # Restore original setting
                    settings.evaluation.unlimited_mode = original_unlimited
            else:
                self.benchmark_service.run_multi_task_benchmark(
                    self.task_names, self.num_samples
                )
            return 0
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            return 1


class CacheCommand(ICommand):
    """Command for cache operations."""

    def __init__(
        self, operation: str, task: Optional[str] = None, method: Optional[str] = None
    ):
        self.operation = operation
        self.task = task
        self.method = method

    def execute(self) -> Optional[int]:
        """Execute cache command."""
        try:
            if self.operation == "clear":
                clear_compression_cache(self.task, self.method)
                if self.task and self.method:
                    print(f"✅ Cleared cache for {self.task}/{self.method}")
                elif self.task:
                    print(f"✅ Cleared cache for {self.task}")
                else:
                    print("✅ Cleared entire cache")
            elif self.operation == "info":
                show_cache_info()
            return 0
        except Exception as e:
            print(f"❌ Cache operation failed: {e}")
            return 1


class HelpCommand(ICommand):
    """Command for showing help information."""

    def execute(self) -> Optional[int]:
        """Execute help command."""
        self.show_help()
        return 0

    def show_help(self):
        """Display help information."""
        print("🚀 PCM-LLM: Prompt Compression Benchmark Tool")
        print("=" * 50)
        print("\nUsage: pcm-llm [command] [options]")
        print("\nCommands:")
        print("  all                 Run all benchmarks (default)")
        print("  reasoning           Run reasoning benchmark")
        print("  summarization       Run summarization benchmark")
        print("  classification      Run classification benchmark")
        print("  cache-info          Show cache status")
        print(
            "  clear-cache [task] [method]  Clear cache (all, or specific task/method)"
        )
        print("  help                Show this help message")
        print("\nOptions:")
        print(
            "  --sample N  or --samples N    Number of samples to run (overrides config)"
        )
        print(
            "  --unlimited                   Disable timeout restrictions (unlimited mode)"
        )
        print("\nExamples:")
        print("  pcm-llm all --sample 10")
        print("  pcm-llm reasoning --samples 5 --unlimited")
        print("  pcm-llm clear-cache reasoning llmlingua2")


class CLIApplication:
    """Main CLI application to parse arguments and execute commands."""

    def __init__(self):
        self.app = get_app()
        self.config_provider = self.app.get_config_provider()
        self.benchmark_config = self.config_provider.get_benchmark_config()
        self.benchmark_service = self.app.get_benchmark_service()

    def create_command(self, args: List[str]) -> ICommand:
        """Factory method to create command objects from arguments."""
        if not args:
            return RunBenchmarkCommand(
                self.benchmark_service, list(self.benchmark_config.tasks.keys())
            )

        # Parse options first
        num_samples = None
        unlimited_mode = None
        command_args = []

        i = 0
        while i < len(args):
            if args[i] in ["--sample", "--samples"]:
                try:
                    num_samples = int(args[i + 1])
                    i += 2  # Skip the option and its value
                except (IndexError, ValueError):
                    print("❌ Invalid number for --sample option.")
                    return HelpCommand()
            elif args[i] == "--unlimited":
                unlimited_mode = True
                i += 1  # Skip the flag
            else:
                command_args.append(args[i])
                i += 1

        # Now determine the command from remaining args
        if not command_args:
            return RunBenchmarkCommand(
                self.benchmark_service,
                list(self.benchmark_config.tasks.keys()),
                num_samples,
                unlimited_mode,
            )

        command = command_args[0].lower()

        if command in ["help", "-h", "--help"]:
            return HelpCommand()
        if command == "all":
            return RunBenchmarkCommand(
                self.benchmark_service,
                list(self.benchmark_config.tasks.keys()),
                num_samples,
                unlimited_mode,
            )
        if command in self.benchmark_config.tasks:
            return RunBenchmarkCommand(
                self.benchmark_service, [command], num_samples, unlimited_mode
            )
        if command == "clear-cache":
            task = command_args[1] if len(command_args) > 1 else None
            method = command_args[2] if len(command_args) > 2 else None
            return CacheCommand("clear", task, method)
        if command == "cache-info":
            return CacheCommand("info")

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
        cli = CLIApplication()
        return cli.run(sys.argv[1:])
    except Exception as e:
        print(f"❌ Application initialization failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
