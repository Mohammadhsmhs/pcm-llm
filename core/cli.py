"""
Command Line Interface for PCM-LLM Benchmark.
"""

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
