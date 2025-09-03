"""
Test configuration for PCM-LLM test suite.
"""

import os
from pathlib import Path
from typing import Dict, Any, List


class TestConfig:
    """Configuration for running tests."""

    def __init__(self):
        """Initialize test configuration."""
        self.project_root = Path(__file__).parent
        self.tests_dir = self.project_root / "tests"
        self.core_dir = self.project_root / "core"
        self.utils_dir = self.project_root / "utils"

        # Test settings
        self.test_settings = {
            "timeout": 300,  # 5 minutes
            "parallel": True,
            "workers": 4,
            "coverage_threshold": 80,
            "slow_test_threshold": 10,  # seconds
        }

        # Test categories
        self.test_categories = {
            "unit": {
                "path": "tests/unit",
                "description": "Unit tests for individual components",
                "expected_count": 15
            },
            "integration": {
                "path": "tests/integration",
                "description": "Integration tests for component interactions",
                "expected_count": 5
            },
            "functional": {
                "path": "tests/functional",
                "description": "Functional tests for end-to-end workflows",
                "expected_count": 3
            }
        }

        # Mock configurations
        self.mock_configs = {
            "llm_responses": {
                "reasoning": "The answer is 42.",
                "summarization": "This is a summary of the text.",
                "classification": "positive"
            },
            "api_responses": {
                "huggingface": {"status": "success", "model": "test-model"},
                "openai": {"status": "success", "model": "gpt-3.5-turbo"}
            }
        }

    def get_test_paths(self) -> Dict[str, Path]:
        """Get all test directory paths."""
        return {
            category: self.project_root / config["path"]
            for category, config in self.test_categories.items()
        }

    def get_coverage_paths(self) -> List[str]:
        """Get paths to include in coverage analysis."""
        return [
            str(self.core_dir),
            str(self.utils_dir)
        ]

    def get_test_settings(self) -> Dict[str, Any]:
        """Get test settings."""
        return self.test_settings

    def get_mock_config(self, key: str) -> Any:
        """Get mock configuration by key."""
        return self.mock_configs.get(key)

    def should_run_slow_tests(self) -> bool:
        """Check if slow tests should be run."""
        return os.getenv("RUN_SLOW_TESTS", "false").lower() == "true"

    def get_ci_settings(self) -> Dict[str, Any]:
        """Get CI-specific test settings."""
        return {
            "parallel": False,  # Disable parallel in CI
            "coverage_report": "xml",
            "junit_report": True,
            "fail_on_missing_tests": True
        }

    def is_ci_environment(self) -> bool:
        """Check if running in CI environment."""
        return os.getenv("CI", "false").lower() == "true"


# Global test configuration instance
test_config = TestConfig()


def get_test_config() -> TestConfig:
    """Get the global test configuration."""
    return test_config
