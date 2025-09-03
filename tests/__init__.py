"""
Test framework and base classes following SOLID principles.
"""

import unittest
from abc import ABC, abstractmethod
from typing import Any, Dict, List
import tempfile
import os
import shutil


class ITestFixture(ABC):
    """Interface for test fixtures."""

    @abstractmethod
    def setup(self) -> None:
        """Set up the test fixture."""
        pass

    @abstractmethod
    def teardown(self) -> None:
        """Clean up the test fixture."""
        pass


class TestFixture(ITestFixture):
    """Base test fixture implementation."""

    def __init__(self):
        self.temp_dir = None
        self.resources = {}

    def setup(self) -> None:
        """Set up temporary directory and resources."""
        self.temp_dir = tempfile.mkdtemp()
        self.resources = {}

    def teardown(self) -> None:
        """Clean up temporary directory and resources."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        self.resources.clear()

    def add_resource(self, key: str, resource: Any) -> None:
        """Add a resource to be managed by the fixture."""
        self.resources[key] = resource

    def get_resource(self, key: str) -> Any:
        """Get a managed resource."""
        return self.resources.get(key)


class ITestRunner(ABC):
    """Interface for test runners."""

    @abstractmethod
    def run_tests(self, test_pattern: str = "*") -> Dict[str, Any]:
        """Run tests matching the pattern."""
        pass

    @abstractmethod
    def get_test_results(self) -> Dict[str, Any]:
        """Get test results."""
        pass


class TestRunner(ITestRunner):
    """Test runner implementation."""

    def __init__(self):
        self.results = {}
        self.test_loader = unittest.TestLoader()
        self.test_runner = unittest.TextTestRunner(verbosity=2)

    def run_tests(self, test_pattern: str = "*") -> Dict[str, Any]:
        """Run tests matching the pattern."""
        # Discover and run tests
        suite = self.test_loader.discover('tests', pattern=f"test_{test_pattern}.py")

        # Run the tests
        result = self.test_runner.run(suite)

        # Store results
        self.results = {
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'skipped': len(result.skipped),
            'success': result.wasSuccessful(),
            'failure_details': result.failures,
            'error_details': result.errors
        }

        return self.results

    def get_test_results(self) -> Dict[str, Any]:
        """Get test results."""
        return self.results


class BaseTestCase(unittest.TestCase):
    """Base test case with common functionality."""

    def setUp(self):
        """Set up test case."""
        self.fixture = TestFixture()
        self.fixture.setup()

    def tearDown(self):
        """Clean up test case."""
        self.fixture.teardown()

    def assertDictContains(self, dict_obj: Dict, key: str, expected_value: Any = None):
        """Assert that dictionary contains key and optionally check value."""
        self.assertIn(key, dict_obj, f"Dictionary does not contain key: {key}")
        if expected_value is not None:
            self.assertEqual(dict_obj[key], expected_value,
                           f"Key '{key}' has value {dict_obj[key]}, expected {expected_value}")

    def assertListContains(self, list_obj: List, item: Any):
        """Assert that list contains item."""
        self.assertIn(item, list_obj, f"List does not contain item: {item}")


# Test categories
class UnitTestCase(BaseTestCase):
    """Base class for unit tests."""
    pass


class IntegrationTestCase(BaseTestCase):
    """Base class for integration tests."""
    pass


class FunctionalTestCase(BaseTestCase):
    """Base class for functional tests."""
    pass


# Test utilities
def create_mock_llm_response(response_text: str = "Mock response"):
    """Create a mock LLM response for testing."""
    return response_text


def create_test_prompt(task_type: str = "reasoning") -> str:
    """Create a test prompt for different task types."""
    prompts = {
        "reasoning": "What is 2 + 2? Provide the numerical answer.",
        "summarization": "Summarize this article: The quick brown fox jumps over the lazy dog.",
        "classification": "Classify this sentiment: I love this product!"
    }
    return prompts.get(task_type, "Test prompt")


def create_test_ground_truth(task_type: str = "reasoning") -> str:
    """Create test ground truth for different task types."""
    ground_truths = {
        "reasoning": "4",
        "summarization": "A fox jumps over a dog.",
        "classification": "positive"
    }
    return ground_truths.get(task_type, "test answer")
