#!/usr/bin/env python3
"""
Test runner for PCM-LLM test suite.
"""

import unittest
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add the tests directory to Python path
tests_dir = project_root / "tests"
sys.path.insert(0, str(tests_dir))


def run_unit_tests():
    """Run unit tests."""
    print("Running unit tests...")
    loader = unittest.TestLoader()
    suite = loader.discover(str(tests_dir / "unit"), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


def run_integration_tests():
    """Run integration tests."""
    print("Running integration tests...")
    loader = unittest.TestLoader()
    suite = loader.discover(str(tests_dir / "integration"), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


def run_functional_tests():
    """Run functional tests."""
    print("Running functional tests...")
    loader = unittest.TestLoader()
    suite = loader.discover(str(tests_dir / "functional"), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


def run_all_tests():
    """Run all tests."""
    print("Running all tests...")
    loader = unittest.TestLoader()
    suite = loader.discover(str(tests_dir), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


def main():
    """Main test runner function."""
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
        if test_type == "unit":
            success = run_unit_tests()
        elif test_type == "integration":
            success = run_integration_tests()
        elif test_type == "functional":
            success = run_functional_tests()
        else:
            print(f"Unknown test type: {test_type}")
            print("Usage: python test_runner.py [unit|integration|functional|all]")
            sys.exit(1)
    else:
        success = run_all_tests()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
