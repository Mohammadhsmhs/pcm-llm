"""
Base test fixture class for managing test resources.
"""

import tempfile
import shutil
import os
from typing import Dict, Any, List


class TestFixture:
    """Base class for test fixtures that manage temporary resources."""

    def __init__(self):
        """Initialize the test fixture."""
        self.temp_dir = None
        self.resources = {}
        self.cleanup_functions = []

    def setup(self):
        """Set up the test fixture."""
        self.temp_dir = tempfile.mkdtemp(prefix="pcm_llm_test_")
        self.resources = {}

    def teardown(self):
        """Clean up the test fixture."""
        # Run custom cleanup functions
        for cleanup_func in self.cleanup_functions:
            try:
                cleanup_func()
            except Exception as e:
                print(f"Warning: Cleanup function failed: {e}")

        # Remove temporary directory
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                print(f"Warning: Failed to remove temp dir {self.temp_dir}: {e}")

        self.temp_dir = None
        self.resources = {}

    def add_resource(self, name: str, resource: Any):
        """Add a resource to be tracked by the fixture."""
        self.resources[name] = resource

    def get_resource(self, name: str) -> Any:
        """Get a resource by name."""
        return self.resources.get(name)

    def add_cleanup_function(self, func):
        """Add a custom cleanup function."""
        self.cleanup_functions.append(func)

    def create_temp_file(self, filename: str, content: str = "") -> str:
        """Create a temporary file with optional content."""
        file_path = os.path.join(self.temp_dir, filename)
        with open(file_path, 'w') as f:
            f.write(content)
        self.add_resource(filename, file_path)
        return file_path

    def create_temp_dir(self, dirname: str) -> str:
        """Create a temporary directory."""
        dir_path = os.path.join(self.temp_dir, dirname)
        os.makedirs(dir_path, exist_ok=True)
        self.add_resource(dirname, dir_path)
        return dir_path

    def __enter__(self):
        """Context manager entry."""
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.teardown()


class TestSuiteFixture(TestFixture):
    """Fixture for managing multiple test fixtures."""

    def __init__(self):
        """Initialize the test suite fixture."""
        super().__init__()
        self.fixtures = []

    def add_fixture(self, fixture: TestFixture):
        """Add a fixture to the suite."""
        self.fixtures.append(fixture)

    def setup(self):
        """Set up all fixtures in the suite."""
        super().setup()
        for fixture in self.fixtures:
            fixture.setup()

    def teardown(self):
        """Clean up all fixtures in the suite."""
        for fixture in reversed(self.fixtures):
            fixture.teardown()
        super().teardown()
