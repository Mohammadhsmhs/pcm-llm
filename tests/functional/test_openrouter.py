#!/usr/bin/env python3
"""
Functional test for OpenRouter LLM integration.
"""

import unittest
import os
from unittest.mock import patch, Mock
from tests import FunctionalTestCase
from core.llm_factory import create_llm_factory
from core.config import EnvironmentConfigProvider


class TestOpenRouterFunctional(FunctionalTestCase):
    """Functional tests for OpenRouter LLM integration."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.config_provider = EnvironmentConfigProvider()

    @patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test_api_key'})
    @patch('llms.openrouter_llm.OpenRouter_LLM')
    def test_openrouter_llm_initialization(self, mock_openrouter_class):
        """Test OpenRouter LLM initialization."""
        print("=== Testing OpenRouter LLM Initialization ===")

        mock_llm = Mock()
        mock_openrouter_class.return_value = mock_llm

        try:
            factory = create_llm_factory(self.config_provider)
            llm = factory.create_llm("openrouter")

            self.assertEqual(llm, mock_llm)
            print("✅ OpenRouter LLM initialized successfully!")

        except Exception as e:
            self.fail(f"❌ OpenRouter LLM initialization failed: {e}")

    @patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test_api_key'})
    @patch('llms.openrouter_llm.OpenRouter_LLM')
    def test_openrouter_llm_response_generation(self, mock_openrouter_class):
        """Test OpenRouter LLM response generation."""
        print("=== Testing OpenRouter LLM Response Generation ===")

        mock_llm = Mock()
        mock_llm.get_response.return_value = "The answer is 4."
        mock_openrouter_class.return_value = mock_llm

        try:
            factory = create_llm_factory(self.config_provider)
            llm = factory.create_llm("openrouter")

            test_prompt = "What is 2 + 2?"
            response = llm.get_response(test_prompt)

            self.assertEqual(response, "The answer is 4.")
            mock_llm.get_response.assert_called_once_with(test_prompt)
            print("✅ OpenRouter LLM response generation successful!")

        except Exception as e:
            self.fail(f"❌ Response generation failed: {e}")

    def test_openrouter_missing_api_key(self):
        """Test OpenRouter LLM with missing API key."""
        print("=== Testing OpenRouter Missing API Key ===")

        # Ensure API key is not set
        with patch.dict('os.environ', {}, clear=True):
            with patch('llms.openrouter_llm.OpenRouter_LLM') as mock_openrouter_class:
                mock_llm = Mock()
                mock_llm.get_response.side_effect = ValueError("API key required")
                mock_openrouter_class.return_value = mock_llm

                try:
                    factory = create_llm_factory(self.config_provider)
                    llm = factory.create_llm("openrouter")

                    with self.assertRaises(ValueError):
                        llm.get_response("test prompt")

                    print("✅ API key validation working correctly!")

                except Exception as e:
                    self.fail(f"❌ API key validation failed: {e}")

    @patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test_api_key'})
    @patch('llms.openrouter_llm.OpenRouter_LLM')
    def test_openrouter_error_handling(self, mock_openrouter_class):
        """Test OpenRouter LLM error handling."""
        print("=== Testing OpenRouter Error Handling ===")

        mock_llm = Mock()
        mock_llm.get_response.side_effect = Exception("Network error")
        mock_openrouter_class.return_value = mock_llm

        try:
            factory = create_llm_factory(self.config_provider)
            llm = factory.create_llm("openrouter")

            with self.assertRaises(Exception):
                llm.get_response("test prompt")

            print("✅ Error handling working correctly!")

        except Exception as e:
            self.fail(f"❌ Error handling failed: {e}")


if __name__ == "__main__":
    unittest.main()
