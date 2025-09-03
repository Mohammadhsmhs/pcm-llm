#!/usr/bin/env python3
"""
Functional test for OpenRouter LLM integration.
"""

import unittest
import os
from unittest.mock import patch, Mock
from tests import FunctionalTestCase
from core.llm_factory import create_llm_factory
from core.config.config_manager import CentralizedConfigProvider as EnvironmentConfigProvider
from llms.providers.openrouter_llm import OpenRouterLLM
from core.config import LLMConfig

class TestOpenRouterFunctional(FunctionalTestCase):
    """Functional tests for OpenRouter LLM integration."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.config_provider = EnvironmentConfigProvider()
        # Set a dummy API key for most tests
        os.environ['OPENROUTER_API_KEY'] = 'test_api_key'
        self.llm_config = self.config_provider.get_llm_config("openrouter")

    def tearDown(self):
        """Clean up after tests."""
        if 'OPENROUTER_API_KEY' in os.environ:
            del os.environ['OPENROUTER_API_KEY']
        super().tearDown()

    @patch('llms.providers.openrouter_llm.OpenAI')
    def test_openrouter_llm_initialization(self, mock_openai_class):
        """Test OpenRouter LLM initialization."""
        print("=== Testing OpenRouter LLM Initialization ===")
        try:
            factory = create_llm_factory(self.config_provider)
            llm = factory.create_llm("openrouter", self.llm_config)
            self.assertIsInstance(llm, OpenRouterLLM)
            mock_openai_class.assert_called_with(
                api_key='test_api_key',
                base_url="https://openrouter.ai/api/v1"
            )
            print("✅ OpenRouter LLM initialized successfully!")
        except Exception as e:
            self.fail(f"❌ OpenRouter LLM initialization failed: {e}")

    @patch('llms.providers.openrouter_llm.OpenAI')
    def test_openrouter_llm_response_generation(self, mock_openai_class):
        """Test OpenRouter LLM response generation."""
        print("=== Testing OpenRouter LLM Response Generation ===")
        
        mock_client_instance = mock_openai_class.return_value
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "The answer is 4."
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client_instance.chat.completions.create.return_value = mock_response

        try:
            factory = create_llm_factory(self.config_provider)
            llm = factory.create_llm("openrouter", self.llm_config)
            test_prompt = "What is 2 + 2?"
            response = llm.get_response(test_prompt)
            self.assertEqual(response, "The answer is 4.")
            mock_client_instance.chat.completions.create.assert_called_once()
            print("✅ OpenRouter LLM response generation successful!")
        except Exception as e:
            self.fail(f"❌ Response generation failed: {e}")

    def test_openrouter_missing_api_key(self):
        """Test OpenRouter LLM with missing API key."""
        print("=== Testing OpenRouter Missing API Key ===")
        # Unset the key for this specific test
        if 'OPENROUTER_API_KEY' in os.environ:
            del os.environ['OPENROUTER_API_KEY']
            
        with self.assertRaises(ValueError) as context:
            # The error should be raised during config creation
            self.config_provider.get_llm_config("openrouter")
        self.assertIn("OPENROUTER_API_KEY is not set", str(context.exception))
        print("✅ API key validation working correctly!")

    @patch('llms.providers.openrouter_llm.OpenAI')
    def test_openrouter_error_handling(self, mock_openai_class):
        """Test OpenRouter LLM error handling."""
        print("=== Testing OpenRouter Error Handling ===")
        mock_client_instance = mock_openai_class.return_value
        mock_client_instance.chat.completions.create.side_effect = Exception("Network error")

        try:
            factory = create_llm_factory(self.config_provider)
            llm = factory.create_llm("openrouter", self.llm_config)
            response = llm.get_response("test prompt")
            self.assertIn("Error calling OpenRouter API: Network error", response)
            print("✅ Error handling working correctly!")
        except Exception as e:
            self.fail(f"❌ Error handling failed unexpectedly: {e}")

if __name__ == "__main__":
    unittest.main()
