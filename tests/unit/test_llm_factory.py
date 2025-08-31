"""
Unit tests for LLM factory.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from tests import UnitTestCase
from core.llm_factory import LLMFactory, create_llm_factory
from core.config import EnvironmentConfigProvider


class TestLLMFactory(UnitTestCase):
    """Test LLMFactory implementation."""

    def setUp(self):
        super().setUp()
        self.config_provider = EnvironmentConfigProvider()
        self.factory = LLMFactory(self.config_provider)

    def test_register_provider(self):
        """Test registering a provider."""
        mock_creator = Mock()
        self.factory.register_provider("test_provider", mock_creator)

        self.assertIn("test_provider", self.factory.get_supported_providers())

    def test_get_supported_providers_empty(self):
        """Test getting supported providers when none registered."""
        providers = self.factory.get_supported_providers()
        self.assertEqual(providers, [])

    def test_get_supported_providers_with_registered(self):
        """Test getting supported providers after registration."""
        mock_creator = Mock()
        self.factory.register_provider("test_provider", mock_creator)

        providers = self.factory.get_supported_providers()
        self.assertEqual(providers, ["test_provider"])

    def test_create_llm_registered_provider(self):
        """Test creating LLM with registered provider."""
        mock_llm = Mock()
        mock_creator = Mock(return_value=mock_llm)
        mock_config = Mock()

        self.factory.register_provider("test_provider", mock_creator)

        # Mock the config provider to return a config for test_provider
        with patch.object(self.config_provider, 'get_llm_config', return_value=mock_config):
            result = self.factory.create_llm("test_provider")

        self.assertEqual(result, mock_llm)
        mock_creator.assert_called_once_with(mock_config)

    def test_create_llm_unregistered_provider(self):
        """Test creating LLM with unregistered provider raises error."""
        with self.assertRaises(ValueError):
            self.factory.create_llm("unregistered_provider")


class TestCreateLLMFactory(UnitTestCase):
    """Test create_llm_factory function."""

    @patch('core.llm_factory.LLMFactory')
    def test_create_llm_factory(self, mock_factory_class):
        """Test creating LLM factory with dependency injection."""
        mock_factory = Mock()
        mock_factory_class.return_value = mock_factory

        config_provider = Mock()
        result = create_llm_factory(config_provider)

        self.assertEqual(result, mock_factory)
        mock_factory_class.assert_called_once_with(config_provider)

    @patch('llms.openai_llm.OpenAI_LLM')
    @patch('llms.huggingface_llm.HuggingFace_LLM')
    @patch('llms.openrouter_llm.OpenRouter_LLM')
    @patch('llms.manual_llm.ManualLLM')
    @patch('core.llm_factory.LLMFactory')
    def test_create_llm_factory_registers_providers(self, mock_factory_class, mock_manual, mock_openrouter, mock_huggingface, mock_openai):
        """Test that create_llm_factory registers available providers."""
        mock_factory = Mock()
        mock_factory_class.return_value = mock_factory

        config_provider = Mock()
        create_llm_factory(config_provider)

        # Verify that register_provider was called for each available provider
        expected_calls = [
            unittest.mock.call("openai", mock_openai),
            unittest.mock.call("huggingface", mock_huggingface),
            unittest.mock.call("openrouter", mock_openrouter),
            unittest.mock.call("manual", mock_manual)
        ]

        mock_factory.register_provider.assert_has_calls(expected_calls, any_order=True)

    @patch('llms.openai_llm', side_effect=ImportError("Module not found"))
    @patch('llms.huggingface_llm', side_effect=ImportError("Module not found"))
    @patch('llms.openrouter_llm', side_effect=ImportError("Module not found"))
    @patch('llms.manual_llm', side_effect=ImportError("Module not found"))
    @patch('core.llm_factory.LLMFactory')
    def test_create_llm_factory_handles_import_errors(self, mock_factory_class, mock_manual, mock_openrouter, mock_huggingface, mock_openai):
        """Test that create_llm_factory handles import errors gracefully."""
        mock_factory = Mock()
        mock_factory_class.return_value = mock_factory

        config_provider = Mock()
        result = create_llm_factory(config_provider)

        # Should still return factory even if imports fail
        self.assertEqual(result, mock_factory)


if __name__ == '__main__':
    unittest.main()
