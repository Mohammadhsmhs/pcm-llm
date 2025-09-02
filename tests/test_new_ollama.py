import unittest
from unittest.mock import patch, MagicMock
from core.config.config_manager import LLMConfig

# Mock the ollama module before it's imported by the class we're testing
import sys
sys.modules['ollama'] = MagicMock()

from llms.providers.ollama_llm import OllamaLLM, list_local_models

class TestNewOllama(unittest.TestCase):

    def setUp(self):
        """Set up the test environment."""
        self.mock_ollama = sys.modules['ollama']
        self.mock_ollama.reset_mock()

    def test_initialization(self):
        """Test that OllamaLLM initializes correctly with a config object."""
        model_name = "test-model"
        config = LLMConfig(provider="ollama", model_name=model_name, temperature=0.1)
        
        with patch('llms.providers.ollama_llm.OllamaLLM._validate_setup') as mock_validate:
            llm = OllamaLLM(config)
            self.assertEqual(llm.model_name, model_name)
            self.assertEqual(llm.config.temperature, 0.1)
            mock_validate.assert_called_once()

    def test_get_response_streaming(self):
        """Test streaming response generation."""
        model_name = "test-model"
        config = LLMConfig(provider="ollama", model_name=model_name, stream_tokens=True)
        
        with patch('llms.providers.ollama_llm.OllamaLLM._validate_setup'):
            llm = OllamaLLM(config)
            
            mock_response = [
                {'response': 'Hello'},
                {'response': ' World'},
            ]
            self.mock_ollama.generate.return_value = mock_response
            
            response = llm.get_response("test prompt")
            
            self.assertEqual(response, "Hello World")
            self.mock_ollama.generate.assert_called_with(
                model=model_name,
                prompt="test prompt",
                stream=True,
                options=unittest.mock.ANY
            )

    def test_get_response_non_streaming(self):
        """Test non-streaming response generation."""
        model_name = "test-model"
        config = LLMConfig(provider="ollama", model_name=model_name, stream_tokens=False)

        with patch('llms.providers.ollama_llm.OllamaLLM._validate_setup'):
            llm = OllamaLLM(config)

            mock_response = {'response': 'Complete response'}
            self.mock_ollama.generate.return_value = mock_response

            response = llm.get_response("test prompt")

            self.assertEqual(response, "Complete response")
            self.mock_ollama.generate.assert_called_with(
                model=model_name,
                prompt="test prompt",
                stream=False,
                options=unittest.mock.ANY
            )

    def test_chat_functionality(self):
        """Test the chat method with tools."""
        model_name = "test-model"
        config = LLMConfig(provider="ollama", model_name=model_name)
        
        with patch('llms.providers.ollama_llm.OllamaLLM._validate_setup'):
            llm = OllamaLLM(config)

            messages = [{"role": "user", "content": "test message"}]
            tools = [{"type": "function", "function": {"name": "test_tool"}}]
            
            mock_response = {"message": {"role": "assistant", "content": "tool response"}}
            self.mock_ollama.chat.return_value = mock_response
            
            response = llm.chat(messages=messages, tools=tools)
            
            self.assertEqual(response, mock_response)
            self.mock_ollama.chat.assert_called_with(
                model=model_name,
                messages=messages,
                tools=tools,
                options=unittest.mock.ANY
            )

    @patch('llms.providers.ollama_llm.ollama')
    def test_list_local_models(self, mock_ollama_pkg):
        """Test the list_local_models utility function."""
        mock_ollama_pkg.list.return_value = {"models": [{"name": "model1"}, {"name": "model2"}]}
        models = list_local_models()
        self.assertEqual(models, ["model1", "model2"])
        mock_ollama_pkg.list.assert_called_once()

if __name__ == "__main__":
    unittest.main()
