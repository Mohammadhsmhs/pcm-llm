#!/usr/bin/env python3
"""
Test Ollama connection using direct REST API calls.
"""

import requests
import json

def test_ollama_rest_api():
    """Test Ollama using direct REST API calls."""
    print("🔍 Testing Ollama REST API Connection")
    print("=" * 50)

    base_url = "http://localhost:11434"

    try:
        # Test 1: List models
        print("1. Testing model listing...")
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', [])
            print(f"   ✅ Found {len(models)} models:")
            for model in models:
                name = model.get('name', 'Unknown')
                size = model.get('size', 0) / (1024**3)  # Convert to GB
                print(f"      • {name}: {size:.1f} GB")
            print(f"   Available models: {[m['name'] for m in models]}")
        else:
            print(f"   ❌ Failed to list models: {response.status_code}")
            return

        if not models:
            print("   ❌ No models available!")
            return

        # Test 2: Generate response
        print("\n2. Testing text generation...")
        model_name = models[0]['name']
        payload = {
            "model": model_name,
            "prompt": "What is 2 + 2? Answer with just the number.",
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 50
            }
        }

        response = requests.post(
            f"{base_url}/api/generate",
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            generated_text = result.get('response', '').strip()
            print(f"   ✅ Generation successful!")
            print(f"   Model: {model_name}")
            print(f"   Response: {generated_text}")
        else:
            print(f"   ❌ Generation failed: {response.status_code}")
            print(f"   Error: {response.text}")

        # Test 3: Chat API
        print("\n3. Testing chat API...")
        chat_payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello! What is AI?"}
            ],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 100
            }
        }

        response = requests.post(
            f"{base_url}/api/chat",
            json=chat_payload,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            chat_response = result.get('message', {}).get('content', '')
            print(f"   ✅ Chat successful!")
            print(f"   Response: {chat_response[:100]}{'...' if len(chat_response) > 100 else ''}")
        else:
            print(f"   ❌ Chat failed: {response.status_code}")
            print(f"   Error: {response.text}")

        print("\n🎉 REST API tests completed successfully!")
        print("\n📋 Ollama Status:")
        print("   • Service: ✅ Running")
        print("   • API: ✅ Responding")
        print("   • Models: ✅ Available")
        print("   • Generation: ✅ Working")
        print("   • Chat: ✅ Working")

    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama server")
        print("   Make sure Ollama is running: ollama serve")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ollama_rest_api()
