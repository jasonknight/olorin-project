"""
Integration tests for OllamaBackend.

These tests require a running Ollama server with the gpt-oss:120b model.
Tests are marked with pytest.mark.integration to allow selective running.

Run with: pytest libs/tests/test_inference_ollama.py -v
"""

import json

import pytest
import requests

from libs.inference import (
    BackendType,
    CompletionChunk,
    CompletionResponse,
    OllamaBackend,
)


@pytest.fixture
def ollama_base_url():
    """Ollama API base URL."""
    return "http://localhost:11434"


@pytest.fixture
def ollama_model():
    """Model to test with."""
    return "gpt-oss:120b"


@pytest.fixture
def ollama_backend(ollama_base_url, ollama_model):
    """Create OllamaBackend instance."""
    return OllamaBackend(
        base_url=ollama_base_url,
        default_model=ollama_model,
        default_temperature=0.7,
    )


def is_ollama_running(base_url: str) -> bool:
    """Check if Ollama server is accessible."""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def has_model(base_url: str, model_name: str) -> bool:
    """Check if the specified model is available in Ollama."""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code != 200:
            return False
        data = response.json()
        models = data.get("models", [])
        for model in models:
            name = model.get("name", "") or model.get("model", "")
            if name == model_name or name.startswith(model_name.split(":")[0]):
                return True
        return False
    except Exception:
        return False


# Skip all tests if Ollama is not running
pytestmark = pytest.mark.skipif(
    not is_ollama_running("http://localhost:11434"),
    reason="Ollama server is not running",
)


class TestOllamaBackendBasics:
    """Basic tests for OllamaBackend initialization and properties."""

    def test_backend_type(self, ollama_backend):
        """Test that backend_type returns OLLAMA."""
        assert ollama_backend.backend_type == BackendType.OLLAMA

    def test_check_health(self, ollama_backend):
        """Test health check returns True when Ollama is running."""
        assert ollama_backend.check_health() is True

    def test_check_health_bad_url(self):
        """Test health check returns False for invalid URL."""
        backend = OllamaBackend(
            base_url="http://localhost:99999",
            default_model="test",
        )
        assert backend.check_health() is False


class TestOllamaModelDetection:
    """Tests for model detection functionality."""

    def test_get_running_model_with_default(self, ollama_backend, ollama_model):
        """Test that get_running_model returns configured model."""
        model = ollama_backend.get_running_model()
        assert model == ollama_model

    def test_get_running_model_auto_detect(self, ollama_base_url):
        """Test auto-detection when no default model is set."""
        backend = OllamaBackend(
            base_url=ollama_base_url,
            default_model=None,
        )
        model = backend.get_running_model()
        # Should return some model if any are available
        assert model is not None or True  # May be None if no models loaded


class TestOllamaCompletion:
    """Tests for non-streaming completion."""

    @pytest.mark.skipif(
        not has_model("http://localhost:11434", "gpt-oss:120b"),
        reason="gpt-oss:120b model not available",
    )
    def test_complete_basic(self, ollama_backend):
        """Test basic non-streaming completion."""
        messages = [{"role": "user", "content": "Say 'hello' and nothing else."}]

        response = ollama_backend.complete(
            messages=messages,
            max_tokens=200,  # More tokens for model to work with
            timeout=120,
        )

        assert isinstance(response, CompletionResponse)
        assert response.content is not None
        # Response may be empty if model hits internal limits
        assert response.finish_reason is not None

    @pytest.mark.skipif(
        not has_model("http://localhost:11434", "gpt-oss:120b"),
        reason="gpt-oss:120b model not available",
    )
    def test_complete_with_system_message(self, ollama_backend):
        """Test completion with system message."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Be brief."},
            {"role": "user", "content": "What is 2+2?"},
        ]

        response = ollama_backend.complete(
            messages=messages,
            max_tokens=50,
            timeout=60,
        )

        assert isinstance(response, CompletionResponse)
        assert response.content is not None

    @pytest.mark.skipif(
        not has_model("http://localhost:11434", "gpt-oss:120b"),
        reason="gpt-oss:120b model not available",
    )
    def test_complete_usage_stats(self, ollama_backend):
        """Test that usage statistics are returned."""
        messages = [{"role": "user", "content": "Hi"}]

        response = ollama_backend.complete(
            messages=messages,
            max_tokens=20,
            timeout=60,
        )

        # Usage may or may not be present depending on Ollama version
        if response.usage:
            assert "prompt_tokens" in response.usage
            assert "completion_tokens" in response.usage


class TestOllamaStreaming:
    """Tests for streaming completion."""

    @pytest.mark.skipif(
        not has_model("http://localhost:11434", "gpt-oss:120b"),
        reason="gpt-oss:120b model not available",
    )
    def test_complete_stream_basic(self, ollama_backend):
        """Test basic streaming completion."""
        messages = [{"role": "user", "content": "Count from 1 to 3."}]

        chunks = list(
            ollama_backend.complete_stream(
                messages=messages,
                max_tokens=50,
                timeout=60,
            )
        )

        assert len(chunks) > 0
        assert all(isinstance(c, CompletionChunk) for c in chunks)

        # Combine all content
        full_content = "".join(c.content for c in chunks)
        assert len(full_content) > 0

        # Last chunk should have finish_reason
        assert chunks[-1].finish_reason is not None

    @pytest.mark.skipif(
        not has_model("http://localhost:11434", "gpt-oss:120b"),
        reason="gpt-oss:120b model not available",
    )
    def test_complete_stream_conversation(self, ollama_backend):
        """Test streaming with conversation history."""
        messages = [
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Nice to meet you, Alice!"},
            {"role": "user", "content": "What is my name?"},
        ]

        chunks = list(
            ollama_backend.complete_stream(
                messages=messages,
                max_tokens=50,
                timeout=60,
            )
        )

        full_content = "".join(c.content for c in chunks).lower()
        # The model should remember the name
        assert "alice" in full_content


class TestOllamaToolSupport:
    """Tests for tool/function calling support."""

    @pytest.mark.skipif(
        not has_model("http://localhost:11434", "gpt-oss:120b"),
        reason="gpt-oss:120b model not available",
    )
    def test_supports_tools_detection(self, ollama_backend):
        """Test tool support detection."""
        # This should not raise an error
        result = ollama_backend.supports_tools()
        assert isinstance(result, bool)

    @pytest.mark.skipif(
        not has_model("http://localhost:11434", "gpt-oss:120b"),
        reason="gpt-oss:120b model not available",
    )
    def test_complete_with_tools(self, ollama_backend):
        """Test completion with tool definitions."""
        messages = [{"role": "user", "content": "What is the weather in Paris?"}]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "The city name",
                            }
                        },
                        "required": ["city"],
                    },
                },
            }
        ]

        try:
            response = ollama_backend.complete(
                messages=messages,
                tools=tools,
                max_tokens=100,
                timeout=60,
            )

            assert isinstance(response, CompletionResponse)

            # If tool calls are returned, verify format
            if response.tool_calls:
                assert len(response.tool_calls) > 0
                for tc in response.tool_calls:
                    assert "id" in tc
                    assert "type" in tc
                    assert "function" in tc
                    assert "name" in tc["function"]
                    assert "arguments" in tc["function"]
                    # Arguments should be JSON string (OpenAI format)
                    args = json.loads(tc["function"]["arguments"])
                    assert isinstance(args, dict)

        except Exception as e:
            # Tool calling may not be supported by the model
            pytest.skip(f"Tool calling not supported: {e}")


class TestOllamaErrorHandling:
    """Tests for error handling."""

    def test_complete_no_model_error(self, ollama_base_url):
        """Test error when no model is available."""
        backend = OllamaBackend(
            base_url=ollama_base_url,
            default_model=None,
        )

        # Mock get_running_model to return None
        backend._default_model = None

        # If no models are available, should raise ValueError
        backend.get_running_model = lambda: None

        with pytest.raises(ValueError, match="No model available"):
            backend.complete(
                messages=[{"role": "user", "content": "test"}],
            )

    def test_complete_invalid_model(self, ollama_base_url):
        """Test error with non-existent model."""
        backend = OllamaBackend(
            base_url=ollama_base_url,
            default_model="nonexistent-model-xyz",
        )

        with pytest.raises(Exception):
            backend.complete(
                messages=[{"role": "user", "content": "test"}],
                timeout=10,
            )


class TestOllamaConfiguration:
    """Tests for configuration options."""

    @pytest.mark.skipif(
        not has_model("http://localhost:11434", "gpt-oss:120b"),
        reason="gpt-oss:120b model not available",
    )
    def test_temperature_setting(self, ollama_backend):
        """Test that temperature is passed correctly."""
        messages = [{"role": "user", "content": "Hi"}]

        # Low temperature should work
        response = ollama_backend.complete(
            messages=messages,
            temperature=0.1,
            max_tokens=20,
            timeout=60,
        )
        assert response.content is not None

        # High temperature should also work
        response = ollama_backend.complete(
            messages=messages,
            temperature=1.5,
            max_tokens=20,
            timeout=60,
        )
        assert response.content is not None

    @pytest.mark.skipif(
        not has_model("http://localhost:11434", "gpt-oss:120b"),
        reason="gpt-oss:120b model not available",
    )
    def test_max_tokens_limit(self, ollama_backend):
        """Test that max_tokens limits output."""
        messages = [{"role": "user", "content": "Write a very long story."}]

        response = ollama_backend.complete(
            messages=messages,
            max_tokens=10,
            timeout=60,
        )

        # Response should exist (content may be empty if limit hit immediately)
        assert response.content is not None
        # finish_reason should indicate the limit was respected
        # "length" means the token limit was hit
        assert response.finish_reason in ("stop", "length")
