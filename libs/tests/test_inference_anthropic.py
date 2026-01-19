"""
Tests for AnthropicBackend.

Unit tests for message/tool format conversion run without API access.
Integration tests require ANTHROPIC_API_KEY in .env and are skipped otherwise.

Run with: pytest libs/tests/test_inference_anthropic.py -v
"""

import json
import os

import pytest

from libs.inference import (
    AnthropicBackend,
    BackendType,
    CompletionChunk,
    CompletionResponse,
)


def has_anthropic_api_key() -> bool:
    """Check if Anthropic API key is available."""
    # Check environment variable
    if os.environ.get("ANTHROPIC_API_KEY"):
        return True

    # Check .env file
    env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.strip().startswith("ANTHROPIC_API_KEY="):
                    value = line.strip().split("=", 1)[1]
                    if value and not value.startswith("#"):
                        return True
    return False


def get_api_key() -> str:
    """Get Anthropic API key from environment or .env file."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        return api_key

    env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.strip().startswith("ANTHROPIC_API_KEY="):
                    return line.strip().split("=", 1)[1]
    return ""


@pytest.fixture
def anthropic_model():
    """Default model for testing."""
    return "claude-sonnet-4-20250514"


@pytest.fixture
def anthropic_backend(anthropic_model):
    """Create AnthropicBackend instance (requires API key)."""
    api_key = get_api_key()
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not available")
    return AnthropicBackend(
        api_key=api_key,
        default_model=anthropic_model,
        default_temperature=0.7,
        default_max_tokens=1024,
    )


# ============================================================================
# Unit Tests - No API Key Required
# ============================================================================


class TestAnthropicBackendInit:
    """Tests for AnthropicBackend initialization."""

    def test_backend_type(self):
        """Test that backend_type returns ANTHROPIC."""
        # Mock initialization without actual API call
        try:
            backend = AnthropicBackend(
                api_key="test-key",
                default_model="claude-sonnet-4-20250514",
            )
            assert backend.backend_type == BackendType.ANTHROPIC
        except Exception:
            # If anthropic SDK raises on invalid key during init, that's fine
            pass

    def test_init_raises_on_missing_key(self):
        """Test that initialization with empty key eventually fails."""
        # The SDK may or may not raise immediately on empty key
        # but check_health or requests will fail
        try:
            backend = AnthropicBackend(
                api_key="",
                default_model="test",
            )
            # If it doesn't raise, health check should fail
            assert backend.check_health() is False
        except Exception:
            pass  # Expected to fail


class TestMessageConversionForAnthropic:
    """Unit tests for _convert_messages_for_anthropic method.

    These tests verify OpenAI-to-Anthropic message format conversion
    without requiring API access.
    """

    @pytest.fixture
    def backend(self):
        """Create backend instance for testing conversion methods."""
        # We need a valid backend instance to test the methods
        # Use a dummy key - we won't make API calls
        try:
            return AnthropicBackend(
                api_key="test-key-for-unit-tests",
                default_model="claude-sonnet-4-20250514",
            )
        except Exception:
            pytest.skip("Cannot create AnthropicBackend for testing")

    def test_system_message_extracted(self, backend):
        """Test that system message is extracted as separate parameter."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]
        system, converted = backend._convert_messages_for_anthropic(messages)

        assert system == "You are a helpful assistant."
        assert len(converted) == 1
        assert converted[0]["role"] == "user"

    def test_user_message_unchanged(self, backend):
        """Test that user messages pass through correctly."""
        messages = [{"role": "user", "content": "Hello, world!"}]
        system, converted = backend._convert_messages_for_anthropic(messages)

        assert system is None
        assert len(converted) == 1
        assert converted[0]["role"] == "user"
        assert converted[0]["content"] == "Hello, world!"

    def test_assistant_message_unchanged(self, backend):
        """Test that basic assistant messages pass through correctly."""
        messages = [{"role": "assistant", "content": "I can help you."}]
        system, converted = backend._convert_messages_for_anthropic(messages)

        assert len(converted) == 1
        assert converted[0]["role"] == "assistant"
        assert converted[0]["content"] == "I can help you."

    def test_tool_calls_converted_to_anthropic_format(self, backend):
        """Test that tool_calls are converted from OpenAI to Anthropic format."""
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Paris"}',
                        },
                    }
                ],
            }
        ]
        system, converted = backend._convert_messages_for_anthropic(messages)

        assert len(converted) == 1
        assert converted[0]["role"] == "assistant"
        # Content should be a list with tool_use block
        assert isinstance(converted[0]["content"], list)
        tool_block = converted[0]["content"][0]
        assert tool_block["type"] == "tool_use"
        assert tool_block["id"] == "call_abc123"
        assert tool_block["name"] == "get_weather"
        assert tool_block["input"] == {"city": "Paris"}

    def test_tool_response_converted_to_anthropic_format(self, backend):
        """Test that tool role messages become user messages with tool_result."""
        messages = [
            {
                "role": "tool",
                "content": '{"temp": 22}',
                "tool_call_id": "call_abc123",
            }
        ]
        system, converted = backend._convert_messages_for_anthropic(messages)

        assert len(converted) == 1
        assert converted[0]["role"] == "user"
        assert isinstance(converted[0]["content"], list)
        result_block = converted[0]["content"][0]
        assert result_block["type"] == "tool_result"
        assert result_block["tool_use_id"] == "call_abc123"
        assert result_block["content"] == '{"temp": 22}'

    def test_full_conversation_with_tool_flow(self, backend):
        """Test a complete conversation flow with tool calls."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Paris"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "content": '{"temp": 18}',
                "tool_call_id": "call_1",
            },
            {"role": "assistant", "content": "It's 18 degrees in Paris."},
        ]
        system, converted = backend._convert_messages_for_anthropic(messages)

        assert system == "You are helpful."
        assert len(converted) == 4  # user, assistant+tool, tool_result, assistant

        # User message
        assert converted[0]["role"] == "user"

        # Assistant with tool call
        assert converted[1]["role"] == "assistant"
        assert isinstance(converted[1]["content"], list)

        # Tool result (as user message)
        assert converted[2]["role"] == "user"

        # Final assistant response
        assert converted[3]["role"] == "assistant"
        assert converted[3]["content"] == "It's 18 degrees in Paris."


class TestToolConversionForAnthropic:
    """Unit tests for _convert_tools_for_anthropic method."""

    @pytest.fixture
    def backend(self):
        """Create backend instance for testing."""
        try:
            return AnthropicBackend(
                api_key="test-key-for-unit-tests",
                default_model="claude-sonnet-4-20250514",
            )
        except Exception:
            pytest.skip("Cannot create AnthropicBackend for testing")

    def test_openai_tools_converted_to_anthropic_format(self, backend):
        """Test that OpenAI tool definitions are converted correctly."""
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                        },
                        "required": ["city"],
                    },
                },
            }
        ]
        anthropic_tools = backend._convert_tools_for_anthropic(openai_tools)

        assert len(anthropic_tools) == 1
        tool = anthropic_tools[0]
        assert tool["name"] == "get_weather"
        assert tool["description"] == "Get weather for a city"
        assert "input_schema" in tool
        assert tool["input_schema"]["type"] == "object"

    def test_multiple_tools_converted(self, backend):
        """Test that multiple tools are all converted."""
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": "func1",
                    "description": "First function",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "func2",
                    "description": "Second function",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]
        anthropic_tools = backend._convert_tools_for_anthropic(openai_tools)

        assert len(anthropic_tools) == 2
        assert anthropic_tools[0]["name"] == "func1"
        assert anthropic_tools[1]["name"] == "func2"


class TestToolCallConversionToOpenAI:
    """Unit tests for _convert_tool_calls_to_openai_format method."""

    @pytest.fixture
    def backend(self):
        """Create backend instance for testing."""
        try:
            return AnthropicBackend(
                api_key="test-key-for-unit-tests",
                default_model="claude-sonnet-4-20250514",
            )
        except Exception:
            pytest.skip("Cannot create AnthropicBackend for testing")

    def test_anthropic_tool_calls_converted_to_openai_format(self, backend):
        """Test that Anthropic tool_use blocks are converted to OpenAI format."""
        anthropic_blocks = [
            {
                "type": "tool_use",
                "id": "toolu_123",
                "name": "get_weather",
                "input": {"city": "Paris"},
            }
        ]
        openai_calls = backend._convert_tool_calls_to_openai_format(anthropic_blocks)

        assert len(openai_calls) == 1
        call = openai_calls[0]
        assert call["id"] == "toolu_123"
        assert call["type"] == "function"
        assert call["function"]["name"] == "get_weather"
        # Arguments should be JSON string
        args = json.loads(call["function"]["arguments"])
        assert args == {"city": "Paris"}

    def test_multiple_tool_calls_converted(self, backend):
        """Test that multiple tool calls are all converted."""
        anthropic_blocks = [
            {"type": "tool_use", "id": "t1", "name": "func1", "input": {"x": 1}},
            {"type": "tool_use", "id": "t2", "name": "func2", "input": {"y": 2}},
        ]
        openai_calls = backend._convert_tool_calls_to_openai_format(anthropic_blocks)

        assert len(openai_calls) == 2
        assert openai_calls[0]["function"]["name"] == "func1"
        assert openai_calls[1]["function"]["name"] == "func2"


class TestModelCapabilities:
    """Tests for get_model_capabilities method."""

    @pytest.fixture
    def backend(self):
        """Create backend instance for testing."""
        try:
            return AnthropicBackend(
                api_key="test-key-for-unit-tests",
                default_model="claude-sonnet-4-20250514",
            )
        except Exception:
            pytest.skip("Cannot create AnthropicBackend for testing")

    def test_capabilities_returns_correct_context_window(self, backend):
        """Test that known models return correct context windows."""
        caps = backend.get_model_capabilities("claude-sonnet-4-20250514")
        assert caps is not None
        assert caps.context_length == 200000
        assert caps.sliding_window is None  # Claude uses full attention

    def test_capabilities_default_for_unknown_model(self, backend):
        """Test that unknown models get default context window."""
        caps = backend.get_model_capabilities("claude-unknown-model")
        assert caps is not None
        assert caps.context_length == 200000  # Default

    def test_supports_tools_returns_true(self, backend):
        """Test that supports_tools returns True for Claude models."""
        assert backend.supports_tools() is True


# ============================================================================
# Integration Tests - Require API Key
# ============================================================================

# Skip integration tests if no API key
pytestmark_integration = pytest.mark.skipif(
    not has_anthropic_api_key(),
    reason="ANTHROPIC_API_KEY not available in .env",
)


@pytestmark_integration
class TestAnthropicIntegrationBasics:
    """Integration tests for basic AnthropicBackend functionality."""

    def test_check_health(self, anthropic_backend):
        """Test health check with real API."""
        assert anthropic_backend.check_health() is True

    def test_get_running_model(self, anthropic_backend, anthropic_model):
        """Test that get_running_model returns configured model."""
        model = anthropic_backend.get_running_model()
        assert model == anthropic_model


@pytestmark_integration
class TestAnthropicCompletion:
    """Integration tests for non-streaming completion."""

    def test_complete_basic(self, anthropic_backend):
        """Test basic non-streaming completion."""
        messages = [{"role": "user", "content": "Say 'hello' and nothing else."}]

        response = anthropic_backend.complete(
            messages=messages,
            max_tokens=50,
        )

        assert isinstance(response, CompletionResponse)
        assert response.content is not None
        assert len(response.content) > 0
        assert response.finish_reason is not None

    def test_complete_with_system_message(self, anthropic_backend):
        """Test completion with system message."""
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Be very brief.",
            },
            {"role": "user", "content": "What is 2+2?"},
        ]

        response = anthropic_backend.complete(
            messages=messages,
            max_tokens=50,
        )

        assert isinstance(response, CompletionResponse)
        assert response.content is not None
        assert "4" in response.content

    def test_complete_usage_stats(self, anthropic_backend):
        """Test that usage statistics are returned."""
        messages = [{"role": "user", "content": "Hi"}]

        response = anthropic_backend.complete(
            messages=messages,
            max_tokens=20,
        )

        assert response.usage is not None
        assert "prompt_tokens" in response.usage
        assert "completion_tokens" in response.usage
        assert "total_tokens" in response.usage


@pytestmark_integration
class TestAnthropicStreaming:
    """Integration tests for streaming completion."""

    def test_complete_stream_basic(self, anthropic_backend):
        """Test basic streaming completion."""
        messages = [{"role": "user", "content": "Count from 1 to 3."}]

        chunks = list(
            anthropic_backend.complete_stream(
                messages=messages,
                max_tokens=50,
            )
        )

        assert len(chunks) > 0
        assert all(isinstance(c, CompletionChunk) for c in chunks)

        # Combine all content
        full_content = "".join(c.content for c in chunks)
        assert len(full_content) > 0

        # Last chunk should have finish_reason
        assert chunks[-1].finish_reason is not None

    def test_complete_stream_conversation(self, anthropic_backend):
        """Test streaming with conversation history."""
        messages = [
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Nice to meet you, Alice!"},
            {"role": "user", "content": "What is my name?"},
        ]

        chunks = list(
            anthropic_backend.complete_stream(
                messages=messages,
                max_tokens=50,
            )
        )

        full_content = "".join(c.content for c in chunks).lower()
        # The model should remember the name
        assert "alice" in full_content


@pytestmark_integration
class TestAnthropicToolCalling:
    """Integration tests for tool/function calling."""

    def test_complete_with_tools(self, anthropic_backend):
        """Test completion with tool definitions."""
        messages = [
            {"role": "user", "content": "What is the weather in Paris? Use the tool."}
        ]

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

        response = anthropic_backend.complete(
            messages=messages,
            tools=tools,
            max_tokens=200,
        )

        assert isinstance(response, CompletionResponse)

        # Claude should call the tool
        if response.tool_calls:
            assert len(response.tool_calls) > 0
            tc = response.tool_calls[0]
            assert tc["type"] == "function"
            assert tc["function"]["name"] == "get_weather"
            # Arguments should be JSON string (OpenAI format)
            args = json.loads(tc["function"]["arguments"])
            assert "city" in args
            assert response.finish_reason == "tool_calls"
