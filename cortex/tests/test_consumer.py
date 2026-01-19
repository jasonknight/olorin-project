#!/usr/bin/env python3
"""
Unit tests for ExoConsumer class.

Tests chat history integration, message processing, and reset functionality
with mocked Kafka and inference dependencies.

Note: The refactored consumer combines context and prompt into a single user message
(as documented in CLAUDE.md) to avoid context loss with models like Deepseek R1.
"""

import os
import sys
import json
import tempfile
from unittest.mock import patch, Mock

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from libs.chat_store import ChatStore
from libs.context_formatter import ContextFormatter


def create_mock_inference():
    """Create a mock inference client."""
    mock = Mock()
    mock.backend_type = Mock(value="exo")
    mock.backend = Mock()
    mock.backend.client = Mock()
    return mock


class TestResetCommandDetection:
    """Tests for reset command detection logic."""

    def test_exact_match_reset(self, mock_config):
        """Should detect exact '/reset' command."""
        with (
            patch("consumer.KafkaConsumer"),
            patch("consumer.KafkaProducer"),
            patch(
                "consumer.get_inference_client", return_value=create_mock_inference()
            ),
            patch("consumer.ContextStore"),
            patch("consumer.ChatStore"),
            patch("consumer.ToolClient"),
        ):
            from consumer import ExoConsumer

            consumer = ExoConsumer(mock_config)
            consumer.config = mock_config

            assert consumer._is_reset_command("/reset") is True
            assert consumer._is_reset_command("reset conversation") is True
            assert consumer._is_reset_command("new conversation") is True

    def test_case_insensitive_reset(self, mock_config):
        """Reset detection should be case insensitive."""
        with (
            patch("consumer.KafkaConsumer"),
            patch("consumer.KafkaProducer"),
            patch(
                "consumer.get_inference_client", return_value=create_mock_inference()
            ),
            patch("consumer.ContextStore"),
            patch("consumer.ChatStore"),
            patch("consumer.ToolClient"),
        ):
            from consumer import ExoConsumer

            consumer = ExoConsumer(mock_config)
            consumer.config = mock_config

            assert consumer._is_reset_command("/RESET") is True
            assert consumer._is_reset_command("Reset Conversation") is True
            assert consumer._is_reset_command("NEW CONVERSATION") is True

    def test_prefix_match_reset(self, mock_config):
        """Should detect reset commands with trailing text."""
        with (
            patch("consumer.KafkaConsumer"),
            patch("consumer.KafkaProducer"),
            patch(
                "consumer.get_inference_client", return_value=create_mock_inference()
            ),
            patch("consumer.ContextStore"),
            patch("consumer.ChatStore"),
            patch("consumer.ToolClient"),
        ):
            from consumer import ExoConsumer

            consumer = ExoConsumer(mock_config)
            consumer.config = mock_config

            assert consumer._is_reset_command("/reset please") is True
            assert consumer._is_reset_command("/reset now") is True

    def test_non_reset_commands(self, mock_config):
        """Should not detect normal prompts as reset commands."""
        with (
            patch("consumer.KafkaConsumer"),
            patch("consumer.KafkaProducer"),
            patch(
                "consumer.get_inference_client", return_value=create_mock_inference()
            ),
            patch("consumer.ContextStore"),
            patch("consumer.ChatStore"),
            patch("consumer.ToolClient"),
        ):
            from consumer import ExoConsumer

            consumer = ExoConsumer(mock_config)
            consumer.config = mock_config

            assert consumer._is_reset_command("Hello, how are you?") is False
            assert (
                consumer._is_reset_command("Tell me about resetting passwords") is False
            )
            assert consumer._is_reset_command("What is a conversation?") is False


class TestContextFormatting:
    """Tests for RAG context formatting using ContextFormatter."""

    def test_format_single_context_chunk(self):
        """Should format single context chunk correctly."""
        formatter = ContextFormatter(system_prompt="You are a helpful assistant.")

        chunks = [
            {
                "content": "Python is a programming language.",
                "source": "docs.md",
                "h1": "Introduction",
            }
        ]

        combined = formatter.format_context_with_prompt(chunks, "What is Python?")

        # Should return a single combined string
        assert isinstance(combined, str)
        assert "Python is a programming language." in combined
        assert "docs.md" in combined
        assert "Introduction" in combined
        assert "What is Python?" in combined
        assert "<context>" in combined
        assert "</context>" in combined

    def test_format_multiple_context_chunks(self):
        """Should format multiple context chunks correctly."""
        formatter = ContextFormatter()

        chunks = [
            {
                "content": "First chunk",
                "source": "file1.md",
                "h1": "Header1",
            },
            {
                "content": "Second chunk",
                "source": "file2.md",
                "h1": "Header2",
                "h2": "Sub",
            },
        ]

        combined = formatter.format_context_with_prompt(chunks, "Summarize the content")

        assert isinstance(combined, str)
        assert "First chunk" in combined
        assert "Second chunk" in combined
        assert "file1.md" in combined
        assert "file2.md" in combined
        assert "Summarize the content" in combined

    def test_format_context_with_missing_metadata(self):
        """Should handle chunks with missing metadata gracefully."""
        formatter = ContextFormatter()

        chunks = [{"content": "Content only"}]  # No source, h1, etc.

        combined = formatter.format_context_with_prompt(chunks, "What is this?")

        assert isinstance(combined, str)
        assert "Content only" in combined
        assert "What is this?" in combined


class TestMessageHistoryBuilding:
    """Tests for building messages array with conversation history."""

    def test_build_messages_without_history(self, mock_config):
        """Should build single message when chat store is disabled."""
        with (
            patch("consumer.KafkaConsumer"),
            patch("consumer.KafkaProducer"),
            patch(
                "consumer.get_inference_client", return_value=create_mock_inference()
            ),
            patch("consumer.ContextStore"),
            patch("consumer.ChatStore"),
            patch("consumer.ToolClient"),
        ):
            from consumer import ExoConsumer

            consumer = ExoConsumer(mock_config)
            consumer.chat_store = None  # Simulate disabled chat history
            consumer.available_tools = []  # No tools

            messages = consumer._build_messages_with_history(
                prompt="Hello", message_id="test_001", context_chunks=None
            )

            assert len(messages) == 1
            assert messages[0]["role"] == "user"
            assert messages[0]["content"] == "Hello"

    def test_build_messages_with_context_no_history(self, mock_config):
        """Should combine context and prompt in single user message."""
        with (
            patch("consumer.KafkaConsumer"),
            patch("consumer.KafkaProducer"),
            patch(
                "consumer.get_inference_client", return_value=create_mock_inference()
            ),
            patch("consumer.ContextStore"),
            patch("consumer.ChatStore"),
            patch("consumer.ToolClient"),
        ):
            from consumer import ExoConsumer

            consumer = ExoConsumer(mock_config)
            consumer.chat_store = None
            consumer.available_tools = []

            context_chunks = [
                {
                    "content": "Context info",
                    "source": "doc.md",
                }
            ]

            messages = consumer._build_messages_with_history(
                prompt="Question?", message_id="test_002", context_chunks=context_chunks
            )

            # Should have: combined context + prompt in single user message
            assert len(messages) == 1
            assert messages[0]["role"] == "user"
            assert "Context info" in messages[0]["content"]
            assert "Question?" in messages[0]["content"]
            assert "<context>" in messages[0]["content"]

    def test_build_messages_with_history(self, mock_config):
        """Should include conversation history in messages array."""
        with (
            patch("consumer.KafkaConsumer"),
            patch("consumer.KafkaProducer"),
            patch(
                "consumer.get_inference_client", return_value=create_mock_inference()
            ),
            patch("consumer.ContextStore"),
            patch("consumer.ToolClient"),
        ):
            from consumer import ExoConsumer

            with tempfile.TemporaryDirectory() as tmpdir:
                chat_db_path = os.path.join(tmpdir, "chat.db")
                mock_config.chat_db_path = chat_db_path

                consumer = ExoConsumer(mock_config)
                consumer.available_tools = []

                # Manually set up chat store with history
                consumer.chat_store = ChatStore(chat_db_path)
                conv_id = consumer.chat_store.get_or_create_active_conversation()
                consumer.chat_store.add_user_message(conv_id, "Previous question")
                consumer.chat_store.add_assistant_message(conv_id, "Previous answer")

                messages = consumer._build_messages_with_history(
                    prompt="New question", message_id="test_003", context_chunks=None
                )

                # Should have: 2 history + 1 new = 3
                assert len(messages) == 3
                assert messages[0]["content"] == "Previous question"
                assert messages[0]["role"] == "user"
                assert messages[1]["content"] == "Previous answer"
                assert messages[1]["role"] == "assistant"
                assert messages[2]["content"] == "New question"
                assert messages[2]["role"] == "user"

    def test_build_messages_with_history_and_context(self, mock_config):
        """Should include both history and context in correct order."""
        with (
            patch("consumer.KafkaConsumer"),
            patch("consumer.KafkaProducer"),
            patch(
                "consumer.get_inference_client", return_value=create_mock_inference()
            ),
            patch("consumer.ContextStore"),
            patch("consumer.ToolClient"),
        ):
            from consumer import ExoConsumer

            with tempfile.TemporaryDirectory() as tmpdir:
                chat_db_path = os.path.join(tmpdir, "chat.db")
                mock_config.chat_db_path = chat_db_path

                consumer = ExoConsumer(mock_config)
                consumer.available_tools = []
                consumer.chat_store = ChatStore(chat_db_path)

                conv_id = consumer.chat_store.get_or_create_active_conversation()
                consumer.chat_store.add_user_message(conv_id, "First message")
                consumer.chat_store.add_assistant_message(conv_id, "First response")

                context_chunks = [
                    {
                        "id": "ctx_001",
                        "content": "Relevant context",
                        "source": "ref.md",
                    }
                ]

                messages = consumer._build_messages_with_history(
                    prompt="Question with context",
                    message_id="test_004",
                    context_chunks=context_chunks,
                )

                # Should have: 2 history + 1 combined context+prompt = 3
                assert len(messages) == 3
                assert messages[0]["content"] == "First message"
                assert messages[1]["content"] == "First response"
                # Last message should be combined context + prompt
                assert "Relevant context" in messages[2]["content"]
                assert "Question with context" in messages[2]["content"]


class TestResetCommandHandling:
    """Tests for conversation reset handling."""

    def test_handle_reset_sends_confirmation(self, mock_config, mock_kafka_producer):
        """Reset should send confirmation message to Kafka."""
        with (
            patch("consumer.KafkaConsumer"),
            patch("consumer.KafkaProducer", return_value=mock_kafka_producer),
            patch(
                "consumer.get_inference_client", return_value=create_mock_inference()
            ),
            patch("consumer.ContextStore"),
            patch("consumer.ToolClient"),
        ):
            from consumer import ExoConsumer

            with tempfile.TemporaryDirectory() as tmpdir:
                chat_db_path = os.path.join(tmpdir, "chat.db")
                mock_config.chat_db_path = chat_db_path

                consumer = ExoConsumer(mock_config)
                consumer.producer = mock_kafka_producer
                consumer.chat_store = ChatStore(chat_db_path)

                # Create an existing conversation
                consumer.chat_store.get_or_create_active_conversation()

                result = consumer._handle_reset_command("test_msg_reset")

                assert result is True
                assert len(mock_kafka_producer.sent_messages) == 1

                sent_msg = mock_kafka_producer.sent_messages[0]["value"]
                assert "reset" in sent_msg["text"].lower()
                assert sent_msg["is_reset_confirmation"] is True

    def test_handle_reset_creates_new_conversation(
        self, mock_config, mock_kafka_producer
    ):
        """Reset should close old conversation and create new one."""
        with (
            patch("consumer.KafkaConsumer"),
            patch("consumer.KafkaProducer", return_value=mock_kafka_producer),
            patch(
                "consumer.get_inference_client", return_value=create_mock_inference()
            ),
            patch("consumer.ContextStore"),
            patch("consumer.ToolClient"),
        ):
            from consumer import ExoConsumer

            with tempfile.TemporaryDirectory() as tmpdir:
                chat_db_path = os.path.join(tmpdir, "chat.db")
                mock_config.chat_db_path = chat_db_path

                consumer = ExoConsumer(mock_config)
                consumer.producer = mock_kafka_producer
                consumer.chat_store = ChatStore(chat_db_path)

                old_conv = consumer.chat_store.get_or_create_active_conversation()
                consumer._handle_reset_command("test_msg")
                new_conv = consumer.chat_store.get_active_conversation_id()

                assert new_conv != old_conv

    def test_handle_reset_when_chat_disabled(self, mock_config):
        """Reset should return False when chat history is disabled."""
        with (
            patch("consumer.KafkaConsumer"),
            patch("consumer.KafkaProducer"),
            patch(
                "consumer.get_inference_client", return_value=create_mock_inference()
            ),
            patch("consumer.ContextStore"),
            patch("consumer.ChatStore"),
            patch("consumer.ToolClient"),
        ):
            from consumer import ExoConsumer

            consumer = ExoConsumer(mock_config)
            consumer.chat_store = None  # Simulate disabled

            result = consumer._handle_reset_command("test_msg")
            assert result is False


class TestMessageParsing:
    """Tests for Kafka message parsing."""

    def test_parse_json_message(self):
        """Should correctly parse JSON formatted messages."""
        # Test the parsing logic extracted from process_message
        message = '{"text": "Hello", "id": "msg_123", "context_available": true, "contexts_stored": 2}'
        parsed = json.loads(message)

        prompt = parsed.get("text", "") or parsed.get("prompt", "")
        assert prompt == "Hello"
        assert parsed.get("id") == "msg_123"
        assert parsed.get("context_available") is True
        assert parsed.get("contexts_stored") == 2

    def test_parse_plain_text_message(self):
        """Should handle plain text messages gracefully."""
        # Plain text that's not valid JSON
        message = "Just a plain text message"

        try:
            json.loads(message)
            assert False, "Should have raised JSONDecodeError"
        except json.JSONDecodeError:
            # This is expected - plain text should be treated as the prompt itself
            prompt = message
            assert prompt == "Just a plain text message"


class TestIntegration:
    """Integration tests with real ChatStore but mocked external services."""

    def test_full_conversation_flow(self, mock_config, mock_kafka_producer):
        """Test a full conversation: multiple messages, then reset."""
        with (
            patch("consumer.KafkaConsumer"),
            patch("consumer.KafkaProducer", return_value=mock_kafka_producer),
            patch(
                "consumer.get_inference_client", return_value=create_mock_inference()
            ),
            patch("consumer.ContextStore"),
            patch("consumer.ToolClient"),
        ):
            from consumer import ExoConsumer

            with tempfile.TemporaryDirectory() as tmpdir:
                chat_db_path = os.path.join(tmpdir, "chat.db")
                mock_config.chat_db_path = chat_db_path

                consumer = ExoConsumer(mock_config)
                consumer.producer = mock_kafka_producer
                consumer.chat_store = ChatStore(chat_db_path)
                consumer.available_tools = []

                # Simulate conversation
                conv_id = consumer.chat_store.get_or_create_active_conversation()

                # Add some messages
                consumer.chat_store.add_user_message(conv_id, "Hello, I'm Alice")
                consumer.chat_store.add_assistant_message(conv_id, "Hello Alice!")
                consumer.chat_store.add_user_message(conv_id, "What's my name?")
                consumer.chat_store.add_assistant_message(
                    conv_id, "Your name is Alice."
                )

                # Build messages - should include all history
                messages = consumer._build_messages_with_history(
                    prompt="Tell me a joke", message_id="msg_005"
                )

                # 4 history + 1 new = 5
                assert len(messages) == 5

                # Now reset
                consumer._handle_reset_command("reset_msg")

                # Build messages again - should be fresh
                messages = consumer._build_messages_with_history(
                    prompt="Hello again", message_id="msg_006"
                )

                # Only the new message (history was reset)
                assert len(messages) == 1
                assert messages[0]["content"] == "Hello again"

    def test_context_injection_preserves_history(self, mock_config):
        """Context should be combined with prompt after history."""
        with (
            patch("consumer.KafkaConsumer"),
            patch("consumer.KafkaProducer"),
            patch(
                "consumer.get_inference_client", return_value=create_mock_inference()
            ),
            patch("consumer.ContextStore"),
            patch("consumer.ToolClient"),
        ):
            from consumer import ExoConsumer

            with tempfile.TemporaryDirectory() as tmpdir:
                chat_db_path = os.path.join(tmpdir, "chat.db")
                mock_config.chat_db_path = chat_db_path

                consumer = ExoConsumer(mock_config)
                consumer.chat_store = ChatStore(chat_db_path)
                consumer.available_tools = []

                # Build up conversation history
                conv_id = consumer.chat_store.get_or_create_active_conversation()
                consumer.chat_store.add_user_message(conv_id, "History message 1")
                consumer.chat_store.add_assistant_message(conv_id, "History response 1")

                # Now request with context
                context = [
                    {
                        "id": "ctx_001",
                        "content": "RAG context",
                        "source": "doc.md",
                    }
                ]

                messages = consumer._build_messages_with_history(
                    prompt="Question about docs",
                    message_id="msg_007",
                    context_chunks=context,
                )

                # Verify order: history (2) + combined context+prompt (1) = 3
                assert len(messages) == 3
                assert messages[0]["content"] == "History message 1"
                assert messages[1]["content"] == "History response 1"
                # Combined message has context and prompt
                assert "RAG context" in messages[2]["content"]
                assert "Question about docs" in messages[2]["content"]


class TestContextPersistenceInChatHistory:
    """Tests for RAG context being stored in chat history for conversation continuity."""

    def test_context_stored_in_chat_history(self, mock_config):
        """Context-with-prompt should be stored in chat history."""
        with (
            patch("consumer.KafkaConsumer"),
            patch("consumer.KafkaProducer"),
            patch(
                "consumer.get_inference_client", return_value=create_mock_inference()
            ),
            patch("consumer.ContextStore"),
            patch("consumer.ToolClient"),
        ):
            from consumer import ExoConsumer

            with tempfile.TemporaryDirectory() as tmpdir:
                chat_db_path = os.path.join(tmpdir, "chat.db")
                mock_config.chat_db_path = chat_db_path

                consumer = ExoConsumer(mock_config)
                consumer.chat_store = ChatStore(chat_db_path)
                consumer.available_tools = []

                context_chunks = [
                    {
                        "id": "ctx_001",
                        "content": "Python is a programming language.",
                        "source": "docs.md",
                    }
                ]

                # Build messages with context - this should store context in chat history
                consumer._build_messages_with_history(
                    prompt="What is Python?",
                    message_id="test_ctx_001",
                    context_chunks=context_chunks,
                )

                # Get the conversation and verify stored messages
                conv_id = consumer.chat_store.get_active_conversation_id()
                stored_messages = consumer.chat_store.get_conversation_messages(conv_id)

                # Should have 1 message with combined context + prompt
                assert len(stored_messages) == 1
                assert stored_messages[0]["role"] == "user"
                assert (
                    "Python is a programming language" in stored_messages[0]["content"]
                )
                assert "What is Python?" in stored_messages[0]["content"]

    def test_followup_message_sees_context_in_history(self, mock_config):
        """Follow-up messages should see previously stored context in their history."""
        with (
            patch("consumer.KafkaConsumer"),
            patch("consumer.KafkaProducer"),
            patch(
                "consumer.get_inference_client", return_value=create_mock_inference()
            ),
            patch("consumer.ContextStore"),
            patch("consumer.ToolClient"),
        ):
            from consumer import ExoConsumer

            with tempfile.TemporaryDirectory() as tmpdir:
                chat_db_path = os.path.join(tmpdir, "chat.db")
                mock_config.chat_db_path = chat_db_path

                consumer = ExoConsumer(mock_config)
                consumer.chat_store = ChatStore(chat_db_path)
                consumer.available_tools = []

                # First message with context
                context_chunks = [
                    {
                        "id": "ctx_001",
                        "content": "Python was created by Guido van Rossum.",
                        "source": "history.md",
                    }
                ]

                consumer._build_messages_with_history(
                    prompt="Tell me about Python's history",
                    message_id="test_ctx_002",
                    context_chunks=context_chunks,
                )

                # Simulate assistant response being stored
                conv_id = consumer.chat_store.get_active_conversation_id()
                consumer.chat_store.add_assistant_message(
                    conv_id, "Python was created by Guido van Rossum in 1991."
                )

                # Now send a follow-up WITHOUT new context
                followup_messages = consumer._build_messages_with_history(
                    prompt="Can you tell me more about Guido?",
                    message_id="test_ctx_003",
                    context_chunks=None,  # No new context
                )

                # The follow-up should see the original context+prompt in its history
                # History: combined context+prompt (1) + assistant response (1) + new prompt (1) = 3
                assert len(followup_messages) == 3

                # Verify the context is in the history
                assert "Guido van Rossum" in followup_messages[0]["content"]
                assert "Guido van Rossum in 1991" in followup_messages[1]["content"]
                assert (
                    followup_messages[2]["content"]
                    == "Can you tell me more about Guido?"
                )

    def test_context_cleared_on_conversation_reset(
        self, mock_config, mock_kafka_producer
    ):
        """Context should be cleared when conversation is reset."""
        with (
            patch("consumer.KafkaConsumer"),
            patch("consumer.KafkaProducer", return_value=mock_kafka_producer),
            patch(
                "consumer.get_inference_client", return_value=create_mock_inference()
            ),
            patch("consumer.ContextStore"),
            patch("consumer.ToolClient"),
        ):
            from consumer import ExoConsumer

            with tempfile.TemporaryDirectory() as tmpdir:
                chat_db_path = os.path.join(tmpdir, "chat.db")
                mock_config.chat_db_path = chat_db_path

                consumer = ExoConsumer(mock_config)
                consumer.producer = mock_kafka_producer
                consumer.chat_store = ChatStore(chat_db_path)
                consumer.available_tools = []

                # First message with context
                context_chunks = [
                    {
                        "id": "ctx_001",
                        "content": "Important context info.",
                        "source": "doc.md",
                    }
                ]
                consumer._build_messages_with_history(
                    prompt="Question with context",
                    message_id="reset_001",
                    context_chunks=context_chunks,
                )

                # Verify context is in history
                old_conv_id = consumer.chat_store.get_active_conversation_id()
                old_messages = consumer.chat_store.get_conversation_messages(
                    old_conv_id
                )
                assert len(old_messages) == 1
                assert "Important context info" in old_messages[0]["content"]

                # Reset conversation
                consumer._handle_reset_command("reset_msg")

                # New message after reset - should NOT see old context
                new_messages = consumer._build_messages_with_history(
                    prompt="Fresh start", message_id="reset_002", context_chunks=None
                )

                # Should only have the new message
                assert len(new_messages) == 1
                assert new_messages[0]["content"] == "Fresh start"

                # Old conversation should still have its messages (preserved but inactive)
                old_messages_after_reset = (
                    consumer.chat_store.get_conversation_messages(old_conv_id)
                )
                assert len(old_messages_after_reset) == 1


class TestExecuteToolCalls:
    """Tests for _execute_tool_calls method."""

    def test_string_result_passed_through(self, mock_config):
        """Tool results that are strings should be passed through unchanged."""
        with (
            patch("consumer.KafkaConsumer"),
            patch("consumer.KafkaProducer"),
            patch(
                "consumer.get_inference_client", return_value=create_mock_inference()
            ),
            patch("consumer.ContextStore"),
            patch("consumer.ChatStore"),
            patch("consumer.ToolClient"),
        ):
            from consumer import ExoConsumer

            consumer = ExoConsumer(mock_config)

            # Mock tool client to return a string result
            mock_tool_client = Mock()
            mock_tool_client.call_tool.return_value = {
                "success": True,
                "result": "File written successfully",
            }
            consumer.tool_client = mock_tool_client

            tool_calls = [
                {
                    "id": "call_123",
                    "function": {
                        "name": "write",
                        "arguments": '{"content": "test", "filename": "test.txt"}',
                    },
                }
            ]

            results = consumer._execute_tool_calls(tool_calls, None, None)

            assert len(results) == 1
            assert results[0]["role"] == "tool"
            assert results[0]["content"] == "File written successfully"

    def test_dict_result_converted_to_json_string(self, mock_config):
        """Tool results that are dicts should be converted to JSON strings."""
        with (
            patch("consumer.KafkaConsumer"),
            patch("consumer.KafkaProducer"),
            patch(
                "consumer.get_inference_client", return_value=create_mock_inference()
            ),
            patch("consumer.ContextStore"),
            patch("consumer.ChatStore"),
            patch("consumer.ToolClient"),
        ):
            from consumer import ExoConsumer

            consumer = ExoConsumer(mock_config)

            # Mock tool client to return a dict result (like embeddings tool)
            mock_tool_client = Mock()
            mock_tool_client.call_tool.return_value = {
                "success": True,
                "result": {
                    "embeddings": [[0.1, 0.2, 0.3]],
                    "model": "nomic-embed-text-v1.5",
                    "dimension": 768,
                },
            }
            consumer.tool_client = mock_tool_client

            tool_calls = [
                {
                    "id": "call_456",
                    "function": {
                        "name": "embeddings",
                        "arguments": '{"texts": ["test"], "mode": "query"}',
                    },
                }
            ]

            results = consumer._execute_tool_calls(tool_calls, None, None)

            assert len(results) == 1
            assert results[0]["role"] == "tool"
            # Should be a JSON string, not a dict
            assert isinstance(results[0]["content"], str)
            # Should be valid JSON
            parsed = json.loads(results[0]["content"])
            assert parsed["embeddings"] == [[0.1, 0.2, 0.3]]
            assert parsed["model"] == "nomic-embed-text-v1.5"

    def test_list_result_converted_to_json_string(self, mock_config):
        """Tool results that are lists should be converted to JSON strings."""
        with (
            patch("consumer.KafkaConsumer"),
            patch("consumer.KafkaProducer"),
            patch(
                "consumer.get_inference_client", return_value=create_mock_inference()
            ),
            patch("consumer.ContextStore"),
            patch("consumer.ChatStore"),
            patch("consumer.ToolClient"),
        ):
            from consumer import ExoConsumer

            consumer = ExoConsumer(mock_config)

            # Mock tool client to return a list result
            mock_tool_client = Mock()
            mock_tool_client.call_tool.return_value = {
                "success": True,
                "result": ["item1", "item2", "item3"],
            }
            consumer.tool_client = mock_tool_client

            tool_calls = [
                {
                    "id": "call_789",
                    "function": {
                        "name": "search",
                        "arguments": '{"query": "test"}',
                    },
                }
            ]

            results = consumer._execute_tool_calls(tool_calls, None, None)

            assert len(results) == 1
            assert isinstance(results[0]["content"], str)
            assert json.loads(results[0]["content"]) == ["item1", "item2", "item3"]

    def test_error_result_formatted_as_string(self, mock_config):
        """Tool errors should be formatted as error strings."""
        with (
            patch("consumer.KafkaConsumer"),
            patch("consumer.KafkaProducer"),
            patch(
                "consumer.get_inference_client", return_value=create_mock_inference()
            ),
            patch("consumer.ContextStore"),
            patch("consumer.ChatStore"),
            patch("consumer.ToolClient"),
        ):
            from consumer import ExoConsumer

            consumer = ExoConsumer(mock_config)

            # Mock tool client to return an error
            mock_tool_client = Mock()
            mock_tool_client.call_tool.return_value = {
                "success": False,
                "error": {
                    "type": "ValidationError",
                    "message": "Missing required parameter",
                },
            }
            consumer.tool_client = mock_tool_client

            tool_calls = [
                {
                    "id": "call_err",
                    "function": {
                        "name": "write",
                        "arguments": "{}",
                    },
                }
            ]

            results = consumer._execute_tool_calls(tool_calls, None, None)

            assert len(results) == 1
            assert results[0]["role"] == "tool"
            assert (
                "Error: ValidationError: Missing required parameter"
                in results[0]["content"]
            )
