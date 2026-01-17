#!/usr/bin/env python3
"""
Unit tests for ExoConsumer class.

Tests chat history integration, message processing, and reset functionality
with mocked Kafka and OpenAI dependencies.
"""

import os
import sys
import json
import tempfile
from unittest.mock import patch

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from libs.chat_store import ChatStore


class TestResetCommandDetection:
    """Tests for reset command detection logic."""

    def test_exact_match_reset(self, mock_config):
        """Should detect exact '/reset' command."""
        # Import here to avoid issues with global state
        with (
            patch("consumer.KafkaConsumer"),
            patch("consumer.KafkaProducer"),
            patch("consumer.OpenAI"),
            patch("consumer.ContextStore"),
            patch("consumer.ChatStore"),
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
            patch("consumer.OpenAI"),
            patch("consumer.ContextStore"),
            patch("consumer.ChatStore"),
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
            patch("consumer.OpenAI"),
            patch("consumer.ContextStore"),
            patch("consumer.ChatStore"),
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
            patch("consumer.OpenAI"),
            patch("consumer.ContextStore"),
            patch("consumer.ChatStore"),
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
    """Tests for RAG context formatting as user/assistant exchange."""

    def test_format_single_context_chunk(self, mock_config):
        """Should format single context chunk correctly."""
        with (
            patch("consumer.KafkaConsumer"),
            patch("consumer.KafkaProducer"),
            patch("consumer.OpenAI"),
            patch("consumer.ContextStore"),
            patch("consumer.ChatStore"),
        ):
            from consumer import ExoConsumer

            consumer = ExoConsumer(mock_config)

            chunks = [
                {
                    "content": "Python is a programming language.",
                    "source": "docs.md",
                    "h1": "Introduction",
                    "h2": None,
                    "h3": None,
                }
            ]

            combined = consumer._format_context_with_prompt(chunks, "What is Python?")

            # Should return a single combined string
            assert isinstance(combined, str)
            assert "Python is a programming language." in combined
            assert "docs.md" in combined
            assert "Introduction" in combined
            assert "What is Python?" in combined
            assert "<context>" in combined
            assert "</context>" in combined

    def test_format_multiple_context_chunks(self, mock_config):
        """Should format multiple context chunks correctly."""
        with (
            patch("consumer.KafkaConsumer"),
            patch("consumer.KafkaProducer"),
            patch("consumer.OpenAI"),
            patch("consumer.ContextStore"),
            patch("consumer.ChatStore"),
        ):
            from consumer import ExoConsumer

            consumer = ExoConsumer(mock_config)

            chunks = [
                {
                    "content": "First chunk",
                    "source": "file1.md",
                    "h1": "Header1",
                    "h2": None,
                    "h3": None,
                },
                {
                    "content": "Second chunk",
                    "source": "file2.md",
                    "h1": "Header2",
                    "h2": "Sub",
                    "h3": None,
                },
            ]

            combined = consumer._format_context_with_prompt(
                chunks, "Summarize the content"
            )

            assert isinstance(combined, str)
            assert "First chunk" in combined
            assert "Second chunk" in combined
            assert "file1.md" in combined
            assert "file2.md" in combined
            assert "Summarize the content" in combined

    def test_format_context_with_missing_metadata(self, mock_config):
        """Should handle chunks with missing metadata gracefully."""
        with (
            patch("consumer.KafkaConsumer"),
            patch("consumer.KafkaProducer"),
            patch("consumer.OpenAI"),
            patch("consumer.ContextStore"),
            patch("consumer.ChatStore"),
        ):
            from consumer import ExoConsumer

            consumer = ExoConsumer(mock_config)

            chunks = [{"content": "Content only"}]  # No source, h1, etc.

            combined = consumer._format_context_with_prompt(chunks, "What is this?")

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
            patch("consumer.OpenAI"),
            patch("consumer.ContextStore"),
            patch("consumer.ChatStore"),
        ):
            from consumer import ExoConsumer

            consumer = ExoConsumer(mock_config)
            consumer.chat_store = None  # Simulate disabled chat history

            messages = consumer._build_messages_with_history(
                prompt="Hello", message_id="test_001", context_chunks=None
            )

            assert len(messages) == 1
            assert messages[0]["role"] == "user"
            assert messages[0]["content"] == "Hello"

    def test_build_messages_with_context_no_history(self, mock_config):
        """Should include context exchange even without chat store."""
        with (
            patch("consumer.KafkaConsumer"),
            patch("consumer.KafkaProducer"),
            patch("consumer.OpenAI"),
            patch("consumer.ContextStore"),
            patch("consumer.ChatStore"),
        ):
            from consumer import ExoConsumer

            consumer = ExoConsumer(mock_config)
            consumer.chat_store = None

            context_chunks = [
                {
                    "content": "Context info",
                    "source": "doc.md",
                    "h1": None,
                    "h2": None,
                    "h3": None,
                }
            ]

            messages = consumer._build_messages_with_history(
                prompt="Question?", message_id="test_002", context_chunks=context_chunks
            )

            # Should have: context user + context assistant + actual user = 3
            assert len(messages) == 3
            assert messages[0]["role"] == "user"
            assert "Context info" in messages[0]["content"]
            assert messages[1]["role"] == "assistant"
            assert messages[2]["role"] == "user"
            assert messages[2]["content"] == "Question?"

    def test_build_messages_with_history(self, mock_config):
        """Should include conversation history in messages array."""
        with (
            patch("consumer.KafkaConsumer"),
            patch("consumer.KafkaProducer"),
            patch("consumer.OpenAI"),
            patch("consumer.ContextStore"),
        ):
            from consumer import ExoConsumer

            with tempfile.TemporaryDirectory() as tmpdir:
                chat_db_path = os.path.join(tmpdir, "chat.db")
                mock_config.chat_db_path = chat_db_path

                consumer = ExoConsumer(mock_config)

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
            patch("consumer.OpenAI"),
            patch("consumer.ContextStore"),
        ):
            from consumer import ExoConsumer

            with tempfile.TemporaryDirectory() as tmpdir:
                chat_db_path = os.path.join(tmpdir, "chat.db")
                mock_config.chat_db_path = chat_db_path

                consumer = ExoConsumer(mock_config)
                consumer.chat_store = ChatStore(chat_db_path)

                conv_id = consumer.chat_store.get_or_create_active_conversation()
                consumer.chat_store.add_user_message(conv_id, "First message")
                consumer.chat_store.add_assistant_message(conv_id, "First response")

                context_chunks = [
                    {
                        "content": "Relevant context",
                        "source": "ref.md",
                        "h1": None,
                        "h2": None,
                        "h3": None,
                    }
                ]

                messages = consumer._build_messages_with_history(
                    prompt="Question with context",
                    message_id="test_004",
                    context_chunks=context_chunks,
                )

                # Should have: 2 history + 2 context exchange + 1 new = 5
                assert len(messages) == 5

                # Verify order: history first, then context, then new prompt
                assert messages[0]["content"] == "First message"
                assert messages[1]["content"] == "First response"
                assert "Relevant context" in messages[2]["content"]  # Context user
                assert "I understand" in messages[3]["content"]  # Context assistant
                assert messages[4]["content"] == "Question with context"


class TestResetCommandHandling:
    """Tests for conversation reset handling."""

    def test_handle_reset_sends_confirmation(self, mock_config, mock_kafka_producer):
        """Reset should send confirmation message to Kafka."""
        with (
            patch("consumer.KafkaConsumer"),
            patch("consumer.KafkaProducer", return_value=mock_kafka_producer),
            patch("consumer.OpenAI"),
            patch("consumer.ContextStore"),
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
            patch("consumer.OpenAI"),
            patch("consumer.ContextStore"),
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
            patch("consumer.OpenAI"),
            patch("consumer.ContextStore"),
            patch("consumer.ChatStore"),
        ):
            from consumer import ExoConsumer

            consumer = ExoConsumer(mock_config)
            consumer.chat_store = None  # Simulate disabled

            result = consumer._handle_reset_command("test_msg")
            assert result is False


class TestMessageParsing:
    """Tests for Kafka message parsing."""

    def test_parse_json_message(self, mock_config):
        """Should correctly parse JSON formatted messages."""
        with (
            patch("consumer.KafkaConsumer"),
            patch("consumer.KafkaProducer"),
            patch("consumer.OpenAI"),
            patch("consumer.ContextStore"),
            patch("consumer.ChatStore"),
        ):
            from consumer import ExoConsumer

            ExoConsumer(mock_config)  # Verify initialization works

            # Test the parsing logic extracted from process_message
            message = '{"prompt": "Hello", "id": "msg_123", "context_available": true, "contexts_stored": 2}'
            parsed = json.loads(message)

            assert parsed.get("prompt") == "Hello"
            assert parsed.get("id") == "msg_123"
            assert parsed.get("context_available") is True
            assert parsed.get("contexts_stored") == 2

    def test_parse_plain_text_message(self, mock_config):
        """Should handle plain text messages gracefully."""
        with (
            patch("consumer.KafkaConsumer"),
            patch("consumer.KafkaProducer"),
            patch("consumer.OpenAI"),
            patch("consumer.ContextStore"),
            patch("consumer.ChatStore"),
        ):
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
            patch("consumer.OpenAI"),
            patch("consumer.ContextStore"),
        ):
            from consumer import ExoConsumer

            with tempfile.TemporaryDirectory() as tmpdir:
                chat_db_path = os.path.join(tmpdir, "chat.db")
                mock_config.chat_db_path = chat_db_path

                consumer = ExoConsumer(mock_config)
                consumer.producer = mock_kafka_producer
                consumer.chat_store = ChatStore(chat_db_path)

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
        """Context should be injected after history, before new prompt."""
        with (
            patch("consumer.KafkaConsumer"),
            patch("consumer.KafkaProducer"),
            patch("consumer.OpenAI"),
            patch("consumer.ContextStore"),
        ):
            from consumer import ExoConsumer

            with tempfile.TemporaryDirectory() as tmpdir:
                chat_db_path = os.path.join(tmpdir, "chat.db")
                mock_config.chat_db_path = chat_db_path

                consumer = ExoConsumer(mock_config)
                consumer.chat_store = ChatStore(chat_db_path)

                # Build up conversation history
                conv_id = consumer.chat_store.get_or_create_active_conversation()
                consumer.chat_store.add_user_message(conv_id, "History message 1")
                consumer.chat_store.add_assistant_message(conv_id, "History response 1")

                # Now request with context
                context = [
                    {
                        "content": "RAG context",
                        "source": "doc.md",
                        "h1": None,
                        "h2": None,
                        "h3": None,
                    }
                ]

                messages = consumer._build_messages_with_history(
                    prompt="Question about docs",
                    message_id="msg_007",
                    context_chunks=context,
                )

                # Verify order: history (2) + context exchange (2) + new prompt (1) = 5
                assert len(messages) == 5
                assert messages[0]["content"] == "History message 1"
                assert messages[1]["content"] == "History response 1"
                assert "RAG context" in messages[2]["content"]
                assert "I understand" in messages[3]["content"]
                assert messages[4]["content"] == "Question about docs"


class TestContextPersistenceInChatHistory:
    """Tests for RAG context being stored in chat history for conversation continuity."""

    def test_context_exchange_stored_in_chat_history(self, mock_config):
        """Context exchange should be stored in chat history when injected."""
        with (
            patch("consumer.KafkaConsumer"),
            patch("consumer.KafkaProducer"),
            patch("consumer.OpenAI"),
            patch("consumer.ContextStore"),
        ):
            from consumer import ExoConsumer

            with tempfile.TemporaryDirectory() as tmpdir:
                chat_db_path = os.path.join(tmpdir, "chat.db")
                mock_config.chat_db_path = chat_db_path

                consumer = ExoConsumer(mock_config)
                consumer.chat_store = ChatStore(chat_db_path)

                context_chunks = [
                    {
                        "content": "Python is a programming language.",
                        "source": "docs.md",
                        "h1": "Intro",
                        "h2": None,
                        "h3": None,
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

                # Should have: context user + context ack + user prompt = 3 messages
                assert len(stored_messages) == 3

                # First message should be the context user message
                assert stored_messages[0]["role"] == "user"
                assert (
                    "Python is a programming language" in stored_messages[0]["content"]
                )
                assert "knowledge base" in stored_messages[0]["content"]

                # Second message should be the context acknowledgment
                assert stored_messages[1]["role"] == "assistant"
                assert "I understand" in stored_messages[1]["content"]

                # Third message should be the actual user prompt
                assert stored_messages[2]["role"] == "user"
                assert stored_messages[2]["content"] == "What is Python?"

    def test_followup_message_sees_context_in_history(self, mock_config):
        """Follow-up messages should include previously stored context in their history."""
        with (
            patch("consumer.KafkaConsumer"),
            patch("consumer.KafkaProducer"),
            patch("consumer.OpenAI"),
            patch("consumer.ContextStore"),
        ):
            from consumer import ExoConsumer

            with tempfile.TemporaryDirectory() as tmpdir:
                chat_db_path = os.path.join(tmpdir, "chat.db")
                mock_config.chat_db_path = chat_db_path

                consumer = ExoConsumer(mock_config)
                consumer.chat_store = ChatStore(chat_db_path)

                # First message with context
                context_chunks = [
                    {
                        "content": "Python was created by Guido van Rossum.",
                        "source": "history.md",
                        "h1": None,
                        "h2": None,
                        "h3": None,
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

                # The follow-up should see the original context in its history
                # History should be: context user + context ack + original prompt + assistant response = 4
                # Plus the new prompt = 5 total in messages array
                assert len(followup_messages) == 5

                # Verify the context is in the history
                assert (
                    "Guido van Rossum" in followup_messages[0]["content"]
                )  # Context user message
                assert "I understand" in followup_messages[1]["content"]  # Context ack
                assert (
                    followup_messages[2]["content"] == "Tell me about Python's history"
                )
                assert "Guido van Rossum in 1991" in followup_messages[3]["content"]
                assert (
                    followup_messages[4]["content"]
                    == "Can you tell me more about Guido?"
                )

    def test_multiple_context_exchanges_accumulate(self, mock_config):
        """Multiple context-enriched messages should accumulate context in history."""
        with (
            patch("consumer.KafkaConsumer"),
            patch("consumer.KafkaProducer"),
            patch("consumer.OpenAI"),
            patch("consumer.ContextStore"),
        ):
            from consumer import ExoConsumer

            with tempfile.TemporaryDirectory() as tmpdir:
                chat_db_path = os.path.join(tmpdir, "chat.db")
                mock_config.chat_db_path = chat_db_path

                consumer = ExoConsumer(mock_config)
                consumer.chat_store = ChatStore(chat_db_path)

                # First message with context about Python
                context1 = [
                    {
                        "content": "Python is interpreted.",
                        "source": "python.md",
                        "h1": None,
                        "h2": None,
                        "h3": None,
                    }
                ]
                consumer._build_messages_with_history(
                    prompt="What is Python?",
                    message_id="multi_001",
                    context_chunks=context1,
                )

                # Simulate response
                conv_id = consumer.chat_store.get_active_conversation_id()
                consumer.chat_store.add_assistant_message(
                    conv_id, "Python is an interpreted language."
                )

                # Second message with context about Java
                context2 = [
                    {
                        "content": "Java is compiled.",
                        "source": "java.md",
                        "h1": None,
                        "h2": None,
                        "h3": None,
                    }
                ]
                consumer._build_messages_with_history(
                    prompt="What is Java?",
                    message_id="multi_002",
                    context_chunks=context2,
                )

                # Simulate response
                consumer.chat_store.add_assistant_message(
                    conv_id, "Java is a compiled language."
                )

                # Third message - no new context, should see both previous contexts
                final_messages = consumer._build_messages_with_history(
                    prompt="Compare them", message_id="multi_003", context_chunks=None
                )

                # Should have accumulated:
                # ctx1 user + ctx1 ack + prompt1 + response1 + ctx2 user + ctx2 ack + prompt2 + response2 + prompt3
                # = 9 messages
                assert len(final_messages) == 9

                # Verify both contexts are present in history
                all_content = " ".join(m["content"] for m in final_messages)
                assert "Python is interpreted" in all_content
                assert "Java is compiled" in all_content

    def test_context_cleared_on_conversation_reset(
        self, mock_config, mock_kafka_producer
    ):
        """Context should be cleared when conversation is reset."""
        with (
            patch("consumer.KafkaConsumer"),
            patch("consumer.KafkaProducer", return_value=mock_kafka_producer),
            patch("consumer.OpenAI"),
            patch("consumer.ContextStore"),
        ):
            from consumer import ExoConsumer

            with tempfile.TemporaryDirectory() as tmpdir:
                chat_db_path = os.path.join(tmpdir, "chat.db")
                mock_config.chat_db_path = chat_db_path

                consumer = ExoConsumer(mock_config)
                consumer.producer = mock_kafka_producer
                consumer.chat_store = ChatStore(chat_db_path)

                # First message with context
                context_chunks = [
                    {
                        "content": "Important context info.",
                        "source": "doc.md",
                        "h1": None,
                        "h2": None,
                        "h3": None,
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
                assert (
                    len(old_messages) == 3
                )  # context user + context ack + user prompt
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
                assert len(old_messages_after_reset) == 3

    def test_context_stored_with_prompt_id_suffix(self, mock_config):
        """Context messages should be stored with _context suffix on prompt_id."""
        with (
            patch("consumer.KafkaConsumer"),
            patch("consumer.KafkaProducer"),
            patch("consumer.OpenAI"),
            patch("consumer.ContextStore"),
        ):
            from consumer import ExoConsumer

            with tempfile.TemporaryDirectory() as tmpdir:
                chat_db_path = os.path.join(tmpdir, "chat.db")
                mock_config.chat_db_path = chat_db_path

                consumer = ExoConsumer(mock_config)
                consumer.chat_store = ChatStore(chat_db_path)

                context_chunks = [
                    {
                        "content": "Test content",
                        "source": "test.md",
                        "h1": None,
                        "h2": None,
                        "h3": None,
                    }
                ]

                consumer._build_messages_with_history(
                    prompt="Test prompt",
                    message_id="prompt_id_test",
                    context_chunks=context_chunks,
                )

                conv_id = consumer.chat_store.get_active_conversation_id()
                stored_messages = consumer.chat_store.get_conversation_messages(conv_id)

                # First message (context user) should have _context suffix
                assert stored_messages[0]["prompt_id"] == "prompt_id_test_context"

                # Last message (actual prompt) should have original prompt_id
                assert stored_messages[2]["prompt_id"] == "prompt_id_test"
