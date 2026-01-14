#!/usr/bin/env python3
"""
Unit tests for ExoConsumer class.

Tests chat history integration, message processing, and reset functionality
with mocked Kafka and OpenAI dependencies.
"""

import pytest
import os
import sys
import json
import tempfile
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from datetime import datetime

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from libs.chat_store import ChatStore
from libs.context_store import ContextStore


class TestResetCommandDetection:
    """Tests for reset command detection logic."""

    def test_exact_match_reset(self, mock_config):
        """Should detect exact '/reset' command."""
        # Import here to avoid issues with global state
        with patch('consumer.KafkaConsumer'), \
             patch('consumer.KafkaProducer'), \
             patch('consumer.OpenAI'), \
             patch('consumer.ContextStore'), \
             patch('consumer.ChatStore'):

            from consumer import ExoConsumer

            consumer = ExoConsumer(mock_config)
            consumer.config = mock_config

            assert consumer._is_reset_command('/reset') is True
            assert consumer._is_reset_command('reset conversation') is True
            assert consumer._is_reset_command('new conversation') is True

    def test_case_insensitive_reset(self, mock_config):
        """Reset detection should be case insensitive."""
        with patch('consumer.KafkaConsumer'), \
             patch('consumer.KafkaProducer'), \
             patch('consumer.OpenAI'), \
             patch('consumer.ContextStore'), \
             patch('consumer.ChatStore'):

            from consumer import ExoConsumer

            consumer = ExoConsumer(mock_config)
            consumer.config = mock_config

            assert consumer._is_reset_command('/RESET') is True
            assert consumer._is_reset_command('Reset Conversation') is True
            assert consumer._is_reset_command('NEW CONVERSATION') is True

    def test_prefix_match_reset(self, mock_config):
        """Should detect reset commands with trailing text."""
        with patch('consumer.KafkaConsumer'), \
             patch('consumer.KafkaProducer'), \
             patch('consumer.OpenAI'), \
             patch('consumer.ContextStore'), \
             patch('consumer.ChatStore'):

            from consumer import ExoConsumer

            consumer = ExoConsumer(mock_config)
            consumer.config = mock_config

            assert consumer._is_reset_command('/reset please') is True
            assert consumer._is_reset_command('/reset now') is True

    def test_non_reset_commands(self, mock_config):
        """Should not detect normal prompts as reset commands."""
        with patch('consumer.KafkaConsumer'), \
             patch('consumer.KafkaProducer'), \
             patch('consumer.OpenAI'), \
             patch('consumer.ContextStore'), \
             patch('consumer.ChatStore'):

            from consumer import ExoConsumer

            consumer = ExoConsumer(mock_config)
            consumer.config = mock_config

            assert consumer._is_reset_command('Hello, how are you?') is False
            assert consumer._is_reset_command('Tell me about resetting passwords') is False
            assert consumer._is_reset_command('What is a conversation?') is False


class TestContextFormatting:
    """Tests for RAG context formatting as user/assistant exchange."""

    def test_format_single_context_chunk(self, mock_config):
        """Should format single context chunk correctly."""
        with patch('consumer.KafkaConsumer'), \
             patch('consumer.KafkaProducer'), \
             patch('consumer.OpenAI'), \
             patch('consumer.ContextStore'), \
             patch('consumer.ChatStore'):

            from consumer import ExoConsumer

            consumer = ExoConsumer(mock_config)

            chunks = [{
                'content': 'Python is a programming language.',
                'source': 'docs.md',
                'h1': 'Introduction',
                'h2': None,
                'h3': None
            }]

            messages = consumer._format_context_as_exchange(chunks)

            assert len(messages) == 2
            assert messages[0]['role'] == 'user'
            assert messages[1]['role'] == 'assistant'
            assert 'Python is a programming language.' in messages[0]['content']
            assert 'docs.md' in messages[0]['content']
            assert 'Introduction' in messages[0]['content']
            assert "I understand" in messages[1]['content']

    def test_format_multiple_context_chunks(self, mock_config):
        """Should format multiple context chunks correctly."""
        with patch('consumer.KafkaConsumer'), \
             patch('consumer.KafkaProducer'), \
             patch('consumer.OpenAI'), \
             patch('consumer.ContextStore'), \
             patch('consumer.ChatStore'):

            from consumer import ExoConsumer

            consumer = ExoConsumer(mock_config)

            chunks = [
                {'content': 'First chunk', 'source': 'file1.md', 'h1': 'Header1', 'h2': None, 'h3': None},
                {'content': 'Second chunk', 'source': 'file2.md', 'h1': 'Header2', 'h2': 'Sub', 'h3': None}
            ]

            messages = consumer._format_context_as_exchange(chunks)

            assert len(messages) == 2
            assert 'First chunk' in messages[0]['content']
            assert 'Second chunk' in messages[0]['content']
            assert 'file1.md' in messages[0]['content']
            assert 'file2.md' in messages[0]['content']

    def test_format_context_with_missing_metadata(self, mock_config):
        """Should handle chunks with missing metadata gracefully."""
        with patch('consumer.KafkaConsumer'), \
             patch('consumer.KafkaProducer'), \
             patch('consumer.OpenAI'), \
             patch('consumer.ContextStore'), \
             patch('consumer.ChatStore'):

            from consumer import ExoConsumer

            consumer = ExoConsumer(mock_config)

            chunks = [{'content': 'Content only'}]  # No source, h1, etc.

            messages = consumer._format_context_as_exchange(chunks)

            assert len(messages) == 2
            assert 'Content only' in messages[0]['content']


class TestMessageHistoryBuilding:
    """Tests for building messages array with conversation history."""

    def test_build_messages_without_history(self, mock_config):
        """Should build single message when chat store is disabled."""
        with patch('consumer.KafkaConsumer'), \
             patch('consumer.KafkaProducer'), \
             patch('consumer.OpenAI'), \
             patch('consumer.ContextStore'), \
             patch('consumer.ChatStore'):

            from consumer import ExoConsumer

            consumer = ExoConsumer(mock_config)
            consumer.chat_store = None  # Simulate disabled chat history

            messages = consumer._build_messages_with_history(
                prompt="Hello",
                message_id="test_001",
                context_chunks=None
            )

            assert len(messages) == 1
            assert messages[0]['role'] == 'user'
            assert messages[0]['content'] == 'Hello'

    def test_build_messages_with_context_no_history(self, mock_config):
        """Should include context exchange even without chat store."""
        with patch('consumer.KafkaConsumer'), \
             patch('consumer.KafkaProducer'), \
             patch('consumer.OpenAI'), \
             patch('consumer.ContextStore'), \
             patch('consumer.ChatStore'):

            from consumer import ExoConsumer

            consumer = ExoConsumer(mock_config)
            consumer.chat_store = None

            context_chunks = [{'content': 'Context info', 'source': 'doc.md', 'h1': None, 'h2': None, 'h3': None}]

            messages = consumer._build_messages_with_history(
                prompt="Question?",
                message_id="test_002",
                context_chunks=context_chunks
            )

            # Should have: context user + context assistant + actual user = 3
            assert len(messages) == 3
            assert messages[0]['role'] == 'user'
            assert 'Context info' in messages[0]['content']
            assert messages[1]['role'] == 'assistant'
            assert messages[2]['role'] == 'user'
            assert messages[2]['content'] == 'Question?'

    def test_build_messages_with_history(self, mock_config):
        """Should include conversation history in messages array."""
        with patch('consumer.KafkaConsumer'), \
             patch('consumer.KafkaProducer'), \
             patch('consumer.OpenAI'), \
             patch('consumer.ContextStore'):

            from consumer import ExoConsumer

            with tempfile.TemporaryDirectory() as tmpdir:
                chat_db_path = os.path.join(tmpdir, 'chat.db')
                mock_config.chat_db_path = chat_db_path

                consumer = ExoConsumer(mock_config)

                # Manually set up chat store with history
                consumer.chat_store = ChatStore(chat_db_path)
                conv_id = consumer.chat_store.get_or_create_active_conversation()
                consumer.chat_store.add_user_message(conv_id, "Previous question")
                consumer.chat_store.add_assistant_message(conv_id, "Previous answer")

                messages = consumer._build_messages_with_history(
                    prompt="New question",
                    message_id="test_003",
                    context_chunks=None
                )

                # Should have: 2 history + 1 new = 3
                assert len(messages) == 3
                assert messages[0]['content'] == 'Previous question'
                assert messages[0]['role'] == 'user'
                assert messages[1]['content'] == 'Previous answer'
                assert messages[1]['role'] == 'assistant'
                assert messages[2]['content'] == 'New question'
                assert messages[2]['role'] == 'user'

    def test_build_messages_with_history_and_context(self, mock_config):
        """Should include both history and context in correct order."""
        with patch('consumer.KafkaConsumer'), \
             patch('consumer.KafkaProducer'), \
             patch('consumer.OpenAI'), \
             patch('consumer.ContextStore'):

            from consumer import ExoConsumer

            with tempfile.TemporaryDirectory() as tmpdir:
                chat_db_path = os.path.join(tmpdir, 'chat.db')
                mock_config.chat_db_path = chat_db_path

                consumer = ExoConsumer(mock_config)
                consumer.chat_store = ChatStore(chat_db_path)

                conv_id = consumer.chat_store.get_or_create_active_conversation()
                consumer.chat_store.add_user_message(conv_id, "First message")
                consumer.chat_store.add_assistant_message(conv_id, "First response")

                context_chunks = [{'content': 'Relevant context', 'source': 'ref.md', 'h1': None, 'h2': None, 'h3': None}]

                messages = consumer._build_messages_with_history(
                    prompt="Question with context",
                    message_id="test_004",
                    context_chunks=context_chunks
                )

                # Should have: 2 history + 2 context exchange + 1 new = 5
                assert len(messages) == 5

                # Verify order: history first, then context, then new prompt
                assert messages[0]['content'] == 'First message'
                assert messages[1]['content'] == 'First response'
                assert 'Relevant context' in messages[2]['content']  # Context user
                assert 'I understand' in messages[3]['content']       # Context assistant
                assert messages[4]['content'] == 'Question with context'


class TestResetCommandHandling:
    """Tests for conversation reset handling."""

    def test_handle_reset_sends_confirmation(self, mock_config, mock_kafka_producer):
        """Reset should send confirmation message to Kafka."""
        with patch('consumer.KafkaConsumer'), \
             patch('consumer.KafkaProducer', return_value=mock_kafka_producer), \
             patch('consumer.OpenAI'), \
             patch('consumer.ContextStore'):

            from consumer import ExoConsumer

            with tempfile.TemporaryDirectory() as tmpdir:
                chat_db_path = os.path.join(tmpdir, 'chat.db')
                mock_config.chat_db_path = chat_db_path

                consumer = ExoConsumer(mock_config)
                consumer.producer = mock_kafka_producer
                consumer.chat_store = ChatStore(chat_db_path)

                # Create an existing conversation
                consumer.chat_store.get_or_create_active_conversation()

                result = consumer._handle_reset_command("test_msg_reset")

                assert result is True
                assert len(mock_kafka_producer.sent_messages) == 1

                sent_msg = mock_kafka_producer.sent_messages[0]['value']
                assert 'reset' in sent_msg['text'].lower()
                assert sent_msg['is_reset_confirmation'] is True

    def test_handle_reset_creates_new_conversation(self, mock_config, mock_kafka_producer):
        """Reset should close old conversation and create new one."""
        with patch('consumer.KafkaConsumer'), \
             patch('consumer.KafkaProducer', return_value=mock_kafka_producer), \
             patch('consumer.OpenAI'), \
             patch('consumer.ContextStore'):

            from consumer import ExoConsumer

            with tempfile.TemporaryDirectory() as tmpdir:
                chat_db_path = os.path.join(tmpdir, 'chat.db')
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
        with patch('consumer.KafkaConsumer'), \
             patch('consumer.KafkaProducer'), \
             patch('consumer.OpenAI'), \
             patch('consumer.ContextStore'), \
             patch('consumer.ChatStore'):

            from consumer import ExoConsumer

            consumer = ExoConsumer(mock_config)
            consumer.chat_store = None  # Simulate disabled

            result = consumer._handle_reset_command("test_msg")
            assert result is False


class TestMessageParsing:
    """Tests for Kafka message parsing."""

    def test_parse_json_message(self, mock_config):
        """Should correctly parse JSON formatted messages."""
        with patch('consumer.KafkaConsumer'), \
             patch('consumer.KafkaProducer'), \
             patch('consumer.OpenAI'), \
             patch('consumer.ContextStore'), \
             patch('consumer.ChatStore'):

            from consumer import ExoConsumer

            consumer = ExoConsumer(mock_config)

            # Test the parsing logic extracted from process_message
            message = '{"prompt": "Hello", "id": "msg_123", "context_available": true, "contexts_stored": 2}'
            parsed = json.loads(message)

            assert parsed.get('prompt') == 'Hello'
            assert parsed.get('id') == 'msg_123'
            assert parsed.get('context_available') is True
            assert parsed.get('contexts_stored') == 2

    def test_parse_plain_text_message(self, mock_config):
        """Should handle plain text messages gracefully."""
        with patch('consumer.KafkaConsumer'), \
             patch('consumer.KafkaProducer'), \
             patch('consumer.OpenAI'), \
             patch('consumer.ContextStore'), \
             patch('consumer.ChatStore'):

            # Plain text that's not valid JSON
            message = "Just a plain text message"

            try:
                parsed = json.loads(message)
            except json.JSONDecodeError:
                # This is expected - plain text should be treated as the prompt itself
                prompt = message
                assert prompt == "Just a plain text message"


class TestIntegration:
    """Integration tests with real ChatStore but mocked external services."""

    def test_full_conversation_flow(self, mock_config, mock_kafka_producer):
        """Test a full conversation: multiple messages, then reset."""
        with patch('consumer.KafkaConsumer'), \
             patch('consumer.KafkaProducer', return_value=mock_kafka_producer), \
             patch('consumer.OpenAI'), \
             patch('consumer.ContextStore'):

            from consumer import ExoConsumer

            with tempfile.TemporaryDirectory() as tmpdir:
                chat_db_path = os.path.join(tmpdir, 'chat.db')
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
                consumer.chat_store.add_assistant_message(conv_id, "Your name is Alice.")

                # Build messages - should include all history
                messages = consumer._build_messages_with_history(
                    prompt="Tell me a joke",
                    message_id="msg_005"
                )

                # 4 history + 1 new = 5
                assert len(messages) == 5

                # Now reset
                consumer._handle_reset_command("reset_msg")

                # Build messages again - should be fresh
                messages = consumer._build_messages_with_history(
                    prompt="Hello again",
                    message_id="msg_006"
                )

                # Only the new message (history was reset)
                assert len(messages) == 1
                assert messages[0]['content'] == "Hello again"

    def test_context_injection_preserves_history(self, mock_config):
        """Context should be injected after history, before new prompt."""
        with patch('consumer.KafkaConsumer'), \
             patch('consumer.KafkaProducer'), \
             patch('consumer.OpenAI'), \
             patch('consumer.ContextStore'):

            from consumer import ExoConsumer

            with tempfile.TemporaryDirectory() as tmpdir:
                chat_db_path = os.path.join(tmpdir, 'chat.db')
                mock_config.chat_db_path = chat_db_path

                consumer = ExoConsumer(mock_config)
                consumer.chat_store = ChatStore(chat_db_path)

                # Build up conversation history
                conv_id = consumer.chat_store.get_or_create_active_conversation()
                consumer.chat_store.add_user_message(conv_id, "History message 1")
                consumer.chat_store.add_assistant_message(conv_id, "History response 1")

                # Now request with context
                context = [{'content': 'RAG context', 'source': 'doc.md', 'h1': None, 'h2': None, 'h3': None}]

                messages = consumer._build_messages_with_history(
                    prompt="Question about docs",
                    message_id="msg_007",
                    context_chunks=context
                )

                # Verify order: history (2) + context exchange (2) + new prompt (1) = 5
                assert len(messages) == 5
                assert messages[0]['content'] == "History message 1"
                assert messages[1]['content'] == "History response 1"
                assert 'RAG context' in messages[2]['content']
                assert 'I understand' in messages[3]['content']
                assert messages[4]['content'] == "Question about docs"
