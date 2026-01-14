#!/usr/bin/env python3
"""
Unit tests for ChatStore class.

Tests conversation management, message storage, and history retrieval
without requiring any external services.
"""

import pytest
import os
import sys
import sqlite3
import threading
from datetime import datetime

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from libs.chat_store import ChatStore


class TestChatStoreInitialization:
    """Tests for ChatStore initialization and database setup."""

    def test_creates_database_file(self, temp_db_path):
        """ChatStore should create database file if it doesn't exist."""
        assert not os.path.exists(temp_db_path)
        store = ChatStore(temp_db_path)
        assert os.path.exists(temp_db_path)

    def test_creates_parent_directory(self):
        """ChatStore should create parent directories if they don't exist."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = os.path.join(tmpdir, 'a', 'b', 'c', 'chat.db')
            store = ChatStore(nested_path)
            assert os.path.exists(nested_path)

    def test_creates_required_tables(self, temp_db_path):
        """ChatStore should create conversations and messages tables."""
        store = ChatStore(temp_db_path)

        conn = sqlite3.connect(temp_db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()

        assert 'conversations' in tables
        assert 'messages' in tables

    def test_creates_indexes(self, temp_db_path):
        """ChatStore should create performance indexes."""
        store = ChatStore(temp_db_path)

        conn = sqlite3.connect(temp_db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        )
        indexes = [row[0] for row in cursor.fetchall()]
        conn.close()

        assert 'idx_messages_conversation_id' in indexes
        assert 'idx_conversations_active' in indexes


class TestConversationManagement:
    """Tests for conversation creation and management."""

    def test_get_or_create_creates_new_conversation(self, chat_store):
        """Should create a new conversation if none exists."""
        conv_id = chat_store.get_or_create_active_conversation()
        assert conv_id is not None
        assert len(conv_id) == 36  # UUID format

    def test_get_or_create_returns_existing_active(self, chat_store):
        """Should return existing active conversation, not create new."""
        conv_id_1 = chat_store.get_or_create_active_conversation()
        conv_id_2 = chat_store.get_or_create_active_conversation()
        assert conv_id_1 == conv_id_2

    def test_get_active_conversation_id_when_none(self, chat_store):
        """Should return None when no active conversation exists."""
        result = chat_store.get_active_conversation_id()
        assert result is None

    def test_get_active_conversation_id_when_exists(self, chat_store):
        """Should return conversation ID when one exists."""
        created_id = chat_store.get_or_create_active_conversation()
        retrieved_id = chat_store.get_active_conversation_id()
        assert created_id == retrieved_id

    def test_reset_conversation_creates_new(self, chat_store):
        """Reset should close current conversation and create new."""
        old_conv = chat_store.get_or_create_active_conversation()
        new_conv = chat_store.reset_conversation()

        assert new_conv != old_conv
        assert chat_store.get_active_conversation_id() == new_conv

    def test_reset_preserves_old_messages(self, chat_store):
        """Reset should preserve messages in old conversation."""
        old_conv = chat_store.get_or_create_active_conversation()
        chat_store.add_user_message(old_conv, "Hello")
        chat_store.add_assistant_message(old_conv, "Hi there!")

        new_conv = chat_store.reset_conversation()

        # Old conversation should still have messages
        old_messages = chat_store.get_conversation_messages(old_conv)
        assert len(old_messages) == 2

        # New conversation should be empty
        new_messages = chat_store.get_conversation_messages(new_conv)
        assert len(new_messages) == 0


class TestMessageManagement:
    """Tests for message storage and retrieval."""

    def test_add_user_message(self, chat_store):
        """Should add user message to conversation."""
        conv_id = chat_store.get_or_create_active_conversation()
        msg_id = chat_store.add_user_message(conv_id, "Hello, world!")

        assert msg_id is not None
        assert len(msg_id) == 36  # UUID format

    def test_add_assistant_message(self, chat_store):
        """Should add assistant message to conversation."""
        conv_id = chat_store.get_or_create_active_conversation()
        msg_id = chat_store.add_assistant_message(conv_id, "Hello! How can I help?")

        assert msg_id is not None

    def test_add_user_message_with_prompt_id(self, chat_store):
        """Should store prompt_id with user message."""
        conv_id = chat_store.get_or_create_active_conversation()
        msg_id = chat_store.add_user_message(conv_id, "Test", prompt_id="kafka_msg_123")

        messages = chat_store.get_conversation_messages(conv_id)
        assert messages[0]['prompt_id'] == "kafka_msg_123"

    def test_get_conversation_messages_empty(self, chat_store):
        """Should return empty list for conversation with no messages."""
        conv_id = chat_store.get_or_create_active_conversation()
        messages = chat_store.get_conversation_messages(conv_id)
        assert messages == []

    def test_get_conversation_messages_ordered(self, chat_store):
        """Messages should be returned in chronological order."""
        conv_id = chat_store.get_or_create_active_conversation()

        chat_store.add_user_message(conv_id, "First")
        chat_store.add_assistant_message(conv_id, "Second")
        chat_store.add_user_message(conv_id, "Third")

        messages = chat_store.get_conversation_messages(conv_id)

        assert len(messages) == 3
        assert messages[0]['content'] == "First"
        assert messages[0]['role'] == "user"
        assert messages[1]['content'] == "Second"
        assert messages[1]['role'] == "assistant"
        assert messages[2]['content'] == "Third"
        assert messages[2]['role'] == "user"

    def test_messages_have_required_fields(self, chat_store):
        """Each message should have all required fields."""
        conv_id = chat_store.get_or_create_active_conversation()
        chat_store.add_user_message(conv_id, "Test message")

        messages = chat_store.get_conversation_messages(conv_id)
        msg = messages[0]

        assert 'id' in msg
        assert 'conversation_id' in msg
        assert 'role' in msg
        assert 'content' in msg
        assert 'created_at' in msg

    def test_updates_conversation_message_count(self, chat_store):
        """Adding messages should update conversation message_count."""
        conv_id = chat_store.get_or_create_active_conversation()

        chat_store.add_user_message(conv_id, "One")
        chat_store.add_assistant_message(conv_id, "Two")

        stats = chat_store.get_statistics()
        assert stats['total_messages'] == 2


class TestStatistics:
    """Tests for statistics and utility methods."""

    def test_get_statistics_empty(self, chat_store):
        """Should return zero counts for empty database."""
        stats = chat_store.get_statistics()

        assert stats['total_conversations'] == 0
        assert stats['active_conversations'] == 0
        assert stats['total_messages'] == 0

    def test_get_statistics_with_data(self, chat_store):
        """Should return accurate counts with data."""
        conv1 = chat_store.get_or_create_active_conversation()
        chat_store.add_user_message(conv1, "Hello")
        chat_store.add_assistant_message(conv1, "Hi")

        conv2 = chat_store.reset_conversation()
        chat_store.add_user_message(conv2, "New conversation")

        stats = chat_store.get_statistics()

        assert stats['total_conversations'] == 2
        assert stats['active_conversations'] == 1
        assert stats['total_messages'] == 3
        assert stats['user_messages'] == 2
        assert stats['assistant_messages'] == 1

    def test_get_conversation_count(self, chat_store):
        """Should return correct conversation count."""
        assert chat_store.get_conversation_count() == 0

        chat_store.get_or_create_active_conversation()
        assert chat_store.get_conversation_count() == 1

        chat_store.reset_conversation()
        assert chat_store.get_conversation_count() == 2

    def test_get_message_count_total(self, chat_store):
        """Should return total message count."""
        conv_id = chat_store.get_or_create_active_conversation()

        assert chat_store.get_message_count() == 0

        chat_store.add_user_message(conv_id, "One")
        chat_store.add_assistant_message(conv_id, "Two")

        assert chat_store.get_message_count() == 2

    def test_get_message_count_per_conversation(self, chat_store):
        """Should return message count for specific conversation."""
        conv1 = chat_store.get_or_create_active_conversation()
        chat_store.add_user_message(conv1, "Conv1 msg")

        conv2 = chat_store.reset_conversation()
        chat_store.add_user_message(conv2, "Conv2 msg1")
        chat_store.add_assistant_message(conv2, "Conv2 msg2")

        assert chat_store.get_message_count(conv1) == 1
        assert chat_store.get_message_count(conv2) == 2

    def test_clear_all(self, chat_store):
        """Should remove all data from database."""
        conv_id = chat_store.get_or_create_active_conversation()
        chat_store.add_user_message(conv_id, "Test")
        chat_store.add_assistant_message(conv_id, "Response")

        convs_deleted, msgs_deleted = chat_store.clear_all()

        assert convs_deleted == 1
        assert msgs_deleted == 2
        assert chat_store.get_conversation_count() == 0
        assert chat_store.get_message_count() == 0


class TestThreadSafety:
    """Tests for thread-safe operation."""

    def test_concurrent_message_adds(self, temp_db_path):
        """Should handle concurrent message additions safely."""
        store = ChatStore(temp_db_path)
        conv_id = store.get_or_create_active_conversation()

        errors = []
        message_count = 50

        def add_messages(thread_num):
            try:
                for i in range(10):
                    store.add_user_message(conv_id, f"Thread {thread_num} msg {i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_messages, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert store.get_message_count(conv_id) == message_count

    def test_separate_connections_per_thread(self, temp_db_path):
        """Each thread should get its own database connection."""
        store = ChatStore(temp_db_path)
        connections = []

        def get_connection():
            conn = store._get_connection()
            connections.append(id(conn))

        threads = [threading.Thread(target=get_connection) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Each thread should have a unique connection
        assert len(set(connections)) == 3


class TestCleanup:
    """Tests for cleanup operations."""

    def test_delete_old_conversations(self, chat_store):
        """Should delete old inactive conversations."""
        # Create and close a conversation
        old_conv = chat_store.get_or_create_active_conversation()
        chat_store.add_user_message(old_conv, "Old message")
        chat_store.reset_conversation()

        # Should not delete recent conversations
        deleted = chat_store.delete_old_conversations(older_than_days=30)
        assert deleted == 0

        # Old conversation should still exist
        messages = chat_store.get_conversation_messages(old_conv)
        assert len(messages) == 1
