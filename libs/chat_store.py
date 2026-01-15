#!/usr/bin/env python3
"""
Chat store module using SQLite to manage conversation history.
Stores user and assistant messages for multi-turn conversations.
"""

import sqlite3
import hashlib
import os
import uuid
import threading
from datetime import datetime
from typing import Optional, List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class ChatStore:
    """Manages chat conversation history in SQLite database."""

    def __init__(self, db_path: str):
        """
        Initialize chat store with SQLite database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._local = threading.local()  # Thread-local storage for connections

        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Initialize database schema
        self._init_db()
        logger.info(f"Chat store initialized with database: {db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "connection") or self._local.connection is None:
            self._local.connection = sqlite3.connect(self.db_path)
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection

    def _init_db(self):
        """Create database schema if it doesn't exist."""
        conn = self._get_connection()

        # Conversations table - tracks conversation sessions
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                created_at TIMESTAMP NOT NULL,
                last_message_at TIMESTAMP NOT NULL,
                message_count INTEGER DEFAULT 0,
                is_active INTEGER DEFAULT 1
            )
        """)

        # Messages table - stores individual messages
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                prompt_id TEXT,
                created_at TIMESTAMP NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
        """)

        # Index for looking up messages by conversation_id
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_conversation_id
            ON messages(conversation_id)
        """)

        # Index for timestamp-based queries
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_created_at
            ON messages(created_at)
        """)

        # Index for active conversation lookup
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_active
            ON conversations(is_active)
        """)

        # Index for last message timestamp
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_last_message
            ON conversations(last_message_at)
        """)

        conn.commit()
        logger.debug("Database schema initialized")

    @staticmethod
    def _compute_hash(content: str) -> str:
        """Compute SHA-256 hash of content."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    # =========================================================================
    # Conversation Management
    # =========================================================================

    def get_or_create_active_conversation(self) -> str:
        """
        Get the current active conversation ID, or create one if none exists.

        Returns:
            conversation_id (UUID string)
        """
        conn = self._get_connection()

        # Check for existing active conversation
        cursor = conn.execute(
            "SELECT id FROM conversations WHERE is_active = 1 ORDER BY last_message_at DESC LIMIT 1"
        )
        row = cursor.fetchone()

        if row:
            logger.debug(f"Found active conversation: {row['id']}")
            return row["id"]

        # Create new conversation
        conversation_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        conn.execute(
            """
            INSERT INTO conversations (id, created_at, last_message_at, message_count, is_active)
            VALUES (?, ?, ?, 0, 1)
            """,
            (conversation_id, now, now),
        )
        conn.commit()

        logger.info(f"Created new conversation: {conversation_id}")
        return conversation_id

    def get_active_conversation_id(self) -> Optional[str]:
        """
        Get the ID of the current active conversation.

        Returns:
            conversation_id or None if no active conversation
        """
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT id FROM conversations WHERE is_active = 1 ORDER BY last_message_at DESC LIMIT 1"
        )
        row = cursor.fetchone()
        return row["id"] if row else None

    def reset_conversation(self) -> str:
        """
        Close the current active conversation and create a new one.

        Returns:
            new conversation_id (UUID string)
        """
        conn = self._get_connection()

        # Mark all active conversations as inactive
        conn.execute("UPDATE conversations SET is_active = 0 WHERE is_active = 1")
        conn.commit()

        logger.info("Closed all active conversations")

        # Create and return new conversation
        return self.get_or_create_active_conversation()

    # =========================================================================
    # Message Management
    # =========================================================================

    def add_user_message(
        self, conversation_id: str, content: str, prompt_id: Optional[str] = None
    ) -> str:
        """
        Add a user message to the conversation.

        Args:
            conversation_id: ID of the conversation
            content: Message content
            prompt_id: Optional original Kafka message ID

        Returns:
            message_id (UUID string)
        """
        return self._add_message(conversation_id, "user", content, prompt_id)

    def add_assistant_message(self, conversation_id: str, content: str) -> str:
        """
        Add an assistant message to the conversation.

        Args:
            conversation_id: ID of the conversation
            content: Message content

        Returns:
            message_id (UUID string)
        """
        return self._add_message(conversation_id, "assistant", content)

    def _add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        prompt_id: Optional[str] = None,
    ) -> str:
        """
        Internal method to add a message to the conversation.

        Args:
            conversation_id: ID of the conversation
            role: Message role ('user' or 'assistant')
            content: Message content
            prompt_id: Optional original Kafka message ID

        Returns:
            message_id (UUID string)
        """
        conn = self._get_connection()
        message_id = str(uuid.uuid4())
        content_hash = self._compute_hash(content)
        now = datetime.now().isoformat()

        # Insert message
        conn.execute(
            """
            INSERT INTO messages (id, conversation_id, role, content, content_hash, prompt_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (message_id, conversation_id, role, content, content_hash, prompt_id, now),
        )

        # Update conversation metadata
        conn.execute(
            """
            UPDATE conversations
            SET last_message_at = ?, message_count = message_count + 1
            WHERE id = ?
            """,
            (now, conversation_id),
        )

        conn.commit()

        logger.debug(
            f"Added {role} message {message_id} to conversation {conversation_id}"
        )
        return message_id

    def update_message(self, message_id: str, content: str) -> bool:
        """
        Update the content of an existing message (used for streaming updates).

        Args:
            message_id: ID of the message to update
            content: New message content

        Returns:
            True if message was updated, False if not found
        """
        conn = self._get_connection()
        content_hash = self._compute_hash(content)
        now = datetime.now().isoformat()

        cursor = conn.execute(
            """
            UPDATE messages
            SET content = ?, content_hash = ?, created_at = ?
            WHERE id = ?
            """,
            (content, content_hash, now, message_id),
        )
        conn.commit()

        if cursor.rowcount > 0:
            logger.debug(f"Updated message {message_id}")
            return True
        return False

    def get_conversation_messages(self, conversation_id: str) -> List[Dict]:
        """
        Get all messages for a conversation in chronological order.

        Args:
            conversation_id: ID of the conversation

        Returns:
            List of dicts with 'role', 'content', 'created_at', etc.
        """
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT id, conversation_id, role, content, prompt_id, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY created_at ASC
            """,
            (conversation_id,),
        )
        return [dict(row) for row in cursor.fetchall()]

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_statistics(self) -> Dict:
        """
        Get statistics about stored conversations and messages.

        Returns:
            Dictionary with statistics
        """
        conn = self._get_connection()

        # Get conversation stats
        cursor = conn.execute(
            """
            SELECT
                COUNT(*) as total_conversations,
                SUM(CASE WHEN is_active = 1 THEN 1 ELSE 0 END) as active_conversations,
                MIN(created_at) as oldest_conversation,
                MAX(last_message_at) as newest_activity
            FROM conversations
            """
        )
        conv_row = cursor.fetchone()

        # Get message stats
        cursor = conn.execute(
            """
            SELECT
                COUNT(*) as total_messages,
                SUM(CASE WHEN role = 'user' THEN 1 ELSE 0 END) as user_messages,
                SUM(CASE WHEN role = 'assistant' THEN 1 ELSE 0 END) as assistant_messages
            FROM messages
            """
        )
        msg_row = cursor.fetchone()

        return {
            "total_conversations": conv_row["total_conversations"] or 0,
            "active_conversations": conv_row["active_conversations"] or 0,
            "oldest_conversation": conv_row["oldest_conversation"],
            "newest_activity": conv_row["newest_activity"],
            "total_messages": msg_row["total_messages"] or 0,
            "user_messages": msg_row["user_messages"] or 0,
            "assistant_messages": msg_row["assistant_messages"] or 0,
        }

    def get_conversation_count(self) -> int:
        """Get total number of conversations."""
        conn = self._get_connection()
        cursor = conn.execute("SELECT COUNT(*) as count FROM conversations")
        return cursor.fetchone()["count"]

    def get_message_count(self, conversation_id: Optional[str] = None) -> int:
        """
        Get message count (total or for specific conversation).

        Args:
            conversation_id: Optional conversation ID to filter by

        Returns:
            Number of messages
        """
        conn = self._get_connection()
        if conversation_id:
            cursor = conn.execute(
                "SELECT COUNT(*) as count FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            )
        else:
            cursor = conn.execute("SELECT COUNT(*) as count FROM messages")
        return cursor.fetchone()["count"]

    def clear_all(self) -> Tuple[int, int]:
        """
        Clear all conversations and messages from the database.

        Returns:
            Tuple of (conversations_deleted, messages_deleted)
        """
        conn = self._get_connection()

        # Delete messages first (FK constraint)
        cursor = conn.execute("DELETE FROM messages")
        messages_deleted = cursor.rowcount

        # Then delete conversations
        cursor = conn.execute("DELETE FROM conversations")
        conversations_deleted = cursor.rowcount

        conn.commit()

        logger.info(
            f"Cleared {conversations_deleted} conversations and {messages_deleted} messages"
        )
        return conversations_deleted, messages_deleted

    def delete_old_conversations(self, older_than_days: int = 30) -> int:
        """
        Delete inactive conversations older than specified days.

        Args:
            older_than_days: Delete conversations older than this many days

        Returns:
            Number of conversations deleted
        """
        from datetime import timedelta

        cutoff = (datetime.now() - timedelta(days=older_than_days)).isoformat()

        conn = self._get_connection()

        # Delete messages from old inactive conversations
        conn.execute(
            """
            DELETE FROM messages WHERE conversation_id IN (
                SELECT id FROM conversations
                WHERE last_message_at < ? AND is_active = 0
            )
            """,
            (cutoff,),
        )

        # Delete old inactive conversations
        cursor = conn.execute(
            "DELETE FROM conversations WHERE last_message_at < ? AND is_active = 0",
            (cutoff,),
        )
        deleted = cursor.rowcount

        conn.commit()

        logger.info(
            f"Deleted {deleted} inactive conversations older than {older_than_days} days"
        )
        return deleted
