#!/usr/bin/env python3
"""
Chat store module using SQLite to manage conversation history.
Stores user and assistant messages for multi-turn conversations.
"""

import json
import sqlite3
import hashlib
import os
import uuid
import threading
from datetime import datetime
from typing import Any, Optional, List, Dict, Tuple
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
                message_type TEXT DEFAULT 'message',
                metadata TEXT,
                updated_at TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
        """)

        # Migration: add new columns to existing databases
        self._migrate_messages_table(conn)

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

        # Index for message type queries
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_message_type
            ON messages(message_type)
        """)

        # Conversation contexts table - tracks which context chunks have been
        # injected into each conversation to prevent duplicate injection
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation_contexts (
                conversation_id TEXT NOT NULL,
                context_id TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (conversation_id, context_id),
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
        """)

        # Index for looking up contexts by conversation
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversation_contexts_conversation_id
            ON conversation_contexts(conversation_id)
        """)

        # Index for looking up by content hash (for deduplication by content)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversation_contexts_content_hash
            ON conversation_contexts(content_hash)
        """)

        conn.commit()
        logger.debug("Database schema initialized")

    def _migrate_messages_table(self, conn: sqlite3.Connection):
        """Add new columns to existing messages table if they don't exist."""
        cursor = conn.execute("PRAGMA table_info(messages)")
        existing_columns = {row[1] for row in cursor.fetchall()}

        migrations = [
            ("message_type", "TEXT DEFAULT 'message'"),
            ("metadata", "TEXT"),
            ("updated_at", "TIMESTAMP"),
        ]

        for column_name, column_def in migrations:
            if column_name not in existing_columns:
                try:
                    conn.execute(
                        f"ALTER TABLE messages ADD COLUMN {column_name} {column_def}"
                    )
                    logger.info(f"Migrated messages table: added {column_name} column")
                except sqlite3.OperationalError as e:
                    # Column might already exist from a concurrent migration
                    if "duplicate column name" not in str(e).lower():
                        raise

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

        Also clears the context tracking for the old conversation(s) to ensure
        fresh context injection in the new conversation.

        Returns:
            new conversation_id (UUID string)
        """
        conn = self._get_connection()

        # Get active conversation IDs before closing them (for context cleanup)
        cursor = conn.execute("SELECT id FROM conversations WHERE is_active = 1")
        active_ids = [row["id"] for row in cursor.fetchall()]

        # Clear context tracking for active conversations
        if active_ids:
            placeholders = ",".join("?" * len(active_ids))
            conn.execute(
                f"DELETE FROM conversation_contexts WHERE conversation_id IN ({placeholders})",
                active_ids,
            )
            logger.info(
                f"Cleared context tracking for {len(active_ids)} conversation(s)"
            )

        # Mark all active conversations as inactive
        conn.execute("UPDATE conversations SET is_active = 0 WHERE is_active = 1")
        conn.commit()

        logger.info("Closed all active conversations")

        # Create and return new conversation
        return self.get_or_create_active_conversation()

    # =========================================================================
    # Context Tracking (for deduplication)
    # =========================================================================

    def add_conversation_context(
        self,
        conversation_id: str,
        context_id: str,
        content_hash: str,
    ) -> bool:
        """
        Track that a context chunk has been injected into a conversation.

        This prevents the same context from being injected multiple times
        in the same conversation.

        Args:
            conversation_id: ID of the conversation
            context_id: Unique ID of the context chunk
            content_hash: SHA-256 hash of the context content

        Returns:
            True if tracked (new context), False if already tracked
        """
        conn = self._get_connection()
        now = datetime.now().isoformat()

        try:
            conn.execute(
                """
                INSERT INTO conversation_contexts (conversation_id, context_id, content_hash, added_at)
                VALUES (?, ?, ?, ?)
                """,
                (conversation_id, context_id, content_hash, now),
            )
            conn.commit()
            logger.debug(
                f"Tracked context {context_id[:8]}... in conversation {conversation_id[:8]}..."
            )
            return True
        except sqlite3.IntegrityError:
            # Already tracked (primary key constraint)
            logger.debug(
                f"Context {context_id[:8]}... already tracked in conversation {conversation_id[:8]}..."
            )
            return False

    def add_conversation_contexts_batch(
        self,
        conversation_id: str,
        contexts: list[tuple[str, str]],
    ) -> int:
        """
        Track multiple context chunks in a single transaction.

        Args:
            conversation_id: ID of the conversation
            contexts: List of (context_id, content_hash) tuples

        Returns:
            Number of new contexts tracked (excludes duplicates)
        """
        if not contexts:
            return 0

        conn = self._get_connection()
        now = datetime.now().isoformat()
        added_count = 0

        for context_id, content_hash in contexts:
            try:
                conn.execute(
                    """
                    INSERT INTO conversation_contexts (conversation_id, context_id, content_hash, added_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (conversation_id, context_id, content_hash, now),
                )
                added_count += 1
            except sqlite3.IntegrityError:
                # Already tracked, skip
                pass

        conn.commit()
        logger.debug(
            f"Tracked {added_count}/{len(contexts)} contexts in conversation {conversation_id[:8]}..."
        )
        return added_count

    def get_conversation_context_ids(self, conversation_id: str) -> set[str]:
        """
        Get all context IDs that have been injected into a conversation.

        Args:
            conversation_id: ID of the conversation

        Returns:
            Set of context IDs
        """
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT context_id FROM conversation_contexts WHERE conversation_id = ?",
            (conversation_id,),
        )
        return {row["context_id"] for row in cursor.fetchall()}

    def get_conversation_context_hashes(self, conversation_id: str) -> set[str]:
        """
        Get all content hashes that have been injected into a conversation.

        This allows deduplication by content even if context IDs differ.

        Args:
            conversation_id: ID of the conversation

        Returns:
            Set of content hashes
        """
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT content_hash FROM conversation_contexts WHERE conversation_id = ?",
            (conversation_id,),
        )
        return {row["content_hash"] for row in cursor.fetchall()}

    def clear_conversation_contexts(self, conversation_id: str) -> int:
        """
        Clear all tracked contexts for a conversation.

        Args:
            conversation_id: ID of the conversation

        Returns:
            Number of context records deleted
        """
        conn = self._get_connection()
        cursor = conn.execute(
            "DELETE FROM conversation_contexts WHERE conversation_id = ?",
            (conversation_id,),
        )
        conn.commit()
        deleted = cursor.rowcount
        if deleted > 0:
            logger.info(
                f"Cleared {deleted} tracked contexts from conversation {conversation_id[:8]}..."
            )
        return deleted

    # =========================================================================
    # Message Management
    # =========================================================================

    def add_user_message(
        self,
        conversation_id: str,
        content: str,
        prompt_id: Optional[str] = None,
        message_type: str = "message",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add a user message to the conversation.

        Args:
            conversation_id: ID of the conversation
            content: Message content
            prompt_id: Optional original Kafka message ID
            message_type: Type of message ('message', 'context_user', 'system')
            metadata: Optional JSON-serializable metadata dict

        Returns:
            message_id (UUID string)
        """
        return self._add_message(
            conversation_id, "user", content, prompt_id, message_type, metadata
        )

    def add_assistant_message(
        self,
        conversation_id: str,
        content: str,
        message_type: str = "message",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add an assistant message to the conversation.

        Args:
            conversation_id: ID of the conversation
            content: Message content
            message_type: Type of message ('message', 'context_ack', 'system')
            metadata: Optional JSON-serializable metadata dict

        Returns:
            message_id (UUID string)
        """
        return self._add_message(
            conversation_id, "assistant", content, None, message_type, metadata
        )

    def add_tool_call_message(
        self,
        conversation_id: str,
        tool_calls: List[Dict[str, Any]],
        prompt_id: Optional[str] = None,
    ) -> str:
        """
        Add an assistant message containing tool calls.

        This stores the assistant's decision to call tools. The content is empty
        because the assistant is requesting tool execution rather than responding.

        Args:
            conversation_id: ID of the conversation
            tool_calls: List of tool call objects from the API response, each with:
                - id: The tool call ID
                - function: {name: str, arguments: str (JSON)}
                - type: "function"
            prompt_id: Optional original Kafka message ID

        Returns:
            message_id (UUID string)
        """
        metadata = {"tool_calls": tool_calls}
        return self._add_message(
            conversation_id,
            "assistant",
            "",  # Content is empty for tool call messages
            prompt_id,
            "tool_call",
            metadata,
        )

    def add_tool_result_message(
        self,
        conversation_id: str,
        tool_call_id: str,
        tool_name: str,
        result: str,
    ) -> str:
        """
        Add a tool result message to the conversation.

        This stores the result of executing a tool. The role is 'tool' which
        indicates this is a tool response rather than a user or assistant message.

        Args:
            conversation_id: ID of the conversation
            tool_call_id: The ID of the tool call this is responding to
            tool_name: Name of the tool that was called
            result: The result content (typically JSON string or text)

        Returns:
            message_id (UUID string)
        """
        metadata = {"tool_call_id": tool_call_id, "tool_name": tool_name}
        return self._add_message(
            conversation_id,
            "tool",
            result,
            None,
            "tool_result",
            metadata,
        )

    def _add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        prompt_id: Optional[str] = None,
        message_type: str = "message",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Internal method to add a message to the conversation.

        Args:
            conversation_id: ID of the conversation
            role: Message role ('user' or 'assistant')
            content: Message content
            prompt_id: Optional original Kafka message ID
            message_type: Type of message ('message', 'context_user', 'context_ack', 'system')
            metadata: Optional JSON-serializable metadata dict

        Returns:
            message_id (UUID string)
        """
        conn = self._get_connection()
        message_id = str(uuid.uuid4())
        content_hash = self._compute_hash(content)
        now = datetime.now().isoformat()
        metadata_json = json.dumps(metadata) if metadata else None

        # Insert message
        conn.execute(
            """
            INSERT INTO messages (id, conversation_id, role, content, content_hash,
                                  prompt_id, created_at, message_type, metadata, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message_id,
                conversation_id,
                role,
                content,
                content_hash,
                prompt_id,
                now,
                message_type,
                metadata_json,
                now,
            ),
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
            f"Added {role} message {message_id} (type={message_type}) to conversation {conversation_id}"
        )
        return message_id

    def update_message(self, message_id: str, content: str) -> bool:
        """
        Update the content of an existing message (used for streaming updates).

        Only updates content and updated_at timestamp, preserving the original
        created_at timestamp for stable message ordering.

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
            SET content = ?, content_hash = ?, updated_at = ?
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
            List of dicts with 'role', 'content', 'created_at', 'message_type', 'metadata', etc.
        """
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT id, conversation_id, role, content, prompt_id, created_at,
                   message_type, metadata, updated_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY created_at ASC
            """,
            (conversation_id,),
        )
        rows = []
        for row in cursor.fetchall():
            row_dict = dict(row)
            # Parse metadata JSON if present
            if row_dict.get("metadata"):
                try:
                    row_dict["metadata"] = json.loads(row_dict["metadata"])
                except json.JSONDecodeError:
                    pass  # Keep as string if not valid JSON
            rows.append(row_dict)
        return rows

    def get_conversation_messages_for_api(self, conversation_id: str) -> List[Dict]:
        """
        Get messages formatted for OpenAI API calls.

        Converts stored messages to the format expected by OpenAI's chat completions API,
        including proper handling of tool calls and tool results.

        Args:
            conversation_id: ID of the conversation

        Returns:
            List of message dicts ready for API consumption
        """
        raw_messages = self.get_conversation_messages(conversation_id)
        api_messages = []

        for msg in raw_messages:
            role = msg.get("role")
            content = msg.get("content", "")
            metadata = msg.get("metadata", {})
            message_type = msg.get("message_type", "message")

            # Skip context injection messages (they're re-injected at query time)
            if message_type in ("context_user", "context_ack"):
                continue

            if role == "tool":
                # Tool result message
                api_msg = {
                    "role": "tool",
                    "content": content,
                    "tool_call_id": metadata.get("tool_call_id", ""),
                }
                api_messages.append(api_msg)
            elif role == "assistant" and message_type == "tool_call":
                # Assistant message with tool calls
                tool_calls = metadata.get("tool_calls", [])
                api_msg = {
                    "role": "assistant",
                    "content": content if content else None,
                    "tool_calls": tool_calls,
                }
                api_messages.append(api_msg)
            else:
                # Regular user or assistant message
                api_msg = {"role": role, "content": content}
                api_messages.append(api_msg)

        return api_messages

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

    def clear_all(self) -> Tuple[int, int, int]:
        """
        Clear all conversations, messages, and context tracking from the database.

        Returns:
            Tuple of (conversations_deleted, messages_deleted, contexts_deleted)
        """
        conn = self._get_connection()

        # Delete context tracking first (FK constraint)
        cursor = conn.execute("DELETE FROM conversation_contexts")
        contexts_deleted = cursor.rowcount

        # Delete messages (FK constraint)
        cursor = conn.execute("DELETE FROM messages")
        messages_deleted = cursor.rowcount

        # Then delete conversations
        cursor = conn.execute("DELETE FROM conversations")
        conversations_deleted = cursor.rowcount

        conn.commit()

        logger.info(
            f"Cleared {conversations_deleted} conversations, {messages_deleted} messages, "
            f"and {contexts_deleted} tracked contexts"
        )
        return conversations_deleted, messages_deleted, contexts_deleted

    def delete_old_conversations(self, older_than_days: int = 30) -> int:
        """
        Delete inactive conversations older than specified days.

        Also deletes associated messages and context tracking.

        Args:
            older_than_days: Delete conversations older than this many days

        Returns:
            Number of conversations deleted
        """
        from datetime import timedelta

        cutoff = (datetime.now() - timedelta(days=older_than_days)).isoformat()

        conn = self._get_connection()

        # Delete context tracking from old inactive conversations
        conn.execute(
            """
            DELETE FROM conversation_contexts WHERE conversation_id IN (
                SELECT id FROM conversations
                WHERE last_message_at < ? AND is_active = 0
            )
            """,
            (cutoff,),
        )

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
