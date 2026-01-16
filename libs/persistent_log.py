"""
Persistent Log Library for Olorin Project

Provides a PersistentLog class for logging all messages flowing through the system
to a SQLite database for debugging and tracking purposes.
"""

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


def _find_project_root() -> Path:
    """Find the project root by looking for settings.json file."""
    current = Path(__file__).resolve().parent

    # Go up from libs/ to project root
    if current.name == "libs":
        project_root = current.parent
        if (project_root / "settings.json").exists():
            return project_root

    # Fallback: search upward for settings.json
    search = current
    for _ in range(5):
        if (search / "settings.json").exists():
            return search
        if search.parent == search:
            break
        search = search.parent

    return Path.cwd()


# SQL schema for the persistent log table
_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    component TEXT NOT NULL,
    direction TEXT NOT NULL,
    message_id TEXT,
    conversation_id TEXT,
    topic TEXT,
    content TEXT,
    metadata TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);
CREATE INDEX IF NOT EXISTS idx_messages_component ON messages(component);
CREATE INDEX IF NOT EXISTS idx_messages_direction ON messages(direction);
CREATE INDEX IF NOT EXISTS idx_messages_message_id ON messages(message_id);
CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);
"""


class PersistentLog:
    """
    Persistent logging for all messages in the Olorin system.

    Stores messages to SQLite for debugging and tracking purposes.
    Thread-safe with connection pooling per thread.
    """

    _local = threading.local()

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize PersistentLog.

        Args:
            db_path: Path to SQLite database. Defaults to ./data/persistent_log.db
        """
        if db_path is None:
            project_root = _find_project_root()
            data_dir = project_root / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(data_dir / "persistent_log.db")

        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "connection") or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                self.db_path, check_same_thread=False
            )
            self._local.connection.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrent access
            self._local.connection.execute("PRAGMA journal_mode=WAL")
        return self._local.connection

    def _init_db(self):
        """Initialize database schema."""
        conn = self._get_connection()
        conn.executescript(_SCHEMA)
        conn.commit()

    def log(
        self,
        component: str,
        direction: str,
        content: Any,
        message_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        topic: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> int:
        """
        Log a message to the persistent log.

        Args:
            component: Component name (e.g., "enrichener", "cortex")
            direction: Direction of message (e.g., "received", "produced", "ai_response")
            content: Message content (will be JSON serialized if not a string)
            message_id: Optional message ID
            conversation_id: Optional conversation/session ID
            topic: Optional Kafka topic
            metadata: Optional additional metadata dict

        Returns:
            The ID of the inserted log entry
        """
        timestamp = datetime.now().isoformat()

        # Serialize content if not a string
        if not isinstance(content, str):
            try:
                content = json.dumps(content, default=str)
            except (TypeError, ValueError):
                content = str(content)

        # Serialize metadata
        metadata_json = None
        if metadata:
            try:
                metadata_json = json.dumps(metadata, default=str)
            except (TypeError, ValueError):
                metadata_json = str(metadata)

        conn = self._get_connection()
        cursor = conn.execute(
            """
            INSERT INTO messages (timestamp, component, direction, message_id,
                                  conversation_id, topic, content, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                component,
                direction,
                message_id,
                conversation_id,
                topic,
                content,
                metadata_json,
            ),
        )
        conn.commit()
        return cursor.lastrowid

    def get_recent(
        self,
        limit: int = 50,
        component: Optional[str] = None,
        direction: Optional[str] = None,
        message_id: Optional[str] = None,
    ) -> list[dict]:
        """
        Get recent log entries.

        Args:
            limit: Maximum number of entries to return
            component: Filter by component name
            direction: Filter by direction
            message_id: Filter by message ID

        Returns:
            List of log entries as dicts
        """
        conn = self._get_connection()

        query = "SELECT * FROM messages WHERE 1=1"
        params = []

        if component:
            query += " AND component = ?"
            params.append(component)
        if direction:
            query += " AND direction = ?"
            params.append(direction)
        if message_id:
            query += " AND message_id = ?"
            params.append(message_id)

        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        cursor = conn.execute(query, params)
        rows = cursor.fetchall()

        return [dict(row) for row in rows]

    def get_by_message_id(self, message_id: str) -> list[dict]:
        """
        Get all log entries for a specific message ID.

        Args:
            message_id: The message ID to search for

        Returns:
            List of log entries as dicts, ordered by timestamp
        """
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM messages WHERE message_id = ? ORDER BY id ASC",
            (message_id,),
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def get_by_conversation_id(self, conversation_id: str) -> list[dict]:
        """
        Get all log entries for a specific conversation ID.

        Args:
            conversation_id: The conversation ID to search for

        Returns:
            List of log entries as dicts, ordered by timestamp
        """
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id ASC",
            (conversation_id,),
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def clear(self) -> int:
        """
        Clear all log entries.

        Returns:
            Number of entries deleted
        """
        conn = self._get_connection()
        cursor = conn.execute("DELETE FROM messages")
        count = cursor.rowcount
        conn.commit()
        return count

    def count(self) -> int:
        """Get total number of log entries."""
        conn = self._get_connection()
        cursor = conn.execute("SELECT COUNT(*) FROM messages")
        return cursor.fetchone()[0]


# Singleton instance
_instance: Optional[PersistentLog] = None
_instance_lock = threading.Lock()


def get_persistent_log(db_path: Optional[str] = None) -> PersistentLog:
    """
    Get singleton PersistentLog instance.

    Args:
        db_path: Optional path to database (only used on first call)

    Returns:
        PersistentLog singleton instance
    """
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = PersistentLog(db_path)
    return _instance
