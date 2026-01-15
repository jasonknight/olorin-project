"""
Centralized State Management Library for Olorin Project

Provides a State class for persistent key-value storage with typed columns.
Uses SQLite for storage with separate columns for each data type, enabling
efficient access from statically-typed languages like Rust and C++.

Design:
- Each value is stored in a type-appropriate column (value_int, value_float, etc.)
- A value_type column indicates which column contains the data
- This avoids serialization overhead and enables type-safe reads
"""

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union


class ValueType(Enum):
    """Enumeration of supported value types for cross-language compatibility."""

    NULL = "null"
    INT = "int"
    FLOAT = "float"
    STRING = "string"
    BOOL = "bool"
    JSON = "json"
    BYTES = "bytes"


# SQL schema for the state table
_SCHEMA = """
CREATE TABLE IF NOT EXISTS state (
    key TEXT PRIMARY KEY NOT NULL,
    value_type TEXT NOT NULL DEFAULT 'null',
    value_int INTEGER,
    value_float REAL,
    value_string TEXT,
    value_bool INTEGER,
    value_json TEXT,
    value_bytes BLOB,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_state_type ON state(value_type);
CREATE INDEX IF NOT EXISTS idx_state_updated ON state(updated_at);
"""


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


class State:
    """
    Centralized state management with SQLite backend and typed columns.

    Provides type-safe getters and setters for sharing state across all
    Olorin components. Uses separate columns for each data type to enable
    efficient access from statically-typed languages.

    Usage:
        state = State()

        # Set values (type auto-detected)
        state.set("broca.audio_pid", 12345)
        state.set("broca.is_playing", True)
        state.set("system.info", {"version": "1.0", "components": ["broca", "cortex"]})

        # Get values with type safety
        pid = state.get_int("broca.audio_pid")  # -> Optional[int]
        playing = state.get_bool("broca.is_playing")  # -> Optional[bool]
        info = state.get_json("system.info")  # -> Optional[dict]

        # With defaults
        pid = state.get_int("broca.audio_pid", default=0)

        # Delete values
        state.delete("broca.audio_pid")

        # List keys by prefix
        broca_keys = state.keys(prefix="broca.")

    Thread Safety:
        This class is thread-safe. Each thread gets its own database connection
        via thread-local storage, and SQLite is configured with WAL mode.

    Schema:
        CREATE TABLE state (
            key TEXT PRIMARY KEY,
            value_type TEXT NOT NULL,  -- 'null', 'int', 'float', 'string', 'bool', 'json', 'bytes'
            value_int INTEGER,
            value_float REAL,
            value_string TEXT,
            value_bool INTEGER,        -- 0 or 1
            value_json TEXT,
            value_bytes BLOB,
            created_at TEXT,
            updated_at TEXT
        );
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        """
        Initialize the state manager.

        Args:
            db_path: Path to SQLite database file. Defaults to ./data/state.db
                     relative to project root.
        """
        self._project_root = _find_project_root()

        if db_path is None:
            self._db_path = self._project_root / "data" / "state.db"
        else:
            path = Path(db_path)
            if path.is_absolute():
                self._db_path = path
            else:
                self._db_path = self._project_root / path

        # Ensure parent directory exists
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # Thread-local storage for connections
        self._local = threading.local()
        self._lock = threading.Lock()

        # Initialize schema
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create a thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
                timeout=30.0,
            )
            # Enable WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=30000")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    @contextmanager
    def _cursor(self) -> Iterator[sqlite3.Cursor]:
        """Context manager for database operations."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()

    def _init_schema(self) -> None:
        """Initialize the database schema."""
        with self._cursor() as cursor:
            cursor.executescript(_SCHEMA)

    def set(self, key: str, value: Any) -> None:
        """
        Set a value with automatic type detection.

        Args:
            key: The state key (supports dot notation for namespacing)
            value: The value to store (type auto-detected)

        Supported types:
            - None -> null
            - int -> int
            - float -> float
            - str -> string
            - bool -> bool
            - dict/list -> json
            - bytes -> bytes
        """
        if value is None:
            self._set_typed(key, ValueType.NULL, None)
        elif isinstance(value, bool):  # Check bool before int (bool is subclass of int)
            self._set_typed(key, ValueType.BOOL, value)
        elif isinstance(value, int):
            self._set_typed(key, ValueType.INT, value)
        elif isinstance(value, float):
            self._set_typed(key, ValueType.FLOAT, value)
        elif isinstance(value, str):
            self._set_typed(key, ValueType.STRING, value)
        elif isinstance(value, bytes):
            self._set_typed(key, ValueType.BYTES, value)
        elif isinstance(value, (dict, list)):
            self._set_typed(key, ValueType.JSON, value)
        else:
            # Fallback: serialize as JSON
            self._set_typed(key, ValueType.JSON, value)

    def _set_typed(self, key: str, value_type: ValueType, value: Any) -> None:
        """Internal method to set a typed value."""
        now = datetime.utcnow().isoformat()

        # Prepare column values
        value_int = None
        value_float = None
        value_string = None
        value_bool = None
        value_json = None
        value_bytes = None

        if value_type == ValueType.INT:
            value_int = value
        elif value_type == ValueType.FLOAT:
            value_float = value
        elif value_type == ValueType.STRING:
            value_string = value
        elif value_type == ValueType.BOOL:
            value_bool = 1 if value else 0
        elif value_type == ValueType.JSON:
            value_json = json.dumps(value)
        elif value_type == ValueType.BYTES:
            value_bytes = value

        with self._cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO state (
                    key, value_type, value_int, value_float, value_string,
                    value_bool, value_json, value_bytes, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_type = excluded.value_type,
                    value_int = excluded.value_int,
                    value_float = excluded.value_float,
                    value_string = excluded.value_string,
                    value_bool = excluded.value_bool,
                    value_json = excluded.value_json,
                    value_bytes = excluded.value_bytes,
                    updated_at = excluded.updated_at
                """,
                (
                    key,
                    value_type.value,
                    value_int,
                    value_float,
                    value_string,
                    value_bool,
                    value_json,
                    value_bytes,
                    now,
                    now,
                ),
            )

    def set_int(self, key: str, value: int) -> None:
        """Explicitly set an integer value."""
        self._set_typed(key, ValueType.INT, value)

    def set_float(self, key: str, value: float) -> None:
        """Explicitly set a float value."""
        self._set_typed(key, ValueType.FLOAT, value)

    def set_string(self, key: str, value: str) -> None:
        """Explicitly set a string value."""
        self._set_typed(key, ValueType.STRING, value)

    def set_bool(self, key: str, value: bool) -> None:
        """Explicitly set a boolean value."""
        self._set_typed(key, ValueType.BOOL, value)

    def set_json(self, key: str, value: Union[dict, list]) -> None:
        """Explicitly set a JSON value (dict or list)."""
        self._set_typed(key, ValueType.JSON, value)

    def set_bytes(self, key: str, value: bytes) -> None:
        """Explicitly set a bytes value."""
        self._set_typed(key, ValueType.BYTES, value)

    def set_null(self, key: str) -> None:
        """Explicitly set a null value (different from delete)."""
        self._set_typed(key, ValueType.NULL, None)

    def _get_raw(self, key: str) -> Optional[sqlite3.Row]:
        """Get raw row data for a key."""
        with self._cursor() as cursor:
            cursor.execute("SELECT * FROM state WHERE key = ?", (key,))
            return cursor.fetchone()

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value with automatic type conversion.

        Returns the value in its native Python type based on value_type.

        Args:
            key: The state key
            default: Default value if key doesn't exist

        Returns:
            The value in its native type, or default if not found
        """
        row = self._get_raw(key)
        if row is None:
            return default

        value_type = row["value_type"]

        if value_type == "null":
            return None
        elif value_type == "int":
            return row["value_int"]
        elif value_type == "float":
            return row["value_float"]
        elif value_type == "string":
            return row["value_string"]
        elif value_type == "bool":
            return bool(row["value_bool"])
        elif value_type == "json":
            return json.loads(row["value_json"]) if row["value_json"] else None
        elif value_type == "bytes":
            return row["value_bytes"]
        else:
            return default

    def get_int(self, key: str, default: Optional[int] = None) -> Optional[int]:
        """
        Get an integer value.

        Args:
            key: The state key
            default: Default value if not found or wrong type

        Returns:
            The integer value, or default
        """
        row = self._get_raw(key)
        if row is None:
            return default

        if row["value_type"] == "int":
            return row["value_int"]
        elif row["value_type"] == "float":
            return (
                int(row["value_float"]) if row["value_float"] is not None else default
            )
        elif row["value_type"] == "string":
            try:
                return int(row["value_string"])
            except (ValueError, TypeError):
                return default
        return default

    def get_float(self, key: str, default: Optional[float] = None) -> Optional[float]:
        """
        Get a float value.

        Args:
            key: The state key
            default: Default value if not found or wrong type

        Returns:
            The float value, or default
        """
        row = self._get_raw(key)
        if row is None:
            return default

        if row["value_type"] == "float":
            return row["value_float"]
        elif row["value_type"] == "int":
            return float(row["value_int"]) if row["value_int"] is not None else default
        elif row["value_type"] == "string":
            try:
                return float(row["value_string"])
            except (ValueError, TypeError):
                return default
        return default

    def get_string(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get a string value.

        Args:
            key: The state key
            default: Default value if not found or wrong type

        Returns:
            The string value, or default
        """
        row = self._get_raw(key)
        if row is None:
            return default

        if row["value_type"] == "string":
            return row["value_string"]
        elif row["value_type"] in ("int", "float", "bool"):
            # Convert primitives to string
            val = self.get(key)
            return str(val) if val is not None else default
        return default

    def get_bool(self, key: str, default: bool = False) -> bool:
        """
        Get a boolean value.

        Args:
            key: The state key
            default: Default value if not found or wrong type

        Returns:
            The boolean value, or default
        """
        row = self._get_raw(key)
        if row is None:
            return default

        if row["value_type"] == "bool":
            return bool(row["value_bool"])
        elif row["value_type"] == "int":
            return bool(row["value_int"])
        elif row["value_type"] == "string":
            return row["value_string"].lower() in ("true", "yes", "1", "on")
        return default

    def get_json(
        self, key: str, default: Optional[Union[dict, list]] = None
    ) -> Optional[Union[dict, list]]:
        """
        Get a JSON value (dict or list).

        Args:
            key: The state key
            default: Default value if not found or wrong type

        Returns:
            The parsed JSON value, or default
        """
        row = self._get_raw(key)
        if row is None:
            return default

        if row["value_type"] == "json" and row["value_json"]:
            return json.loads(row["value_json"])
        return default

    def get_bytes(self, key: str, default: Optional[bytes] = None) -> Optional[bytes]:
        """
        Get a bytes value.

        Args:
            key: The state key
            default: Default value if not found or wrong type

        Returns:
            The bytes value, or default
        """
        row = self._get_raw(key)
        if row is None:
            return default

        if row["value_type"] == "bytes":
            return row["value_bytes"]
        return default

    def get_type(self, key: str) -> Optional[ValueType]:
        """
        Get the type of a stored value.

        Args:
            key: The state key

        Returns:
            The ValueType enum, or None if key doesn't exist
        """
        row = self._get_raw(key)
        if row is None:
            return None
        try:
            return ValueType(row["value_type"])
        except ValueError:
            return None

    def exists(self, key: str) -> bool:
        """Check if a key exists in the state."""
        with self._cursor() as cursor:
            cursor.execute("SELECT 1 FROM state WHERE key = ?", (key,))
            return cursor.fetchone() is not None

    def delete(self, key: str) -> bool:
        """
        Delete a key from the state.

        Args:
            key: The state key

        Returns:
            True if key was deleted, False if it didn't exist
        """
        with self._cursor() as cursor:
            cursor.execute("DELETE FROM state WHERE key = ?", (key,))
            return cursor.rowcount > 0

    def delete_prefix(self, prefix: str) -> int:
        """
        Delete all keys with a given prefix.

        Args:
            prefix: Key prefix to match (e.g., "broca." deletes all broca.* keys)

        Returns:
            Number of keys deleted
        """
        with self._cursor() as cursor:
            cursor.execute("DELETE FROM state WHERE key LIKE ?", (prefix + "%",))
            return cursor.rowcount

    def keys(self, prefix: Optional[str] = None) -> List[str]:
        """
        Get all keys, optionally filtered by prefix.

        Args:
            prefix: Optional prefix to filter keys (e.g., "broca.")

        Returns:
            List of matching keys
        """
        with self._cursor() as cursor:
            if prefix:
                cursor.execute(
                    "SELECT key FROM state WHERE key LIKE ? ORDER BY key",
                    (prefix + "%",),
                )
            else:
                cursor.execute("SELECT key FROM state ORDER BY key")
            return [row["key"] for row in cursor.fetchall()]

    def items(self, prefix: Optional[str] = None) -> List[Tuple[str, Any]]:
        """
        Get all key-value pairs, optionally filtered by prefix.

        Args:
            prefix: Optional prefix to filter keys

        Returns:
            List of (key, value) tuples
        """
        keys_list = self.keys(prefix)
        return [(k, self.get(k)) for k in keys_list]

    def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a key including type and timestamps.

        Args:
            key: The state key

        Returns:
            Dict with 'type', 'created_at', 'updated_at', or None if not found
        """
        row = self._get_raw(key)
        if row is None:
            return None

        return {
            "type": row["value_type"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def clear(self) -> int:
        """
        Delete all state entries.

        Returns:
            Number of entries deleted
        """
        with self._cursor() as cursor:
            cursor.execute("DELETE FROM state")
            return cursor.rowcount

    def close(self) -> None:
        """Close the database connection for the current thread."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    @property
    def db_path(self) -> Path:
        """Return the path to the state database."""
        return self._db_path

    @property
    def project_root(self) -> Path:
        """Return the project root directory."""
        return self._project_root

    def __enter__(self) -> "State":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - close connection."""
        self.close()


# Convenience singleton for simple usage
_default_state: Optional[State] = None
_singleton_lock = threading.Lock()


def get_state(db_path: Optional[Union[str, Path]] = None) -> State:
    """
    Get the default State singleton.

    Args:
        db_path: Path to database (only applies on first call)

    Returns:
        The default State instance
    """
    global _default_state
    if _default_state is None:
        with _singleton_lock:
            if _default_state is None:
                _default_state = State(db_path)
    return _default_state
