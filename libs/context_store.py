#!/usr/bin/env python3
"""
Context store module using SQLite to store retrieved context chunks.
Stores context chunks with metadata for later use by cortex.
"""

import sqlite3
import hashlib
import os
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class ContextStore:
    """Stores context chunks retrieved from ChromaDB in SQLite database."""

    def __init__(self, db_path: str):
        """
        Initialize context store with SQLite database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path

        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Initialize database
        self._init_db()
        logger.info(f"Context store initialized with database: {db_path}")

    def _init_db(self):
        """Create database schema if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            # Main context table - stores individual context chunks
            conn.execute("""
                CREATE TABLE IF NOT EXISTS contexts (
                    id TEXT PRIMARY KEY,
                    prompt_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    source TEXT,
                    h1 TEXT,
                    h2 TEXT,
                    h3 TEXT,
                    chunk_index INTEGER,
                    distance REAL,
                    added_at TIMESTAMP NOT NULL
                )
            """)

            # Index for looking up contexts by prompt_id
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_prompt_id
                ON contexts(prompt_id)
            """)

            # Index for timestamp-based queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_added_at
                ON contexts(added_at)
            """)

            # Index for source-based queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_source
                ON contexts(source)
            """)

            # Index for deduplication lookups (prompt_id + content_hash)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_prompt_content_hash
                ON contexts(prompt_id, content_hash)
            """)

            conn.commit()
            logger.debug("Database schema initialized")

    @staticmethod
    def _compute_hash(content: str) -> str:
        """Compute SHA-256 hash of content for deduplication."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _context_exists(
        self, conn: sqlite3.Connection, prompt_id: str, content_hash: str
    ) -> bool:
        """Check if a context with the same hash already exists for this prompt."""
        cursor = conn.execute(
            "SELECT 1 FROM contexts WHERE prompt_id = ? AND content_hash = ? LIMIT 1",
            (prompt_id, content_hash),
        )
        return cursor.fetchone() is not None

    def add_context(
        self,
        prompt_id: str,
        content: str,
        source: Optional[str] = None,
        h1: Optional[str] = None,
        h2: Optional[str] = None,
        h3: Optional[str] = None,
        chunk_index: Optional[int] = None,
        distance: Optional[float] = None,
    ) -> Optional[str]:
        """
        Add a context chunk to the database if it doesn't already exist.

        Args:
            prompt_id: ID of the prompt this context is associated with
            content: The context text content
            source: Source file/document name
            h1: First-level header from source
            h2: Second-level header from source
            h3: Third-level header from source
            chunk_index: Index of the chunk within the source
            distance: Semantic distance from query (lower = more relevant)

        Returns:
            The generated context ID, or None if duplicate was skipped
        """
        content_hash = self._compute_hash(content)
        added_at = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            # Check for duplicate
            if self._context_exists(conn, prompt_id, content_hash):
                logger.debug(f"Skipping duplicate context for prompt {prompt_id}")
                return None

            context_id = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO contexts
                (id, prompt_id, content, content_hash, source, h1, h2, h3, chunk_index, distance, added_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    context_id,
                    prompt_id,
                    content,
                    content_hash,
                    source,
                    h1,
                    h2,
                    h3,
                    chunk_index,
                    distance,
                    added_at,
                ),
            )
            conn.commit()

        logger.debug(f"Added context {context_id} for prompt {prompt_id}")
        return context_id

    def add_contexts_batch(
        self, prompt_id: str, chunks: List[Dict]
    ) -> Tuple[List[str], int]:
        """
        Add multiple context chunks in a single transaction, skipping duplicates.

        Args:
            prompt_id: ID of the prompt these contexts are associated with
            chunks: List of dicts with 'text', 'metadata', and optionally 'distance'

        Returns:
            Tuple of (list of generated context IDs, number of duplicates skipped)
        """
        context_ids = []
        skipped = 0
        added_at = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            for chunk in chunks:
                content = chunk.get("text", "")
                content_hash = self._compute_hash(content)

                # Check for duplicate
                if self._context_exists(conn, prompt_id, content_hash):
                    skipped += 1
                    continue

                context_id = str(uuid.uuid4())
                metadata = chunk.get("metadata", {})

                conn.execute(
                    """
                    INSERT INTO contexts
                    (id, prompt_id, content, content_hash, source, h1, h2, h3, chunk_index, distance, added_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        context_id,
                        prompt_id,
                        content,
                        content_hash,
                        metadata.get("source"),
                        metadata.get("h1"),
                        metadata.get("h2"),
                        metadata.get("h3"),
                        metadata.get("chunk_index"),
                        chunk.get("distance"),
                        added_at,
                    ),
                )
                context_ids.append(context_id)

            conn.commit()

        if skipped > 0:
            logger.info(
                f"Added {len(context_ids)} contexts for prompt {prompt_id} ({skipped} duplicates skipped)"
            )
        else:
            logger.info(f"Added {len(context_ids)} contexts for prompt {prompt_id}")
        return context_ids, skipped

    def get_contexts_for_prompt(self, prompt_id: str) -> List[Dict]:
        """
        Get all context chunks for a given prompt.

        Args:
            prompt_id: ID of the prompt

        Returns:
            List of context dictionaries ordered by distance (most relevant first)
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT id, prompt_id, content, source, h1, h2, h3,
                       chunk_index, distance, added_at
                FROM contexts
                WHERE prompt_id = ?
                ORDER BY distance ASC NULLS LAST
                """,
                (prompt_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_context_by_id(self, context_id: str) -> Optional[Dict]:
        """
        Get a specific context by ID.

        Args:
            context_id: ID of the context

        Returns:
            Context dictionary or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT id, prompt_id, content, source, h1, h2, h3,
                       chunk_index, distance, added_at
                FROM contexts
                WHERE id = ?
                """,
                (context_id,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def delete_contexts_for_prompt(self, prompt_id: str) -> int:
        """
        Delete all contexts for a given prompt.

        Args:
            prompt_id: ID of the prompt

        Returns:
            Number of contexts deleted
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM contexts WHERE prompt_id = ?", (prompt_id,)
            )
            conn.commit()
            deleted = cursor.rowcount

        logger.debug(f"Deleted {deleted} contexts for prompt {prompt_id}")
        return deleted

    def delete_old_contexts(self, older_than_hours: int = 24) -> int:
        """
        Delete contexts older than specified hours.

        Args:
            older_than_hours: Delete contexts older than this many hours

        Returns:
            Number of contexts deleted
        """
        cutoff = datetime.now().isoformat()
        # Calculate cutoff time
        from datetime import timedelta

        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
        cutoff = cutoff_time.isoformat()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM contexts WHERE added_at < ?", (cutoff,))
            conn.commit()
            deleted = cursor.rowcount

        logger.info(f"Deleted {deleted} contexts older than {older_than_hours} hours")
        return deleted

    def get_statistics(self) -> Dict:
        """
        Get statistics about stored contexts.

        Returns:
            Dictionary with statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT
                    COUNT(*) as total_contexts,
                    COUNT(DISTINCT prompt_id) as unique_prompts,
                    COUNT(DISTINCT source) as unique_sources,
                    MIN(added_at) as oldest,
                    MAX(added_at) as newest
                FROM contexts
                """
            )
            row = cursor.fetchone()

            return {
                "total_contexts": row[0] or 0,
                "unique_prompts": row[1] or 0,
                "unique_sources": row[2] or 0,
                "oldest": row[3],
                "newest": row[4],
            }

    def clear_all(self) -> int:
        """
        Clear all contexts from the database.

        Returns:
            Number of contexts deleted
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM contexts")
            conn.commit()
            deleted = cursor.rowcount

        logger.info(f"Cleared all {deleted} contexts from database")
        return deleted
