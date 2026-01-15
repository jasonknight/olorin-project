#!/usr/bin/env python3
"""
File tracking module using SQLite to track processed documents.
Tracks file paths, content hashes, and processing timestamps.
"""

import sqlite3
import hashlib
import os
from datetime import datetime
from typing import Optional, List, Dict
import logging
from clean_md import clean_markdown

logger = logging.getLogger(__name__)


class FileTracker:
    """Tracks processed files using SQLite database."""

    def __init__(self, db_path: str):
        """
        Initialize file tracker with SQLite database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path

        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Initialize database
        self._init_db()
        logger.info(f"File tracker initialized with database: {db_path}")

    def _init_db(self):
        """Create database schema if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS processed_files (
                    file_path TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    processed_at TIMESTAMP NOT NULL,
                    chunk_count INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'success',
                    retries INTEGER DEFAULT 0,
                    error_message TEXT
                )
            """)

            # Migration: Add retries column if it doesn't exist
            cursor = conn.execute("PRAGMA table_info(processed_files)")
            columns = [row[1] for row in cursor.fetchall()]
            if "retries" not in columns:
                conn.execute(
                    "ALTER TABLE processed_files ADD COLUMN retries INTEGER DEFAULT 0"
                )
                logger.info("Added 'retries' column to processed_files table")
            if "error_message" not in columns:
                conn.execute(
                    "ALTER TABLE processed_files ADD COLUMN error_message TEXT"
                )
                logger.info("Added 'error_message' column to processed_files table")

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_processed_at
                ON processed_files(processed_at)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status
                ON processed_files(status)
            """)

            conn.commit()
            logger.debug("Database schema initialized")

    @staticmethod
    def compute_file_hash(file_path: str) -> str:
        """
        Compute SHA256 hash of file contents.

        Args:
            file_path: Path to file

        Returns:
            Hex string of SHA256 hash
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read in chunks for memory efficiency
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def preprocess_file(
        self,
        file_path: str,
        model: str = "llama3.2",
        chunk_size: int = 4000,
        repetition_threshold: int = 3,
    ) -> bool:
        """
        Preprocess a file before tracking. For markdown files, this calls
        clean_md.py to clean and optimize the content.

        Args:
            file_path: Path to file to preprocess
            model: Ollama model to use for cleaning (default: llama3.2)
            chunk_size: Maximum chunk size for Ollama processing
            repetition_threshold: Threshold for detecting repetitive content

        Returns:
            True if file was preprocessed (modified), False if skipped
        """
        # Check if file is a markdown file
        if not file_path.lower().endswith(".md"):
            logger.debug(f"Skipping non-markdown file: {file_path}")
            return False

        # Check if file exists
        if not os.path.exists(file_path):
            logger.warning(f"File not found for preprocessing: {file_path}")
            return False

        try:
            logger.info(f"Preprocessing markdown file: {file_path}")

            # Calculate hash before cleaning
            hash_before = self.compute_file_hash(file_path)

            # Clean the markdown file (modifies file in place)
            cleaned_content = clean_markdown(
                file_path,
                model=model,
                chunk_size=chunk_size,
                repetition_threshold=repetition_threshold,
            )

            # Write cleaned content back to file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(cleaned_content)

            # Calculate hash after cleaning
            hash_after = self.compute_file_hash(file_path)

            if hash_before != hash_after:
                logger.info(f"File was modified during preprocessing: {file_path}")
                return True
            else:
                logger.info(f"File unchanged during preprocessing: {file_path}")
                return False

        except Exception as e:
            logger.error(f"Error preprocessing file {file_path}: {e}")
            return False

    def is_file_processed(self, file_path: str, max_retries: int = 5) -> bool:
        """
        Check if a file has been successfully processed.

        Files with status='error' that have been retried fewer than max_retries
        times are considered NOT processed (so they will be retried).

        Args:
            file_path: Path to file
            max_retries: Maximum number of retries for errored files (default: 5)

        Returns:
            True if file has been successfully processed or has exceeded max retries
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT status, retries FROM processed_files WHERE file_path = ?",
                (file_path,),
            )
            row = cursor.fetchone()
            if row is None:
                return False

            status, retries = row
            retries = retries or 0

            # If status is error and we haven't exceeded max retries, allow retry
            if status == "error" and retries < max_retries:
                return False

            return True

    def has_file_changed(self, file_path: str) -> bool:
        """
        Check if a file's content has changed since last processing.

        Args:
            file_path: Path to file

        Returns:
            True if file has changed or is new, False if unchanged
        """
        if not self.is_file_processed(file_path):
            return True

        current_hash = self.compute_file_hash(file_path)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT content_hash FROM processed_files WHERE file_path = ?",
                (file_path,),
            )
            row = cursor.fetchone()
            if row is None:
                return True

            stored_hash = row[0]
            return current_hash != stored_hash

    def mark_processed(
        self,
        file_path: str,
        chunk_count: int = 0,
        status: str = "success",
        error_message: str = None,
    ):
        """
        Mark a file as processed.

        Args:
            file_path: Path to file
            chunk_count: Number of chunks created from file
            status: Processing status ('success', 'error', 'partial')
            error_message: Error message if status is 'error'
        """
        content_hash = self.compute_file_hash(file_path)
        file_size = os.path.getsize(file_path)
        processed_at = datetime.now().isoformat()

        # Get current retry count if this is a retry
        retries = 0
        if status == "error":
            existing = self.get_file_info(file_path)
            if existing and existing.get("status") == "error":
                retries = (existing.get("retries") or 0) + 1

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO processed_files
                (file_path, content_hash, file_size, processed_at, chunk_count, status, retries, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    file_path,
                    content_hash,
                    file_size,
                    processed_at,
                    chunk_count,
                    status,
                    retries,
                    error_message,
                ),
            )
            conn.commit()

        if status == "error":
            logger.debug(
                f"Marked file as {status} (retry {retries}): {file_path} - {error_message}"
            )
        else:
            logger.debug(f"Marked file as {status}: {file_path} ({chunk_count} chunks)")

    def get_file_info(self, file_path: str) -> Optional[Dict]:
        """
        Get information about a processed file.

        Args:
            file_path: Path to file

        Returns:
            Dictionary with file info or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT file_path, content_hash, file_size,
                       processed_at, chunk_count, status, retries, error_message
                FROM processed_files
                WHERE file_path = ?
                """,
                (file_path,),
            )
            row = cursor.fetchone()

            if row:
                return dict(row)
            return None

    def get_all_processed_files(self) -> List[Dict]:
        """
        Get information about all processed files.

        Returns:
            List of dictionaries with file info
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT file_path, content_hash, file_size,
                       processed_at, chunk_count, status, retries, error_message
                FROM processed_files
                ORDER BY processed_at DESC
                """
            )
            return [dict(row) for row in cursor.fetchall()]

    def remove_file_record(self, file_path: str):
        """
        Remove a file's record from the database.
        Useful if file is deleted or needs reprocessing.

        Args:
            file_path: Path to file
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM processed_files WHERE file_path = ?", (file_path,)
            )
            conn.commit()

        logger.debug(f"Removed file record: {file_path}")

    def get_statistics(self) -> Dict:
        """
        Get statistics about processed files.

        Returns:
            Dictionary with statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT
                    COUNT(*) as total_files,
                    SUM(chunk_count) as total_chunks,
                    SUM(file_size) as total_size,
                    COUNT(CASE WHEN status = 'success' THEN 1 END) as successful,
                    COUNT(CASE WHEN status = 'error' THEN 1 END) as errors
                FROM processed_files
                """
            )
            row = cursor.fetchone()

            return {
                "total_files": row[0] or 0,
                "total_chunks": row[1] or 0,
                "total_size_bytes": row[2] or 0,
                "successful": row[3] or 0,
                "errors": row[4] or 0,
            }
