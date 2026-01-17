#!/usr/bin/env python3
"""
Abstract base class for document format trackers.

Provides common functionality for monitoring directories, tracking processed files,
converting documents to markdown, and integrating with Ollama and Kafka/Broca.
"""

import json
import os
import re
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Set, Tuple

from kafka import KafkaProducer
from kafka.errors import KafkaError

# Add parent directory to path for libs import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from libs.config import Config
from libs.olorin_logging import OlorinLogger

from file_tracker import FileTracker


class DocumentTrackerBase(ABC):
    """
    Abstract base class for document format trackers.

    Subclasses must override:
    - EXTENSIONS: Set of file extensions to track (e.g., {".pdf"})
    - TRACKER_NAME: Human-readable name (e.g., "PDF")
    - LOG_FILENAME: Log file name (e.g., "hippocampus-pdf-tracker.log")
    - TRACKING_DB_CONFIG_KEY: Config key for tracking DB (e.g., "PDF_TRACKING_DB")

    And implement abstract methods:
    - find_files(): Find all files with supported extensions
    - to_markdown(): Convert file to markdown content
    - process_file(): Process a single file
    - _extract_metadata(): Extract format-specific metadata
    - _get_title_prompt_examples(): Return examples for title extraction prompt
    """

    # Class attributes - MUST be overridden by subclasses
    EXTENSIONS: ClassVar[Set[str]] = set()
    TRACKER_NAME: ClassVar[str] = ""
    LOG_FILENAME: ClassVar[str] = ""
    TRACKING_DB_CONFIG_KEY: ClassVar[str] = ""

    def __init__(
        self,
        input_dir: str = None,
        output_dir: str = None,
        tracking_db: str = None,
        poll_interval: int = None,
        reprocess_on_change: bool = True,
        min_content_chars: int = None,
        min_content_density: float = None,
        min_word_count: int = None,
        use_ollama: bool = True,
        ollama_model: str = None,
        ollama_threshold: float = None,
        include_metadata: bool = True,
        force_reprocess_pattern: str = None,
        notify_broca: bool = None,
    ):
        """
        Initialize document tracker.

        Args:
            input_dir: Directory to monitor for documents (default from config)
            output_dir: Directory to save markdown files (defaults to input_dir)
            tracking_db: Path to SQLite tracking database (default from config)
            poll_interval: Seconds between directory scans (default from config)
            reprocess_on_change: Reprocess if document content changes
            min_content_chars: Minimum characters for content to be considered
            min_content_density: Minimum ratio of alphanumeric to total chars
            min_word_count: Minimum word count for valid content
            use_ollama: Enable local LLM for content filtering
            ollama_model: Ollama model to use
            ollama_threshold: Minimum relevance score (0-1)
            include_metadata: Include YAML frontmatter with document metadata
            force_reprocess_pattern: Regex pattern to match filenames for forced reprocessing
            notify_broca: Send TTS notification to Broca when processing starts
        """
        # Load configuration
        self.config = Config()

        # Get paths from config if not provided
        if input_dir is None:
            input_dir = self.config.get_path("INPUT_DIR", "~/Documents/AI_IN")
        if tracking_db is None:
            tracking_db = self.config.get_path(
                self.TRACKING_DB_CONFIG_KEY,
                f"./hippocampus/data/{self.TRACKER_NAME.lower()}_tracking.db",
            )

        self.input_dir = os.path.expanduser(input_dir)
        self.output_dir = (
            os.path.expanduser(output_dir) if output_dir else self.input_dir
        )

        # Get defaults from config or use fallbacks
        if poll_interval is None:
            poll_interval = self.config.get_int("POLL_INTERVAL", 5)
        if min_content_chars is None:
            min_content_chars = self.config.get_int("TRACKER_MIN_CONTENT_CHARS", 100)
        if min_content_density is None:
            min_content_density = self.config.get_float(
                "TRACKER_MIN_CONTENT_DENSITY", 0.3
            )
        if min_word_count is None:
            min_word_count = self.config.get_int("TRACKER_MIN_WORD_COUNT", 20)
        if ollama_model is None:
            ollama_model = self.config.get("TRACKER_OLLAMA_MODEL", "llama3.2:1b")
        if ollama_threshold is None:
            ollama_threshold = self.config.get_float("TRACKER_OLLAMA_THRESHOLD", 0.5)
        if notify_broca is None:
            notify_broca = self.config.get_bool("TRACKER_NOTIFY_BROCA", True)

        self.poll_interval = poll_interval
        self.reprocess_on_change = reprocess_on_change

        # Content configuration
        self.min_content_chars = min_content_chars
        self.min_content_density = min_content_density
        self.min_word_count = min_word_count
        self.use_ollama = use_ollama
        self.ollama_model = ollama_model
        self.ollama_threshold = ollama_threshold
        self.include_metadata = include_metadata

        # Force reprocess pattern
        self.force_reprocess_regex = None
        if force_reprocess_pattern:
            try:
                self.force_reprocess_regex = re.compile(force_reprocess_pattern)
            except re.error as e:
                raise ValueError(
                    f"Invalid regex pattern: {force_reprocess_pattern} - {e}"
                )

        # Broca notification settings
        self.notify_broca = notify_broca
        self.broca_producer = None
        self.broca_topic = self.config.get("BROCA_KAFKA_TOPIC", "ai_out")
        self.kafka_servers = self.config.get(
            "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"
        )

        # Setup logging
        self._init_logger()

        # Check Ollama availability
        self.ollama_path: Optional[str] = None
        self.ollama_available = False
        if self.use_ollama:
            self.ollama_available = self._check_ollama()
            if self.ollama_available:
                self.logger.info(f"Ollama detected - using model: {self.ollama_model}")
            else:
                self.logger.warning(
                    "Ollama not available - falling back to heuristics only"
                )

        # Initialize Broca Kafka producer
        if self.notify_broca:
            self._init_kafka()

        # Create directories if they don't exist
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize file tracker
        self.file_tracker = FileTracker(tracking_db)

        self.logger.info(f"{self.TRACKER_NAME} Tracker initialized")
        self.logger.info(f"Monitoring directory: {self.input_dir}")
        self.logger.info(f"Output directory: {self.output_dir}")
        if self.force_reprocess_regex:
            self.logger.info(f"Force reprocess pattern: {force_reprocess_pattern}")

    def _init_logger(self) -> None:
        """Initialize the logger."""
        default_log_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "logs"
        )
        log_file = os.path.join(default_log_dir, self.LOG_FILENAME)
        self.logger = OlorinLogger(log_file=log_file, log_level="INFO", name=__name__)

    def _init_kafka(self) -> None:
        """Initialize Kafka producer for Broca notifications."""
        try:
            self.broca_producer = KafkaProducer(
                bootstrap_servers=self.kafka_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                api_version_auto_timeout_ms=5000,
                retries=3,
            )
            self.logger.info(f"Broca notifications enabled (topic: {self.broca_topic})")
        except Exception as e:
            self.logger.warning(f"Could not connect to Kafka for Broca: {e}")
            self.broca_producer = None

    def _get_ollama_path(self) -> Optional[str]:
        """
        Find the ollama binary path.

        Returns:
            Path to ollama binary or None
        """
        # Try paths from config first, then defaults
        config_paths = self.config.get_list("TRACKER_OLLAMA_PATHS", None)
        default_paths = [
            "ollama",
            "/usr/local/bin/ollama",
            "/opt/homebrew/bin/ollama",
            "/Applications/Ollama.app/Contents/Resources/ollama",
        ]
        paths = config_paths if config_paths else default_paths

        for path in paths:
            try:
                result = subprocess.run(
                    [path, "--version"], capture_output=True, timeout=2
                )
                if result.returncode == 0:
                    return path
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue

        return None

    def _check_ollama(self) -> bool:
        """
        Check if Ollama is available on the system.

        Returns:
            True if Ollama is available and model exists
        """
        self.ollama_path = self._get_ollama_path()

        if not self.ollama_path:
            return False

        try:
            result = subprocess.run(
                [self.ollama_path, "list"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                if self.ollama_model in result.stdout:
                    return True
                # Try to pull the model
                self.logger.info(f"Pulling Ollama model: {self.ollama_model}")
                pull_result = subprocess.run(
                    [self.ollama_path, "pull", self.ollama_model],
                    capture_output=True,
                    timeout=300,
                )
                return pull_result.returncode == 0
            return False
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            return False

    def _send_broca_notification(self, message: str) -> bool:
        """
        Send a notification message to Broca for TTS.

        Args:
            message: Text to speak

        Returns:
            True if message was sent successfully
        """
        if not self.broca_producer:
            return False

        try:
            msg = {
                "text": message,
                "id": datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
            }
            future = self.broca_producer.send(self.broca_topic, value=msg)
            future.get(timeout=5)
            self.logger.debug(f"Sent Broca notification: {message}")
            return True

        except KafkaError as e:
            self.logger.warning(f"Failed to send Broca notification: {e}")
            return False
        except Exception as e:
            self.logger.warning(f"Unexpected error sending to Broca: {e}")
            return False

    def _calculate_content_density(self, text: str) -> float:
        """
        Calculate the ratio of meaningful content to total characters.

        Args:
            text: Text to analyze

        Returns:
            Ratio of alphanumeric characters to total (0-1)
        """
        if not text:
            return 0.0

        total_chars = len(text)
        alphanumeric_chars = sum(1 for c in text if c.isalnum())

        return alphanumeric_chars / total_chars if total_chars > 0 else 0.0

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text by removing artifacts and excessive whitespace.

        Args:
            text: Text to normalize

        Returns:
            Cleaned text
        """
        # Remove form feed and other control characters
        text = text.replace("\f", "\n")
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Fix hyphenated words across lines
        text = re.sub(r"-\s*\n\s*", "", text)

        # Normalize whitespace
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

        # Remove standalone page numbers
        text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

        return text.strip()

    def _is_content_substantial(self, text: str) -> bool:
        """
        Determine if content meets minimum quality thresholds.

        Args:
            text: Text to analyze

        Returns:
            True if content meets minimum thresholds
        """
        if not text or not text.strip():
            return False

        cleaned_text = text.strip()

        # Check minimum length
        if len(cleaned_text) < self.min_content_chars:
            return False

        # Check content density
        density = self._calculate_content_density(cleaned_text)
        if density < self.min_content_density:
            return False

        # Check for minimum number of words
        words = re.findall(r"\b\w+\b", cleaned_text)
        if len(words) < self.min_word_count:
            return False

        return True

    def _is_boilerplate_content_llm(self, text: str) -> Tuple[bool, str]:
        """
        Use local LLM to determine if content is boilerplate (TOC, copyright, etc.).

        Args:
            text: Text to evaluate

        Returns:
            Tuple of (is_boilerplate, content_type) where content_type is one of:
            'toc', 'copyright', 'substantive', 'unknown'
        """
        if not self.ollama_available:
            return False, "unknown"

        # Truncate very long text for faster inference
        text_sample = text[:800] if len(text) > 800 else text

        prompt = f"""Analyze this content from a document. Classify it into one of these categories:

TOC - Table of contents (chapter listings, section numbers with page numbers)
COPYRIGHT - Copyright notices, legal disclaimers, rights reserved statements
SUBSTANTIVE - Actual document content worth keeping
UNKNOWN - Cannot determine

Content:
{text_sample}

Respond with ONLY one word: TOC, COPYRIGHT, SUBSTANTIVE, or UNKNOWN"""

        try:
            result = subprocess.run(
                [self.ollama_path, "run", self.ollama_model, prompt],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                response = result.stdout.strip().upper()

                if "TOC" in response or "TABLE OF CONTENTS" in response:
                    return True, "toc"
                elif "COPYRIGHT" in response:
                    return True, "copyright"
                elif "SUBSTANTIVE" in response:
                    return False, "substantive"

            return False, "unknown"

        except (subprocess.TimeoutExpired, Exception) as e:
            self.logger.debug(f"LLM boilerplate check failed: {e}")
            return False, "unknown"

    def _guess_title_from_filename(self, filename: str) -> str:
        """
        Use Ollama to guess a clean title/author from the document filename.

        Args:
            filename: The document filename (without path or extension)

        Returns:
            A clean, speakable title string
        """
        if not self.ollama_available:
            return filename.replace("_", " ").replace("-", " ").strip()

        examples = self._get_title_prompt_examples()

        prompt = f"""Given this document filename, extract the likely title and author (if present).
Return ONLY a clean, speakable phrase suitable for text-to-speech.

Examples:
{examples}

Filename: {filename}

Clean title:"""

        try:
            result = subprocess.run(
                [self.ollama_path, "run", self.ollama_model, prompt],
                capture_output=True,
                text=True,
                timeout=15,
            )

            if result.returncode == 0:
                clean_title = result.stdout.strip().strip("\"'")
                if clean_title and len(clean_title) < 200:
                    return clean_title

        except (subprocess.TimeoutExpired, Exception) as e:
            self.logger.debug(f"Title extraction failed: {e}")

        return filename.replace("_", " ").replace("-", " ").strip()

    def should_process_file(self, file_path: str) -> bool:
        """
        Determine if a document should be processed.

        Args:
            file_path: Path to document

        Returns:
            True if file should be processed
        """
        if not os.path.exists(file_path):
            return False

        # Check if filename matches force reprocess pattern
        if self.force_reprocess_regex:
            filename = Path(file_path).name
            if self.force_reprocess_regex.search(filename):
                self.logger.info(f"Force reprocessing (matches pattern): {filename}")
                return True

        if not self.file_tracker.is_file_processed(file_path):
            return True

        if self.reprocess_on_change:
            return self.file_tracker.has_file_changed(file_path)

        return False

    def run_single_scan(self) -> Dict[str, int]:
        """
        Run a single scan of the input directory.

        Returns:
            Dictionary with scan statistics
        """
        stats = {
            "files_found": 0,
            "files_processed": 0,
            "files_skipped": 0,
            "files_failed": 0,
        }

        # Find all files with supported extensions
        files = self.find_files()
        stats["files_found"] = len(files)

        if not files:
            self.logger.debug(f"No {self.TRACKER_NAME} files found in {self.input_dir}")
            return stats

        # Process each file
        for file_path in files:
            if self.should_process_file(file_path):
                if self.process_file(file_path):
                    stats["files_processed"] += 1
                else:
                    stats["files_failed"] += 1
            else:
                stats["files_skipped"] += 1
                self.logger.debug(f"Skipping (already processed): {file_path}")

        return stats

    def run_continuous(self) -> None:
        """
        Run continuous monitoring loop.
        """
        self.logger.info(
            f"Starting continuous {self.TRACKER_NAME} monitoring "
            f"(poll interval: {self.poll_interval}s)"
        )
        self.logger.info("Press Ctrl+C to stop")

        scan_count = 0

        try:
            while True:
                scan_count += 1
                self.logger.debug(f"Scan #{scan_count}")

                stats = self.run_single_scan()

                if stats["files_processed"] > 0 or stats["files_failed"] > 0:
                    self.logger.info(
                        f"Scan #{scan_count} complete: "
                        f"{stats['files_processed']} processed, "
                        f"{stats['files_failed']} failed, "
                        f"{stats['files_skipped']} skipped"
                    )

                    # Show tracker statistics
                    tracker_stats = self.file_tracker.get_statistics()
                    self.logger.info(
                        f"Total: {tracker_stats['total_files']} "
                        f"{self.TRACKER_NAME} files processed"
                    )

                # Wait before next scan
                time.sleep(self.poll_interval)

        except KeyboardInterrupt:
            self.logger.info("\nShutting down gracefully...")
            self._shutdown()

    def _shutdown(self) -> None:
        """Cleanup and shutdown."""
        stats = self.file_tracker.get_statistics()
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Final Statistics:")
        self.logger.info(
            f"  Total {self.TRACKER_NAME} files processed: {stats['total_files']}"
        )
        self.logger.info(f"  Successful: {stats['successful']}")
        self.logger.info(f"  Errors: {stats['errors']}")
        self.logger.info("=" * 60)

        # Cleanup Kafka producer
        if self.broca_producer:
            try:
                self.broca_producer.flush()
                self.broca_producer.close()
            except Exception:
                pass

        self.logger.info(f"{self.TRACKER_NAME} Tracker stopped")

    # =========================================================================
    # Abstract methods - MUST be implemented by subclasses
    # =========================================================================

    @abstractmethod
    def find_files(self) -> List[str]:
        """
        Find all files with supported extensions in the input directory.

        Returns:
            List of absolute file paths
        """
        pass

    @abstractmethod
    def to_markdown(self, file_path: str) -> str:
        """
        Convert document to markdown text.

        Args:
            file_path: Path to document

        Returns:
            Markdown formatted text
        """
        pass

    @abstractmethod
    def process_file(self, file_path: str) -> bool:
        """
        Process a single document file.

        Args:
            file_path: Path to document

        Returns:
            True if processing succeeded
        """
        pass

    @abstractmethod
    def _extract_metadata(self, file_path: str) -> Dict[str, str]:
        """
        Extract format-specific metadata from document.

        Args:
            file_path: Path to document

        Returns:
            Dictionary containing metadata fields
        """
        pass

    @abstractmethod
    def _get_title_prompt_examples(self) -> str:
        """
        Return examples for the title extraction LLM prompt.

        Returns:
            String with example transformations
        """
        pass

    # =========================================================================
    # Helper methods for subclasses
    # =========================================================================

    def _find_files_by_extensions(self) -> List[str]:
        """
        Find all files matching the class EXTENSIONS in the input directory.

        This is a helper method that subclasses can use to implement find_files().

        Returns:
            List of absolute file paths
        """
        found_files = []
        input_path = Path(self.input_dir)

        for ext in self.EXTENSIONS:
            # Check lowercase extension
            for doc_file in input_path.rglob(f"*{ext}"):
                if doc_file.is_file():
                    found_files.append(str(doc_file.absolute()))

            # Check uppercase extension
            for doc_file in input_path.rglob(f"*{ext.upper()}"):
                if doc_file.is_file():
                    file_path = str(doc_file.absolute())
                    if file_path not in found_files:
                        found_files.append(file_path)

        return found_files

    def _build_yaml_frontmatter(self, metadata: Dict[str, str]) -> str:
        """
        Build YAML frontmatter from metadata dictionary.

        Args:
            metadata: Dictionary with metadata fields

        Returns:
            YAML frontmatter string
        """
        parts = ["---\n"]

        # Standard fields
        if "title" in metadata:
            parts.append(f'title: "{metadata["title"]}"\n')
        if "author" in metadata:
            parts.append(f'author: "{metadata["author"]}"\n')
        if "publish_date" in metadata:
            parts.append(f'publish_date: "{metadata["publish_date"]}"\n')
        if "language" in metadata:
            parts.append(f'language: "{metadata["language"]}"\n')
        if "source" in metadata:
            parts.append(f'source: "{metadata["source"]}"\n')
        if "file_date" in metadata:
            parts.append(f'file_date: "{metadata["file_date"]}"\n')
        if "word_count" in metadata:
            parts.append(f"word_count: {metadata['word_count']}\n")

        # Keywords
        if "keywords" in metadata:
            keywords = metadata["keywords"]
            if isinstance(keywords, list):
                if len(keywords) == 1:
                    parts.append(f'keywords: ["{keywords[0]}"]\n')
                else:
                    parts.append("keywords:\n")
                    for keyword in keywords:
                        parts.append(f'  - "{keyword}"\n')
            else:
                parts.append(f'keywords: ["{keywords}"]\n')

        parts.append("---\n\n")

        return "".join(parts)

    def _remove_boilerplate(self, text: str) -> str:
        """
        Remove common boilerplate patterns from text.

        Args:
            text: Text to clean

        Returns:
            Text with boilerplate removed
        """
        patterns = [
            r"(?i)^\s*page \d+ of \d+\s*$",
            r"(?i)^\s*confidential\s*$",
            r"(?i)^\s*copyright ©.*$",
            r"(?i)^\s*all rights reserved.*$",
            r"(?i)^\s*\[.*?\]\s*$",  # Standalone bracketed content
        ]

        for pattern in patterns:
            text = re.sub(pattern, "", text, flags=re.MULTILINE)

        return text
