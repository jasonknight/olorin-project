#!/usr/bin/env python3
"""
Ebook monitoring and conversion script.
Monitors ~/Documents/AI_IN for EPUB and MOBI files, converts them to markdown,
and tracks processed files to avoid reprocessing.
"""

import os
import sys
import time
import re
import subprocess
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from html.parser import HTMLParser

# Add parent directory to path for libs import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from libs.olorin_logging import OlorinLogger
from libs.config import Config

# Try to import ebooklib for EPUB support
try:
    import ebooklib
    from ebooklib import epub

    EPUB_AVAILABLE = True
except ImportError:
    print("ebooklib not found. Installing...")
    os.system(f"{sys.executable} -m pip install ebooklib")
    try:
        import ebooklib
        from ebooklib import epub

        EPUB_AVAILABLE = True
    except ImportError:
        EPUB_AVAILABLE = False
        print("Warning: ebooklib could not be installed. EPUB support disabled.")

# Try to import mobi for MOBI support
try:
    import mobi

    MOBI_AVAILABLE = True
except ImportError:
    print("mobi not found. Installing...")
    os.system(f"{sys.executable} -m pip install mobi")
    try:
        import mobi

        MOBI_AVAILABLE = True
    except ImportError:
        MOBI_AVAILABLE = False
        print("Warning: mobi could not be installed. MOBI support disabled.")

from file_tracker import FileTracker


class HTMLStripper(HTMLParser):
    """Simple HTML to text converter."""

    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = []
        self.in_style = False
        self.in_script = False

    def handle_starttag(self, tag, attrs):
        if tag == "style":
            self.in_style = True
        elif tag == "script":
            self.in_script = True
        elif tag in ("p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6", "li"):
            self.text.append("\n")

    def handle_endtag(self, tag):
        if tag == "style":
            self.in_style = False
        elif tag == "script":
            self.in_script = False
        elif tag in ("p", "div", "h1", "h2", "h3", "h4", "h5", "h6"):
            self.text.append("\n")

    def handle_data(self, data):
        if not self.in_style and not self.in_script:
            self.text.append(data)

    def get_text(self):
        return "".join(self.text)


def strip_html(html_content: str) -> str:
    """Strip HTML tags and return plain text."""
    stripper = HTMLStripper()
    try:
        stripper.feed(html_content)
        return stripper.get_text()
    except Exception:
        # Fallback: simple regex-based stripping
        text = re.sub(
            r"<style[^>]*>.*?</style>",
            "",
            html_content,
            flags=re.DOTALL | re.IGNORECASE,
        )
        text = re.sub(
            r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE
        )
        text = re.sub(r"<[^>]+>", " ", text)
        return text


class EbookTracker:
    """
    Monitors directory for EPUB and MOBI files and converts them to markdown.
    """

    def __init__(
        self,
        input_dir: str = None,
        output_dir: str = None,
        tracking_db: str = None,
        poll_interval: int = 5,
        reprocess_on_change: bool = True,
        min_chapter_chars: int = 100,
        min_content_density: float = 0.3,
        use_ollama: bool = True,
        ollama_model: str = "llama3.2:1b",
        ollama_threshold: float = 0.5,
        include_metadata: bool = True,
        force_reprocess_pattern: str = None,
    ):
        """
        Initialize Ebook tracker.

        Args:
            input_dir: Directory to monitor for ebooks (default from config)
            output_dir: Directory to save markdown files (defaults to input_dir)
            tracking_db: Path to SQLite tracking database (default from config)
            poll_interval: Seconds between directory scans
            reprocess_on_change: Reprocess if ebook content changes
            min_chapter_chars: Minimum characters for chapter to be considered
            min_content_density: Minimum ratio of alphanumeric to total chars
            use_ollama: Enable local LLM for relevance detection
            ollama_model: Ollama model to use (e.g., llama3.2:1b, phi3:mini)
            ollama_threshold: Minimum relevance score (0-1) to include chapter
            include_metadata: Include YAML frontmatter with document metadata
            force_reprocess_pattern: Regex pattern to match filenames for forced reprocessing
        """
        # Get paths from config if not provided
        config = Config()
        if input_dir is None:
            input_dir = config.get_path("INPUT_DIR", "~/Documents/AI_IN")
        if tracking_db is None:
            tracking_db = config.get_path(
                "EBOOK_TRACKING_DB", "./hippocampus/data/ebook_tracking.db"
            )

        self.input_dir = os.path.expanduser(input_dir)
        self.output_dir = (
            os.path.expanduser(output_dir) if output_dir else self.input_dir
        )
        self.poll_interval = poll_interval
        self.reprocess_on_change = reprocess_on_change

        # Cleaning configuration
        self.min_chapter_chars = min_chapter_chars
        self.min_content_density = min_content_density
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

        # Setup logging
        default_log_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "logs"
        )
        log_file = os.path.join(default_log_dir, "hippocampus-ebook-tracker.log")
        self.logger = OlorinLogger(log_file=log_file, log_level="INFO", name=__name__)

        # Check Ollama availability
        self.ollama_available = False
        if self.use_ollama:
            self.ollama_available = self._check_ollama()
            if self.ollama_available:
                self.logger.info(f"Ollama detected - using model: {self.ollama_model}")
            else:
                self.logger.warning(
                    "Ollama not available - falling back to heuristics only"
                )

        # Create directories if they don't exist
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize file tracker
        self.file_tracker = FileTracker(tracking_db)

        # Log availability
        self.logger.info("Ebook Tracker initialized")
        self.logger.info(f"Monitoring directory: {self.input_dir}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"EPUB support: {'enabled' if EPUB_AVAILABLE else 'disabled'}")
        self.logger.info(f"MOBI support: {'enabled' if MOBI_AVAILABLE else 'disabled'}")
        if self.force_reprocess_regex:
            self.logger.info(f"Force reprocess pattern: {force_reprocess_pattern}")

    def find_ebook_files(self) -> List[str]:
        """
        Find all EPUB and MOBI files in the input directory.

        Returns:
            List of absolute file paths
        """
        ebook_files = []
        input_path = Path(self.input_dir)

        # Find EPUB files
        if EPUB_AVAILABLE:
            for ebook_file in input_path.rglob("*.epub"):
                if ebook_file.is_file():
                    ebook_files.append(str(ebook_file.absolute()))
            for ebook_file in input_path.rglob("*.EPUB"):
                if ebook_file.is_file():
                    ebook_files.append(str(ebook_file.absolute()))

        # Find MOBI files
        if MOBI_AVAILABLE:
            for ebook_file in input_path.rglob("*.mobi"):
                if ebook_file.is_file():
                    ebook_files.append(str(ebook_file.absolute()))
            for ebook_file in input_path.rglob("*.MOBI"):
                if ebook_file.is_file():
                    ebook_files.append(str(ebook_file.absolute()))

        return ebook_files

    def _get_ollama_path(self) -> Optional[str]:
        """
        Find the ollama binary path.

        Returns:
            Path to ollama binary or None
        """
        paths = [
            "ollama",
            "/usr/local/bin/ollama",
            "/opt/homebrew/bin/ollama",
            "/Applications/Ollama.app/Contents/Resources/ollama",
        ]

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

    def _is_chapter_substantial(self, text: str) -> bool:
        """
        Determine if a chapter has substantial content using heuristics.

        Args:
            text: Chapter text to analyze

        Returns:
            True if chapter meets minimum quality thresholds
        """
        if not text or not text.strip():
            return False

        cleaned_text = text.strip()
        if len(cleaned_text) < self.min_chapter_chars:
            return False

        density = self._calculate_content_density(cleaned_text)
        if density < self.min_content_density:
            return False

        whitespace_ratio = (
            (len(text) - len(text.replace(" ", "").replace("\n", ""))) / len(text)
            if len(text) > 0
            else 0
        )
        if whitespace_ratio > 0.8:
            return False

        words = re.findall(r"\b\w+\b", cleaned_text)
        if len(words) < 20:
            return False

        return True

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text by removing artifacts and excessive whitespace.

        Args:
            text: Text to normalize

        Returns:
            Cleaned text
        """
        text = text.replace("\f", "\n")
        text = re.sub(r"-\s*\n\s*", "", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
        text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

        return text.strip()

    def _remove_boilerplate(self, text: str) -> str:
        """
        Remove common boilerplate patterns.

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
        ]

        for pattern in patterns:
            text = re.sub(pattern, "", text, flags=re.MULTILINE)

        return text

    def _is_boilerplate_content_llm(self, text: str) -> Tuple[bool, str]:
        """
        Use local LLM to determine if content is boilerplate.

        Args:
            text: Text to evaluate

        Returns:
            Tuple of (is_boilerplate, content_type)
        """
        if not self.ollama_available:
            return False, "unknown"

        text_sample = text[:800] if len(text) > 800 else text

        prompt = f"""Analyze this content from an ebook. Classify it into one of these categories:

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

    def _check_chapter_relevance_llm(self, text: str) -> Tuple[bool, float]:
        """
        Use local LLM to determine if chapter content is relevant.

        Args:
            text: Chapter text to evaluate

        Returns:
            Tuple of (is_relevant, confidence_score)
        """
        if not self.ollama_available:
            return True, 1.0

        text_sample = text[:1000] if len(text) > 1000 else text

        prompt = f"""Analyze this chapter from an ebook. Determine if it contains substantive information worth indexing.

Chapter content:
{text_sample}

Does this chapter contain meaningful content (NOT just headers, page numbers, or boilerplate)?
Answer with ONLY 'YES' or 'NO' followed by a confidence score 0-1.
Format: YES 0.9 or NO 0.3"""

        try:
            result = subprocess.run(
                [self.ollama_path, "run", self.ollama_model, prompt],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                response = result.stdout.strip().upper()

                match = re.search(r"(YES|NO)\s+(0?\.\d+|1\.0|[01])", response)
                if match:
                    is_relevant = match.group(1) == "YES"
                    confidence = float(match.group(2))
                    return is_relevant, confidence

            return True, 0.5

        except (subprocess.TimeoutExpired, Exception) as e:
            self.logger.debug(f"LLM check failed: {e}")
            return True, 0.5

    def _extract_epub_metadata(
        self, book: epub.EpubBook, ebook_path: str
    ) -> Dict[str, str]:
        """
        Extract metadata from EPUB document.

        Args:
            book: ebooklib EpubBook object
            ebook_path: Path to EPUB file

        Returns:
            Dictionary containing metadata fields
        """
        metadata = {}

        # Extract title
        title = book.get_metadata("DC", "title")
        if title and len(title) > 0:
            metadata["title"] = title[0][0]
        else:
            metadata["title"] = Path(ebook_path).stem

        # Extract author
        creator = book.get_metadata("DC", "creator")
        if creator and len(creator) > 0:
            metadata["author"] = creator[0][0]
        else:
            metadata["author"] = "unknown"

        # Extract date
        date = book.get_metadata("DC", "date")
        if date and len(date) > 0:
            date_str = date[0][0]
            # Try to parse and format date
            try:
                if len(date_str) >= 10:
                    metadata["publish_date"] = date_str[:10]
                else:
                    metadata["publish_date"] = date_str
            except Exception:
                metadata["publish_date"] = "unknown"
        else:
            metadata["publish_date"] = "unknown"

        # Extract subject/keywords
        subjects = book.get_metadata("DC", "subject")
        if subjects and len(subjects) > 0:
            metadata["keywords"] = [s[0] for s in subjects]
        else:
            metadata["keywords"] = ["unknown"]

        # Extract language
        language = book.get_metadata("DC", "language")
        if language and len(language) > 0:
            metadata["language"] = language[0][0]
        else:
            metadata["language"] = "unknown"

        return metadata

    def _extract_mobi_metadata(self, ebook_path: str) -> Dict[str, str]:
        """
        Extract metadata from MOBI document.

        Args:
            ebook_path: Path to MOBI file

        Returns:
            Dictionary containing metadata fields
        """
        metadata = {
            "title": Path(ebook_path).stem,
            "author": "unknown",
            "publish_date": "unknown",
            "keywords": ["unknown"],
            "language": "unknown",
        }

        try:
            tempdir, filepath = mobi.extract(ebook_path)

            # Try to read OPF file for metadata
            opf_files = list(Path(tempdir).rglob("*.opf"))
            if opf_files:
                with open(opf_files[0], "r", encoding="utf-8", errors="ignore") as f:
                    opf_content = f.read()

                    # Extract title
                    title_match = re.search(
                        r"<dc:title[^>]*>([^<]+)</dc:title>", opf_content, re.IGNORECASE
                    )
                    if title_match:
                        metadata["title"] = title_match.group(1).strip()

                    # Extract author
                    author_match = re.search(
                        r"<dc:creator[^>]*>([^<]+)</dc:creator>",
                        opf_content,
                        re.IGNORECASE,
                    )
                    if author_match:
                        metadata["author"] = author_match.group(1).strip()

                    # Extract date
                    date_match = re.search(
                        r"<dc:date[^>]*>([^<]+)</dc:date>", opf_content, re.IGNORECASE
                    )
                    if date_match:
                        metadata["publish_date"] = date_match.group(1).strip()[:10]

                    # Extract subject
                    subjects = re.findall(
                        r"<dc:subject[^>]*>([^<]+)</dc:subject>",
                        opf_content,
                        re.IGNORECASE,
                    )
                    if subjects:
                        metadata["keywords"] = [s.strip() for s in subjects]

                    # Extract language
                    lang_match = re.search(
                        r"<dc:language[^>]*>([^<]+)</dc:language>",
                        opf_content,
                        re.IGNORECASE,
                    )
                    if lang_match:
                        metadata["language"] = lang_match.group(1).strip()

            # Clean up temp directory
            import shutil

            shutil.rmtree(tempdir, ignore_errors=True)

        except Exception as e:
            self.logger.debug(f"Error extracting MOBI metadata: {e}")

        return metadata

    def should_process_file(self, file_path: str) -> bool:
        """
        Determine if an ebook should be processed.

        Args:
            file_path: Path to ebook file

        Returns:
            True if file should be processed
        """
        if not os.path.exists(file_path):
            return False

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

    def epub_to_markdown(self, epub_path: str) -> str:
        """
        Convert EPUB to markdown text with intelligent cleaning.

        Args:
            epub_path: Path to EPUB file

        Returns:
            Cleaned markdown formatted text
        """
        markdown_parts = []

        book = epub.read_epub(epub_path)

        # Extract metadata
        if self.include_metadata:
            metadata = self._extract_epub_metadata(book, epub_path)

            markdown_parts.append("---\n")
            markdown_parts.append(f'title: "{metadata["title"]}"\n')
            markdown_parts.append(f'author: "{metadata["author"]}"\n')
            markdown_parts.append(f'publish_date: "{metadata["publish_date"]}"\n')
            markdown_parts.append(f'language: "{metadata["language"]}"\n')

            if len(metadata["keywords"]) == 1:
                markdown_parts.append(f'keywords: ["{metadata["keywords"][0]}"]\n')
            else:
                markdown_parts.append("keywords:\n")
                for keyword in metadata["keywords"]:
                    markdown_parts.append(f'  - "{keyword}"\n')

            markdown_parts.append("---\n\n")
            markdown_parts.append(f"# {metadata['title']}\n\n")
        else:
            filename = Path(epub_path).stem
            markdown_parts.append(f"# {filename}\n\n")
            markdown_parts.append("---\n\n")

        # Extract chapters
        chapters_text = []
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                try:
                    html_content = item.get_content().decode("utf-8", errors="ignore")
                    text = strip_html(html_content)
                    if text.strip():
                        chapters_text.append(text)
                except Exception as e:
                    self.logger.debug(f"Error extracting chapter: {e}")

        # Filter and clean chapters
        substantial_chapters = []
        skipped_empty = 0
        skipped_heuristic = 0
        skipped_llm = 0
        skipped_boilerplate = 0

        for chapter_num, text in enumerate(chapters_text):
            if not text.strip():
                skipped_empty += 1
                continue

            text = self._normalize_text(text)
            text = self._remove_boilerplate(text)
            text = self._normalize_text(text)

            # Check for boilerplate using LLM
            if self.ollama_available:
                is_boilerplate, content_type = self._is_boilerplate_content_llm(text)
                if is_boilerplate:
                    skipped_boilerplate += 1
                    self.logger.debug(
                        f"Chapter {chapter_num + 1}: Skipped (detected as {content_type})"
                    )
                    continue

            if not self._is_chapter_substantial(text):
                skipped_heuristic += 1
                self.logger.debug(f"Chapter {chapter_num + 1}: Skipped (heuristic)")
                continue

            if self.ollama_available:
                is_relevant, confidence = self._check_chapter_relevance_llm(text)
                if not is_relevant or confidence < self.ollama_threshold:
                    skipped_llm += 1
                    self.logger.debug(
                        f"Chapter {chapter_num + 1}: Skipped by LLM "
                        f"(relevance={is_relevant}, confidence={confidence:.2f})"
                    )
                    continue

            substantial_chapters.append(text)

        # Log filtering results
        total_chapters = len(chapters_text)
        kept_chapters = len(substantial_chapters)
        self.logger.info(
            f"Filtered {epub_path}: {kept_chapters}/{total_chapters} chapters kept "
            f"(empty={skipped_empty}, boilerplate={skipped_boilerplate}, "
            f"heuristic={skipped_heuristic}, llm={skipped_llm})"
        )

        if substantial_chapters:
            content = "\n\n".join(substantial_chapters)
            content = self._normalize_text(content)
            markdown_parts.append(content)
        else:
            markdown_parts.append("_No substantial content found in document._\n")

        return "".join(markdown_parts)

    def mobi_to_markdown(self, mobi_path: str) -> str:
        """
        Convert MOBI to markdown text with intelligent cleaning.

        Args:
            mobi_path: Path to MOBI file

        Returns:
            Cleaned markdown formatted text
        """
        import shutil

        markdown_parts = []

        # Extract MOBI to temp directory
        tempdir, filepath = mobi.extract(mobi_path)

        try:
            # Extract metadata
            if self.include_metadata:
                metadata = self._extract_mobi_metadata(mobi_path)

                markdown_parts.append("---\n")
                markdown_parts.append(f'title: "{metadata["title"]}"\n')
                markdown_parts.append(f'author: "{metadata["author"]}"\n')
                markdown_parts.append(f'publish_date: "{metadata["publish_date"]}"\n')
                markdown_parts.append(f'language: "{metadata["language"]}"\n')

                if len(metadata["keywords"]) == 1:
                    markdown_parts.append(f'keywords: ["{metadata["keywords"][0]}"]\n')
                else:
                    markdown_parts.append("keywords:\n")
                    for keyword in metadata["keywords"]:
                        markdown_parts.append(f'  - "{keyword}"\n')

                markdown_parts.append("---\n\n")
                markdown_parts.append(f"# {metadata['title']}\n\n")
            else:
                filename = Path(mobi_path).stem
                markdown_parts.append(f"# {filename}\n\n")
                markdown_parts.append("---\n\n")

            # Find and read HTML/text content from extracted files
            chapters_text = []

            # Look for HTML files
            html_files = list(Path(tempdir).rglob("*.html")) + list(
                Path(tempdir).rglob("*.htm")
            )
            html_files.sort()  # Sort to maintain order

            for html_file in html_files:
                try:
                    with open(html_file, "r", encoding="utf-8", errors="ignore") as f:
                        html_content = f.read()
                    text = strip_html(html_content)
                    if text.strip():
                        chapters_text.append(text)
                except Exception as e:
                    self.logger.debug(f"Error reading HTML file {html_file}: {e}")

            # If no HTML files, try the main extracted file
            if not chapters_text and filepath:
                try:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    # Check if it's HTML
                    if "<html" in content.lower() or "<body" in content.lower():
                        text = strip_html(content)
                    else:
                        text = content
                    if text.strip():
                        chapters_text.append(text)
                except Exception as e:
                    self.logger.debug(f"Error reading main file {filepath}: {e}")

            # Filter and clean chapters
            substantial_chapters = []
            skipped_empty = 0
            skipped_heuristic = 0
            skipped_llm = 0
            skipped_boilerplate = 0

            for chapter_num, text in enumerate(chapters_text):
                if not text.strip():
                    skipped_empty += 1
                    continue

                text = self._normalize_text(text)
                text = self._remove_boilerplate(text)
                text = self._normalize_text(text)

                if self.ollama_available:
                    is_boilerplate, content_type = self._is_boilerplate_content_llm(
                        text
                    )
                    if is_boilerplate:
                        skipped_boilerplate += 1
                        self.logger.debug(
                            f"Chapter {chapter_num + 1}: Skipped (detected as {content_type})"
                        )
                        continue

                if not self._is_chapter_substantial(text):
                    skipped_heuristic += 1
                    self.logger.debug(f"Chapter {chapter_num + 1}: Skipped (heuristic)")
                    continue

                if self.ollama_available:
                    is_relevant, confidence = self._check_chapter_relevance_llm(text)
                    if not is_relevant or confidence < self.ollama_threshold:
                        skipped_llm += 1
                        self.logger.debug(
                            f"Chapter {chapter_num + 1}: Skipped by LLM "
                            f"(relevance={is_relevant}, confidence={confidence:.2f})"
                        )
                        continue

                substantial_chapters.append(text)

            # Log filtering results
            total_chapters = len(chapters_text)
            kept_chapters = len(substantial_chapters)
            self.logger.info(
                f"Filtered {mobi_path}: {kept_chapters}/{total_chapters} chapters kept "
                f"(empty={skipped_empty}, boilerplate={skipped_boilerplate}, "
                f"heuristic={skipped_heuristic}, llm={skipped_llm})"
            )

            if substantial_chapters:
                content = "\n\n".join(substantial_chapters)
                content = self._normalize_text(content)
                markdown_parts.append(content)
            else:
                markdown_parts.append("_No substantial content found in document._\n")

        finally:
            # Clean up temp directory
            shutil.rmtree(tempdir, ignore_errors=True)

        return "".join(markdown_parts)

    def process_ebook(self, ebook_path: str) -> bool:
        """
        Process a single ebook file.

        Args:
            ebook_path: Path to ebook file

        Returns:
            True if processing succeeded
        """
        try:
            self.logger.info(f"Processing: {ebook_path}")

            # Determine file type and convert
            ext = Path(ebook_path).suffix.lower()

            if ext == ".epub":
                if not EPUB_AVAILABLE:
                    self.logger.error(f"EPUB support not available: {ebook_path}")
                    return False
                markdown_content = self.epub_to_markdown(ebook_path)
            elif ext == ".mobi":
                if not MOBI_AVAILABLE:
                    self.logger.error(f"MOBI support not available: {ebook_path}")
                    return False
                markdown_content = self.mobi_to_markdown(ebook_path)
            else:
                self.logger.error(f"Unsupported file type: {ext}")
                return False

            if not markdown_content.strip():
                self.logger.warning(f"No content extracted from: {ebook_path}")
                return False

            # Generate output filename
            ebook_filename = Path(ebook_path).stem
            output_path = os.path.join(self.output_dir, f"{ebook_filename}.md")

            # Write markdown file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)

            # Mark as processed
            self.file_tracker.mark_processed(
                ebook_path, chunk_count=0, status="success"
            )

            self.logger.info(
                f"Successfully converted {Path(ebook_path).name} -> {Path(output_path).name}"
            )

            return True

        except Exception as e:
            self.logger.error(f"Error processing {ebook_path}: {e}", exc_info=True)
            try:
                self.file_tracker.mark_processed(
                    ebook_path, chunk_count=0, status="error"
                )
            except Exception:
                pass

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

        ebook_files = self.find_ebook_files()
        stats["files_found"] = len(ebook_files)

        if not ebook_files:
            self.logger.debug(f"No ebook files found in {self.input_dir}")
            return stats

        for file_path in ebook_files:
            if self.should_process_file(file_path):
                if self.process_ebook(file_path):
                    stats["files_processed"] += 1
                else:
                    stats["files_failed"] += 1
            else:
                stats["files_skipped"] += 1
                self.logger.debug(f"Skipping (already processed): {file_path}")

        return stats

    def run_continuous(self):
        """
        Run continuous monitoring loop.
        """
        self.logger.info(
            f"Starting continuous ebook monitoring "
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

                    tracker_stats = self.file_tracker.get_statistics()
                    self.logger.info(
                        f"Total: {tracker_stats['total_files']} ebooks processed"
                    )

                time.sleep(self.poll_interval)

        except KeyboardInterrupt:
            self.logger.info("\nShutting down gracefully...")
            self._shutdown()

    def _shutdown(self):
        """Cleanup and shutdown."""
        stats = self.file_tracker.get_statistics()
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Final Statistics:")
        self.logger.info(f"  Total ebooks processed: {stats['total_files']}")
        self.logger.info(f"  Successful: {stats['successful']}")
        self.logger.info(f"  Errors: {stats['errors']}")
        self.logger.info("=" * 60)
        self.logger.info("Ebook Tracker stopped")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Ebook (EPUB/MOBI) monitoring and conversion to markdown",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run in continuous monitoring mode (default)
  python ebook_tracker.py

  # Force reprocess all ebooks starting with "book" (one-shot mode)
  python ebook_tracker.py --force-reprocess "^book"

  # Force reprocess ebooks containing "2024" (one-shot mode)
  python ebook_tracker.py --force-reprocess "2024"

  # Force reprocess specific ebook (one-shot mode)
  python ebook_tracker.py --force-reprocess "exact_filename\\.epub$"

Note: When --force-reprocess is provided, the script runs once and exits.
      Without it, the script runs in continuous monitoring mode.
        """,
    )

    parser.add_argument(
        "--force-reprocess",
        type=str,
        metavar="PATTERN",
        help="Regex pattern to match ebook filenames for forced reprocessing. "
        "Runs in one-shot mode (single scan then exit) instead of continuous monitoring. "
        'Example patterns: "^book" or "2024" or "filename\\.epub$"',
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default="~/Documents/AI_IN",
        help="Directory to monitor for ebooks (default: ~/Documents/AI_IN)",
    )

    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Disable YAML frontmatter metadata in output",
    )

    parser.add_argument(
        "--poll-interval",
        type=int,
        default=5,
        help="Seconds between directory scans (default: 5)",
    )

    args = parser.parse_args()

    # Check if any format support is available
    if not EPUB_AVAILABLE and not MOBI_AVAILABLE:
        print("Error: Neither EPUB nor MOBI support is available.")
        print("Please install required libraries:")
        print("  pip install ebooklib mobi")
        sys.exit(1)

    try:
        tracker = EbookTracker(
            input_dir=args.input_dir,
            poll_interval=args.poll_interval,
            reprocess_on_change=True,
            min_chapter_chars=100,
            min_content_density=0.3,
            use_ollama=True,
            ollama_model="llama3.2:1b",
            ollama_threshold=0.5,
            include_metadata=not args.no_metadata,
            force_reprocess_pattern=args.force_reprocess,
        )

        if args.force_reprocess:
            tracker.logger.info("Running in one-shot mode (force reprocess enabled)")
            stats = tracker.run_single_scan()

            tracker.logger.info("=" * 60)
            tracker.logger.info("Single scan complete:")
            tracker.logger.info(f"  Files found: {stats['files_found']}")
            tracker.logger.info(f"  Files processed: {stats['files_processed']}")
            tracker.logger.info(f"  Files failed: {stats['files_failed']}")
            tracker.logger.info(f"  Files skipped: {stats['files_skipped']}")
            tracker.logger.info("=" * 60)

            tracker_stats = tracker.file_tracker.get_statistics()
            tracker.logger.info(
                f"Total ebooks in database: {tracker_stats['total_files']}"
            )

            sys.exit(0)
        else:
            tracker.run_continuous()

    except Exception as e:
        import logging

        logging.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
