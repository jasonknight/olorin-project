#!/usr/bin/env python3
"""
PDF monitoring and conversion script.
Monitors ~/Documents/AI_IN for PDF files, converts them to markdown,
and tracks processed files to avoid reprocessing.
"""

import os
import sys
import time
import re
import subprocess
import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
from collections import Counter
from datetime import datetime

from kafka import KafkaProducer
from kafka.errors import KafkaError

# Add parent directory to path for libs import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from libs.olorin_logging import OlorinLogger
from libs.config import Config

try:
    import fitz  # PyMuPDF
except ImportError:
    print("PyMuPDF not found. Installing...")
    os.system(f"{sys.executable} -m pip install PyMuPDF")
    import fitz

from file_tracker import FileTracker

# Optional: macOS Vision framework for OCR fallback
VISION_AVAILABLE = False
try:
    import Vision
    from Foundation import NSData

    VISION_AVAILABLE = True
except ImportError:
    Vision = None
    NSData = None


class PDFTracker:
    """
    Monitors directory for PDF files and converts them to markdown.
    """

    def __init__(
        self,
        input_dir: str = None,
        output_dir: str = None,
        tracking_db: str = None,
        poll_interval: int = 5,
        reprocess_on_change: bool = True,
        min_page_chars: int = 100,
        min_content_density: float = 0.3,
        use_ollama: bool = True,
        ollama_model: str = "llama3.2:1b",
        ollama_threshold: float = 0.5,
        include_metadata: bool = True,
        force_reprocess_pattern: str = None,
        notify_broca: bool = True,
    ):
        """
        Initialize PDF tracker.

        Args:
            input_dir: Directory to monitor for PDFs (default from config)
            output_dir: Directory to save markdown files (defaults to input_dir)
            tracking_db: Path to SQLite tracking database (default from config)
            poll_interval: Seconds between directory scans
            reprocess_on_change: Reprocess if PDF content changes
            min_page_chars: Minimum characters for page to be considered
            min_content_density: Minimum ratio of alphanumeric to total chars
            use_ollama: Enable local LLM for relevance detection
            ollama_model: Ollama model to use (e.g., llama3.2:1b, phi3:mini)
            ollama_threshold: Minimum relevance score (0-1) to include page
            include_metadata: Include YAML frontmatter with document metadata
            force_reprocess_pattern: Regex pattern to match filenames for forced reprocessing
            notify_broca: Send TTS notification to Broca when processing starts
        """
        # Get paths from config if not provided
        config = Config()
        if input_dir is None:
            input_dir = config.get_path("INPUT_DIR", "~/Documents/AI_IN")
        if tracking_db is None:
            tracking_db = config.get_path(
                "PDF_TRACKING_DB", "./hippocampus/data/pdf_tracking.db"
            )

        self.input_dir = os.path.expanduser(input_dir)
        self.output_dir = (
            os.path.expanduser(output_dir) if output_dir else self.input_dir
        )
        self.poll_interval = poll_interval
        self.reprocess_on_change = reprocess_on_change

        # Cleaning configuration
        self.min_page_chars = min_page_chars
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

        # Broca notification settings
        self.notify_broca = notify_broca
        self.broca_producer = None
        self.broca_topic = config.get("BROCA_KAFKA_TOPIC", "ai_out")
        self.kafka_servers = config.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

        # Setup logging
        default_log_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "logs"
        )
        log_file = os.path.join(default_log_dir, "hippocampus-pdf-tracker.log")
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

        # Check Vision OCR availability (macOS)
        self.vision_available = VISION_AVAILABLE
        if self.vision_available:
            self.logger.info("macOS Vision OCR available for image-based PDFs")
        else:
            self.logger.debug(
                "macOS Vision OCR not available (install pyobjc-framework-Vision)"
            )

        # Initialize Broca Kafka producer
        if self.notify_broca:
            try:
                self.broca_producer = KafkaProducer(
                    bootstrap_servers=self.kafka_servers,
                    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                    api_version_auto_timeout_ms=5000,
                    retries=3,
                )
                self.logger.info(
                    f"Broca notifications enabled (topic: {self.broca_topic})"
                )
            except Exception as e:
                self.logger.warning(f"Could not connect to Kafka for Broca: {e}")
                self.broca_producer = None

        # Create directories if they don't exist
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize file tracker
        self.file_tracker = FileTracker(tracking_db)

        self.logger.info("PDF Tracker initialized")
        self.logger.info(f"Monitoring directory: {self.input_dir}")
        self.logger.info(f"Output directory: {self.output_dir}")
        if self.force_reprocess_regex:
            self.logger.info(f"Force reprocess pattern: {force_reprocess_pattern}")

    def find_pdf_files(self) -> List[str]:
        """
        Find all PDF files in the input directory.

        Returns:
            List of absolute file paths
        """
        pdf_files = []
        input_path = Path(self.input_dir)

        for pdf_file in input_path.rglob("*.pdf"):
            if pdf_file.is_file():
                pdf_files.append(str(pdf_file.absolute()))

        # Also check for .PDF extension
        for pdf_file in input_path.rglob("*.PDF"):
            if pdf_file.is_file():
                pdf_files.append(str(pdf_file.absolute()))

        return pdf_files

    def _get_ollama_path(self) -> Optional[str]:
        """
        Find the ollama binary path.

        Returns:
            Path to ollama binary or None
        """
        # Try common locations
        paths = [
            "ollama",  # In PATH
            "/usr/local/bin/ollama",
            "/opt/homebrew/bin/ollama",
            "/Applications/Ollama.app/Contents/Resources/ollama",  # macOS app
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
        # Find ollama binary
        self.ollama_path = self._get_ollama_path()

        if not self.ollama_path:
            return False

        try:
            result = subprocess.run(
                [self.ollama_path, "list"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                # Check if our model is available
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

    def _guess_title_from_filename(self, filename: str) -> str:
        """
        Use Ollama to guess a clean title/author from the PDF filename.

        Many PDF filenames contain extra info like dates, edition numbers,
        scan info, etc. This uses an LLM to extract the likely title and author.

        Args:
            filename: The PDF filename (without path or extension)

        Returns:
            A clean, speakable title string (e.g., "The Great Gatsby by F. Scott Fitzgerald")
        """
        if not self.ollama_available:
            # Fallback: just clean up underscores/dashes
            return filename.replace("_", " ").replace("-", " ").strip()

        prompt = f"""Given this PDF filename, extract the likely book/document title and author (if present).
Return ONLY a clean, speakable phrase suitable for text-to-speech.

Examples:
- "Clean_Code_A_Handbook_Robert_C_Martin_2008" → "Clean Code by Robert C. Martin"
- "the-pragmatic-programmer-2nd-edition-scan" → "The Pragmatic Programmer"
- "1984_George_Orwell_ebook" → "1984 by George Orwell"
- "financial_report_q3_2024_final_v2" → "Financial Report Q3 2024"

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
                clean_title = result.stdout.strip()
                # Remove any quotes that might have been added
                clean_title = clean_title.strip("\"'")
                # Sanity check: if response is too long or empty, fallback
                if clean_title and len(clean_title) < 200:
                    return clean_title

        except (subprocess.TimeoutExpired, Exception) as e:
            self.logger.debug(f"Title extraction failed: {e}")

        # Fallback
        return filename.replace("_", " ").replace("-", " ").strip()

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

    def _extract_with_vision(self, page, dpi: int = 150) -> Optional[str]:
        """
        Use macOS Vision framework for OCR on a PDF page.

        Args:
            page: PyMuPDF page object
            dpi: Resolution for rendering (higher = better quality but slower)

        Returns:
            Extracted text or None if Vision is not available
        """
        if not self.vision_available:
            return None

        try:
            # Render page to image
            pix = page.get_pixmap(dpi=dpi)
            img_data = pix.tobytes("png")

            # Create NSData from image bytes
            ns_data = NSData.dataWithBytes_length_(img_data, len(img_data))

            # Create Vision request handler
            handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(
                ns_data, None
            )

            # Create text recognition request
            request = Vision.VNRecognizeTextRequest.alloc().init()
            request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)

            # Perform request
            success, error = handler.performRequests_error_([request], None)

            if not success:
                self.logger.debug(f"Vision OCR failed: {error}")
                return None

            # Extract text from results
            results = request.results()
            if not results:
                return ""

            lines = []
            for observation in results:
                candidates = observation.topCandidates_(1)
                if candidates:
                    lines.append(candidates[0].string())

            return "\n".join(lines)

        except Exception as e:
            self.logger.debug(f"Vision OCR error: {e}")
            return None

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

    def _is_page_substantial(self, text: str) -> bool:
        """
        Determine if a page has substantial content using heuristics.

        Args:
            text: Page text to analyze

        Returns:
            True if page meets minimum quality thresholds
        """
        if not text or not text.strip():
            return False

        # Check minimum length
        cleaned_text = text.strip()
        if len(cleaned_text) < self.min_page_chars:
            return False

        # Check content density
        density = self._calculate_content_density(cleaned_text)
        if density < self.min_content_density:
            return False

        # Check if page is mostly whitespace
        whitespace_ratio = (
            len(text) - len(text.replace(" ", "").replace("\n", ""))
        ) / len(text)
        if whitespace_ratio > 0.8:
            return False

        # Check for minimum number of words
        words = re.findall(r"\b\w+\b", cleaned_text)
        if len(words) < 20:
            return False

        return True

    def _detect_repeated_patterns(self, pages_text: List[str]) -> Set[str]:
        """
        Detect headers/footers that repeat across pages.

        Args:
            pages_text: List of page texts

        Returns:
            Set of repeated pattern strings to remove
        """
        if len(pages_text) < 3:
            return set()

        repeated_patterns = set()

        # Check first and last lines of each page
        first_lines = []
        last_lines = []

        for text in pages_text:
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            if lines:
                if len(lines[0]) < 100:  # Headers are usually short
                    first_lines.append(lines[0])
                if len(lines) > 1 and len(lines[-1]) < 100:
                    last_lines.append(lines[-1])

        # Find patterns that appear in >50% of pages
        threshold = len(pages_text) * 0.5

        for line_list in [first_lines, last_lines]:
            line_counts = Counter(line_list)
            for line, count in line_counts.items():
                if count >= threshold:
                    repeated_patterns.add(line)

        return repeated_patterns

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text by removing artifacts and excessive whitespace.

        Args:
            text: Text to normalize

        Returns:
            Cleaned text
        """
        # Remove form feed characters
        text = text.replace("\f", "\n")

        # Fix hyphenated words across lines
        text = re.sub(r"-\s*\n\s*", "", text)

        # Normalize whitespace
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

        # Remove standalone page numbers (lines with only digits)
        text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

        # Remove common table-of-contents patterns
        text = re.sub(r"^.*\.{4,}.*\d+\s*$", "", text, flags=re.MULTILINE)

        return text.strip()

    def _remove_boilerplate(self, text: str) -> str:
        """
        Remove common boilerplate patterns.

        Args:
            text: Text to clean

        Returns:
            Text with boilerplate removed
        """
        # Common patterns to remove
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

    def _extract_metadata(self, doc: fitz.Document, pdf_path: str) -> Dict[str, str]:
        """
        Extract metadata from PDF document.

        Args:
            doc: PyMuPDF document object
            pdf_path: Path to PDF file

        Returns:
            Dictionary containing metadata fields
        """
        metadata = {}
        pdf_metadata = doc.metadata or {}

        # Extract title (try metadata first, fallback to filename)
        title = pdf_metadata.get("title", "").strip()
        if not title or len(title) < 2:
            title = Path(pdf_path).stem
        metadata["title"] = title if title else "unknown"

        # Extract author
        author = pdf_metadata.get("author", "").strip()
        metadata["author"] = author if author else "unknown"

        # Extract creation/modification date
        # Try multiple date fields
        date_str = None
        for date_field in ["creationDate", "modDate", "created", "modified"]:
            date_value = pdf_metadata.get(date_field, "").strip()
            if date_value:
                date_str = date_value
                break

        # Parse date if found
        publish_date = "unknown"
        if date_str:
            try:
                # PyMuPDF dates are often in format: D:20240101120000+00'00'
                if date_str.startswith("D:"):
                    date_str = date_str[2:]
                # Extract year, month, day
                if len(date_str) >= 8:
                    year = date_str[0:4]
                    month = date_str[4:6]
                    day = date_str[6:8]
                    publish_date = f"{year}-{month}-{day}"
            except Exception:
                pass
        metadata["publish_date"] = publish_date

        # Extract keywords/subject
        keywords = []

        # Check for keywords field
        keywords_str = pdf_metadata.get("keywords", "").strip()
        if keywords_str:
            # Split by common delimiters
            keywords.extend(
                [k.strip() for k in re.split(r"[,;]", keywords_str) if k.strip()]
            )

        # Check for subject field
        subject = pdf_metadata.get("subject", "").strip()
        if subject and subject not in keywords:
            keywords.append(subject)

        # If no keywords found, use "unknown"
        metadata["keywords"] = keywords if keywords else ["unknown"]

        return metadata

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

        prompt = f"""Analyze this content from a PDF document. Classify it into one of these categories:

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

                # Parse response
                if "TOC" in response or "TABLE OF CONTENTS" in response:
                    return True, "toc"
                elif "COPYRIGHT" in response:
                    return True, "copyright"
                elif "SUBSTANTIVE" in response:
                    return False, "substantive"

            # Fallback: not boilerplate
            return False, "unknown"

        except (subprocess.TimeoutExpired, Exception) as e:
            self.logger.debug(f"LLM boilerplate check failed: {e}")
            return False, "unknown"

    def _check_page_relevance_llm(self, text: str) -> Tuple[bool, float]:
        """
        Use local LLM to determine if page content is relevant.

        Args:
            text: Page text to evaluate

        Returns:
            Tuple of (is_relevant, confidence_score)
        """
        if not self.ollama_available:
            return True, 1.0

        # Truncate very long pages for faster inference
        text_sample = text[:1000] if len(text) > 1000 else text

        prompt = f"""Analyze this page from a PDF document. Determine if it contains substantive information worth indexing.

Page content:
{text_sample}

Does this page contain meaningful content (NOT just headers, page numbers, or boilerplate)?
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

                # Parse response
                match = re.search(r"(YES|NO)\s+(0?\.\d+|1\.0|[01])", response)
                if match:
                    is_relevant = match.group(1) == "YES"
                    confidence = float(match.group(2))
                    return is_relevant, confidence

            # Fallback to heuristic
            return True, 0.5

        except (subprocess.TimeoutExpired, Exception) as e:
            self.logger.debug(f"LLM check failed: {e}")
            return True, 0.5

    def should_process_file(self, file_path: str) -> bool:
        """
        Determine if a PDF should be processed.

        Args:
            file_path: Path to PDF file

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

    def pdf_to_markdown(self, pdf_path: str) -> str:
        """
        Convert PDF to markdown text with intelligent cleaning.

        Uses hybrid approach:
        1. Extract all pages
        2. Detect and remove repeated headers/footers
        3. Apply heuristic filtering (content density, length)
        4. Optional LLM relevance check
        5. Create continuous text flow (no page markers)

        Args:
            pdf_path: Path to PDF file

        Returns:
            Cleaned markdown formatted text
        """
        markdown_parts = []

        # Open PDF
        doc = fitz.open(pdf_path)

        # Extract metadata
        filename = Path(pdf_path).stem

        # Build header with optional YAML frontmatter
        if self.include_metadata:
            # Extract structured metadata
            metadata = self._extract_metadata(doc, pdf_path)

            # Create YAML frontmatter
            markdown_parts.append("---\n")
            markdown_parts.append(f'title: "{metadata["title"]}"\n')
            markdown_parts.append(f'author: "{metadata["author"]}"\n')
            markdown_parts.append(f'publish_date: "{metadata["publish_date"]}"\n')

            # Format keywords as YAML list
            if len(metadata["keywords"]) == 1:
                markdown_parts.append(f'keywords: ["{metadata["keywords"][0]}"]\n')
            else:
                markdown_parts.append("keywords:\n")
                for keyword in metadata["keywords"]:
                    markdown_parts.append(f'  - "{keyword}"\n')

            markdown_parts.append("---\n\n")
            markdown_parts.append(f"# {metadata['title']}\n\n")
        else:
            # Legacy format without YAML frontmatter
            old_metadata = doc.metadata
            markdown_parts.append(f"# {filename}\n\n")

            if (
                old_metadata.get("title")
                and old_metadata["title"].lower() != filename.lower()
            ):
                markdown_parts.append(f"**Title:** {old_metadata['title']}\n\n")
            if old_metadata.get("author"):
                markdown_parts.append(f"**Author:** {old_metadata['author']}\n\n")

            markdown_parts.append("---\n\n")

        # First pass: Extract all page texts
        pages_text = []
        ocr_pages = 0
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()

            # If standard extraction returned empty, try OCR
            if not text.strip() and self.vision_available:
                ocr_text = self._extract_with_vision(page)
                if ocr_text and ocr_text.strip():
                    text = ocr_text
                    ocr_pages += 1
                    self.logger.debug(f"Page {page_num + 1}: Used Vision OCR")

            pages_text.append(text)

        doc.close()

        if ocr_pages > 0:
            self.logger.info(f"Used Vision OCR for {ocr_pages}/{len(pages_text)} pages")

        # Detect repeated patterns (headers/footers)
        self.logger.debug(f"Analyzing {len(pages_text)} pages for repeated patterns...")
        repeated_patterns = self._detect_repeated_patterns(pages_text)

        if repeated_patterns:
            self.logger.debug(
                f"Found {len(repeated_patterns)} repeated patterns to remove"
            )

        # Second pass: Filter and clean pages
        substantial_pages = []
        skipped_empty = 0
        skipped_heuristic = 0
        skipped_llm = 0
        skipped_boilerplate = 0

        for page_num, text in enumerate(pages_text):
            # Skip empty pages
            if not text.strip():
                skipped_empty += 1
                continue

            # Normalize text
            text = self._normalize_text(text)

            # Remove boilerplate
            text = self._remove_boilerplate(text)

            # Remove repeated patterns
            for pattern in repeated_patterns:
                text = text.replace(pattern, "")

            # Clean up after removals
            text = self._normalize_text(text)

            # Check if page is TOC or copyright using LLM
            if self.ollama_available:
                is_boilerplate, content_type = self._is_boilerplate_content_llm(text)
                if is_boilerplate:
                    skipped_boilerplate += 1
                    self.logger.debug(
                        f"Page {page_num + 1}: Skipped (detected as {content_type})"
                    )
                    continue

            # Apply heuristic filtering
            if not self._is_page_substantial(text):
                skipped_heuristic += 1
                self.logger.debug(f"Page {page_num + 1}: Skipped (heuristic)")
                continue

            # Optional LLM relevance check for borderline cases
            if self.ollama_available:
                is_relevant, confidence = self._check_page_relevance_llm(text)

                if not is_relevant or confidence < self.ollama_threshold:
                    skipped_llm += 1
                    self.logger.debug(
                        f"Page {page_num + 1}: Skipped by LLM "
                        f"(relevance={is_relevant}, confidence={confidence:.2f})"
                    )
                    continue

            # Page passed all filters
            substantial_pages.append(text)

        # Log filtering results
        total_pages = len(pages_text)
        kept_pages = len(substantial_pages)
        self.logger.info(
            f"Filtered {pdf_path}: {kept_pages}/{total_pages} pages kept "
            f"(empty={skipped_empty}, boilerplate={skipped_boilerplate}, "
            f"heuristic={skipped_heuristic}, llm={skipped_llm})"
        )

        # Join substantial pages into continuous text flow
        if substantial_pages:
            # Add natural paragraph breaks between pages
            content = "\n\n".join(substantial_pages)

            # Final normalization
            content = self._normalize_text(content)

            markdown_parts.append(content)
        else:
            markdown_parts.append("_No substantial content found in document._\n")

        return "".join(markdown_parts)

    def process_pdf(self, pdf_path: str) -> bool:
        """
        Process a single PDF file.

        Args:
            pdf_path: Path to PDF file

        Returns:
            True if processing succeeded
        """
        try:
            # Check if this is a retry
            file_info = self.file_tracker.get_file_info(pdf_path)
            if file_info and file_info.get("status") == "error":
                retry_count = (file_info.get("retries") or 0) + 1
                self.logger.info(f"Retrying (attempt {retry_count}/5): {pdf_path}")
                if file_info.get("error_message"):
                    self.logger.info(f"Previous error: {file_info['error_message']}")
            else:
                self.logger.info(f"Processing: {pdf_path}")

            # Send notification to Broca when starting to process
            if self.notify_broca and self.broca_producer:
                filename = Path(pdf_path).stem
                clean_title = self._guess_title_from_filename(filename)
                notification = f"Now processing: {clean_title}"
                self._send_broca_notification(notification)

            # Convert PDF to markdown
            markdown_content = self.pdf_to_markdown(pdf_path)

            if not markdown_content.strip():
                self.logger.warning(f"No content extracted from: {pdf_path}")
                return False

            # Generate output filename
            pdf_filename = Path(pdf_path).stem
            output_path = os.path.join(self.output_dir, f"{pdf_filename}.md")

            # Write markdown file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)

            # Mark as processed
            self.file_tracker.mark_processed(pdf_path, chunk_count=0, status="success")

            self.logger.info(
                f"Successfully converted {Path(pdf_path).name} -> {Path(output_path).name}"
            )

            return True

        except Exception as e:
            error_msg = str(e)
            self.logger.error(
                f"Error processing {pdf_path}: {error_msg}", exc_info=True
            )
            try:
                self.file_tracker.mark_processed(
                    pdf_path, chunk_count=0, status="error", error_message=error_msg
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

        # Find all PDF files
        pdf_files = self.find_pdf_files()
        stats["files_found"] = len(pdf_files)

        if not pdf_files:
            self.logger.debug(f"No PDF files found in {self.input_dir}")
            return stats

        # Process each file
        for file_path in pdf_files:
            if self.should_process_file(file_path):
                if self.process_pdf(file_path):
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
            f"Starting continuous PDF monitoring (poll interval: {self.poll_interval}s)"
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
                        f"Total: {tracker_stats['total_files']} PDFs processed"
                    )

                # Wait before next scan
                time.sleep(self.poll_interval)

        except KeyboardInterrupt:
            self.logger.info("\nShutting down gracefully...")
            self._shutdown()

    def _shutdown(self):
        """Cleanup and shutdown."""
        stats = self.file_tracker.get_statistics()
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Final Statistics:")
        self.logger.info(f"  Total PDFs processed: {stats['total_files']}")
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

        self.logger.info("PDF Tracker stopped")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="PDF monitoring and conversion to markdown",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run in continuous monitoring mode (default)
  python pdf_tracker.py

  # Force reprocess all PDFs starting with "report" (one-shot mode)
  python pdf_tracker.py --force-reprocess "^report"

  # Force reprocess PDFs containing "2024" (one-shot mode)
  python pdf_tracker.py --force-reprocess "2024"

  # Force reprocess specific PDF (one-shot mode)
  python pdf_tracker.py --force-reprocess "exact_filename\\.pdf$"

Note: When --force-reprocess is provided, the script runs once and exits.
      Without it, the script runs in continuous monitoring mode.
        """,
    )

    parser.add_argument(
        "--force-reprocess",
        type=str,
        metavar="PATTERN",
        help="Regex pattern to match PDF filenames for forced reprocessing. "
        "Runs in one-shot mode (single scan then exit) instead of continuous monitoring. "
        'Example patterns: "^report" or "2024" or "filename\\.pdf$"',
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory to monitor for PDFs (default: from config)",
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

    parser.add_argument(
        "--no-notify",
        action="store_true",
        help="Disable Broca TTS notifications when processing starts",
    )

    args = parser.parse_args()

    try:
        tracker = PDFTracker(
            input_dir=args.input_dir,
            poll_interval=args.poll_interval,
            reprocess_on_change=True,
            # Content filtering settings
            min_page_chars=100,  # Minimum characters per page
            min_content_density=0.3,  # Minimum ratio of alphanumeric chars
            # Local LLM settings
            use_ollama=True,  # Enable Ollama for relevance detection
            ollama_model="llama3.2:1b",  # Fast, small model
            ollama_threshold=0.5,  # Minimum confidence to include page
            # Metadata settings
            include_metadata=not args.no_metadata,
            # Force reprocess pattern
            force_reprocess_pattern=args.force_reprocess,
            # Broca notifications
            notify_broca=not args.no_notify,
        )

        # If force-reprocess is set, run single scan and exit
        if args.force_reprocess:
            tracker.logger.info("Running in one-shot mode (force reprocess enabled)")
            stats = tracker.run_single_scan()

            # Show results
            tracker.logger.info("=" * 60)
            tracker.logger.info("Single scan complete:")
            tracker.logger.info(f"  Files found: {stats['files_found']}")
            tracker.logger.info(f"  Files processed: {stats['files_processed']}")
            tracker.logger.info(f"  Files failed: {stats['files_failed']}")
            tracker.logger.info(f"  Files skipped: {stats['files_skipped']}")
            tracker.logger.info("=" * 60)

            # Show tracker statistics
            tracker_stats = tracker.file_tracker.get_statistics()
            tracker.logger.info(
                f"Total PDFs in database: {tracker_stats['total_files']}"
            )

            sys.exit(0)
        else:
            # Run in continuous monitoring mode
            tracker.run_continuous()

    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
