#!/usr/bin/env python3
"""
Ebook monitoring and conversion script.
Monitors ~/Documents/AI_IN for EPUB and MOBI files, converts them to markdown,
and tracks processed files to avoid reprocessing.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent directory to path for libs import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from document_tracker_base import DocumentTrackerBase

# Try to import ebooklib for EPUB support
try:
    import ebooklib
    from ebooklib import epub

    EPUB_AVAILABLE = True
except ImportError:
    EPUB_AVAILABLE = False

# Try to import mobi for MOBI support
try:
    import mobi

    MOBI_AVAILABLE = True
except ImportError:
    MOBI_AVAILABLE = False


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


class EbookTracker(DocumentTrackerBase):
    """
    Monitors directory for EPUB and MOBI files and converts them to markdown.
    """

    # Class attributes required by base class
    EXTENSIONS = {".epub", ".mobi"}
    TRACKER_NAME = "Ebook"
    LOG_FILENAME = "hippocampus-ebook-tracker.log"
    TRACKING_DB_CONFIG_KEY = "EBOOK_TRACKING_DB"

    def __init__(self, **kwargs):
        """
        Initialize Ebook tracker.

        Args:
            **kwargs: Arguments passed to DocumentTrackerBase
        """
        # Initialize base class
        super().__init__(**kwargs)

        # Log availability
        self.logger.info(f"EPUB support: {'enabled' if EPUB_AVAILABLE else 'disabled'}")
        self.logger.info(f"MOBI support: {'enabled' if MOBI_AVAILABLE else 'disabled'}")

    # =========================================================================
    # Abstract method implementations
    # =========================================================================

    def find_files(self) -> List[str]:
        """Find all EPUB and MOBI files in the input directory."""
        ebook_files = []
        input_path = Path(self.input_dir)

        # Find EPUB files
        if EPUB_AVAILABLE:
            for ebook_file in input_path.rglob("*.epub"):
                if ebook_file.is_file():
                    ebook_files.append(str(ebook_file.absolute()))
            for ebook_file in input_path.rglob("*.EPUB"):
                if ebook_file.is_file():
                    path = str(ebook_file.absolute())
                    if path not in ebook_files:
                        ebook_files.append(path)

        # Find MOBI files
        if MOBI_AVAILABLE:
            for ebook_file in input_path.rglob("*.mobi"):
                if ebook_file.is_file():
                    ebook_files.append(str(ebook_file.absolute()))
            for ebook_file in input_path.rglob("*.MOBI"):
                if ebook_file.is_file():
                    path = str(ebook_file.absolute())
                    if path not in ebook_files:
                        ebook_files.append(path)

        return ebook_files

    def to_markdown(self, ebook_path: str) -> str:
        """
        Convert ebook to markdown text with intelligent cleaning.

        Args:
            ebook_path: Path to ebook file

        Returns:
            Cleaned markdown formatted text
        """
        ext = Path(ebook_path).suffix.lower()

        if ext == ".epub":
            return self._epub_to_markdown(ebook_path)
        elif ext == ".mobi":
            return self._mobi_to_markdown(ebook_path)
        else:
            return ""

    def process_file(self, ebook_path: str) -> bool:
        """
        Process a single ebook file.

        Args:
            ebook_path: Path to ebook file

        Returns:
            True if processing succeeded
        """
        try:
            self.logger.info(f"Processing: {ebook_path}")

            # Send notification to Broca when starting to process
            if self.notify_broca and self.broca_producer:
                filename = Path(ebook_path).stem
                clean_title = self._guess_title_from_filename(filename)
                notification = f"Now processing: {clean_title}"
                self._send_broca_notification(notification)

            # Determine file type and check availability
            ext = Path(ebook_path).suffix.lower()

            if ext == ".epub" and not EPUB_AVAILABLE:
                self.logger.error(f"EPUB support not available: {ebook_path}")
                return False
            elif ext == ".mobi" and not MOBI_AVAILABLE:
                self.logger.error(f"MOBI support not available: {ebook_path}")
                return False

            # Convert to markdown
            markdown_content = self.to_markdown(ebook_path)

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

    def _extract_metadata(self, ebook_path: str) -> Dict[str, str]:
        """
        Extract metadata from ebook based on its extension.

        Args:
            ebook_path: Path to ebook file

        Returns:
            Dictionary containing metadata fields
        """
        ext = Path(ebook_path).suffix.lower()

        if ext == ".epub":
            return self._extract_epub_metadata(ebook_path)
        elif ext == ".mobi":
            return self._extract_mobi_metadata(ebook_path)
        else:
            return {
                "title": Path(ebook_path).stem,
                "author": "unknown",
                "publish_date": "unknown",
                "keywords": ["unknown"],
                "language": "unknown",
            }

    def _get_title_prompt_examples(self) -> str:
        """Return examples for the title extraction LLM prompt."""
        return """- "Clean_Code_A_Handbook_Robert_C_Martin_2008" → "Clean Code by Robert C. Martin"
- "the-pragmatic-programmer-2nd-edition-scan" → "The Pragmatic Programmer"
- "1984_George_Orwell_ebook" → "1984 by George Orwell"""

    # =========================================================================
    # Ebook-specific methods
    # =========================================================================

    def _epub_to_markdown(self, epub_path: str) -> str:
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
            metadata = self._extract_epub_metadata_from_book(book, epub_path)
            markdown_parts.append(self._build_yaml_frontmatter(metadata))
            markdown_parts.append(f"# {metadata['title']}\n\n")
        else:
            filename = Path(epub_path).stem
            markdown_parts.append(f"# {filename}\n\n")
            markdown_parts.append("---\n\n")

        # Extract and filter chapters
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
        substantial_chapters, stats = self._filter_chapters(chapters_text)

        # Log filtering results
        self.logger.info(
            f"Filtered {epub_path}: {stats['kept']}/{stats['total']} chapters kept "
            f"(empty={stats['empty']}, boilerplate={stats['boilerplate']}, "
            f"heuristic={stats['heuristic']}, llm={stats['llm']})"
        )

        if substantial_chapters:
            content = "\n\n".join(substantial_chapters)
            content = self._normalize_text(content)
            markdown_parts.append(content)
        else:
            markdown_parts.append("_No substantial content found in document._\n")

        return "".join(markdown_parts)

    def _mobi_to_markdown(self, mobi_path: str) -> str:
        """
        Convert MOBI to markdown text with intelligent cleaning.

        Args:
            mobi_path: Path to MOBI file

        Returns:
            Cleaned markdown formatted text
        """
        markdown_parts = []

        # Extract MOBI to temp directory
        tempdir, filepath = mobi.extract(mobi_path)

        try:
            # Extract metadata
            if self.include_metadata:
                metadata = self._extract_mobi_metadata(mobi_path)
                markdown_parts.append(self._build_yaml_frontmatter(metadata))
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
            substantial_chapters, stats = self._filter_chapters(chapters_text)

            # Log filtering results
            self.logger.info(
                f"Filtered {mobi_path}: {stats['kept']}/{stats['total']} chapters kept "
                f"(empty={stats['empty']}, boilerplate={stats['boilerplate']}, "
                f"heuristic={stats['heuristic']}, llm={stats['llm']})"
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

    def _filter_chapters(
        self, chapters_text: List[str]
    ) -> Tuple[List[str], Dict[str, int]]:
        """
        Filter chapters using heuristics and LLM.

        Args:
            chapters_text: List of chapter texts

        Returns:
            Tuple of (substantial_chapters, stats_dict)
        """
        substantial_chapters = []
        stats = {
            "total": len(chapters_text),
            "kept": 0,
            "empty": 0,
            "boilerplate": 0,
            "heuristic": 0,
            "llm": 0,
        }

        for chapter_num, text in enumerate(chapters_text):
            if not text.strip():
                stats["empty"] += 1
                continue

            text = self._normalize_text(text)
            text = self._remove_boilerplate(text)
            text = self._normalize_text(text)

            # Check for boilerplate using LLM
            if self.ollama_available:
                is_boilerplate, content_type = self._is_boilerplate_content_llm(text)
                if is_boilerplate:
                    stats["boilerplate"] += 1
                    self.logger.debug(
                        f"Chapter {chapter_num + 1}: Skipped (detected as {content_type})"
                    )
                    continue

            if not self._is_content_substantial(text):
                stats["heuristic"] += 1
                self.logger.debug(f"Chapter {chapter_num + 1}: Skipped (heuristic)")
                continue

            if self.ollama_available:
                is_relevant, confidence = self._check_chapter_relevance_llm(text)
                if not is_relevant or confidence < self.ollama_threshold:
                    stats["llm"] += 1
                    self.logger.debug(
                        f"Chapter {chapter_num + 1}: Skipped by LLM "
                        f"(relevance={is_relevant}, confidence={confidence:.2f})"
                    )
                    continue

            substantial_chapters.append(text)

        stats["kept"] = len(substantial_chapters)
        return substantial_chapters, stats

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

    def _extract_epub_metadata(self, epub_path: str) -> Dict[str, str]:
        """
        Extract metadata from EPUB file.

        Args:
            epub_path: Path to EPUB file

        Returns:
            Dictionary containing metadata fields
        """
        try:
            book = epub.read_epub(epub_path)
            return self._extract_epub_metadata_from_book(book, epub_path)
        except Exception:
            return {
                "title": Path(epub_path).stem,
                "author": "unknown",
                "publish_date": "unknown",
                "keywords": ["unknown"],
                "language": "unknown",
            }

    def _extract_epub_metadata_from_book(self, book, epub_path: str) -> Dict[str, str]:
        """
        Extract metadata from an already-loaded EPUB book object.

        Args:
            book: ebooklib EpubBook object
            epub_path: Path to EPUB file

        Returns:
            Dictionary containing metadata fields
        """
        metadata = {}

        # Extract title
        title = book.get_metadata("DC", "title")
        if title and len(title) > 0:
            metadata["title"] = title[0][0]
        else:
            metadata["title"] = Path(epub_path).stem

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
            shutil.rmtree(tempdir, ignore_errors=True)

        except Exception as e:
            self.logger.debug(f"Error extracting MOBI metadata: {e}")

        return metadata


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
        default=None,
        help="Directory to monitor for ebooks (default: from config)",
    )

    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Disable YAML frontmatter metadata in output",
    )

    parser.add_argument(
        "--poll-interval",
        type=int,
        default=None,
        help="Seconds between directory scans (default: from config)",
    )

    parser.add_argument(
        "--no-notify",
        action="store_true",
        help="Disable Broca TTS notifications when processing starts",
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
            include_metadata=not args.no_metadata,
            force_reprocess_pattern=args.force_reprocess,
            notify_broca=not args.no_notify,
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
