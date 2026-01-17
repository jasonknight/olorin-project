#!/usr/bin/env python3
"""
PDF monitoring and conversion script.
Monitors ~/Documents/AI_IN for PDF files, converts them to markdown,
and tracks processed files to avoid reprocessing.
"""

import argparse
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Add parent directory to path for libs import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from document_tracker_base import DocumentTrackerBase

try:
    import fitz  # PyMuPDF
except ImportError:
    print("PyMuPDF not found. Please install it:")
    print("  pip install PyMuPDF")
    sys.exit(1)

# Optional: macOS Vision framework for OCR fallback
VISION_AVAILABLE = False
try:
    import Vision
    from Foundation import NSData

    VISION_AVAILABLE = True
except ImportError:
    Vision = None
    NSData = None


class PDFTracker(DocumentTrackerBase):
    """
    Monitors directory for PDF files and converts them to markdown.
    """

    # Class attributes required by base class
    EXTENSIONS = {".pdf"}
    TRACKER_NAME = "PDF"
    LOG_FILENAME = "hippocampus-pdf-tracker.log"
    TRACKING_DB_CONFIG_KEY = "PDF_TRACKING_DB"

    def __init__(self, **kwargs):
        """
        Initialize PDF tracker.

        Args:
            **kwargs: Arguments passed to DocumentTrackerBase
        """
        # Initialize base class
        super().__init__(**kwargs)

        # Check Vision OCR availability (macOS)
        self.vision_available = VISION_AVAILABLE
        if self.vision_available:
            self.logger.info("macOS Vision OCR available for image-based PDFs")
        else:
            self.logger.debug(
                "macOS Vision OCR not available (install pyobjc-framework-Vision)"
            )

    # =========================================================================
    # Abstract method implementations
    # =========================================================================

    def find_files(self) -> List[str]:
        """Find all PDF files in the input directory."""
        return self._find_files_by_extensions()

    def to_markdown(self, pdf_path: str) -> str:
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
        if self.include_metadata:
            metadata = self._extract_metadata_from_doc(doc, pdf_path)
            markdown_parts.append(self._build_yaml_frontmatter(metadata))
            markdown_parts.append(f"# {metadata['title']}\n\n")
        else:
            filename = Path(pdf_path).stem
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

        # Filter and clean pages
        substantial_pages, stats = self._filter_pages(pages_text, repeated_patterns)

        # Log filtering results
        self.logger.info(
            f"Filtered {pdf_path}: {stats['kept']}/{stats['total']} pages kept "
            f"(empty={stats['empty']}, boilerplate={stats['boilerplate']}, "
            f"heuristic={stats['heuristic']}, llm={stats['llm']})"
        )

        # Join substantial pages into continuous text flow
        if substantial_pages:
            content = "\n\n".join(substantial_pages)
            content = self._normalize_text(content)
            markdown_parts.append(content)
        else:
            markdown_parts.append("_No substantial content found in document._\n")

        return "".join(markdown_parts)

    def process_file(self, pdf_path: str) -> bool:
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
            markdown_content = self.to_markdown(pdf_path)

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

    def _extract_metadata(self, pdf_path: str) -> Dict[str, str]:
        """
        Extract metadata from PDF file.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary containing metadata fields
        """
        try:
            doc = fitz.open(pdf_path)
            metadata = self._extract_metadata_from_doc(doc, pdf_path)
            doc.close()
            return metadata
        except Exception:
            return {
                "title": Path(pdf_path).stem,
                "author": "unknown",
                "publish_date": "unknown",
                "keywords": ["unknown"],
            }

    def _get_title_prompt_examples(self) -> str:
        """Return examples for the title extraction LLM prompt."""
        return """- "Clean_Code_A_Handbook_Robert_C_Martin_2008" → "Clean Code by Robert C. Martin"
- "the-pragmatic-programmer-2nd-edition-scan" → "The Pragmatic Programmer"
- "1984_George_Orwell_ebook" → "1984 by George Orwell"
- "financial_report_q3_2024_final_v2" → "Financial Report Q3 2024"""

    # =========================================================================
    # PDF-specific methods
    # =========================================================================

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

    def _filter_pages(
        self, pages_text: List[str], repeated_patterns: Set[str]
    ) -> Tuple[List[str], Dict[str, int]]:
        """
        Filter pages using heuristics and LLM.

        Args:
            pages_text: List of page texts
            repeated_patterns: Set of patterns to remove

        Returns:
            Tuple of (substantial_pages, stats_dict)
        """
        substantial_pages = []
        stats = {
            "total": len(pages_text),
            "kept": 0,
            "empty": 0,
            "boilerplate": 0,
            "heuristic": 0,
            "llm": 0,
        }

        for page_num, text in enumerate(pages_text):
            # Skip empty pages
            if not text.strip():
                stats["empty"] += 1
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
                    stats["boilerplate"] += 1
                    self.logger.debug(
                        f"Page {page_num + 1}: Skipped (detected as {content_type})"
                    )
                    continue

            # Apply heuristic filtering
            if not self._is_content_substantial(text):
                stats["heuristic"] += 1
                self.logger.debug(f"Page {page_num + 1}: Skipped (heuristic)")
                continue

            # Optional LLM relevance check for borderline cases
            if self.ollama_available:
                is_relevant, confidence = self._check_page_relevance_llm(text)

                if not is_relevant or confidence < self.ollama_threshold:
                    stats["llm"] += 1
                    self.logger.debug(
                        f"Page {page_num + 1}: Skipped by LLM "
                        f"(relevance={is_relevant}, confidence={confidence:.2f})"
                    )
                    continue

            # Page passed all filters
            substantial_pages.append(text)

        stats["kept"] = len(substantial_pages)
        return substantial_pages, stats

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

    def _extract_metadata_from_doc(
        self, doc: fitz.Document, pdf_path: str
    ) -> Dict[str, str]:
        """
        Extract metadata from an already-opened PDF document.

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
        default=None,
        help="Seconds between directory scans (default: from config)",
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
            include_metadata=not args.no_metadata,
            force_reprocess_pattern=args.force_reprocess,
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
