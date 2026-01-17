#!/usr/bin/env python3
"""
Office document monitoring and conversion script.
Monitors ~/Documents/AI_IN for .doc, .docx, and .odt files, converts them to markdown,
and tracks processed files to avoid reprocessing.

Uses pandoc for conversion with metadata extraction from document properties.
"""

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path for libs import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from document_tracker_base import DocumentTrackerBase

# Optional imports for metadata extraction
DOCX_AVAILABLE = False
ODF_AVAILABLE = False

try:
    from docx import Document as DocxDocument

    DOCX_AVAILABLE = True
except ImportError:
    pass

try:
    from odf.opendocument import load as load_odf
    from odf.namespaces import DCNS

    ODF_AVAILABLE = True
except ImportError:
    pass


class OfficeTracker(DocumentTrackerBase):
    """
    Monitors directory for Office documents and converts them to markdown.
    Supports .doc, .docx, and .odt formats using pandoc.
    """

    # Class attributes required by base class
    EXTENSIONS = {".doc", ".docx", ".odt"}
    TRACKER_NAME = "Office"
    LOG_FILENAME = "hippocampus-office-tracker.log"
    TRACKING_DB_CONFIG_KEY = "OFFICE_TRACKING_DB"

    def __init__(self, **kwargs):
        """
        Initialize Office document tracker.

        Args:
            **kwargs: Arguments passed to DocumentTrackerBase
        """
        # Initialize base class
        super().__init__(**kwargs)

        # Check pandoc availability (Office-specific requirement)
        self.pandoc_path = self._get_pandoc_path()
        if self.pandoc_path:
            self.logger.info(f"Pandoc detected: {self.pandoc_path}")
        else:
            self.logger.error(
                "Pandoc not found! Office document conversion requires pandoc."
            )
            self.logger.error(
                "Install pandoc: brew install pandoc (macOS) or apt install pandoc (Linux)"
            )

        # Log metadata extraction capabilities
        if DOCX_AVAILABLE:
            self.logger.info("python-docx available for .docx metadata extraction")
        else:
            self.logger.warning(
                "python-docx not available - .docx metadata limited to filename"
            )

        if ODF_AVAILABLE:
            self.logger.info("odfpy available for .odt metadata extraction")
        else:
            self.logger.warning(
                "odfpy not available - .odt metadata limited to filename"
            )

        self.logger.info(f"Supported formats: {', '.join(sorted(self.EXTENSIONS))}")

    # =========================================================================
    # Abstract method implementations
    # =========================================================================

    def find_files(self) -> List[str]:
        """Find all Office documents in the input directory."""
        return self._find_files_by_extensions()

    def to_markdown(self, file_path: str) -> str:
        """
        Convert Office document to markdown with metadata.

        Args:
            file_path: Path to document

        Returns:
            Markdown formatted text
        """
        markdown_parts = []

        # Extract metadata
        metadata = self._extract_metadata(file_path)

        # Convert document content with pandoc
        content = self._convert_with_pandoc(file_path)

        if not content:
            self.logger.warning(f"No content extracted from: {file_path}")
            content = "_Document conversion failed or document is empty._\n"

        # Normalize content
        content = self._normalize_text(content)

        # Check if content is substantial
        if not self._is_content_substantial(content):
            self.logger.warning(f"Content below minimum threshold: {file_path}")

        # Check for boilerplate using LLM
        if self.ollama_available and content:
            is_boilerplate, content_type = self._is_boilerplate_content_llm(content)
            if is_boilerplate:
                self.logger.info(
                    f"Content detected as {content_type}, but including anyway"
                )

        # Build markdown with YAML frontmatter
        if self.include_metadata:
            markdown_parts.append(self._build_yaml_frontmatter(metadata))
            markdown_parts.append(f"# {metadata['title']}\n\n")
        else:
            markdown_parts.append(f"# {Path(file_path).stem}\n\n")
            markdown_parts.append("---\n\n")

        markdown_parts.append(content)

        return "".join(markdown_parts)

    def process_file(self, file_path: str) -> bool:
        """
        Process a single Office document.

        Args:
            file_path: Path to document

        Returns:
            True if processing succeeded
        """
        try:
            self.logger.info(f"Processing: {file_path}")

            # Send notification to Broca when starting to process
            if self.notify_broca and self.broca_producer:
                filename = Path(file_path).stem
                clean_title = self._guess_title_from_filename(filename)
                notification = f"Now processing: {clean_title}"
                self._send_broca_notification(notification)

            # Convert to markdown
            markdown_content = self.to_markdown(file_path)

            if not markdown_content.strip():
                self.logger.warning(f"No content extracted from: {file_path}")
                return False

            # Generate output filename
            doc_filename = Path(file_path).stem
            output_path = os.path.join(self.output_dir, f"{doc_filename}.md")

            # Write markdown file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)

            # Mark as processed
            self.file_tracker.mark_processed(file_path, chunk_count=0, status="success")

            self.logger.info(
                f"Successfully converted {Path(file_path).name} -> {Path(output_path).name}"
            )

            return True

        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {e}", exc_info=True)
            try:
                self.file_tracker.mark_processed(
                    file_path, chunk_count=0, status="error"
                )
            except Exception:
                pass

            return False

    def _extract_metadata(self, file_path: str) -> Dict[str, str]:
        """
        Extract metadata from an Office document based on its extension.

        Args:
            file_path: Path to document

        Returns:
            Dictionary containing metadata fields
        """
        ext = Path(file_path).suffix.lower()

        if ext == ".docx":
            return self._extract_metadata_docx(file_path)
        elif ext == ".odt":
            return self._extract_metadata_odt(file_path)
        elif ext == ".doc":
            return self._extract_metadata_doc(file_path)
        else:
            return {
                "title": Path(file_path).stem,
                "author": "unknown",
                "publish_date": "unknown",
                "keywords": ["unknown"],
            }

    def _get_title_prompt_examples(self) -> str:
        """Return examples for the title extraction LLM prompt."""
        return """- "meeting_notes_q3_2024_final_v2" → "Meeting Notes Q3 2024"
- "John_Smith_Resume_2024" → "Resume by John Smith"
- "project_proposal_acme_corp" → "Project Proposal for Acme Corp"""

    # =========================================================================
    # Override should_process_file to check pandoc availability
    # =========================================================================

    def should_process_file(self, file_path: str) -> bool:
        """
        Determine if a document should be processed.

        Args:
            file_path: Path to document

        Returns:
            True if file should be processed
        """
        # Check pandoc availability first
        if not self.pandoc_path:
            return False

        # Delegate to base class implementation
        return super().should_process_file(file_path)

    # =========================================================================
    # Office-specific methods
    # =========================================================================

    def _get_pandoc_path(self) -> Optional[str]:
        """
        Find the pandoc binary path.

        Returns:
            Path to pandoc binary or None
        """
        paths = [
            "pandoc",  # In PATH
            "/usr/local/bin/pandoc",
            "/opt/homebrew/bin/pandoc",
            "/usr/bin/pandoc",
        ]

        for path in paths:
            try:
                result = subprocess.run(
                    [path, "--version"], capture_output=True, timeout=5
                )
                if result.returncode == 0:
                    return path
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue

        return None

    def _convert_with_pandoc(self, input_path: str) -> Optional[str]:
        """
        Convert Office document to markdown using pandoc.

        Args:
            input_path: Path to input document

        Returns:
            Markdown content as string, or None on failure
        """
        if not self.pandoc_path:
            self.logger.error("Pandoc not available for conversion")
            return None

        try:
            # Use pandoc to convert to markdown
            result = subprocess.run(
                [self.pandoc_path, input_path, "-t", "markdown", "--wrap=none"],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                return result.stdout
            else:
                self.logger.error(f"Pandoc conversion failed: {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            self.logger.error(f"Pandoc conversion timed out for: {input_path}")
            return None
        except Exception as e:
            self.logger.error(f"Pandoc conversion error: {e}")
            return None

    def _extract_metadata_docx(self, docx_path: str) -> Dict[str, str]:
        """
        Extract metadata from a .docx file using python-docx.

        Args:
            docx_path: Path to .docx file

        Returns:
            Dictionary containing metadata fields
        """
        metadata = {
            "title": Path(docx_path).stem,
            "author": "unknown",
            "publish_date": "unknown",
            "keywords": ["unknown"],
        }

        if not DOCX_AVAILABLE:
            return metadata

        try:
            doc = DocxDocument(docx_path)
            core_props = doc.core_properties

            if core_props.title and core_props.title.strip():
                metadata["title"] = core_props.title.strip()

            if core_props.author and core_props.author.strip():
                metadata["author"] = core_props.author.strip()

            # Try multiple date fields
            date_value = core_props.created or core_props.modified
            if date_value:
                if isinstance(date_value, datetime):
                    metadata["publish_date"] = date_value.strftime("%Y-%m-%d")
                else:
                    metadata["publish_date"] = str(date_value)[:10]

            # Keywords
            if core_props.keywords and core_props.keywords.strip():
                keywords = [
                    k.strip()
                    for k in re.split(r"[,;]", core_props.keywords)
                    if k.strip()
                ]
                if keywords:
                    metadata["keywords"] = keywords

            # Subject as additional keyword
            if core_props.subject and core_props.subject.strip():
                subject = core_props.subject.strip()
                if subject not in metadata["keywords"]:
                    if metadata["keywords"] == ["unknown"]:
                        metadata["keywords"] = [subject]
                    else:
                        metadata["keywords"].append(subject)

        except Exception as e:
            self.logger.debug(f"Error extracting .docx metadata: {e}")

        return metadata

    def _extract_metadata_odt(self, odt_path: str) -> Dict[str, str]:
        """
        Extract metadata from a .odt file using odfpy.

        Args:
            odt_path: Path to .odt file

        Returns:
            Dictionary containing metadata fields
        """
        metadata = {
            "title": Path(odt_path).stem,
            "author": "unknown",
            "publish_date": "unknown",
            "keywords": ["unknown"],
        }

        if not ODF_AVAILABLE:
            return metadata

        try:
            doc = load_odf(odt_path)
            meta_element = doc.meta

            if meta_element:
                # Title
                for elem in meta_element.childNodes:
                    if elem.qname == (DCNS, "title"):
                        text = "".join(
                            str(c) for c in elem.childNodes if hasattr(c, "__str__")
                        )
                        if text.strip():
                            metadata["title"] = text.strip()
                            break

                # Author (creator)
                for elem in meta_element.childNodes:
                    if elem.qname == (DCNS, "creator"):
                        text = "".join(
                            str(c) for c in elem.childNodes if hasattr(c, "__str__")
                        )
                        if text.strip():
                            metadata["author"] = text.strip()
                            break

                # Date
                for elem in meta_element.childNodes:
                    if elem.qname == (DCNS, "date"):
                        text = "".join(
                            str(c) for c in elem.childNodes if hasattr(c, "__str__")
                        )
                        if text.strip():
                            metadata["publish_date"] = text.strip()[:10]
                            break

                # Subject/Keywords
                keywords = []
                for elem in meta_element.childNodes:
                    if elem.qname == (DCNS, "subject"):
                        text = "".join(
                            str(c) for c in elem.childNodes if hasattr(c, "__str__")
                        )
                        if text.strip():
                            keywords.append(text.strip())

                if keywords:
                    metadata["keywords"] = keywords

        except Exception as e:
            self.logger.debug(f"Error extracting .odt metadata: {e}")

        return metadata

    def _extract_metadata_doc(self, doc_path: str) -> Dict[str, str]:
        """
        Extract metadata from a legacy .doc file.
        Limited extraction - mainly uses filename.

        Args:
            doc_path: Path to .doc file

        Returns:
            Dictionary containing metadata fields
        """
        # Legacy .doc format has limited metadata accessibility without
        # additional libraries. Default to filename-based metadata.
        return {
            "title": Path(doc_path).stem,
            "author": "unknown",
            "publish_date": "unknown",
            "keywords": ["unknown"],
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Office document monitoring and conversion to markdown",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run in continuous monitoring mode (default)
  python office_tracker.py

  # Force reprocess all documents starting with "report" (one-shot mode)
  python office_tracker.py --force-reprocess "^report"

  # Force reprocess documents containing "2024" (one-shot mode)
  python office_tracker.py --force-reprocess "2024"

  # Force reprocess specific document (one-shot mode)
  python office_tracker.py --force-reprocess "exact_filename\\.docx$"

Note: When --force-reprocess is provided, the script runs once and exits.
      Without it, the script runs in continuous monitoring mode.
        """,
    )

    parser.add_argument(
        "--force-reprocess",
        type=str,
        metavar="PATTERN",
        help="Regex pattern to match document filenames for forced reprocessing. "
        "Runs in one-shot mode (single scan then exit) instead of continuous monitoring.",
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory to monitor for documents (default: from config)",
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
        tracker = OfficeTracker(
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
                f"Total documents in database: {tracker_stats['total_files']}"
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
