#!/usr/bin/env python3
"""
Office document monitoring and conversion script.
Monitors ~/Documents/AI_IN for .doc, .docx, and .odt files, converts them to markdown,
and tracks processed files to avoid reprocessing.

Uses pandoc for conversion with metadata extraction from document properties.
"""

import os
import sys
import time
import re
import subprocess
import argparse
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

# Add parent directory to path for libs import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from libs.olorin_logging import OlorinLogger

from file_tracker import FileTracker

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
    from odf import meta as odf_meta
    from odf.namespaces import DCNS
    ODF_AVAILABLE = True
except ImportError:
    pass


class OfficeTracker:
    """
    Monitors directory for Office documents and converts them to markdown.
    Supports .doc, .docx, and .odt formats using pandoc.
    """

    # Supported extensions
    EXTENSIONS = {'.doc', '.docx', '.odt'}

    def __init__(
        self,
        input_dir: str = "~/Documents/AI_IN",
        output_dir: str = None,
        tracking_db: str = "./data/office_tracking.db",
        poll_interval: int = 5,
        reprocess_on_change: bool = True,
        min_content_chars: int = 100,
        use_ollama: bool = True,
        ollama_model: str = "llama3.2:1b",
        include_metadata: bool = True,
        force_reprocess_pattern: str = None
    ):
        """
        Initialize Office document tracker.

        Args:
            input_dir: Directory to monitor for documents
            output_dir: Directory to save markdown files (defaults to input_dir)
            tracking_db: Path to SQLite tracking database
            poll_interval: Seconds between directory scans
            reprocess_on_change: Reprocess if document content changes
            min_content_chars: Minimum characters for valid document
            use_ollama: Enable local LLM for content filtering
            ollama_model: Ollama model to use
            include_metadata: Include YAML frontmatter with document metadata
            force_reprocess_pattern: Regex pattern to match filenames for forced reprocessing
        """
        self.input_dir = os.path.expanduser(input_dir)
        self.output_dir = os.path.expanduser(output_dir) if output_dir else self.input_dir
        self.poll_interval = poll_interval
        self.reprocess_on_change = reprocess_on_change

        # Content configuration
        self.min_content_chars = min_content_chars
        self.use_ollama = use_ollama
        self.ollama_model = ollama_model
        self.include_metadata = include_metadata

        # Force reprocess pattern
        self.force_reprocess_regex = None
        if force_reprocess_pattern:
            try:
                self.force_reprocess_regex = re.compile(force_reprocess_pattern)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern: {force_reprocess_pattern} - {e}")

        # Setup logging
        default_log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')
        log_file = os.path.join(default_log_dir, 'hippocampus-office-tracker.log')
        self.logger = OlorinLogger(log_file=log_file, log_level='INFO', name=__name__)

        # Check pandoc availability
        self.pandoc_path = self._get_pandoc_path()
        if self.pandoc_path:
            self.logger.info(f"Pandoc detected: {self.pandoc_path}")
        else:
            self.logger.error("Pandoc not found! Office document conversion requires pandoc.")
            self.logger.error("Install pandoc: brew install pandoc (macOS) or apt install pandoc (Linux)")

        # Check Ollama availability
        self.ollama_available = False
        if self.use_ollama:
            self.ollama_available = self._check_ollama()
            if self.ollama_available:
                self.logger.info(f"Ollama detected - using model: {self.ollama_model}")
            else:
                self.logger.warning("Ollama not available - skipping LLM filtering")

        # Log metadata extraction capabilities
        if DOCX_AVAILABLE:
            self.logger.info("python-docx available for .docx metadata extraction")
        else:
            self.logger.warning("python-docx not available - .docx metadata limited to filename")

        if ODF_AVAILABLE:
            self.logger.info("odfpy available for .odt metadata extraction")
        else:
            self.logger.warning("odfpy not available - .odt metadata limited to filename")

        # Create directories if they don't exist
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize file tracker
        self.file_tracker = FileTracker(tracking_db)

        self.logger.info(f"Office Tracker initialized")
        self.logger.info(f"Monitoring directory: {self.input_dir}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Supported formats: {', '.join(sorted(self.EXTENSIONS))}")
        if self.force_reprocess_regex:
            self.logger.info(f"Force reprocess pattern: {force_reprocess_pattern}")

    def _get_pandoc_path(self) -> Optional[str]:
        """
        Find the pandoc binary path.

        Returns:
            Path to pandoc binary or None
        """
        paths = [
            'pandoc',  # In PATH
            '/usr/local/bin/pandoc',
            '/opt/homebrew/bin/pandoc',
            '/usr/bin/pandoc',
        ]

        for path in paths:
            try:
                result = subprocess.run(
                    [path, '--version'],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode == 0:
                    return path
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue

        return None

    def _get_ollama_path(self) -> Optional[str]:
        """
        Find the ollama binary path.

        Returns:
            Path to ollama binary or None
        """
        paths = [
            'ollama',
            '/usr/local/bin/ollama',
            '/opt/homebrew/bin/ollama',
            '/Applications/Ollama.app/Contents/Resources/ollama',
        ]

        for path in paths:
            try:
                result = subprocess.run(
                    [path, '--version'],
                    capture_output=True,
                    timeout=2
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
                [self.ollama_path, 'list'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                if self.ollama_model in result.stdout:
                    return True
                # Try to pull the model
                self.logger.info(f"Pulling Ollama model: {self.ollama_model}")
                pull_result = subprocess.run(
                    [self.ollama_path, 'pull', self.ollama_model],
                    capture_output=True,
                    timeout=300
                )
                return pull_result.returncode == 0
            return False
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            return False

    def find_office_files(self) -> List[str]:
        """
        Find all Office documents in the input directory.

        Returns:
            List of absolute file paths
        """
        office_files = []
        input_path = Path(self.input_dir)

        for ext in self.EXTENSIONS:
            # Check lowercase extension
            for doc_file in input_path.rglob(f"*{ext}"):
                if doc_file.is_file():
                    office_files.append(str(doc_file.absolute()))

            # Check uppercase extension
            for doc_file in input_path.rglob(f"*{ext.upper()}"):
                if doc_file.is_file():
                    file_path = str(doc_file.absolute())
                    if file_path not in office_files:
                        office_files.append(file_path)

        return office_files

    def _extract_metadata_docx(self, docx_path: str) -> Dict[str, str]:
        """
        Extract metadata from a .docx file using python-docx.

        Args:
            docx_path: Path to .docx file

        Returns:
            Dictionary containing metadata fields
        """
        metadata = {
            'title': Path(docx_path).stem,
            'author': 'unknown',
            'publish_date': 'unknown',
            'keywords': ['unknown']
        }

        if not DOCX_AVAILABLE:
            return metadata

        try:
            doc = DocxDocument(docx_path)
            core_props = doc.core_properties

            if core_props.title and core_props.title.strip():
                metadata['title'] = core_props.title.strip()

            if core_props.author and core_props.author.strip():
                metadata['author'] = core_props.author.strip()

            # Try multiple date fields
            date_value = core_props.created or core_props.modified
            if date_value:
                if isinstance(date_value, datetime):
                    metadata['publish_date'] = date_value.strftime('%Y-%m-%d')
                else:
                    metadata['publish_date'] = str(date_value)[:10]

            # Keywords
            if core_props.keywords and core_props.keywords.strip():
                keywords = [k.strip() for k in re.split(r'[,;]', core_props.keywords) if k.strip()]
                if keywords:
                    metadata['keywords'] = keywords

            # Subject as additional keyword
            if core_props.subject and core_props.subject.strip():
                subject = core_props.subject.strip()
                if subject not in metadata['keywords']:
                    if metadata['keywords'] == ['unknown']:
                        metadata['keywords'] = [subject]
                    else:
                        metadata['keywords'].append(subject)

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
            'title': Path(odt_path).stem,
            'author': 'unknown',
            'publish_date': 'unknown',
            'keywords': ['unknown']
        }

        if not ODF_AVAILABLE:
            return metadata

        try:
            doc = load_odf(odt_path)
            meta_element = doc.meta

            if meta_element:
                # Title
                title_elements = meta_element.getElementsByType(odf_meta.Title) if hasattr(odf_meta, 'Title') else []
                for elem in meta_element.childNodes:
                    if elem.qname == (DCNS, 'title'):
                        text = ''.join(str(c) for c in elem.childNodes if hasattr(c, '__str__'))
                        if text.strip():
                            metadata['title'] = text.strip()
                            break

                # Author (creator)
                for elem in meta_element.childNodes:
                    if elem.qname == (DCNS, 'creator'):
                        text = ''.join(str(c) for c in elem.childNodes if hasattr(c, '__str__'))
                        if text.strip():
                            metadata['author'] = text.strip()
                            break

                # Date
                for elem in meta_element.childNodes:
                    if elem.qname == (DCNS, 'date'):
                        text = ''.join(str(c) for c in elem.childNodes if hasattr(c, '__str__'))
                        if text.strip():
                            metadata['publish_date'] = text.strip()[:10]
                            break

                # Subject/Keywords
                keywords = []
                for elem in meta_element.childNodes:
                    if elem.qname == (DCNS, 'subject'):
                        text = ''.join(str(c) for c in elem.childNodes if hasattr(c, '__str__'))
                        if text.strip():
                            keywords.append(text.strip())

                if keywords:
                    metadata['keywords'] = keywords

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
            'title': Path(doc_path).stem,
            'author': 'unknown',
            'publish_date': 'unknown',
            'keywords': ['unknown']
        }

    def _extract_metadata(self, file_path: str) -> Dict[str, str]:
        """
        Extract metadata from an Office document based on its extension.

        Args:
            file_path: Path to document

        Returns:
            Dictionary containing metadata fields
        """
        ext = Path(file_path).suffix.lower()

        if ext == '.docx':
            return self._extract_metadata_docx(file_path)
        elif ext == '.odt':
            return self._extract_metadata_odt(file_path)
        elif ext == '.doc':
            return self._extract_metadata_doc(file_path)
        else:
            return {
                'title': Path(file_path).stem,
                'author': 'unknown',
                'publish_date': 'unknown',
                'keywords': ['unknown']
            }

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
                [self.pandoc_path, input_path, '-t', 'markdown', '--wrap=none'],
                capture_output=True,
                text=True,
                timeout=60
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

        # Check for minimum number of words
        words = re.findall(r'\b\w+\b', cleaned_text)
        if len(words) < 10:
            return False

        return True

    def _is_boilerplate_content_llm(self, text: str) -> Tuple[bool, str]:
        """
        Use local LLM to determine if content is boilerplate.

        Args:
            text: Text to evaluate

        Returns:
            Tuple of (is_boilerplate, content_type)
        """
        if not self.ollama_available:
            return False, 'unknown'

        text_sample = text[:800] if len(text) > 800 else text

        prompt = f"""Analyze this content from a document. Classify it:

TOC - Table of contents
COPYRIGHT - Copyright notices, legal disclaimers
SUBSTANTIVE - Actual document content worth keeping
UNKNOWN - Cannot determine

Content:
{text_sample}

Respond with ONLY one word: TOC, COPYRIGHT, SUBSTANTIVE, or UNKNOWN"""

        try:
            result = subprocess.run(
                [self.ollama_path, 'run', self.ollama_model, prompt],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                response = result.stdout.strip().upper()

                if 'TOC' in response or 'TABLE OF CONTENTS' in response:
                    return True, 'toc'
                elif 'COPYRIGHT' in response:
                    return True, 'copyright'
                elif 'SUBSTANTIVE' in response:
                    return False, 'substantive'

            return False, 'unknown'

        except (subprocess.TimeoutExpired, Exception) as e:
            self.logger.debug(f"LLM boilerplate check failed: {e}")
            return False, 'unknown'

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text by removing artifacts and excessive whitespace.

        Args:
            text: Text to normalize

        Returns:
            Cleaned text
        """
        # Remove form feed characters
        text = text.replace('\f', '\n')

        # Normalize whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

        return text.strip()

    def office_to_markdown(self, file_path: str) -> str:
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
                self.logger.info(f"Content detected as {content_type}, but including anyway")

        # Build markdown with YAML frontmatter
        if self.include_metadata:
            markdown_parts.append("---\n")
            markdown_parts.append(f"title: \"{metadata['title']}\"\n")
            markdown_parts.append(f"author: \"{metadata['author']}\"\n")
            markdown_parts.append(f"publish_date: \"{metadata['publish_date']}\"\n")

            if len(metadata['keywords']) == 1:
                markdown_parts.append(f"keywords: [\"{metadata['keywords'][0]}\"]\n")
            else:
                markdown_parts.append("keywords:\n")
                for keyword in metadata['keywords']:
                    markdown_parts.append(f"  - \"{keyword}\"\n")

            markdown_parts.append("---\n\n")
            markdown_parts.append(f"# {metadata['title']}\n\n")
        else:
            markdown_parts.append(f"# {Path(file_path).stem}\n\n")
            markdown_parts.append("---\n\n")

        markdown_parts.append(content)

        return ''.join(markdown_parts)

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

        # Check if pandoc is available
        if not self.pandoc_path:
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

    def process_office(self, file_path: str) -> bool:
        """
        Process a single Office document.

        Args:
            file_path: Path to document

        Returns:
            True if processing succeeded
        """
        try:
            self.logger.info(f"Processing: {file_path}")

            # Convert to markdown
            markdown_content = self.office_to_markdown(file_path)

            if not markdown_content.strip():
                self.logger.warning(f"No content extracted from: {file_path}")
                return False

            # Generate output filename
            doc_filename = Path(file_path).stem
            output_path = os.path.join(self.output_dir, f"{doc_filename}.md")

            # Write markdown file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            # Mark as processed
            self.file_tracker.mark_processed(
                file_path,
                chunk_count=0,
                status='success'
            )

            self.logger.info(
                f"Successfully converted {Path(file_path).name} -> {Path(output_path).name}"
            )

            return True

        except Exception as e:
            self.logger.error(
                f"Error processing {file_path}: {e}",
                exc_info=True
            )
            try:
                self.file_tracker.mark_processed(
                    file_path,
                    chunk_count=0,
                    status='error'
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
            'files_found': 0,
            'files_processed': 0,
            'files_skipped': 0,
            'files_failed': 0
        }

        # Find all Office documents
        office_files = self.find_office_files()
        stats['files_found'] = len(office_files)

        if not office_files:
            self.logger.debug(f"No Office documents found in {self.input_dir}")
            return stats

        # Process each file
        for file_path in office_files:
            if self.should_process_file(file_path):
                if self.process_office(file_path):
                    stats['files_processed'] += 1
                else:
                    stats['files_failed'] += 1
            else:
                stats['files_skipped'] += 1
                self.logger.debug(f"Skipping (already processed): {file_path}")

        return stats

    def run_continuous(self):
        """
        Run continuous monitoring loop.
        """
        self.logger.info(
            f"Starting continuous Office document monitoring "
            f"(poll interval: {self.poll_interval}s)"
        )
        self.logger.info("Press Ctrl+C to stop")

        scan_count = 0

        try:
            while True:
                scan_count += 1
                self.logger.debug(f"Scan #{scan_count}")

                stats = self.run_single_scan()

                if stats['files_processed'] > 0 or stats['files_failed'] > 0:
                    self.logger.info(
                        f"Scan #{scan_count} complete: "
                        f"{stats['files_processed']} processed, "
                        f"{stats['files_failed']} failed, "
                        f"{stats['files_skipped']} skipped"
                    )

                    # Show tracker statistics
                    tracker_stats = self.file_tracker.get_statistics()
                    self.logger.info(
                        f"Total: {tracker_stats['total_files']} documents processed"
                    )

                # Wait before next scan
                time.sleep(self.poll_interval)

        except KeyboardInterrupt:
            self.logger.info("\nShutting down gracefully...")
            self._shutdown()

    def _shutdown(self):
        """Cleanup and shutdown."""
        stats = self.file_tracker.get_statistics()
        self.logger.info("\n" + "="*60)
        self.logger.info("Final Statistics:")
        self.logger.info(f"  Total documents processed: {stats['total_files']}")
        self.logger.info(f"  Successful: {stats['successful']}")
        self.logger.info(f"  Errors: {stats['errors']}")
        self.logger.info("="*60)
        self.logger.info("Office Tracker stopped")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Office document monitoring and conversion to markdown',
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
        """
    )

    parser.add_argument(
        '--force-reprocess',
        type=str,
        metavar='PATTERN',
        help='Regex pattern to match document filenames for forced reprocessing. '
             'Runs in one-shot mode (single scan then exit) instead of continuous monitoring.'
    )

    parser.add_argument(
        '--input-dir',
        type=str,
        default="~/Documents/AI_IN",
        help='Directory to monitor for documents (default: ~/Documents/AI_IN)'
    )

    parser.add_argument(
        '--no-metadata',
        action='store_true',
        help='Disable YAML frontmatter metadata in output'
    )

    parser.add_argument(
        '--poll-interval',
        type=int,
        default=5,
        help='Seconds between directory scans (default: 5)'
    )

    args = parser.parse_args()

    try:
        tracker = OfficeTracker(
            input_dir=args.input_dir,
            poll_interval=args.poll_interval,
            reprocess_on_change=True,
            min_content_chars=100,
            use_ollama=True,
            ollama_model="llama3.2:1b",
            include_metadata=not args.no_metadata,
            force_reprocess_pattern=args.force_reprocess
        )

        # If force-reprocess is set, run single scan and exit
        if args.force_reprocess:
            tracker.logger.info("Running in one-shot mode (force reprocess enabled)")
            stats = tracker.run_single_scan()

            # Show results
            tracker.logger.info("="*60)
            tracker.logger.info("Single scan complete:")
            tracker.logger.info(f"  Files found: {stats['files_found']}")
            tracker.logger.info(f"  Files processed: {stats['files_processed']}")
            tracker.logger.info(f"  Files failed: {stats['files_failed']}")
            tracker.logger.info(f"  Files skipped: {stats['files_skipped']}")
            tracker.logger.info("="*60)

            # Show tracker statistics
            tracker_stats = tracker.file_tracker.get_statistics()
            tracker.logger.info(f"Total documents in database: {tracker_stats['total_files']}")

            sys.exit(0)
        else:
            # Run in continuous monitoring mode
            tracker.run_continuous()

    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
