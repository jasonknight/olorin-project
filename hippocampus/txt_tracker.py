#!/usr/bin/env python3
"""
Text file monitoring and conversion script.
Monitors ~/Documents/AI_IN for TXT files, converts them to markdown,
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
from datetime import datetime

# Add parent directory to path for libs import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from libs.olorin_logging import OlorinLogger

from file_tracker import FileTracker


class TxtTracker:
    """
    Monitors directory for TXT files and converts them to markdown.
    """

    def __init__(
        self,
        input_dir: str = "~/Documents/AI_IN",
        output_dir: str = None,
        tracking_db: str = "./data/txt_tracking.db",
        poll_interval: int = 5,
        reprocess_on_change: bool = True,
        min_content_chars: int = 50,
        min_content_density: float = 0.2,
        use_ollama: bool = True,
        ollama_model: str = "llama3.2:1b",
        ollama_threshold: float = 0.5,
        include_metadata: bool = True,
        force_reprocess_pattern: str = None,
        detect_structure: bool = True
    ):
        """
        Initialize TXT tracker.

        Args:
            input_dir: Directory to monitor for text files
            output_dir: Directory to save markdown files (defaults to input_dir)
            tracking_db: Path to SQLite tracking database
            poll_interval: Seconds between directory scans
            reprocess_on_change: Reprocess if text file content changes
            min_content_chars: Minimum characters for content to be considered
            min_content_density: Minimum ratio of alphanumeric to total chars
            use_ollama: Enable local LLM for structure detection
            ollama_model: Ollama model to use (e.g., llama3.2:1b, phi3:mini)
            ollama_threshold: Minimum relevance score (0-1) to include content
            include_metadata: Include YAML frontmatter with document metadata
            force_reprocess_pattern: Regex pattern to match filenames for forced reprocessing
            detect_structure: Attempt to detect and convert text structure to markdown
        """
        self.input_dir = os.path.expanduser(input_dir)
        self.output_dir = os.path.expanduser(output_dir) if output_dir else self.input_dir
        self.poll_interval = poll_interval
        self.reprocess_on_change = reprocess_on_change

        # Cleaning configuration
        self.min_content_chars = min_content_chars
        self.min_content_density = min_content_density
        self.use_ollama = use_ollama
        self.ollama_model = ollama_model
        self.ollama_threshold = ollama_threshold
        self.include_metadata = include_metadata
        self.detect_structure = detect_structure

        # Force reprocess pattern
        self.force_reprocess_regex = None
        if force_reprocess_pattern:
            try:
                self.force_reprocess_regex = re.compile(force_reprocess_pattern)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern: {force_reprocess_pattern} - {e}")

        # Setup logging
        default_log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')
        log_file = os.path.join(default_log_dir, 'hippocampus-txt-tracker.log')
        self.logger = OlorinLogger(log_file=log_file, log_level='INFO', name=__name__)

        # Check Ollama availability
        self.ollama_available = False
        if self.use_ollama:
            self.ollama_available = self._check_ollama()
            if self.ollama_available:
                self.logger.info(f"Ollama detected - using model: {self.ollama_model}")
            else:
                self.logger.warning("Ollama not available - falling back to heuristics only")

        # Create directories if they don't exist
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize file tracker
        self.file_tracker = FileTracker(tracking_db)

        self.logger.info(f"TXT Tracker initialized")
        self.logger.info(f"Monitoring directory: {self.input_dir}")
        self.logger.info(f"Output directory: {self.output_dir}")
        if self.force_reprocess_regex:
            self.logger.info(f"Force reprocess pattern: {force_reprocess_pattern}")

    def find_txt_files(self) -> List[str]:
        """
        Find all TXT files in the input directory.

        Returns:
            List of absolute file paths
        """
        txt_files = []
        input_path = Path(self.input_dir)

        for txt_file in input_path.rglob("*.txt"):
            if txt_file.is_file():
                txt_files.append(str(txt_file.absolute()))

        for txt_file in input_path.rglob("*.TXT"):
            if txt_file.is_file():
                txt_files.append(str(txt_file.absolute()))

        return txt_files

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

    def _is_content_substantial(self, text: str) -> bool:
        """
        Determine if content has substantial text using heuristics.

        Args:
            text: Text to analyze

        Returns:
            True if content meets minimum quality thresholds
        """
        if not text or not text.strip():
            return False

        cleaned_text = text.strip()
        if len(cleaned_text) < self.min_content_chars:
            return False

        density = self._calculate_content_density(cleaned_text)
        if density < self.min_content_density:
            return False

        words = re.findall(r'\b\w+\b', cleaned_text)
        if len(words) < 10:
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
        # Remove form feed and other control characters
        text = text.replace('\f', '\n')
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # Normalize whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

        return text.strip()

    def _detect_headings(self, text: str) -> str:
        """
        Detect and convert potential headings to markdown format.

        Args:
            text: Plain text content

        Returns:
            Text with detected headings converted to markdown
        """
        lines = text.split('\n')
        result = []

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Skip empty lines
            if not stripped:
                result.append(line)
                continue

            # Detect underlined headings (==== or ----)
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if re.match(r'^={3,}$', next_line):
                    result.append(f'# {stripped}')
                    lines[i + 1] = ''  # Clear the underline
                    continue
                elif re.match(r'^-{3,}$', next_line):
                    result.append(f'## {stripped}')
                    lines[i + 1] = ''  # Clear the underline
                    continue

            # Detect ALL CAPS lines as potential headings (short lines only)
            if (stripped.isupper() and
                len(stripped) > 3 and
                len(stripped) < 80 and
                not stripped.startswith('#')):
                # Check if it looks like a heading (no punctuation at end except colon)
                if not re.search(r'[.!?,;]$', stripped):
                    result.append(f'## {stripped.title()}')
                    continue

            # Detect numbered section headings like "1. Introduction" or "1.2 Methods"
            heading_match = re.match(r'^(\d+\.(?:\d+\.?)*)\s+([A-Z][^.!?]*?)$', stripped)
            if heading_match:
                level = heading_match.group(1).count('.') + 1
                level = min(level, 4)  # Cap at h4
                heading_text = heading_match.group(2).strip()
                result.append(f'{"#" * level} {heading_match.group(1)} {heading_text}')
                continue

            result.append(line)

        return '\n'.join(result)

    def _detect_lists(self, text: str) -> str:
        """
        Detect and convert potential lists to markdown format.

        Args:
            text: Text content

        Returns:
            Text with detected lists converted to markdown
        """
        lines = text.split('\n')
        result = []

        for line in lines:
            stripped = line.strip()

            # Already a markdown list
            if re.match(r'^[-*+]\s', stripped) or re.match(r'^\d+\.\s', stripped):
                result.append(line)
                continue

            # Detect bullet-like patterns
            # - Lines starting with *, -, o, •, >, etc.
            bullet_match = re.match(r'^[*\-o•>]\s+(.+)$', stripped)
            if bullet_match:
                result.append(f'- {bullet_match.group(1)}')
                continue

            # Detect numbered lists (1), 1., a), a., etc.
            numbered_match = re.match(r'^(?:\d+[.\)]\s*|[a-zA-Z][.\)]\s*)(.+)$', stripped)
            if numbered_match and len(stripped) < 200:  # Avoid false positives on long lines
                # Check if content looks like a list item (doesn't end with period usually, or is short)
                content = numbered_match.group(1)
                if len(content) < 100 or not content.endswith('.'):
                    result.append(f'- {content}')
                    continue

            result.append(line)

        return '\n'.join(result)

    def _detect_code_blocks(self, text: str) -> str:
        """
        Detect and wrap potential code blocks.

        Args:
            text: Text content

        Returns:
            Text with detected code blocks wrapped in markdown code fences
        """
        lines = text.split('\n')
        result = []
        in_code_block = False
        code_buffer = []

        for line in lines:
            # Detect lines that look like code (indented with 4+ spaces or tabs)
            is_code_line = (
                line.startswith('    ') or
                line.startswith('\t') or
                re.match(r'^[ \t]*[{}\[\]();]', line) or  # Lines with common code chars
                re.match(r'^[ \t]*(def |class |function |if |for |while |import |from |var |let |const )', line)
            )

            # Also detect lines with common programming patterns
            if not is_code_line and line.strip():
                code_patterns = [
                    r'^\s*#include\s*<',
                    r'^\s*#define\s+',
                    r'^\s*public\s+|private\s+|protected\s+',
                    r'^\s*return\s+',
                    r'=\s*function\s*\(',
                    r'=>\s*{',
                ]
                for pattern in code_patterns:
                    if re.search(pattern, line):
                        is_code_line = True
                        break

            if is_code_line:
                if not in_code_block:
                    in_code_block = True
                    code_buffer = []
                code_buffer.append(line)
            else:
                if in_code_block:
                    # End of code block
                    if len(code_buffer) >= 2:  # Only wrap if multiple lines
                        result.append('```')
                        result.extend(code_buffer)
                        result.append('```')
                    else:
                        result.extend(code_buffer)
                    in_code_block = False
                    code_buffer = []
                result.append(line)

        # Handle remaining code buffer
        if code_buffer:
            if len(code_buffer) >= 2:
                result.append('```')
                result.extend(code_buffer)
                result.append('```')
            else:
                result.extend(code_buffer)

        return '\n'.join(result)

    def _enhance_with_llm(self, text: str) -> str:
        """
        Use LLM to enhance markdown structure detection.

        Args:
            text: Text to enhance

        Returns:
            Enhanced text with better markdown structure
        """
        if not self.ollama_available:
            return text

        # Only use LLM for shorter texts to avoid timeouts
        if len(text) > 3000:
            return text

        prompt = f"""Convert this plain text to well-formatted markdown.
Preserve all content but add appropriate markdown formatting:
- Use # ## ### for headings
- Use - or * for bullet lists
- Use 1. 2. 3. for numbered lists
- Use ``` for code blocks
- Use **bold** and *italic* where appropriate
- Use > for quotes

Only output the formatted markdown, no explanations.

Text:
{text[:2500]}"""

        try:
            result = subprocess.run(
                [self.ollama_path, 'run', self.ollama_model, prompt],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0 and result.stdout.strip():
                enhanced = result.stdout.strip()
                # Validate output is reasonable
                if len(enhanced) >= len(text) * 0.5:  # At least half the original length
                    return enhanced

        except (subprocess.TimeoutExpired, Exception) as e:
            self.logger.debug(f"LLM enhancement failed: {e}")

        return text

    def _extract_metadata(self, txt_path: str, content: str) -> Dict[str, str]:
        """
        Extract metadata from text file.

        Args:
            txt_path: Path to text file
            content: File content

        Returns:
            Dictionary containing metadata fields
        """
        metadata = {}

        # Title from filename
        metadata['title'] = Path(txt_path).stem

        # Try to detect title from first non-empty line
        lines = content.strip().split('\n')
        for line in lines[:5]:
            stripped = line.strip()
            if stripped and len(stripped) < 100:
                # Looks like a title if it's short and doesn't end with punctuation
                if not re.search(r'[.!?,;:]$', stripped):
                    metadata['title'] = stripped
                    break

        # File modification date
        try:
            mtime = os.path.getmtime(txt_path)
            metadata['file_date'] = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d')
        except Exception:
            metadata['file_date'] = "unknown"

        # Source
        metadata['source'] = "txt"

        # Word count
        words = re.findall(r'\b\w+\b', content)
        metadata['word_count'] = str(len(words))

        return metadata

    def should_process_file(self, file_path: str) -> bool:
        """
        Determine if a text file should be processed.

        Args:
            file_path: Path to text file

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

    def txt_to_markdown(self, txt_path: str) -> str:
        """
        Convert TXT to markdown text with intelligent structure detection.

        Args:
            txt_path: Path to TXT file

        Returns:
            Markdown formatted text
        """
        markdown_parts = []

        # Read file content
        try:
            # Try UTF-8 first, fallback to latin-1
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                with open(txt_path, 'r', encoding='latin-1') as f:
                    content = f.read()
        except Exception as e:
            self.logger.error(f"Error reading file {txt_path}: {e}")
            return ""

        # Normalize text
        content = self._normalize_text(content)

        if not self._is_content_substantial(content):
            self.logger.warning(f"Content not substantial: {txt_path}")
            return ""

        # Extract metadata
        if self.include_metadata:
            metadata = self._extract_metadata(txt_path, content)

            markdown_parts.append("---\n")
            markdown_parts.append(f"title: \"{metadata['title']}\"\n")
            markdown_parts.append(f"source: \"{metadata['source']}\"\n")
            markdown_parts.append(f"file_date: \"{metadata['file_date']}\"\n")
            markdown_parts.append(f"word_count: {metadata['word_count']}\n")
            markdown_parts.append("---\n\n")
            markdown_parts.append(f"# {metadata['title']}\n\n")
        else:
            filename = Path(txt_path).stem
            markdown_parts.append(f"# {filename}\n\n")
            markdown_parts.append("---\n\n")

        # Apply structure detection
        if self.detect_structure:
            # First try LLM enhancement for short texts
            if self.ollama_available and len(content) <= 3000:
                enhanced = self._enhance_with_llm(content)
                if enhanced != content:
                    self.logger.debug("Applied LLM enhancement")
                    content = enhanced
                else:
                    # Fall back to heuristic detection
                    content = self._detect_headings(content)
                    content = self._detect_lists(content)
                    content = self._detect_code_blocks(content)
            else:
                # Use heuristic detection for longer texts
                content = self._detect_headings(content)
                content = self._detect_lists(content)
                content = self._detect_code_blocks(content)

        # Final normalization
        content = self._normalize_text(content)

        markdown_parts.append(content)

        return ''.join(markdown_parts)

    def process_txt(self, txt_path: str) -> bool:
        """
        Process a single TXT file.

        Args:
            txt_path: Path to TXT file

        Returns:
            True if processing succeeded
        """
        try:
            self.logger.info(f"Processing: {txt_path}")

            # Convert TXT to markdown
            markdown_content = self.txt_to_markdown(txt_path)

            if not markdown_content.strip():
                self.logger.warning(f"No content extracted from: {txt_path}")
                return False

            # Generate output filename
            txt_filename = Path(txt_path).stem
            output_path = os.path.join(self.output_dir, f"{txt_filename}.md")

            # Write markdown file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            # Mark as processed
            self.file_tracker.mark_processed(
                txt_path,
                chunk_count=0,
                status='success'
            )

            self.logger.info(
                f"Successfully converted {Path(txt_path).name} -> {Path(output_path).name}"
            )

            return True

        except Exception as e:
            self.logger.error(
                f"Error processing {txt_path}: {e}",
                exc_info=True
            )
            try:
                self.file_tracker.mark_processed(
                    txt_path,
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

        txt_files = self.find_txt_files()
        stats['files_found'] = len(txt_files)

        if not txt_files:
            self.logger.debug(f"No TXT files found in {self.input_dir}")
            return stats

        for file_path in txt_files:
            if self.should_process_file(file_path):
                if self.process_txt(file_path):
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
            f"Starting continuous TXT monitoring "
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

                    tracker_stats = self.file_tracker.get_statistics()
                    self.logger.info(
                        f"Total: {tracker_stats['total_files']} TXT files processed"
                    )

                time.sleep(self.poll_interval)

        except KeyboardInterrupt:
            self.logger.info("\nShutting down gracefully...")
            self._shutdown()

    def _shutdown(self):
        """Cleanup and shutdown."""
        stats = self.file_tracker.get_statistics()
        self.logger.info("\n" + "="*60)
        self.logger.info("Final Statistics:")
        self.logger.info(f"  Total TXT files processed: {stats['total_files']}")
        self.logger.info(f"  Successful: {stats['successful']}")
        self.logger.info(f"  Errors: {stats['errors']}")
        self.logger.info("="*60)
        self.logger.info("TXT Tracker stopped")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='TXT file monitoring and conversion to markdown',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run in continuous monitoring mode (default)
  python txt_tracker.py

  # Force reprocess all TXT files starting with "notes" (one-shot mode)
  python txt_tracker.py --force-reprocess "^notes"

  # Force reprocess TXT files containing "2024" (one-shot mode)
  python txt_tracker.py --force-reprocess "2024"

  # Disable structure detection (just wrap in markdown)
  python txt_tracker.py --no-structure

Note: When --force-reprocess is provided, the script runs once and exits.
      Without it, the script runs in continuous monitoring mode.
        """
    )

    parser.add_argument(
        '--force-reprocess',
        type=str,
        metavar='PATTERN',
        help='Regex pattern to match TXT filenames for forced reprocessing. '
             'Runs in one-shot mode (single scan then exit) instead of continuous monitoring. '
             'Example patterns: "^notes" or "2024" or "filename\\.txt$"'
    )

    parser.add_argument(
        '--input-dir',
        type=str,
        default="~/Documents/AI_IN",
        help='Directory to monitor for TXT files (default: ~/Documents/AI_IN)'
    )

    parser.add_argument(
        '--no-metadata',
        action='store_true',
        help='Disable YAML frontmatter metadata in output'
    )

    parser.add_argument(
        '--no-structure',
        action='store_true',
        help='Disable structure detection (headings, lists, code blocks)'
    )

    parser.add_argument(
        '--poll-interval',
        type=int,
        default=5,
        help='Seconds between directory scans (default: 5)'
    )

    args = parser.parse_args()

    try:
        tracker = TxtTracker(
            input_dir=args.input_dir,
            poll_interval=args.poll_interval,
            reprocess_on_change=True,
            min_content_chars=50,
            min_content_density=0.2,
            use_ollama=True,
            ollama_model="llama3.2:1b",
            ollama_threshold=0.5,
            include_metadata=not args.no_metadata,
            force_reprocess_pattern=args.force_reprocess,
            detect_structure=not args.no_structure
        )

        if args.force_reprocess:
            tracker.logger.info("Running in one-shot mode (force reprocess enabled)")
            stats = tracker.run_single_scan()

            tracker.logger.info("="*60)
            tracker.logger.info("Single scan complete:")
            tracker.logger.info(f"  Files found: {stats['files_found']}")
            tracker.logger.info(f"  Files processed: {stats['files_processed']}")
            tracker.logger.info(f"  Files failed: {stats['files_failed']}")
            tracker.logger.info(f"  Files skipped: {stats['files_skipped']}")
            tracker.logger.info("="*60)

            tracker_stats = tracker.file_tracker.get_statistics()
            tracker.logger.info(f"Total TXT files in database: {tracker_stats['total_files']}")

            sys.exit(0)
        else:
            tracker.run_continuous()

    except Exception as e:
        import logging
        logging.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
