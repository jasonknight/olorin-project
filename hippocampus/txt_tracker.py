#!/usr/bin/env python3
"""
Text file monitoring and conversion script.
Monitors ~/Documents/AI_IN for TXT files, converts them to markdown,
and tracks processed files to avoid reprocessing.
"""

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Add parent directory to path for libs import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from document_tracker_base import DocumentTrackerBase


class TxtTracker(DocumentTrackerBase):
    """
    Monitors directory for TXT files and converts them to markdown.
    """

    # Class attributes required by base class
    EXTENSIONS = {".txt"}
    TRACKER_NAME = "TXT"
    LOG_FILENAME = "hippocampus-txt-tracker.log"
    TRACKING_DB_CONFIG_KEY = "TXT_TRACKING_DB"

    def __init__(
        self,
        detect_structure: bool = True,
        **kwargs,
    ):
        """
        Initialize TXT tracker.

        Args:
            detect_structure: Attempt to detect and convert text structure to markdown
            **kwargs: Arguments passed to DocumentTrackerBase
        """
        # TXT-specific configuration
        self.detect_structure = detect_structure

        # Initialize base class
        super().__init__(**kwargs)

    # =========================================================================
    # Abstract method implementations
    # =========================================================================

    def find_files(self) -> List[str]:
        """Find all TXT files in the input directory."""
        return self._find_files_by_extensions()

    def to_markdown(self, txt_path: str) -> str:
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
                with open(txt_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError:
                with open(txt_path, "r", encoding="latin-1") as f:
                    content = f.read()
        except Exception as e:
            self.logger.error(f"Error reading file {txt_path}: {e}")
            return ""

        # Normalize text
        content = self._normalize_text(content)

        if not self._is_content_substantial(content):
            self.logger.warning(f"Content not substantial: {txt_path}")
            return ""

        # Extract metadata and build frontmatter
        if self.include_metadata:
            metadata = self._extract_metadata(txt_path, content)
            markdown_parts.append(self._build_yaml_frontmatter(metadata))
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

        return "".join(markdown_parts)

    def process_file(self, txt_path: str) -> bool:
        """
        Process a single TXT file.

        Args:
            txt_path: Path to TXT file

        Returns:
            True if processing succeeded
        """
        try:
            self.logger.info(f"Processing: {txt_path}")

            # Send notification to Broca when starting to process
            if self.notify_broca and self.broca_producer:
                filename = Path(txt_path).stem
                clean_title = self._guess_title_from_filename(filename)
                notification = f"Now processing: {clean_title}"
                self._send_broca_notification(notification)

            # Convert TXT to markdown
            markdown_content = self.to_markdown(txt_path)

            if not markdown_content.strip():
                self.logger.warning(f"No content extracted from: {txt_path}")
                return False

            # Generate output filename
            txt_filename = Path(txt_path).stem
            output_path = os.path.join(self.output_dir, f"{txt_filename}.md")

            # Write markdown file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)

            # Mark as processed
            self.file_tracker.mark_processed(txt_path, chunk_count=0, status="success")

            self.logger.info(
                f"Successfully converted {Path(txt_path).name} -> {Path(output_path).name}"
            )

            return True

        except Exception as e:
            self.logger.error(f"Error processing {txt_path}: {e}", exc_info=True)
            try:
                self.file_tracker.mark_processed(
                    txt_path, chunk_count=0, status="error"
                )
            except Exception:
                pass

            return False

    def _extract_metadata(self, txt_path: str, content: str = None) -> Dict[str, str]:
        """
        Extract metadata from text file.

        Args:
            txt_path: Path to text file
            content: File content (optional, will be read if not provided)

        Returns:
            Dictionary containing metadata fields
        """
        metadata = {}

        # Read content if not provided
        if content is None:
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError:
                with open(txt_path, "r", encoding="latin-1") as f:
                    content = f.read()
            except Exception:
                content = ""

        # Title from filename
        metadata["title"] = Path(txt_path).stem

        # Try to detect title from first non-empty line
        lines = content.strip().split("\n")
        for line in lines[:5]:
            stripped = line.strip()
            if stripped and len(stripped) < 100:
                # Looks like a title if it's short and doesn't end with punctuation
                if not re.search(r"[.!?,;:]$", stripped):
                    metadata["title"] = stripped
                    break

        # File modification date
        try:
            mtime = os.path.getmtime(txt_path)
            metadata["file_date"] = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
        except Exception:
            metadata["file_date"] = "unknown"

        # Source
        metadata["source"] = "txt"

        # Word count
        words = re.findall(r"\b\w+\b", content)
        metadata["word_count"] = str(len(words))

        return metadata

    def _get_title_prompt_examples(self) -> str:
        """Return examples for the title extraction LLM prompt."""
        return """- "meeting_notes_2024_q3_final" → "Meeting Notes Q3 2024"
- "readme-project-alpha" → "Readme for Project Alpha"
- "john_doe_cover_letter" → "Cover Letter by John Doe"""

    # =========================================================================
    # TXT-specific methods
    # =========================================================================

    def _detect_headings(self, text: str) -> str:
        """
        Detect and convert potential headings to markdown format.

        Args:
            text: Plain text content

        Returns:
            Text with detected headings converted to markdown
        """
        lines = text.split("\n")
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
                if re.match(r"^={3,}$", next_line):
                    result.append(f"# {stripped}")
                    lines[i + 1] = ""  # Clear the underline
                    continue
                elif re.match(r"^-{3,}$", next_line):
                    result.append(f"## {stripped}")
                    lines[i + 1] = ""  # Clear the underline
                    continue

            # Detect ALL CAPS lines as potential headings (short lines only)
            if (
                stripped.isupper()
                and len(stripped) > 3
                and len(stripped) < 80
                and not stripped.startswith("#")
            ):
                # Check if it looks like a heading (no punctuation at end except colon)
                if not re.search(r"[.!?,;]$", stripped):
                    result.append(f"## {stripped.title()}")
                    continue

            # Detect numbered section headings like "1. Introduction" or "1.2 Methods"
            heading_match = re.match(
                r"^(\d+\.(?:\d+\.?)*)\s+([A-Z][^.!?]*?)$", stripped
            )
            if heading_match:
                level = heading_match.group(1).count(".") + 1
                level = min(level, 4)  # Cap at h4
                heading_text = heading_match.group(2).strip()
                result.append(f"{'#' * level} {heading_match.group(1)} {heading_text}")
                continue

            result.append(line)

        return "\n".join(result)

    def _detect_lists(self, text: str) -> str:
        """
        Detect and convert potential lists to markdown format.

        Args:
            text: Text content

        Returns:
            Text with detected lists converted to markdown
        """
        lines = text.split("\n")
        result = []

        for line in lines:
            stripped = line.strip()

            # Already a markdown list
            if re.match(r"^[-*+]\s", stripped) or re.match(r"^\d+\.\s", stripped):
                result.append(line)
                continue

            # Detect bullet-like patterns
            # - Lines starting with *, -, o, bullet, >, etc.
            bullet_match = re.match(r"^[*\-o\u2022>]\s+(.+)$", stripped)
            if bullet_match:
                result.append(f"- {bullet_match.group(1)}")
                continue

            # Detect numbered lists (1), 1., a), a., etc.
            numbered_match = re.match(
                r"^(?:\d+[.\)]\s*|[a-zA-Z][.\)]\s*)(.+)$", stripped
            )
            if (
                numbered_match and len(stripped) < 200
            ):  # Avoid false positives on long lines
                # Check if content looks like a list item
                content = numbered_match.group(1)
                if len(content) < 100 or not content.endswith("."):
                    result.append(f"- {content}")
                    continue

            result.append(line)

        return "\n".join(result)

    def _detect_code_blocks(self, text: str) -> str:
        """
        Detect and wrap potential code blocks.

        Args:
            text: Text content

        Returns:
            Text with detected code blocks wrapped in markdown code fences
        """
        lines = text.split("\n")
        result = []
        in_code_block = False
        code_buffer = []

        for line in lines:
            # Detect lines that look like code (indented with 4+ spaces or tabs)
            is_code_line = (
                line.startswith("    ")
                or line.startswith("\t")
                or re.match(r"^[ \t]*[{}\[\]();]", line)
                or re.match(
                    r"^[ \t]*(def |class |function |if |for |while |import |from |var |let |const )",
                    line,
                )
            )

            # Also detect lines with common programming patterns
            if not is_code_line and line.strip():
                code_patterns = [
                    r"^\s*#include\s*<",
                    r"^\s*#define\s+",
                    r"^\s*public\s+|private\s+|protected\s+",
                    r"^\s*return\s+",
                    r"=\s*function\s*\(",
                    r"=>\s*{",
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
                        result.append("```")
                        result.extend(code_buffer)
                        result.append("```")
                    else:
                        result.extend(code_buffer)
                    in_code_block = False
                    code_buffer = []
                result.append(line)

        # Handle remaining code buffer
        if code_buffer:
            if len(code_buffer) >= 2:
                result.append("```")
                result.extend(code_buffer)
                result.append("```")
            else:
                result.extend(code_buffer)

        return "\n".join(result)

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
                [self.ollama_path, "run", self.ollama_model, prompt],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0 and result.stdout.strip():
                enhanced = result.stdout.strip()
                # Validate output is reasonable
                if len(enhanced) >= len(text) * 0.5:
                    return enhanced

        except (subprocess.TimeoutExpired, Exception) as e:
            self.logger.debug(f"LLM enhancement failed: {e}")

        return text


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="TXT file monitoring and conversion to markdown",
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
        """,
    )

    parser.add_argument(
        "--force-reprocess",
        type=str,
        metavar="PATTERN",
        help="Regex pattern to match TXT filenames for forced reprocessing. "
        "Runs in one-shot mode (single scan then exit) instead of continuous monitoring.",
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory to monitor for TXT files (default: from config)",
    )

    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Disable YAML frontmatter metadata in output",
    )

    parser.add_argument(
        "--no-structure",
        action="store_true",
        help="Disable structure detection (headings, lists, code blocks)",
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
        tracker = TxtTracker(
            input_dir=args.input_dir,
            poll_interval=args.poll_interval,
            include_metadata=not args.no_metadata,
            force_reprocess_pattern=args.force_reprocess,
            detect_structure=not args.no_structure,
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
                f"Total TXT files in database: {tracker_stats['total_files']}"
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
