#!/usr/bin/env python3
"""
Semantic markdown chunker that preserves document structure.
Splits markdown files intelligently based on headers and paragraphs.
"""

from typing import List, Dict, Tuple, Optional
import re
import logging
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class MarkdownChunker:
    """
    Semantic chunker for markdown documents.
    Preserves structure by respecting headers and logical sections.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100
    ):
        """
        Initialize markdown chunker.

        Args:
            chunk_size: Target size for chunks in characters
            chunk_overlap: Overlap between chunks in characters
            min_chunk_size: Minimum chunk size in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

        # Define headers to split on (in order of priority)
        self.headers_to_split = [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
            ("####", "h4"),
            ("#####", "h5"),
            ("######", "h6"),
        ]

        logger.debug(
            f"Chunker initialized: size={chunk_size}, "
            f"overlap={chunk_overlap}, min={min_chunk_size}"
        )

    def _parse_yaml_frontmatter(self, content: str) -> Tuple[Optional[Dict], str]:
        """
        Parse YAML frontmatter from markdown content.

        Args:
            content: Markdown content that may contain YAML frontmatter

        Returns:
            Tuple of (metadata_dict, content_without_frontmatter)
            If no frontmatter found, returns (None, original_content)
        """
        # Check if content starts with YAML frontmatter delimiter
        if not content.startswith('---'):
            return None, content

        # Find the closing delimiter
        # Look for second occurrence of '---' on its own line
        lines = content.split('\n')
        frontmatter_end = -1

        for i in range(1, len(lines)):
            if lines[i].strip() == '---':
                frontmatter_end = i
                break

        if frontmatter_end == -1:
            # No closing delimiter found
            return None, content

        # Extract frontmatter content (between delimiters)
        frontmatter_lines = lines[1:frontmatter_end]
        frontmatter_text = '\n'.join(frontmatter_lines)

        # Extract content after frontmatter
        content_lines = lines[frontmatter_end + 1:]
        remaining_content = '\n'.join(content_lines)

        # Parse YAML manually (simple key-value parsing)
        metadata = {}
        current_key = None
        current_list = []
        in_list = False

        for line in frontmatter_lines:
            # Check for list items
            if line.strip().startswith('- '):
                if in_list:
                    # Extract value from list item: - "value" or - value
                    value = line.strip()[2:].strip()
                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    current_list.append(value)
                continue

            # Check for key-value pairs
            if ':' in line:
                # Save previous list if any
                if in_list and current_key:
                    metadata[current_key] = current_list
                    current_list = []
                    in_list = False

                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()

                # Check if this starts a list
                if value == '' or value == '[]':
                    in_list = True
                    current_key = key
                    current_list = []
                    if value == '[]':
                        # Empty list notation
                        metadata[key] = []
                        in_list = False
                    continue

                # Check if value is a list in bracket notation: ["item1", "item2"]
                if value.startswith('[') and value.endswith(']'):
                    # Parse bracket list
                    list_content = value[1:-1]
                    if list_content:
                        items = [item.strip().strip('"').strip("'")
                                for item in list_content.split(',')]
                        metadata[key] = items
                    else:
                        metadata[key] = []
                    continue

                # Remove quotes from value if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]

                metadata[key] = value

        # Save final list if any
        if in_list and current_key:
            metadata[current_key] = current_list

        logger.debug(f"Parsed frontmatter metadata: {metadata}")
        return metadata, remaining_content

    def chunk_markdown(self, content: str, source_file: str = "") -> List[Dict]:
        """
        Chunk markdown content semantically.

        Args:
            content: Markdown content to chunk
            source_file: Source file path for metadata

        Returns:
            List of dictionaries containing chunk text and metadata
        """
        if not content.strip():
            logger.warning(f"Empty content for {source_file}")
            return []

        try:
            # Parse YAML frontmatter if present
            document_metadata, content_without_frontmatter = self._parse_yaml_frontmatter(content)

            if document_metadata:
                logger.info(f"Extracted document metadata from {source_file}: {list(document_metadata.keys())}")
            else:
                document_metadata = {}

            # Use content without frontmatter for chunking
            content_to_chunk = content_without_frontmatter

            # First pass: Split by markdown headers
            header_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=self.headers_to_split,
                strip_headers=False  # Keep headers in chunks for context
            )

            header_splits = header_splitter.split_text(content_to_chunk)

            # Second pass: Further split large sections
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=[
                    "\n\n",  # Paragraph breaks
                    "\n",    # Line breaks
                    ". ",    # Sentences
                    " ",     # Words
                    ""       # Characters
                ]
            )

            chunks = []
            for idx, doc in enumerate(header_splits):
                # Get header hierarchy from metadata
                headers = doc.metadata if hasattr(doc, 'metadata') else {}

                # Split large sections further
                if len(doc.page_content) > self.chunk_size:
                    sub_chunks = text_splitter.split_text(doc.page_content)
                else:
                    sub_chunks = [doc.page_content]

                # Create chunk dictionaries with metadata
                for sub_idx, chunk_text in enumerate(sub_chunks):
                    # Skip tiny chunks unless they're the only content
                    if len(chunk_text.strip()) < self.min_chunk_size and len(sub_chunks) > 1:
                        logger.debug(
                            f"Skipping small chunk ({len(chunk_text)} chars) "
                            f"from {source_file}"
                        )
                        continue

                    # Prepare chunk metadata
                    chunk_metadata = {
                        'source': source_file,
                        'chunk_index': len(chunks),
                        'section_index': idx,
                        'sub_index': sub_idx,
                        'char_count': len(chunk_text.strip()),
                        **headers,  # Include header hierarchy
                    }

                    # Add document metadata, converting lists to comma-separated strings
                    # ChromaDB metadata works best with simple types
                    for key, value in document_metadata.items():
                        if isinstance(value, list):
                            # Convert list to comma-separated string
                            chunk_metadata[key] = ', '.join(str(v) for v in value)
                        else:
                            chunk_metadata[key] = value

                    chunk = {
                        'text': chunk_text.strip(),
                        'metadata': chunk_metadata
                    }

                    chunks.append(chunk)

            logger.info(
                f"Created {len(chunks)} chunks from {source_file} "
                f"({len(content)} chars)"
            )

            return chunks

        except Exception as e:
            logger.error(f"Error chunking {source_file}: {e}", exc_info=True)
            # Try to parse frontmatter even in error case
            try:
                document_metadata, _ = self._parse_yaml_frontmatter(content)
                if not document_metadata:
                    document_metadata = {}
            except Exception:
                document_metadata = {}

            # Prepare fallback metadata
            fallback_metadata = {
                'source': source_file,
                'chunk_index': 0,
                'char_count': len(content.strip()),
                'error': 'chunking_failed'
            }

            # Add document metadata, converting lists to comma-separated strings
            for key, value in document_metadata.items():
                if isinstance(value, list):
                    fallback_metadata[key] = ', '.join(str(v) for v in value)
                else:
                    fallback_metadata[key] = value

            # Fallback: create single chunk
            return [{
                'text': content.strip(),
                'metadata': fallback_metadata
            }]

    def extract_title(self, content: str) -> str:
        """
        Extract title from markdown (first h1 or filename).

        Args:
            content: Markdown content

        Returns:
            Extracted title or empty string
        """
        # Look for first h1 header
        match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if match:
            return match.group(1).strip()

        # Look for any header at start
        match = re.search(r'^#{1,6}\s+(.+)$', content, re.MULTILINE)
        if match:
            return match.group(1).strip()

        return ""

    def get_chunk_preview(self, chunk: Dict, max_length: int = 100) -> str:
        """
        Get a preview of chunk text.

        Args:
            chunk: Chunk dictionary
            max_length: Maximum preview length

        Returns:
            Preview string
        """
        text = chunk['text']
        if len(text) <= max_length:
            return text

        return text[:max_length] + "..."
