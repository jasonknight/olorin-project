#!/usr/bin/env python3
"""
Pytest tests for markdown files without frontmatter.
"""

import pytest
from markdown_chunker import MarkdownChunker


class TestNoFrontmatter:
    """Test suite for markdown files without YAML frontmatter."""

    def test_no_frontmatter_basic(self, chunker, sample_markdown_no_frontmatter):
        """Test that files without frontmatter are processed correctly."""
        chunks = chunker.chunk_markdown(
            sample_markdown_no_frontmatter,
            "test_no_frontmatter.md"
        )

        assert len(chunks) > 0, "Should create chunks even without frontmatter"

        first_chunk = chunks[0]
        assert 'metadata' in first_chunk
        assert 'text' in first_chunk

        metadata = first_chunk['metadata']

        # Standard metadata should be present
        assert metadata['source'] == "test_no_frontmatter.md"
        assert 'chunk_index' in metadata
        assert 'char_count' in metadata

    def test_no_frontmatter_fields_absent(self, chunker, sample_markdown_no_frontmatter):
        """Test that frontmatter-specific fields are not present."""
        chunks = chunker.chunk_markdown(
            sample_markdown_no_frontmatter,
            "test_no_frontmatter.md"
        )

        metadata = chunks[0]['metadata']

        # Frontmatter fields should NOT be present
        assert 'title' not in metadata, "Should not have 'title' field"
        assert 'author' not in metadata, "Should not have 'author' field"
        assert 'keywords' not in metadata, "Should not have 'keywords' field"
        assert 'publish_date' not in metadata, "Should not have 'publish_date' field"

    def test_text_content_preserved(self, chunker, sample_markdown_no_frontmatter):
        """Test that text content is preserved correctly."""
        chunks = chunker.chunk_markdown(
            sample_markdown_no_frontmatter,
            "test_no_frontmatter.md"
        )

        # Combine all chunk texts
        combined_text = ' '.join(chunk['text'] for chunk in chunks)

        # Should contain expected content
        assert "Regular Document" in combined_text
        assert "Section 1" in combined_text
        assert "Section 2" in combined_text
        assert "regular markdown file" in combined_text

    def test_markdown_structure_preserved(self, chunker, sample_markdown_no_frontmatter):
        """Test that markdown header structure is preserved in metadata."""
        chunks = chunker.chunk_markdown(
            sample_markdown_no_frontmatter,
            "test_no_frontmatter.md"
        )

        # At least one chunk should have header metadata
        has_h1 = any('h1' in chunk['metadata'] for chunk in chunks)
        has_h2 = any('h2' in chunk['metadata'] for chunk in chunks)

        # Document has h1 and h2 headers
        assert has_h1 or has_h2, "Should preserve header hierarchy in metadata"

    def test_multiple_chunks_without_frontmatter(self, chunker):
        """Test chunking of longer content without frontmatter."""
        # Create content that will definitely be split into multiple chunks
        long_content = """# Main Title

## Section 1

""" + "Lorem ipsum dolor sit amet. " * 100 + """

## Section 2

""" + "Consectetur adipiscing elit. " * 100

        chunks = chunker.chunk_markdown(long_content, "long_doc.md")

        # Should create multiple chunks
        assert len(chunks) > 1, "Long content should be split into multiple chunks"

        # All chunks should have consistent metadata structure
        for chunk in chunks:
            assert 'source' in chunk['metadata']
            assert chunk['metadata']['source'] == "long_doc.md"
            assert 'chunk_index' in chunk['metadata']

    @pytest.mark.parametrize("content", [
        "# Simple Header\n\nSimple paragraph.",
        "Just plain text without any headers.",
        "## H2 First\n\nStarting with h2 instead of h1.",
    ])
    def test_various_markdown_without_frontmatter(self, chunker, content):
        """Test various markdown structures without frontmatter."""
        chunks = chunker.chunk_markdown(content, "test.md")

        assert len(chunks) > 0
        assert 'title' not in chunks[0]['metadata']
        assert 'author' not in chunks[0]['metadata']
        assert 'keywords' not in chunks[0]['metadata']
