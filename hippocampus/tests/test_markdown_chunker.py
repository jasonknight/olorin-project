#!/usr/bin/env python3
"""
Pytest tests for YAML frontmatter parsing in markdown_chunker.py
"""

import pytest
from markdown_chunker import MarkdownChunker


class TestFrontmatterParsing:
    """Test suite for YAML frontmatter parsing functionality."""

    def test_frontmatter_list_format(self, chunker, sample_markdown_with_frontmatter):
        """Test parsing frontmatter with list format (dashes)."""
        chunks = chunker.chunk_markdown(
            sample_markdown_with_frontmatter, "test_file_1.md"
        )

        # Assert basic chunk creation
        assert len(chunks) > 0, "Should create at least one chunk"

        # Assert first chunk has required metadata
        first_chunk = chunks[0]
        assert "metadata" in first_chunk
        assert "text" in first_chunk

        metadata = first_chunk["metadata"]

        # Assert frontmatter fields are present
        assert "title" in metadata
        assert metadata["title"] == "Example Research Paper"

        assert "author" in metadata
        assert metadata["author"] == "John Doe"

        assert "publish_date" in metadata
        assert metadata["publish_date"] == "2024-01-15"

        assert "keywords" in metadata
        # Keywords should be converted to comma-separated string
        keywords = metadata["keywords"]
        assert "machine learning" in keywords
        assert "neural networks" in keywords
        assert "deep learning" in keywords

        # Assert standard chunk metadata
        assert metadata["source"] == "test_file_1.md"
        assert "chunk_index" in metadata
        assert "char_count" in metadata

    def test_frontmatter_bracket_notation(self, chunker, sample_markdown_brackets):
        """Test parsing frontmatter with bracket notation for lists."""
        chunks = chunker.chunk_markdown(sample_markdown_brackets, "test_file_2.md")

        assert len(chunks) > 0

        metadata = chunks[0]["metadata"]

        # Assert bracket notation keywords are parsed
        assert "keywords" in metadata
        keywords = metadata["keywords"]
        assert "AI" in keywords
        assert "NLP" in keywords
        assert "transformers" in keywords

        assert metadata["title"] == "Another Paper"
        assert metadata["author"] == "Jane Smith"

    def test_metadata_in_all_chunks(self, chunker, sample_markdown_with_frontmatter):
        """Test that frontmatter metadata is propagated to all chunks."""
        chunks = chunker.chunk_markdown(
            sample_markdown_with_frontmatter, "test_file_1.md"
        )

        # All chunks should have frontmatter metadata
        for i, chunk in enumerate(chunks):
            metadata = chunk["metadata"]

            assert "title" in metadata, f"Chunk {i} missing title"
            assert "author" in metadata, f"Chunk {i} missing author"
            assert "keywords" in metadata, f"Chunk {i} missing keywords"

            # Verify values are consistent across chunks
            assert metadata["title"] == "Example Research Paper"
            assert metadata["author"] == "John Doe"

    def test_chunk_text_excludes_frontmatter(
        self, chunker, sample_markdown_with_frontmatter
    ):
        """Test that chunk text does not include YAML frontmatter delimiters."""
        chunks = chunker.chunk_markdown(
            sample_markdown_with_frontmatter, "test_file_1.md"
        )

        first_chunk_text = chunks[0]["text"]

        # Frontmatter delimiters should not appear at the start
        # (there might be --- in content, but not the YAML block)
        assert not first_chunk_text.strip().startswith("---")
        # Check that we don't have the raw YAML frontmatter keys
        assert 'title: "Example Research Paper"' not in first_chunk_text

    def test_empty_content(self, chunker):
        """Test handling of empty content."""
        chunks = chunker.chunk_markdown("", "empty.md")
        assert chunks == []

    def test_chunker_initialization(self):
        """Test MarkdownChunker initialization with custom parameters."""
        chunker = MarkdownChunker(
            chunk_size=1000, chunk_overlap=200, min_chunk_size=100
        )

        assert chunker.chunk_size == 1000
        assert chunker.chunk_overlap == 200
        assert chunker.min_chunk_size == 100


class TestFrontmatterEdgeCases:
    """Test edge cases in frontmatter parsing."""

    @pytest.mark.parametrize(
        "frontmatter,expected_title",
        [
            ('title: "Simple Title"', "Simple Title"),
            ("title: 'Single Quotes'", "Single Quotes"),
            ("title: No Quotes", "No Quotes"),
        ],
    )
    def test_title_quote_variations(self, chunker, frontmatter, expected_title):
        """Test different quote styles in frontmatter values."""
        markdown = f"""---
{frontmatter}
---

# Content
"""
        chunks = chunker.chunk_markdown(markdown, "test.md")
        assert len(chunks) > 0
        assert chunks[0]["metadata"]["title"] == expected_title

    def test_frontmatter_with_empty_keywords(self, chunker):
        """Test frontmatter with empty keywords list."""
        markdown = """---
title: "Test"
keywords: []
---

# Content
"""
        chunks = chunker.chunk_markdown(markdown, "test.md")
        # Empty list should result in keywords field present
        assert "keywords" in chunks[0]["metadata"]
