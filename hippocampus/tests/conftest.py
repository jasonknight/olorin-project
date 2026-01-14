#!/usr/bin/env python3
"""
Pytest configuration and fixtures for hippocampus tests.
"""

import pytest
from markdown_chunker import MarkdownChunker


@pytest.fixture
def chunker():
    """Fixture providing a MarkdownChunker instance with test parameters."""
    return MarkdownChunker(chunk_size=500, chunk_overlap=50)


@pytest.fixture
def sample_markdown_with_frontmatter():
    """Fixture providing markdown content with YAML frontmatter (list format)."""
    return """---
title: "Example Research Paper"
author: "John Doe"
publish_date: "2024-01-15"
keywords:
  - "machine learning"
  - "neural networks"
  - "deep learning"
---

# Example Research Paper

## Introduction

This is the introduction section with some content about machine learning.

## Methods

This section describes the methodology used in the research.

## Results

Here are the results of our experiments.
"""


@pytest.fixture
def sample_markdown_brackets():
    """Fixture providing markdown with bracket notation for keywords."""
    return """---
title: "Another Paper"
author: "Jane Smith"
publish_date: "2024-02-20"
keywords: ["AI", "NLP", "transformers"]
---

# Another Paper

Content here.
"""


@pytest.fixture
def sample_markdown_no_frontmatter():
    """Fixture providing markdown without frontmatter."""
    return """# Regular Document

## Section 1

This is a regular markdown file without any YAML frontmatter.

## Section 2

It should still chunk correctly and not have title/author/keywords metadata.
"""
