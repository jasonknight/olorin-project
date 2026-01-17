#!/usr/bin/env python3
"""
Pytest configuration and fixtures for hippocampus tests.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

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


# ============================================================================
# Document Tracker Fixtures
# ============================================================================


@pytest.fixture
def temp_input_dir(tmp_path):
    """Create a temporary input directory for tracker tests."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    return str(input_dir)


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory for tracker tests."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return str(output_dir)


@pytest.fixture
def temp_tracking_db(tmp_path):
    """Create a temporary tracking database path."""
    return str(tmp_path / "tracking.db")


@pytest.fixture
def mock_kafka_producer():
    """Mock KafkaProducer for testing without Kafka connection."""
    with patch("document_tracker_base.KafkaProducer") as mock_producer:
        producer_instance = MagicMock()
        mock_producer.return_value = producer_instance
        future = MagicMock()
        future.get.return_value = None
        producer_instance.send.return_value = future
        yield producer_instance


@pytest.fixture
def mock_subprocess_ollama_available():
    """Mock subprocess for ollama being available."""
    with patch("document_tracker_base.subprocess.run") as mock_run:

        def run_side_effect(cmd, **kwargs):
            result = MagicMock()
            if "--version" in cmd:
                result.returncode = 0
            elif "list" in cmd:
                result.returncode = 0
                result.stdout = "llama3.2:1b"
            elif "run" in cmd:
                result.returncode = 0
                result.stdout = "SUBSTANTIVE"
            else:
                result.returncode = 0
            return result

        mock_run.side_effect = run_side_effect
        yield mock_run


@pytest.fixture
def mock_subprocess_ollama_unavailable():
    """Mock subprocess for ollama not being available."""
    with patch("document_tracker_base.subprocess.run") as mock_run:
        result = MagicMock()
        result.returncode = 1
        mock_run.return_value = result
        yield mock_run


@pytest.fixture
def mock_config(tmp_path):
    """Create a mock settings.json file."""
    import json

    settings_file = tmp_path / "settings.json"
    settings = {
        "hippocampus": {
            "input_dir": str(tmp_path / "input"),
            "poll_interval": 5,
            "trackers": {
                "ollama": {
                    "enabled": True,
                    "model": "llama3.2:1b",
                    "threshold": 0.5,
                },
                "min_content_chars": 100,
                "min_content_density": 0.3,
                "min_word_count": 20,
                "notify_broca": False,
            },
        },
        "kafka": {"bootstrap_servers": "localhost:9092"},
        "broca": {"kafka_topic": "ai_out"},
    }
    settings_file.write_text(json.dumps(settings))

    # Create directories
    (tmp_path / "input").mkdir(exist_ok=True)
    (tmp_path / "logs").mkdir(exist_ok=True)

    return settings_file


@pytest.fixture
def sample_txt_file(temp_input_dir):
    """Create a sample TXT file for testing."""
    txt_path = Path(temp_input_dir) / "sample_document.txt"
    content = """INTRODUCTION

This is a sample text document for testing purposes.
It contains multiple paragraphs with substantial content.

MAIN SECTION

Here is the main content of the document. It discusses
important topics that should be converted to markdown.
The content is substantial enough to pass quality thresholds.

- First point about the topic
- Second point with more details
- Third point for completeness

CONCLUSION

In conclusion, this document demonstrates the text tracking
and conversion functionality of the hippocampus module.
"""
    txt_path.write_text(content)
    return str(txt_path)
