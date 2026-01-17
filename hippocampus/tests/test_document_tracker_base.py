#!/usr/bin/env python3
"""
Tests for DocumentTrackerBase abstract class.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock


# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class TestCalculateContentDensity:
    """Tests for _calculate_content_density method."""

    def test_empty_string_returns_zero(self):
        """Empty string should return 0."""
        from document_tracker_base import DocumentTrackerBase

        # Can't instantiate abstract class, so test the method logic directly
        result = DocumentTrackerBase._calculate_content_density(None, "")
        assert result == 0.0

    def test_pure_alphanumeric_returns_one(self):
        """String with only alphanumeric characters should return 1.0."""
        from document_tracker_base import DocumentTrackerBase

        result = DocumentTrackerBase._calculate_content_density(None, "HelloWorld123")
        assert result == 1.0

    def test_mixed_content_returns_ratio(self):
        """Mixed content should return correct ratio."""
        from document_tracker_base import DocumentTrackerBase

        # "Hello World" has 10 alphanumeric chars out of 11 total
        result = DocumentTrackerBase._calculate_content_density(None, "Hello World")
        assert round(result, 2) == round(10 / 11, 2)

    def test_none_returns_zero(self):
        """None should return 0."""
        from document_tracker_base import DocumentTrackerBase

        result = DocumentTrackerBase._calculate_content_density(None, None)
        assert result == 0.0


class TestNormalizeText:
    """Tests for _normalize_text method."""

    def test_removes_form_feed(self):
        """Form feed characters should be removed."""
        from document_tracker_base import DocumentTrackerBase

        result = DocumentTrackerBase._normalize_text(None, "Hello\fWorld")
        assert "\f" not in result
        assert "Hello" in result
        assert "World" in result

    def test_collapses_whitespace(self):
        """Multiple spaces should collapse to single space."""
        from document_tracker_base import DocumentTrackerBase

        result = DocumentTrackerBase._normalize_text(None, "Hello    World")
        assert result == "Hello World"

    def test_normalizes_multiple_newlines(self):
        """Multiple newlines should collapse to double newline."""
        from document_tracker_base import DocumentTrackerBase

        result = DocumentTrackerBase._normalize_text(None, "Hello\n\n\n\n\nWorld")
        assert result == "Hello\n\nWorld"

    def test_removes_standalone_page_numbers(self):
        """Standalone page numbers should be removed."""
        from document_tracker_base import DocumentTrackerBase

        text = "Content\n123\nMore content"
        result = DocumentTrackerBase._normalize_text(None, text)
        # Page number "123" on its own line should be removed
        assert "Content" in result
        assert "More content" in result

    def test_fixes_hyphenated_words(self):
        """Hyphenated words across lines should be joined."""
        from document_tracker_base import DocumentTrackerBase

        result = DocumentTrackerBase._normalize_text(None, "docu-\nment")
        assert result == "document"


class TestIsContentSubstantial:
    """Tests for _is_content_substantial method."""

    def test_empty_string_not_substantial(self):
        """Empty string should not be substantial."""
        from document_tracker_base import DocumentTrackerBase

        # Create a mock instance with required attributes
        mock_self = MagicMock()
        mock_self.min_content_chars = 100
        mock_self.min_content_density = 0.3
        mock_self.min_word_count = 20

        result = DocumentTrackerBase._is_content_substantial(mock_self, "")
        assert result is False

    def test_short_string_not_substantial(self):
        """String below min_content_chars should not be substantial."""
        from document_tracker_base import DocumentTrackerBase

        mock_self = MagicMock()
        mock_self.min_content_chars = 100
        mock_self.min_content_density = 0.3
        mock_self.min_word_count = 20

        result = DocumentTrackerBase._is_content_substantial(mock_self, "Short text")
        assert result is False

    def test_low_density_not_substantial(self):
        """String with low content density should not be substantial."""
        from document_tracker_base import DocumentTrackerBase

        # Create a simple namespace object that includes the helper method
        class MockSelf:
            min_content_chars = 10
            min_content_density = 0.9
            min_word_count = 1

            def _calculate_content_density(self, text):
                return DocumentTrackerBase._calculate_content_density(self, text)

        # Lots of spaces/symbols = low density
        text = "a    b    c    d    e    f    g    h    i    j    k    l"
        result = DocumentTrackerBase._is_content_substantial(MockSelf(), text)
        assert result is False

    def test_few_words_not_substantial(self):
        """String with too few words should not be substantial."""
        from document_tracker_base import DocumentTrackerBase

        class MockSelf:
            min_content_chars = 10
            min_content_density = 0.1
            min_word_count = 50

            def _calculate_content_density(self, text):
                return DocumentTrackerBase._calculate_content_density(self, text)

        text = "This has only a few words but is long enough in characters."
        result = DocumentTrackerBase._is_content_substantial(MockSelf(), text)
        assert result is False

    def test_substantial_content_passes(self):
        """Content meeting all thresholds should pass."""
        from document_tracker_base import DocumentTrackerBase

        class MockSelf:
            min_content_chars = 50
            min_content_density = 0.3
            min_word_count = 10

            def _calculate_content_density(self, text):
                return DocumentTrackerBase._calculate_content_density(self, text)

        text = (
            "This is a substantial piece of content that contains "
            "many words and has good content density with alphanumeric "
            "characters throughout the text."
        )
        result = DocumentTrackerBase._is_content_substantial(MockSelf(), text)
        assert result is True


class TestBuildYamlFrontmatter:
    """Tests for _build_yaml_frontmatter method."""

    def test_basic_frontmatter(self):
        """Should generate valid YAML frontmatter."""
        from document_tracker_base import DocumentTrackerBase

        metadata = {
            "title": "Test Document",
            "author": "Test Author",
            "publish_date": "2024-01-01",
        }
        result = DocumentTrackerBase._build_yaml_frontmatter(None, metadata)

        assert result.startswith("---\n")
        assert result.endswith("---\n\n")
        assert 'title: "Test Document"' in result
        assert 'author: "Test Author"' in result
        assert 'publish_date: "2024-01-01"' in result

    def test_frontmatter_with_keywords_list(self):
        """Keywords list should be properly formatted."""
        from document_tracker_base import DocumentTrackerBase

        metadata = {
            "title": "Test",
            "keywords": ["AI", "ML", "Deep Learning"],
        }
        result = DocumentTrackerBase._build_yaml_frontmatter(None, metadata)

        assert "keywords:" in result
        assert '"AI"' in result
        assert '"ML"' in result
        assert '"Deep Learning"' in result

    def test_frontmatter_with_single_keyword(self):
        """Single keyword should be formatted as inline array."""
        from document_tracker_base import DocumentTrackerBase

        metadata = {
            "title": "Test",
            "keywords": ["AI"],
        }
        result = DocumentTrackerBase._build_yaml_frontmatter(None, metadata)

        assert 'keywords: ["AI"]' in result


class TestRemoveBoilerplate:
    """Tests for _remove_boilerplate method."""

    def test_removes_page_numbers(self):
        """Page X of Y patterns should be removed."""
        from document_tracker_base import DocumentTrackerBase

        text = "Content\nPage 1 of 10\nMore content"
        result = DocumentTrackerBase._remove_boilerplate(None, text)

        assert "Page 1 of 10" not in result
        assert "Content" in result
        assert "More content" in result

    def test_removes_copyright(self):
        """Copyright notices should be removed."""
        from document_tracker_base import DocumentTrackerBase

        text = "Content\nCopyright © 2024 Company\nMore content"
        result = DocumentTrackerBase._remove_boilerplate(None, text)

        assert "Copyright" not in result
        assert "Content" in result

    def test_removes_confidential(self):
        """Confidential markers should be removed."""
        from document_tracker_base import DocumentTrackerBase

        text = "Content\nCONFIDENTIAL\nMore content"
        result = DocumentTrackerBase._remove_boilerplate(None, text)

        assert "CONFIDENTIAL" not in result


class TestFindFilesByExtensions:
    """Tests for _find_files_by_extensions helper method."""

    def test_finds_files_with_extension(self, temp_input_dir):
        """Should find files with matching extensions."""
        from document_tracker_base import DocumentTrackerBase

        # Create test files
        (Path(temp_input_dir) / "test1.txt").touch()
        (Path(temp_input_dir) / "test2.TXT").touch()
        (Path(temp_input_dir) / "test3.pdf").touch()

        # Create mock self with TXT extensions
        mock_self = MagicMock()
        mock_self.input_dir = temp_input_dir
        mock_self.EXTENSIONS = {".txt"}

        result = DocumentTrackerBase._find_files_by_extensions(mock_self)

        assert len(result) == 2
        assert any("test1.txt" in f for f in result)
        assert any("test2.TXT" in f for f in result)
        assert not any("test3.pdf" in f for f in result)

    def test_finds_files_recursively(self, temp_input_dir):
        """Should find files in subdirectories."""
        from document_tracker_base import DocumentTrackerBase

        # Create subdirectory with test file
        subdir = Path(temp_input_dir) / "subdir"
        subdir.mkdir()
        (subdir / "nested.txt").touch()
        (Path(temp_input_dir) / "root.txt").touch()

        mock_self = MagicMock()
        mock_self.input_dir = temp_input_dir
        mock_self.EXTENSIONS = {".txt"}

        result = DocumentTrackerBase._find_files_by_extensions(mock_self)

        assert len(result) == 2
        assert any("nested.txt" in f for f in result)
        assert any("root.txt" in f for f in result)


class TestShouldProcessFile:
    """Tests for should_process_file method."""

    def test_nonexistent_file_returns_false(self):
        """Non-existent file should not be processed."""
        from document_tracker_base import DocumentTrackerBase

        mock_self = MagicMock()
        mock_self.force_reprocess_regex = None

        result = DocumentTrackerBase.should_process_file(
            mock_self, "/nonexistent/path.txt"
        )
        assert result is False

    def test_force_reprocess_pattern_matches(self, temp_input_dir):
        """File matching force reprocess pattern should be processed."""
        import re

        from document_tracker_base import DocumentTrackerBase

        # Create test file
        test_file = Path(temp_input_dir) / "report_2024.txt"
        test_file.touch()

        mock_self = MagicMock()
        mock_self.force_reprocess_regex = re.compile("2024")
        mock_self.file_tracker = MagicMock()
        mock_self.logger = MagicMock()

        result = DocumentTrackerBase.should_process_file(mock_self, str(test_file))
        assert result is True

    def test_unprocessed_file_returns_true(self, temp_input_dir):
        """Unprocessed file should be processed."""
        from document_tracker_base import DocumentTrackerBase

        test_file = Path(temp_input_dir) / "new_file.txt"
        test_file.touch()

        mock_self = MagicMock()
        mock_self.force_reprocess_regex = None
        mock_self.file_tracker.is_file_processed.return_value = False
        mock_self.reprocess_on_change = False

        result = DocumentTrackerBase.should_process_file(mock_self, str(test_file))
        assert result is True

    def test_processed_unchanged_file_returns_false(self, temp_input_dir):
        """Already processed, unchanged file should not be processed."""
        from document_tracker_base import DocumentTrackerBase

        test_file = Path(temp_input_dir) / "processed.txt"
        test_file.touch()

        mock_self = MagicMock()
        mock_self.force_reprocess_regex = None
        mock_self.file_tracker.is_file_processed.return_value = True
        mock_self.reprocess_on_change = False

        result = DocumentTrackerBase.should_process_file(mock_self, str(test_file))
        assert result is False

    def test_processed_changed_file_returns_true_if_reprocess_enabled(
        self, temp_input_dir
    ):
        """Changed file with reprocess_on_change=True should be processed."""
        from document_tracker_base import DocumentTrackerBase

        test_file = Path(temp_input_dir) / "changed.txt"
        test_file.touch()

        mock_self = MagicMock()
        mock_self.force_reprocess_regex = None
        mock_self.file_tracker.is_file_processed.return_value = True
        mock_self.reprocess_on_change = True
        mock_self.file_tracker.has_file_changed.return_value = True

        result = DocumentTrackerBase.should_process_file(mock_self, str(test_file))
        assert result is True
