"""
Tests for the unified configuration library.

Validates:
- Path resolution (relative paths against project_root, ~ expansion)
- Key mappings (flat keys to nested JSON paths)
- Type-safe getters
"""

import os
import tempfile
import json
from pathlib import Path


# Import from libs (which is at project root level)
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "libs"))
from config import Config, _KEY_TO_PATH


class TestConfigPathResolution:
    """Tests for get_path() method."""

    def test_relative_path_resolves_against_project_root(self):
        """Relative paths like ./data/file.db should resolve against project_root."""
        config = Config()

        # The config library should resolve relative paths against project_root
        tracking_db = config.get_path("TRACKING_DB")
        assert tracking_db is not None
        assert os.path.isabs(tracking_db), "get_path() should return absolute path"
        assert "hippocampus/data/tracking.db" in tracking_db

    def test_tilde_expansion(self):
        """Paths with ~ should expand to home directory."""
        config = Config()

        input_dir = config.get_path("INPUT_DIR")
        assert input_dir is not None
        assert "~" not in input_dir, "~ should be expanded"
        assert os.path.isabs(input_dir), "get_path() should return absolute path"

    def test_absolute_path_unchanged(self):
        """Absolute paths should be returned as-is."""
        config = Config()

        # Set an absolute path override
        absolute_path = "/tmp/test/absolute/path.db"
        config.set("TEST_PATH", absolute_path)

        result = config.get_path("TEST_PATH")
        assert result == absolute_path, "Absolute paths should not be modified"

    def test_default_path_resolution(self):
        """Default paths should also be resolved correctly."""
        config = Config()

        # Using a key that doesn't exist, with a relative default
        result = config.get_path("NONEXISTENT_KEY", "./some/relative/path.txt")
        assert result is not None
        assert os.path.isabs(result), "Default relative paths should be resolved"
        assert result.endswith("some/relative/path.txt")

    def test_all_db_paths_are_absolute(self):
        """All database paths from config should be absolute."""
        config = Config()

        db_keys = [
            ("TRACKING_DB", "./hippocampus/data/tracking.db"),
            ("PDF_TRACKING_DB", "./hippocampus/data/pdf_tracking.db"),
            ("EBOOK_TRACKING_DB", "./hippocampus/data/ebook_tracking.db"),
            ("TXT_TRACKING_DB", "./hippocampus/data/txt_tracking.db"),
            ("OFFICE_TRACKING_DB", "./hippocampus/data/office_tracking.db"),
            ("CONTEXT_DB_PATH", "./hippocampus/data/context.db"),
            ("CHAT_DB_PATH", "./cortex/data/chat.db"),
        ]

        for key, default in db_keys:
            path = config.get_path(key, default)
            assert path is not None, f"{key} should return a path"
            assert os.path.isabs(path), (
                f"{key} should return absolute path, got: {path}"
            )


class TestConfigKeyMappings:
    """Tests for key to JSON path mappings."""

    def test_flat_key_maps_to_nested(self):
        """Flat keys like CHROMADB_PORT should map to nested paths."""
        config = Config()

        # These should return the same value
        port_flat = config.get_int("CHROMADB_PORT")
        port_nested = config.get_int("hippocampus.chromadb.port")

        assert port_flat == port_nested, "Flat and nested keys should return same value"

    def test_tracking_db_mapping(self):
        """TRACKING_DB should map to hippocampus.tracking_db."""
        assert _KEY_TO_PATH.get("TRACKING_DB") == "hippocampus.tracking_db"

    def test_all_tracker_db_mappings_exist(self):
        """All tracker DB flat keys should have mappings."""
        expected_mappings = {
            "TRACKING_DB": "hippocampus.tracking_db",
            "PDF_TRACKING_DB": "hippocampus.pdf_tracking_db",
            "EBOOK_TRACKING_DB": "hippocampus.ebook_tracking_db",
            "TXT_TRACKING_DB": "hippocampus.txt_tracking_db",
            "OFFICE_TRACKING_DB": "hippocampus.office_tracking_db",
            "HIPPOCAMPUS_CONTEXT_DB": "hippocampus.context_db",
        }

        for flat_key, expected_path in expected_mappings.items():
            assert flat_key in _KEY_TO_PATH, f"Missing mapping for {flat_key}"
            assert _KEY_TO_PATH[flat_key] == expected_path, (
                f"Wrong mapping for {flat_key}"
            )


class TestConfigProjectRoot:
    """Tests for project root detection."""

    def test_project_root_contains_settings_json(self):
        """Project root should contain settings.json."""
        config = Config()

        settings_path = config.project_root / "settings.json"
        assert settings_path.exists(), (
            f"settings.json should exist at {config.project_root}"
        )

    def test_project_root_is_absolute(self):
        """Project root should be an absolute path."""
        config = Config()

        assert config.project_root.is_absolute(), "project_root should be absolute"


class TestConfigTypeGetters:
    """Tests for type-safe getter methods."""

    def test_get_int(self):
        """get_int should return integer values."""
        config = Config()

        port = config.get_int("CHROMADB_PORT", 8000)
        assert isinstance(port, int)

    def test_get_bool(self):
        """get_bool should return boolean values."""
        config = Config()

        # Test with a known boolean
        enabled = config.get_bool("CHAT_HISTORY_ENABLED", False)
        assert isinstance(enabled, bool)

    def test_get_float(self):
        """get_float should return float values."""
        config = Config()

        temp = config.get_float("TEMPERATURE", 0.7)
        assert isinstance(temp, float)

    def test_get_list(self):
        """get_list should return list values."""
        config = Config()

        patterns = config.get_list("CHAT_RESET_PATTERNS", [])
        assert isinstance(patterns, list)


class TestConfigWithTempFile:
    """Tests using temporary config files."""

    def test_custom_config_path(self):
        """Config should work with custom settings.json path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_data = {"test": {"db_path": "./data/test.db", "port": 9999}}

            settings_path = Path(tmpdir) / "settings.json"
            with open(settings_path, "w") as f:
                json.dump(config_data, f)

            config = Config(config_path=settings_path)

            # Project root should be the temp directory
            # Use resolve() to handle macOS /var -> /private/var symlink
            assert config.project_root.resolve() == Path(tmpdir).resolve()

            # Relative path should resolve against tmpdir
            db_path = config.get_path("test.db_path")
            expected = str((Path(tmpdir) / "data" / "test.db").resolve())
            assert Path(db_path).resolve() == Path(expected).resolve()

            # Int getter should work
            port = config.get_int("test.port")
            assert port == 9999


class TestConfigOverrides:
    """Tests for in-memory overrides."""

    def test_override_takes_precedence(self):
        """set() overrides should take precedence over file values."""
        config = Config()

        original = config.get("CHROMADB_PORT")
        config.set("CHROMADB_PORT", "12345")

        assert config.get("CHROMADB_PORT") == "12345"

        # Clear and verify original is restored
        config.clear_override("CHROMADB_PORT")
        assert config.get("CHROMADB_PORT") == original

    def test_override_path_resolution(self):
        """Overridden paths should also be resolved."""
        config = Config()

        config.set("TEST_PATH", "./relative/path.txt")
        result = config.get_path("TEST_PATH")

        assert os.path.isabs(result)
        assert result.endswith("relative/path.txt")
