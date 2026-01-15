"""
Unit tests for the control handlers discovery system and handlers.

These tests use mocking to avoid actual state database or process operations.
"""

import signal
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add libs to path for imports
libs_path = Path(__file__).parent.parent.parent
if str(libs_path) not in sys.path:
    sys.path.insert(0, str(libs_path))

from libs.control_handlers import (  # noqa: E402
    _filename_to_command,
    clear_cache,
    discover_handlers,
    get_handler,
    list_commands,
)


# =============================================================================
# Test fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_handler_cache():
    """Reset the handler cache before each test."""
    clear_cache()
    yield
    clear_cache()


# =============================================================================
# Test _filename_to_command
# =============================================================================


class TestFilenameToCommand:
    """Tests for the _filename_to_command function."""

    def test_single_underscore(self):
        """Test conversion with single underscore."""
        assert _filename_to_command("reload_config") == "reload-config"

    def test_multiple_underscores(self):
        """Test conversion with multiple underscores."""
        assert _filename_to_command("stop_broca_audio_play") == "stop-broca-audio-play"

    def test_no_underscores(self):
        """Test conversion with no underscores."""
        assert _filename_to_command("shutdown") == "shutdown"

    def test_leading_underscore_converted(self):
        """Test that leading underscores are also converted."""
        # Note: in practice, files starting with _ are skipped by discover_handlers
        assert _filename_to_command("_private") == "-private"

    def test_consecutive_underscores(self):
        """Test conversion with consecutive underscores."""
        assert _filename_to_command("some__handler") == "some--handler"


# =============================================================================
# Test discover_handlers
# =============================================================================


class TestDiscoverHandlers:
    """Tests for the discover_handlers function."""

    def test_discovers_stop_broca_audio_play(self):
        """Test that discover_handlers finds the stop_broca_audio_play handler."""
        handlers = discover_handlers()
        assert "stop-broca-audio-play" in handlers
        assert callable(handlers["stop-broca-audio-play"])

    def test_returns_dict(self):
        """Test that discover_handlers returns a dictionary."""
        handlers = discover_handlers()
        assert isinstance(handlers, dict)

    def test_caches_result(self):
        """Test that discover_handlers caches its result."""
        handlers1 = discover_handlers()
        handlers2 = discover_handlers()
        # Should be the exact same object due to caching
        assert handlers1 is handlers2

    def test_all_handlers_callable(self):
        """Test that all discovered handlers are callable."""
        handlers = discover_handlers()
        for name, handler in handlers.items():
            assert callable(handler), f"Handler {name} is not callable"


# =============================================================================
# Test get_handler
# =============================================================================


class TestGetHandler:
    """Tests for the get_handler function."""

    def test_returns_callable(self):
        """Test that get_handler returns a callable for known commands."""
        handler = get_handler("stop-broca-audio-play")
        assert callable(handler)

    def test_raises_keyerror_for_unknown(self):
        """Test that get_handler raises KeyError for unknown commands."""
        with pytest.raises(KeyError) as exc_info:
            get_handler("unknown-command")
        assert "unknown-command" in str(exc_info.value)

    def test_raises_keyerror_for_empty_string(self):
        """Test that get_handler raises KeyError for empty string."""
        with pytest.raises(KeyError):
            get_handler("")


# =============================================================================
# Test list_commands
# =============================================================================


class TestListCommands:
    """Tests for the list_commands function."""

    def test_contains_stop_broca_audio_play(self):
        """Test that list_commands includes stop-broca-audio-play."""
        commands = list_commands()
        assert "stop-broca-audio-play" in commands

    def test_returns_list(self):
        """Test that list_commands returns a list."""
        commands = list_commands()
        assert isinstance(commands, list)

    def test_returns_sorted(self):
        """Test that list_commands returns sorted results."""
        commands = list_commands()
        assert commands == sorted(commands)

    def test_all_strings(self):
        """Test that all command names are strings."""
        commands = list_commands()
        for cmd in commands:
            assert isinstance(cmd, str)


# =============================================================================
# Test stop_broca_audio_play handler
# =============================================================================


class TestStopBrocaAudioPlayHandler:
    """Tests for the stop_broca_audio_play handler with mocked dependencies."""

    def test_kills_process_with_sigterm(self, mocker):
        """Test that handler kills process with SIGTERM by default."""
        # Mock the state
        mock_state = MagicMock()
        mock_state.get_int.return_value = 12345
        mock_state.get_bool.return_value = True
        mocker.patch(
            "libs.control_handlers.stop_broca_audio_play.get_state",
            return_value=mock_state,
        )

        # Mock os.kill
        mock_kill = mocker.patch("os.kill")

        # Import and call the handler
        from libs.control_handlers.stop_broca_audio_play import handle

        result = handle({})

        # Verify os.kill was called with SIGTERM
        mock_kill.assert_called_once_with(12345, signal.SIGTERM)

        # Verify result
        assert result["pid_killed"] == 12345
        assert result["was_playing"] is True
        assert "12345" in result["message"]

        # Verify state was updated
        mock_state.delete.assert_called_with("broca.audio_pid")
        mock_state.set_bool.assert_called_with("broca.is_playing", False)

    def test_nothing_playing(self, mocker):
        """Test handler when no audio is playing."""
        # Mock the state - no PID
        mock_state = MagicMock()
        mock_state.get_int.return_value = None
        mock_state.get_bool.return_value = False
        mocker.patch(
            "libs.control_handlers.stop_broca_audio_play.get_state",
            return_value=mock_state,
        )

        # Mock os.kill (should not be called)
        mock_kill = mocker.patch("os.kill")

        from libs.control_handlers.stop_broca_audio_play import handle

        result = handle({})

        # os.kill should not be called
        mock_kill.assert_not_called()

        # Verify result
        assert result["pid_killed"] is None
        assert result["was_playing"] is False
        assert "No audio" in result["message"]

    def test_force_sigkill(self, mocker):
        """Test that handler uses SIGKILL when force=True."""
        mock_state = MagicMock()
        mock_state.get_int.return_value = 12345
        mock_state.get_bool.return_value = True
        mocker.patch(
            "libs.control_handlers.stop_broca_audio_play.get_state",
            return_value=mock_state,
        )

        mock_kill = mocker.patch("os.kill")

        from libs.control_handlers.stop_broca_audio_play import handle

        result = handle({"force": True})

        # Verify SIGKILL was used
        mock_kill.assert_called_once_with(12345, signal.SIGKILL)
        assert result["pid_killed"] == 12345
        assert "SIGKILL" in result["message"]

    def test_process_already_dead(self, mocker):
        """Test handler when process is already terminated."""
        mock_state = MagicMock()
        mock_state.get_int.return_value = 12345
        mock_state.get_bool.return_value = True
        mocker.patch(
            "libs.control_handlers.stop_broca_audio_play.get_state",
            return_value=mock_state,
        )

        # Mock os.kill to raise ProcessLookupError
        mocker.patch("os.kill", side_effect=ProcessLookupError)

        from libs.control_handlers.stop_broca_audio_play import handle

        result = handle({})

        # Should still return the PID and clear state
        assert result["pid_killed"] == 12345
        assert result["was_playing"] is True
        assert "already terminated" in result["message"]

        # State should still be cleared
        mock_state.delete.assert_called_with("broca.audio_pid")
        mock_state.set_bool.assert_called_with("broca.is_playing", False)

    def test_permission_denied(self, mocker):
        """Test handler when permission denied killing process."""
        mock_state = MagicMock()
        mock_state.get_int.return_value = 12345
        mock_state.get_bool.return_value = True
        mocker.patch(
            "libs.control_handlers.stop_broca_audio_play.get_state",
            return_value=mock_state,
        )

        # Mock os.kill to raise PermissionError
        mocker.patch("os.kill", side_effect=PermissionError)

        from libs.control_handlers.stop_broca_audio_play import handle

        result = handle({})

        # Should return None for pid_killed and not clear state
        assert result["pid_killed"] is None
        assert result["was_playing"] is True
        assert "Permission denied" in result["message"]

        # State should NOT be cleared on permission error
        mock_state.delete.assert_not_called()
        mock_state.set_bool.assert_not_called()

    def test_was_playing_false_but_pid_exists(self, mocker):
        """Test edge case where PID exists but is_playing is False."""
        mock_state = MagicMock()
        mock_state.get_int.return_value = 12345
        mock_state.get_bool.return_value = False  # Playing flag is False
        mocker.patch(
            "libs.control_handlers.stop_broca_audio_play.get_state",
            return_value=mock_state,
        )

        mock_kill = mocker.patch("os.kill")

        from libs.control_handlers.stop_broca_audio_play import handle

        result = handle({})

        # Should still kill the process since PID exists
        mock_kill.assert_called_once_with(12345, signal.SIGTERM)
        assert result["pid_killed"] == 12345
        assert result["was_playing"] is False

    def test_empty_payload(self, mocker):
        """Test handler with empty payload defaults force to False."""
        mock_state = MagicMock()
        mock_state.get_int.return_value = 99999
        mock_state.get_bool.return_value = True
        mocker.patch(
            "libs.control_handlers.stop_broca_audio_play.get_state",
            return_value=mock_state,
        )

        mock_kill = mocker.patch("os.kill")

        from libs.control_handlers.stop_broca_audio_play import handle

        handle({})

        # Should use SIGTERM (not SIGKILL)
        mock_kill.assert_called_once_with(99999, signal.SIGTERM)

    def test_force_false_explicit(self, mocker):
        """Test handler with force=False explicitly."""
        mock_state = MagicMock()
        mock_state.get_int.return_value = 11111
        mock_state.get_bool.return_value = True
        mocker.patch(
            "libs.control_handlers.stop_broca_audio_play.get_state",
            return_value=mock_state,
        )

        mock_kill = mocker.patch("os.kill")

        from libs.control_handlers.stop_broca_audio_play import handle

        result = handle({"force": False})

        mock_kill.assert_called_once_with(11111, signal.SIGTERM)
        assert "SIGTERM" in result["message"]


# =============================================================================
# Test clear_cache
# =============================================================================


class TestClearCache:
    """Tests for the clear_cache function."""

    def test_clears_cache(self):
        """Test that clear_cache forces re-discovery."""
        # First discovery
        handlers1 = discover_handlers()

        # Clear cache
        clear_cache()

        # Second discovery should be a new dict object
        handlers2 = discover_handlers()

        # After clearing, they should NOT be the same object
        # (the cache was invalidated and a new dict was created)
        assert handlers1 is not handlers2

        # But both should have the same keys
        assert set(handlers1.keys()) == set(handlers2.keys())

    def test_cache_returns_same_object(self):
        """Test that without clearing, cache returns same dict object."""
        handlers1 = discover_handlers()
        handlers2 = discover_handlers()

        # Without clearing, should be the exact same object
        assert handlers1 is handlers2
