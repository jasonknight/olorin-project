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
        assert _filename_to_command("stop_audio") == "stop-audio"

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

    def test_discovers_stop_audio(self):
        """Test that discover_handlers finds the stop_audio handler."""
        handlers = discover_handlers()
        assert "stop-audio" in handlers
        assert callable(handlers["stop-audio"])

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
        handler = get_handler("stop-audio")
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

    def test_contains_stop_audio(self):
        """Test that list_commands includes stop-audio."""
        commands = list_commands()
        assert "stop-audio" in commands

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
# Test stop_audio handler
# =============================================================================


class TestStopAudioHandler:
    """Tests for the stop_audio handler with mocked dependencies."""

    def test_kills_process_with_sigterm_and_mutes(self, mocker):
        """Test that handler kills process with SIGTERM by default and sets muted."""
        # Mock the state
        mock_state = MagicMock()
        mock_state.get_int.return_value = 12345
        mock_state.get_bool.return_value = True
        mocker.patch(
            "libs.control_handlers.stop_audio.get_state",
            return_value=mock_state,
        )

        # Mock os.kill
        mock_kill = mocker.patch("os.kill")

        # Import and call the handler
        from libs.control_handlers.stop_audio import handle

        result = handle({})

        # Verify os.kill was called with SIGTERM
        mock_kill.assert_called_once_with(12345, signal.SIGTERM)

        # Verify result
        assert result["pid_killed"] == 12345
        assert result["was_playing"] is True
        assert result["muted"] is True
        assert "12345" in result["message"]

        # Verify state was updated (muted set, audio state cleared)
        mock_state.set_bool.assert_any_call("broca.audio_muted", True)
        mock_state.delete.assert_called_with("broca.audio_pid")
        mock_state.set_bool.assert_any_call("broca.is_playing", False)

    def test_nothing_playing_still_mutes(self, mocker):
        """Test handler when no audio is playing - should still set muted."""
        # Mock the state - no PID
        mock_state = MagicMock()
        mock_state.get_int.return_value = None
        mock_state.get_bool.return_value = False
        mocker.patch(
            "libs.control_handlers.stop_audio.get_state",
            return_value=mock_state,
        )

        # Mock os.kill (should not be called)
        mock_kill = mocker.patch("os.kill")

        from libs.control_handlers.stop_audio import handle

        result = handle({})

        # os.kill should not be called
        mock_kill.assert_not_called()

        # Verify result
        assert result["pid_killed"] is None
        assert result["was_playing"] is False
        assert result["muted"] is True
        assert "muted" in result["message"]

        # Verify muted state was set
        mock_state.set_bool.assert_called_with("broca.audio_muted", True)

    def test_force_sigkill(self, mocker):
        """Test that handler uses SIGKILL when force=True."""
        mock_state = MagicMock()
        mock_state.get_int.return_value = 12345
        mock_state.get_bool.return_value = True
        mocker.patch(
            "libs.control_handlers.stop_audio.get_state",
            return_value=mock_state,
        )

        mock_kill = mocker.patch("os.kill")

        from libs.control_handlers.stop_audio import handle

        result = handle({"force": True})

        # Verify SIGKILL was used
        mock_kill.assert_called_once_with(12345, signal.SIGKILL)
        assert result["pid_killed"] == 12345
        assert result["muted"] is True
        assert "SIGKILL" in result["message"]

    def test_process_already_dead(self, mocker):
        """Test handler when process is already terminated."""
        mock_state = MagicMock()
        mock_state.get_int.return_value = 12345
        mock_state.get_bool.return_value = True
        mocker.patch(
            "libs.control_handlers.stop_audio.get_state",
            return_value=mock_state,
        )

        # Mock os.kill to raise ProcessLookupError
        mocker.patch("os.kill", side_effect=ProcessLookupError)

        from libs.control_handlers.stop_audio import handle

        result = handle({})

        # Should still return the PID, clear state, and set muted
        assert result["pid_killed"] == 12345
        assert result["was_playing"] is True
        assert result["muted"] is True
        assert "already terminated" in result["message"]

        # State should still be cleared
        mock_state.delete.assert_called_with("broca.audio_pid")
        mock_state.set_bool.assert_any_call("broca.is_playing", False)

    def test_permission_denied_still_mutes(self, mocker):
        """Test handler when permission denied - should still set muted."""
        mock_state = MagicMock()
        mock_state.get_int.return_value = 12345
        mock_state.get_bool.return_value = True
        mocker.patch(
            "libs.control_handlers.stop_audio.get_state",
            return_value=mock_state,
        )

        # Mock os.kill to raise PermissionError
        mocker.patch("os.kill", side_effect=PermissionError)

        from libs.control_handlers.stop_audio import handle

        result = handle({})

        # Should return None for pid_killed but still set muted
        assert result["pid_killed"] is None
        assert result["was_playing"] is True
        assert result["muted"] is True
        assert "Permission denied" in result["message"]

        # Muted should be set even on permission error
        mock_state.set_bool.assert_called_with("broca.audio_muted", True)

    def test_was_playing_false_but_pid_exists(self, mocker):
        """Test edge case where PID exists but is_playing is False."""
        mock_state = MagicMock()
        mock_state.get_int.return_value = 12345
        mock_state.get_bool.return_value = False  # Playing flag is False
        mocker.patch(
            "libs.control_handlers.stop_audio.get_state",
            return_value=mock_state,
        )

        mock_kill = mocker.patch("os.kill")

        from libs.control_handlers.stop_audio import handle

        result = handle({})

        # Should still kill the process since PID exists
        mock_kill.assert_called_once_with(12345, signal.SIGTERM)
        assert result["pid_killed"] == 12345
        assert result["was_playing"] is False
        assert result["muted"] is True

    def test_empty_payload(self, mocker):
        """Test handler with empty payload defaults force to False."""
        mock_state = MagicMock()
        mock_state.get_int.return_value = 99999
        mock_state.get_bool.return_value = True
        mocker.patch(
            "libs.control_handlers.stop_audio.get_state",
            return_value=mock_state,
        )

        mock_kill = mocker.patch("os.kill")

        from libs.control_handlers.stop_audio import handle

        handle({})

        # Should use SIGTERM (not SIGKILL)
        mock_kill.assert_called_once_with(99999, signal.SIGTERM)

    def test_force_false_explicit(self, mocker):
        """Test handler with force=False explicitly."""
        mock_state = MagicMock()
        mock_state.get_int.return_value = 11111
        mock_state.get_bool.return_value = True
        mocker.patch(
            "libs.control_handlers.stop_audio.get_state",
            return_value=mock_state,
        )

        mock_kill = mocker.patch("os.kill")

        from libs.control_handlers.stop_audio import handle

        result = handle({"force": False})

        mock_kill.assert_called_once_with(11111, signal.SIGTERM)
        assert "SIGTERM" in result["message"]


# =============================================================================
# Test resume_audio handler
# =============================================================================


class TestResumeAudioHandler:
    """Tests for the resume_audio handler with mocked dependencies."""

    def test_resumes_when_muted(self, mocker):
        """Test that handler clears muted state when audio was muted."""
        mock_state = MagicMock()
        mock_state.get_bool.return_value = True  # was muted
        mocker.patch(
            "libs.control_handlers.resume_audio.get_state",
            return_value=mock_state,
        )

        from libs.control_handlers.resume_audio import handle

        result = handle({})

        assert result["was_muted"] is True
        assert result["muted"] is False
        assert "resumed" in result["message"]

        mock_state.set_bool.assert_called_once_with("broca.audio_muted", False)

    def test_resumes_when_not_muted(self, mocker):
        """Test handler when audio was not muted."""
        mock_state = MagicMock()
        mock_state.get_bool.return_value = False  # was not muted
        mocker.patch(
            "libs.control_handlers.resume_audio.get_state",
            return_value=mock_state,
        )

        from libs.control_handlers.resume_audio import handle

        result = handle({})

        assert result["was_muted"] is False
        assert result["muted"] is False
        assert "not muted" in result["message"]

        mock_state.set_bool.assert_called_once_with("broca.audio_muted", False)


# =============================================================================
# Test audio_status handler
# =============================================================================


class TestAudioStatusHandler:
    """Tests for the audio_status handler with mocked dependencies."""

    def test_status_muted_and_idle(self, mocker):
        """Test status when audio is muted and idle."""
        mock_state = MagicMock()
        mock_state.get_bool.side_effect = lambda key, default=False: {
            "broca.audio_muted": True,
            "broca.is_playing": False,
        }.get(key, default)
        mock_state.get_int.return_value = None
        mocker.patch(
            "libs.control_handlers.audio_status.get_state",
            return_value=mock_state,
        )

        from libs.control_handlers.audio_status import handle

        result = handle({})

        assert result["muted"] is True
        assert result["is_playing"] is False
        assert result["audio_pid"] is None
        assert "muted" in result["message"]
        assert "idle" in result["message"]

    def test_status_unmuted_and_playing(self, mocker):
        """Test status when audio is unmuted and playing."""
        mock_state = MagicMock()
        mock_state.get_bool.side_effect = lambda key, default=False: {
            "broca.audio_muted": False,
            "broca.is_playing": True,
        }.get(key, default)
        mock_state.get_int.return_value = 12345
        mocker.patch(
            "libs.control_handlers.audio_status.get_state",
            return_value=mock_state,
        )

        from libs.control_handlers.audio_status import handle

        result = handle({})

        assert result["muted"] is False
        assert result["is_playing"] is True
        assert result["audio_pid"] == 12345
        assert "unmuted" in result["message"]
        assert "playing" in result["message"]
        assert "12345" in result["message"]

    def test_status_unmuted_and_idle(self, mocker):
        """Test status when audio is unmuted and idle."""
        mock_state = MagicMock()
        mock_state.get_bool.side_effect = lambda key, default=False: {
            "broca.audio_muted": False,
            "broca.is_playing": False,
        }.get(key, default)
        mock_state.get_int.return_value = None
        mocker.patch(
            "libs.control_handlers.audio_status.get_state",
            return_value=mock_state,
        )

        from libs.control_handlers.audio_status import handle

        result = handle({})

        assert result["muted"] is False
        assert result["is_playing"] is False
        assert result["audio_pid"] is None
        assert "unmuted" in result["message"]
        assert "idle" in result["message"]


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


# =============================================================================
# Test clear handler
# =============================================================================


class TestClearHandler:
    """Tests for the clear handler with mocked dependencies."""

    def test_discovers_clear_handler(self):
        """Test that discover_handlers finds the clear handler."""
        handlers = discover_handlers()
        assert "clear" in handlers
        assert callable(handlers["clear"])

    def test_clears_both_databases(self, mocker, tmp_path):
        """Test that handler clears both chat and context databases."""
        # Create temp database paths
        chat_db = str(tmp_path / "chat.db")
        context_db = str(tmp_path / "context.db")

        # Mock config to return our temp paths
        mock_config = MagicMock()
        mock_config.get_path.side_effect = lambda key, default: {
            "CHAT_DB_PATH": chat_db,
            "CONTEXT_DB_PATH": context_db,
        }.get(key, default)

        mocker.patch("libs.control_handlers.clear.Config", return_value=mock_config)

        # Mock ChatStore
        mock_chat_store = MagicMock()
        mock_chat_store.clear_all.return_value = (
            3,
            10,
            5,
        )  # 3 convs, 10 messages, 5 context tracking
        mocker.patch(
            "libs.control_handlers.clear.ChatStore", return_value=mock_chat_store
        )

        # Mock ContextStore
        mock_context_store = MagicMock()
        mock_context_store.clear_all.return_value = 25  # 25 contexts
        mocker.patch(
            "libs.control_handlers.clear.ContextStore", return_value=mock_context_store
        )

        # Mock os.path.exists to return True
        mocker.patch("os.path.exists", return_value=True)

        from libs.control_handlers.clear import handle

        result = handle({})

        # Verify both stores were cleared
        mock_chat_store.clear_all.assert_called_once()
        mock_context_store.clear_all.assert_called_once()

        # Verify result
        assert result["conversations_deleted"] == 3
        assert result["messages_deleted"] == 10
        assert result["contexts_deleted"] == 25
        assert result["context_tracking_deleted"] == 5
        assert "43 records" in result["message"]  # 3 + 10 + 25 + 5 = 43

    def test_handles_missing_chat_db(self, mocker, tmp_path):
        """Test handler when chat database doesn't exist."""
        context_db = str(tmp_path / "context.db")

        mock_config = MagicMock()
        mock_config.get_path.side_effect = lambda key, default: {
            "CHAT_DB_PATH": "/nonexistent/chat.db",
            "CONTEXT_DB_PATH": context_db,
        }.get(key, default)

        mocker.patch("libs.control_handlers.clear.Config", return_value=mock_config)

        # Mock ContextStore
        mock_context_store = MagicMock()
        mock_context_store.clear_all.return_value = 5
        mocker.patch(
            "libs.control_handlers.clear.ContextStore", return_value=mock_context_store
        )

        # os.path.exists returns False for chat.db, True for context.db
        def exists_side_effect(path):
            return path == context_db

        mocker.patch("os.path.exists", side_effect=exists_side_effect)

        from libs.control_handlers.clear import handle

        result = handle({})

        # Only context should be cleared
        assert result["conversations_deleted"] == 0
        assert result["messages_deleted"] == 0
        assert result["contexts_deleted"] == 5

    def test_handles_missing_context_db(self, mocker, tmp_path):
        """Test handler when context database doesn't exist."""
        chat_db = str(tmp_path / "chat.db")

        mock_config = MagicMock()
        mock_config.get_path.side_effect = lambda key, default: {
            "CHAT_DB_PATH": chat_db,
            "CONTEXT_DB_PATH": "/nonexistent/context.db",
        }.get(key, default)

        mocker.patch("libs.control_handlers.clear.Config", return_value=mock_config)

        # Mock ChatStore
        mock_chat_store = MagicMock()
        mock_chat_store.clear_all.return_value = (
            2,
            8,
            3,
        )  # 2 convs, 8 msgs, 3 ctx tracking
        mocker.patch(
            "libs.control_handlers.clear.ChatStore", return_value=mock_chat_store
        )

        # os.path.exists returns True for chat.db, False for context.db
        def exists_side_effect(path):
            return path == chat_db

        mocker.patch("os.path.exists", side_effect=exists_side_effect)

        from libs.control_handlers.clear import handle

        result = handle({})

        # Only chat should be cleared
        assert result["conversations_deleted"] == 2
        assert result["messages_deleted"] == 8
        assert result["contexts_deleted"] == 0

    def test_handles_both_databases_missing(self, mocker):
        """Test handler when both databases don't exist."""
        mock_config = MagicMock()
        mock_config.get_path.return_value = "/nonexistent/db.db"

        mocker.patch("libs.control_handlers.clear.Config", return_value=mock_config)
        mocker.patch("os.path.exists", return_value=False)

        from libs.control_handlers.clear import handle

        result = handle({})

        # Nothing should be deleted
        assert result["conversations_deleted"] == 0
        assert result["messages_deleted"] == 0
        assert result["contexts_deleted"] == 0
        assert "0 records" in result["message"]

    def test_empty_payload(self, mocker, tmp_path):
        """Test handler accepts empty payload."""
        chat_db = str(tmp_path / "chat.db")
        context_db = str(tmp_path / "context.db")

        mock_config = MagicMock()
        mock_config.get_path.side_effect = lambda key, default: {
            "CHAT_DB_PATH": chat_db,
            "CONTEXT_DB_PATH": context_db,
        }.get(key, default)

        mocker.patch("libs.control_handlers.clear.Config", return_value=mock_config)
        mocker.patch("os.path.exists", return_value=False)

        from libs.control_handlers.clear import handle

        # Should not raise any error
        result = handle({})
        assert "message" in result
