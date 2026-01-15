"""
Unit tests for the Control API Server.

Tests the HTTP JSON API for slash command discovery and execution.
"""

import json
import sys
import time
from pathlib import Path

import pytest
import urllib.request
import urllib.error

# Add libs to path for imports
libs_path = Path(__file__).parent.parent.parent
if str(libs_path) not in sys.path:
    sys.path.insert(0, str(libs_path))

from libs.control_handlers import (  # noqa: E402
    ArgumentSpec,
    CommandMeta,
    clear_cache,
    get_all_commands_meta,
    get_command_meta,
)
from libs.control_server import ControlServer  # noqa: E402


# =============================================================================
# Test fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_handler_cache():
    """Reset the handler cache before each test."""
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def test_port():
    """Get a unique port for testing."""
    # Use a port in the high range to avoid conflicts
    import random

    return random.randint(19000, 19999)


@pytest.fixture
def server(test_port):
    """Create and start a test server."""
    srv = ControlServer(port=test_port, host="127.0.0.1")
    srv.start_background()
    # Give the server a moment to start
    time.sleep(0.1)
    yield srv
    srv.stop()


# =============================================================================
# Test CommandMeta and ArgumentSpec
# =============================================================================


class TestArgumentSpec:
    """Tests for the ArgumentSpec dataclass."""

    def test_default_values(self):
        """Test that ArgumentSpec has correct defaults."""
        arg = ArgumentSpec(name="test")
        assert arg.name == "test"
        assert arg.type == "str"
        assert arg.required is False
        assert arg.default is None
        assert arg.description == ""

    def test_to_dict(self):
        """Test ArgumentSpec.to_dict() serialization."""
        arg = ArgumentSpec(
            name="force",
            type="bool",
            required=False,
            default=False,
            description="Use force",
        )
        d = arg.to_dict()
        assert d["name"] == "force"
        assert d["type"] == "bool"
        assert d["required"] is False
        assert d["default"] is False
        assert d["description"] == "Use force"


class TestCommandMeta:
    """Tests for the CommandMeta dataclass."""

    def test_default_values(self):
        """Test that CommandMeta has correct defaults."""
        meta = CommandMeta(command="test-cmd", slash_command="/test-cmd")
        assert meta.command == "test-cmd"
        assert meta.slash_command == "/test-cmd"
        assert meta.description == ""
        assert meta.arguments == []
        assert meta.has_handler is True

    def test_to_dict(self):
        """Test CommandMeta.to_dict() serialization."""
        meta = CommandMeta(
            command="stop-audio",
            slash_command="/stop-audio",
            description="Stop audio playback",
            arguments=[ArgumentSpec(name="force", type="bool")],
            has_handler=True,
        )
        d = meta.to_dict()
        assert d["command"] == "stop-audio"
        assert d["slash_command"] == "/stop-audio"
        assert d["description"] == "Stop audio playback"
        assert len(d["arguments"]) == 1
        assert d["arguments"][0]["name"] == "force"
        assert d["has_handler"] is True


# =============================================================================
# Test metadata discovery
# =============================================================================


class TestMetadataDiscovery:
    """Tests for handler metadata discovery."""

    def test_get_command_meta_returns_metadata(self):
        """Test that get_command_meta returns correct metadata."""
        meta = get_command_meta("stop-broca-audio-play")
        assert meta.command == "stop-broca-audio-play"
        assert meta.slash_command == "/stop-broca-audio-play"
        assert meta.description != ""  # Should have description
        assert len(meta.arguments) > 0  # Should have force argument

    def test_get_command_meta_unknown_raises(self):
        """Test that get_command_meta raises KeyError for unknown commands."""
        with pytest.raises(KeyError):
            get_command_meta("unknown-command")

    def test_get_all_commands_meta(self):
        """Test that get_all_commands_meta returns all metadata."""
        all_meta = get_all_commands_meta()
        assert isinstance(all_meta, dict)
        assert "stop-broca-audio-play" in all_meta
        assert isinstance(all_meta["stop-broca-audio-play"], CommandMeta)


# =============================================================================
# Test ControlServer lifecycle
# =============================================================================


class TestControlServerLifecycle:
    """Tests for ControlServer start/stop."""

    def test_start_background(self, test_port):
        """Test that server starts in background."""
        srv = ControlServer(port=test_port, host="127.0.0.1")
        assert not srv.is_running

        srv.start_background()
        time.sleep(0.1)

        assert srv.is_running
        srv.stop()
        assert not srv.is_running

    def test_stop_when_not_running(self, test_port):
        """Test that stop() is safe when server isn't running."""
        srv = ControlServer(port=test_port, host="127.0.0.1")
        # Should not raise
        srv.stop()
        srv.stop()

    def test_url_property(self, test_port):
        """Test that url property returns correct URL."""
        srv = ControlServer(port=test_port, host="127.0.0.1")
        assert srv.url == f"http://127.0.0.1:{test_port}"


# =============================================================================
# Test HTTP endpoints
# =============================================================================


class TestHealthEndpoint:
    """Tests for GET /health endpoint."""

    def test_health_check(self, server, test_port):
        """Test that health endpoint returns OK."""
        url = f"http://127.0.0.1:{test_port}/health"
        with urllib.request.urlopen(url, timeout=5) as response:
            data = json.loads(response.read().decode())
            assert data["status"] == "ok"
            assert "timestamp" in data


class TestCommandsEndpoint:
    """Tests for GET /commands endpoint."""

    def test_list_commands(self, server, test_port):
        """Test that /commands lists all commands."""
        url = f"http://127.0.0.1:{test_port}/commands"
        with urllib.request.urlopen(url, timeout=5) as response:
            data = json.loads(response.read().decode())
            assert data["success"] is True
            assert "commands" in data
            assert isinstance(data["commands"], list)
            assert data["count"] == len(data["commands"])

            # Check for stop-broca-audio-play
            commands = {c["command"]: c for c in data["commands"]}
            assert "stop-broca-audio-play" in commands
            assert (
                commands["stop-broca-audio-play"]["slash_command"]
                == "/stop-broca-audio-play"
            )

    def test_get_specific_command(self, server, test_port):
        """Test that /commands/<name> returns specific command metadata."""
        url = f"http://127.0.0.1:{test_port}/commands/stop-broca-audio-play"
        with urllib.request.urlopen(url, timeout=5) as response:
            data = json.loads(response.read().decode())
            assert data["success"] is True
            assert data["command"]["command"] == "stop-broca-audio-play"
            assert data["command"]["slash_command"] == "/stop-broca-audio-play"

    def test_get_unknown_command(self, server, test_port):
        """Test that /commands/<name> returns 404 for unknown command."""
        url = f"http://127.0.0.1:{test_port}/commands/unknown-command"
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(url, timeout=5)
        assert exc_info.value.code == 404


class TestExecuteEndpoint:
    """Tests for POST /execute endpoint."""

    def test_execute_command_success(self, server, test_port, mocker):
        """Test successful command execution."""
        # Mock the handler
        mock_result = {"pid_killed": 12345, "was_playing": True, "message": "OK"}
        mocker.patch(
            "libs.control_server.get_handler",
            return_value=lambda payload: mock_result,
        )

        url = f"http://127.0.0.1:{test_port}/execute"
        request_body = json.dumps(
            {
                "command": "stop-broca-audio-play",
                "payload": {"force": True},
            }
        ).encode()

        req = urllib.request.Request(
            url,
            data=request_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            assert data["success"] is True
            assert data["command"] == "stop-broca-audio-play"
            assert data["result"] == mock_result
            assert "duration_ms" in data

    def test_execute_missing_command_field(self, server, test_port):
        """Test execute with missing command field."""
        url = f"http://127.0.0.1:{test_port}/execute"
        request_body = json.dumps({"payload": {}}).encode()

        req = urllib.request.Request(
            url,
            data=request_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=5)
        assert exc_info.value.code == 400

    def test_execute_invalid_json(self, server, test_port):
        """Test execute with invalid JSON body."""
        url = f"http://127.0.0.1:{test_port}/execute"
        request_body = b"not valid json"

        req = urllib.request.Request(
            url,
            data=request_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=5)
        assert exc_info.value.code == 400

    def test_execute_unknown_command(self, server, test_port):
        """Test execute with unknown command."""
        url = f"http://127.0.0.1:{test_port}/execute"
        request_body = json.dumps(
            {
                "command": "unknown-command",
                "payload": {},
            }
        ).encode()

        req = urllib.request.Request(
            url,
            data=request_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=5)
        assert exc_info.value.code == 404

    def test_execute_handler_error(self, server, test_port, mocker):
        """Test execute when handler raises an exception."""
        mocker.patch(
            "libs.control_server.get_handler",
            return_value=lambda payload: (_ for _ in ()).throw(
                ValueError("test error")
            ),
        )

        url = f"http://127.0.0.1:{test_port}/execute"
        request_body = json.dumps(
            {
                "command": "stop-broca-audio-play",
                "payload": {},
            }
        ).encode()

        req = urllib.request.Request(
            url,
            data=request_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=5)
        assert exc_info.value.code == 500


class TestNotFoundEndpoint:
    """Tests for unknown endpoints."""

    def test_unknown_path(self, server, test_port):
        """Test that unknown paths return 404."""
        url = f"http://127.0.0.1:{test_port}/unknown/path"
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(url, timeout=5)
        assert exc_info.value.code == 404
