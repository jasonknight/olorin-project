"""
Slash Command API Server for Olorin Project

HTTP JSON API server for cross-language slash command execution.
Exposes control handlers as slash commands that can be queried
and executed from any language (e.g., Rust chat client).

NOTE: This server is for USER-INVOKED SLASH COMMANDS (e.g., /stop, /write, /clear)
triggered by the user via the chat client. This is SEPARATE from AI tool use, which
allows the AI model to call functions during inference. Tool use is handled in
cortex/consumer.py by passing tool definitions to the OpenAI API and processing
tool_calls in responses.

Endpoints:
    GET  /commands          - List all available commands with metadata
    GET  /commands/{name}   - Get metadata for a specific command
    POST /execute           - Execute a command
    GET  /health            - Health check endpoint

Usage:
    from libs.control_server import ControlServer

    # Create and start server (blocking)
    server = ControlServer(port=8765)
    server.start()

    # Or run in a background thread
    server = ControlServer(port=8765)
    server.start_background()
    # ... do other work ...
    server.stop()
"""

import json
import logging
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, Optional
from urllib.parse import parse_qs, urlparse

from libs.control_handlers import (
    discover_handlers,
    get_all_commands_meta,
    get_command_meta,
    get_handler,
)

logger = logging.getLogger(__name__)


def _map_positional_to_named(command: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map positional arguments to named parameters based on ARGUMENTS metadata.

    The Rust client passes positional arguments in a "_positional" array.
    This function maps them to named parameters in order of ARGUMENTS definition.

    Args:
        command: The command name
        payload: The payload dict, possibly containing "_positional" array

    Returns:
        Updated payload with positional args mapped to named parameters
    """
    if "_positional" not in payload:
        return payload

    positional = payload.get("_positional")
    if not isinstance(positional, list) or len(positional) == 0:
        # Strip _positional key but keep other keys
        return {k: v for k, v in payload.items() if k != "_positional"}

    try:
        meta = get_command_meta(command)
    except KeyError:
        # Command not found, return payload as-is
        return payload

    # Create a copy to avoid modifying the original
    result = {k: v for k, v in payload.items() if k != "_positional"}

    # Map positional args to named parameters in order
    for i, value in enumerate(positional):
        if i < len(meta.arguments):
            arg_name = meta.arguments[i].name
            # Only set if not already provided as a named argument
            if arg_name not in result:
                result[arg_name] = value
        else:
            # More positional args than defined parameters - store remainder
            # in "_extra" for handlers that might want them
            if "_extra" not in result:
                result["_extra"] = []
            result["_extra"].append(value)

    return result


class ControlRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the control API."""

    # Reference to server for accessing shared state
    server: "ControlServer"

    def log_message(self, format: str, *args: Any) -> None:
        """Override to use our logger instead of stderr."""
        logger.debug(f"{self.address_string()} - {format % args}")

    def _send_json_response(
        self,
        data: Dict[str, Any],
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        """Send a JSON response with appropriate headers."""
        response_body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(response_body)

    def _send_error_response(
        self,
        message: str,
        status: HTTPStatus = HTTPStatus.BAD_REQUEST,
    ) -> None:
        """Send an error response."""
        self._send_json_response(
            {"success": False, "error": message},
            status=status,
        )

    def _parse_path(self) -> tuple[str, Dict[str, str]]:
        """Parse the request path and query parameters."""
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        return path, query

    def do_OPTIONS(self) -> None:
        """Handle CORS preflight requests."""
        self.send_response(HTTPStatus.OK)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "86400")
        self.end_headers()

    def do_GET(self) -> None:
        """Handle GET requests."""
        path, query = self._parse_path()

        # Health check
        if path == "/health":
            self._send_json_response({"status": "ok", "timestamp": time.time()})
            return

        # List all commands
        if path == "/commands":
            try:
                all_meta = get_all_commands_meta()
                commands = [meta.to_dict() for meta in all_meta.values()]
                commands.sort(key=lambda c: c["command"])
                self._send_json_response(
                    {"success": True, "commands": commands, "count": len(commands)}
                )
            except Exception as e:
                logger.error(f"Error listing commands: {e}")
                self._send_error_response(
                    f"Failed to list commands: {e}",
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            return

        # Get specific command metadata
        if path.startswith("/commands/"):
            command_name = path[len("/commands/") :]
            try:
                meta = get_command_meta(command_name)
                self._send_json_response({"success": True, "command": meta.to_dict()})
            except KeyError:
                self._send_error_response(
                    f"Unknown command: {command_name}",
                    HTTPStatus.NOT_FOUND,
                )
            except Exception as e:
                logger.error(f"Error getting command metadata: {e}")
                self._send_error_response(
                    f"Failed to get command: {e}",
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            return

        # Not found
        self._send_error_response(f"Not found: {path}", HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        """Handle POST requests."""
        path, query = self._parse_path()

        # Execute command
        if path == "/execute":
            self._handle_execute()
            return

        # Not found
        self._send_error_response(f"Not found: {path}", HTTPStatus.NOT_FOUND)

    def _handle_execute(self) -> None:
        """Handle command execution."""
        # Read request body
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self._send_error_response("Request body required")
            return

        try:
            body = self.rfile.read(content_length)
            request = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as e:
            self._send_error_response(f"Invalid JSON: {e}")
            return

        # Extract command and payload
        command = request.get("command")
        if not command:
            self._send_error_response("Missing 'command' field")
            return

        payload = request.get("payload", {})
        if not isinstance(payload, dict):
            self._send_error_response("'payload' must be an object")
            return

        # Map positional arguments to named parameters based on ARGUMENTS metadata
        payload = _map_positional_to_named(command, payload)

        # Execute the command
        start_time = time.time()
        try:
            handler = get_handler(command)
            result = handler(payload)
            duration_ms = (time.time() - start_time) * 1000

            self._send_json_response(
                {
                    "success": True,
                    "command": command,
                    "result": result,
                    "duration_ms": round(duration_ms, 2),
                }
            )
            logger.info(f"Executed command '{command}' in {duration_ms:.2f}ms")

        except KeyError:
            self._send_error_response(
                f"Unknown command: {command}",
                HTTPStatus.NOT_FOUND,
            )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Command '{command}' failed: {e}")
            self._send_error_response(
                f"Command execution failed: {type(e).__name__}: {e}",
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )


class ControlServer:
    """
    HTTP server for the control API.

    Runs a threaded HTTP server that exposes control handlers
    as a JSON API for cross-language integration.
    """

    def __init__(
        self,
        port: int = 8765,
        host: str = "0.0.0.0",
        handler_executor: Optional[Callable[[str, Dict], Dict]] = None,
    ):
        """
        Initialize the control server.

        Args:
            port: Port to listen on (default: 8765)
            host: Host to bind to (default: 0.0.0.0 for all interfaces)
            handler_executor: Optional custom executor for commands.
                             If None, uses direct handler invocation.
        """
        self.port = port
        self.host = host
        self.handler_executor = handler_executor
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()

        # Ensure handlers are discovered on init
        discover_handlers()

    def start(self) -> None:
        """
        Start the server (blocking).

        This method blocks until the server is stopped.
        For non-blocking operation, use start_background().
        """
        if self._running.is_set():
            logger.warning("Server already running")
            return

        logger.info(f"Starting control API server on {self.host}:{self.port}")

        self._server = ThreadingHTTPServer(
            (self.host, self.port),
            ControlRequestHandler,
        )
        self._running.set()

        try:
            self._server.serve_forever()
        except KeyboardInterrupt:
            logger.info("Server interrupted")
        finally:
            self._running.clear()

    def start_background(self) -> None:
        """
        Start the server in a background thread.

        Returns immediately. Use stop() to shut down.
        """
        if self._running.is_set():
            logger.warning("Server already running")
            return

        self._thread = threading.Thread(
            target=self._run_server,
            daemon=True,
            name="control-api-server",
        )
        self._thread.start()

        # Wait for server to start
        self._running.wait(timeout=5.0)
        logger.info(f"Control API server started on {self.host}:{self.port}")

    def _run_server(self) -> None:
        """Internal method to run the server (called from background thread)."""
        self._server = ThreadingHTTPServer(
            (self.host, self.port),
            ControlRequestHandler,
        )
        self._running.set()

        try:
            self._server.serve_forever()
        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            self._running.clear()

    def stop(self, timeout: float = 5.0) -> None:
        """
        Stop the server gracefully.

        Args:
            timeout: Maximum time to wait for shutdown in seconds
        """
        if not self._running.is_set():
            return

        logger.info("Stopping control API server...")

        if self._server is not None:
            self._server.shutdown()

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)

        self._running.clear()
        logger.info("Control API server stopped")

    @property
    def is_running(self) -> bool:
        """Check if the server is running."""
        return self._running.is_set()

    @property
    def url(self) -> str:
        """Get the base URL of the server."""
        return f"http://{self.host}:{self.port}"


# Convenience function for quick testing
def main() -> None:
    """Run the control server standalone for testing."""
    import sys

    # Set up basic logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765

    server = ControlServer(port=port)
    print(f"Starting control API server on port {port}...")
    print(f"  GET  {server.url}/commands")
    print(f"  GET  {server.url}/commands/<name>")
    print(f"  POST {server.url}/execute")
    print(f"  GET  {server.url}/health")
    print("Press Ctrl+C to stop")

    try:
        server.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.stop()


if __name__ == "__main__":
    main()
