"""
Tool Client for AI Tool Use

This module provides a client for communicating with AI tool servers.
Tool servers implement a simple HTTP protocol:
  - GET /health   - Health check
  - GET /describe - Tool metadata
  - POST /call    - Execute the tool

The ToolClient discovers available tools, converts their metadata to OpenAI
format, and handles tool execution during AI inference.

NOTE: This is for AI TOOL USE where the model invokes functions during inference.
This is separate from the slash command system (libs/control_handlers) which
handles user-invoked commands.
"""

import json
import logging
from typing import Any, Optional

import requests

from libs.config import Config

logger = logging.getLogger(__name__)

# Type mapping from simple types to JSON Schema types
_TYPE_MAP = {
    "string": "string",
    "str": "string",
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "double": "number",
    "number": "number",
    "bool": "boolean",
    "boolean": "boolean",
    "array": "array",
    "list": "array",
    "object": "object",
    "dict": "object",
}


class ToolClient:
    """
    Client for discovering and calling AI tool servers.

    Handles:
    - Discovery: Check health and fetch metadata from configured tools
    - Conversion: Convert simple tool metadata to OpenAI function format
    - Execution: Call tools and return results

    Usage:
        client = ToolClient(config)
        tools = client.discover_tools()  # Returns OpenAI-format tool definitions
        result = client.call_tool("write", {"content": "hello", "filename": "test.txt"})
    """

    def __init__(self, config: Config, timeout: float = 5.0):
        """
        Initialize the tool client.

        Args:
            config: Configuration instance with tool settings
            timeout: HTTP request timeout in seconds
        """
        self.config = config
        self.timeout = timeout
        self._tools_cache: dict[str, dict] = {}  # name -> metadata
        self._openai_cache: dict[str, dict] = {}  # name -> OpenAI format

    def discover_tools(self) -> list[dict]:
        """
        Discover all enabled and healthy tool servers.

        For each enabled tool in settings.json:
        1. Check /health endpoint
        2. If healthy, fetch /describe metadata
        3. Convert to OpenAI tool format

        Returns:
            List of OpenAI-format tool definitions for use in API calls
        """
        tools = self.config.get_tools()
        if not tools:
            logger.debug("No tools configured in settings.json")
            return []

        openai_tools = []
        self._tools_cache.clear()
        self._openai_cache.clear()

        for name, tool_config in tools.items():
            port = tool_config.get("port")
            if not port:
                logger.warning(f"Tool '{name}' has no port configured, skipping")
                continue

            base_url = f"http://localhost:{port}"

            # Check health
            if not self._check_health(name, base_url):
                logger.warning(f"Tool '{name}' is not healthy, skipping")
                continue

            # Fetch metadata
            metadata = self._fetch_describe(name, base_url)
            if not metadata:
                logger.warning(f"Tool '{name}' returned no metadata, skipping")
                continue

            # Store in cache
            self._tools_cache[name] = {**metadata, "_base_url": base_url}

            # Convert to OpenAI format
            openai_format = self._to_openai_format(metadata)
            self._openai_cache[name] = openai_format
            openai_tools.append(openai_format)

            logger.info(f"Discovered tool '{name}' at {base_url}")

        logger.info(f"Discovered {len(openai_tools)} tools")
        return openai_tools

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        Call a tool by name with the given arguments.

        Args:
            name: The tool name (e.g., "write")
            arguments: The arguments to pass to the tool

        Returns:
            Dict with 'success', 'result' or 'error' keys

        Raises:
            ValueError: If tool is not known
            requests.RequestException: If HTTP call fails
        """
        if name not in self._tools_cache:
            raise ValueError(f"Unknown tool: {name}")

        tool_meta = self._tools_cache[name]
        base_url = tool_meta["_base_url"]

        try:
            response = requests.post(
                f"{base_url}/call",
                json=arguments,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"Tool '{name}' returned: {result.get('success', False)}")
            return result
        except requests.Timeout:
            logger.error(f"Tool '{name}' timed out")
            return {
                "success": False,
                "error": {"type": "Timeout", "message": f"Tool '{name}' timed out"},
            }
        except requests.RequestException as e:
            logger.error(f"Tool '{name}' request failed: {e}")
            return {
                "success": False,
                "error": {"type": "RequestError", "message": str(e)},
            }
        except json.JSONDecodeError as e:
            logger.error(f"Tool '{name}' returned invalid JSON: {e}")
            return {
                "success": False,
                "error": {"type": "JSONError", "message": str(e)},
            }

    def get_tool_names(self) -> list[str]:
        """Return list of discovered tool names."""
        return list(self._tools_cache.keys())

    def is_tool_available(self, name: str) -> bool:
        """Check if a tool is available (discovered and healthy)."""
        return name in self._tools_cache

    def _check_health(self, name: str, base_url: str) -> bool:
        """
        Check if a tool server is healthy.

        Args:
            name: Tool name (for logging)
            base_url: The tool server base URL

        Returns:
            True if healthy, False otherwise
        """
        try:
            response = requests.get(f"{base_url}/health", timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return data.get("status") == "ok"
        except Exception as e:
            logger.debug(f"Tool '{name}' health check failed: {e}")
            return False

    def _fetch_describe(self, name: str, base_url: str) -> Optional[dict]:
        """
        Fetch tool metadata from /describe endpoint.

        Args:
            name: Tool name (for logging)
            base_url: The tool server base URL

        Returns:
            Tool metadata dict or None if failed
        """
        try:
            response = requests.get(f"{base_url}/describe", timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Tool '{name}' describe failed: {e}")
            return None

    def _to_openai_format(self, metadata: dict) -> dict:
        """
        Convert simple tool metadata to OpenAI function calling format.

        Input format:
            {
                "name": "write",
                "description": "...",
                "parameters": [
                    {"name": "content", "type": "string", "required": true, "description": "..."}
                ]
            }

        Output format:
            {
                "type": "function",
                "function": {
                    "name": "write",
                    "description": "...",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string", "description": "..."}
                        },
                        "required": ["content"]
                    }
                }
            }
        """
        name = metadata.get("name", "unknown")
        description = metadata.get("description", "")
        params = metadata.get("parameters", [])

        # Build properties and required list
        properties = {}
        required = []

        for param in params:
            param_name = param.get("name")
            if not param_name:
                continue

            param_type = param.get("type", "string")
            json_type = _TYPE_MAP.get(param_type.lower(), "string")

            properties[param_name] = {
                "type": json_type,
                "description": param.get("description", ""),
            }

            if param.get("required", False):
                required.append(param_name)

        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
