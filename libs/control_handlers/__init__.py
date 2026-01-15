"""
Control Handler Discovery and Dispatch System for Olorin Project

This module provides automatic discovery and registration of control handlers.
Each handler is a Python module with a `handle(payload: dict) -> dict` function.

Handlers can optionally define metadata for slash command exposure:
    DESCRIPTION: str - Human-readable description of the command
    ARGUMENTS: List[dict] - List of argument specifications, each with:
        - name: str - Argument name (used as key in payload)
        - type: str - Type hint ("bool", "int", "float", "str", "list", "dict")
        - required: bool - Whether the argument is required (default: False)
        - default: Any - Default value if not provided
        - description: str - Human-readable description

Usage:
    from libs.control_handlers import discover_handlers, get_handler, list_commands
    from libs.control_handlers import get_command_meta, get_all_commands_meta

    # Discover all handlers
    handlers = discover_handlers()

    # Get a specific handler
    handler = get_handler("stop-broca-audio-play")
    result = handler({"force": True})

    # List available commands
    commands = list_commands()

    # Get metadata for slash command integration
    meta = get_command_meta("stop-broca-audio-play")
    all_meta = get_all_commands_meta()
"""

import importlib
import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Type alias for handler functions
HandlerFunc = Callable[[dict], dict]


@dataclass
class ArgumentSpec:
    """Specification for a command argument."""

    name: str
    type: str = "str"  # "bool", "int", "float", "str", "list", "dict"
    required: bool = False
    default: Any = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "type": self.type,
            "required": self.required,
            "default": self.default,
            "description": self.description,
        }


@dataclass
class CommandMeta:
    """Metadata for a control command (for slash command exposure)."""

    command: str  # Command name (e.g., "stop-broca-audio-play")
    slash_command: str  # Slash command format (e.g., "/stop-broca-audio-play")
    description: str = ""
    arguments: List[ArgumentSpec] = field(default_factory=list)
    has_handler: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "command": self.command,
            "slash_command": self.slash_command,
            "description": self.description,
            "arguments": [arg.to_dict() for arg in self.arguments],
            "has_handler": self.has_handler,
        }


# Cache for discovered handlers and metadata
_handlers_cache: Optional[Dict[str, HandlerFunc]] = None
_metadata_cache: Optional[Dict[str, CommandMeta]] = None


def _filename_to_command(filename: str) -> str:
    """
    Convert filename to command name (underscores to dashes).

    Args:
        filename: The filename without extension (e.g., "stop_broca_audio_play")

    Returns:
        Command name with dashes (e.g., "stop-broca-audio-play")

    Examples:
        >>> _filename_to_command("stop_broca_audio_play")
        'stop-broca-audio-play'
        >>> _filename_to_command("reload_config")
        'reload-config'
    """
    return filename.replace("_", "-")


def _extract_metadata(module: Any, command_name: str) -> CommandMeta:
    """
    Extract metadata from a handler module.

    Args:
        module: The imported module
        command_name: The command name (e.g., "stop-broca-audio-play")

    Returns:
        CommandMeta instance with extracted or default metadata
    """
    # Extract description
    description = getattr(module, "DESCRIPTION", "")

    # Extract and parse arguments
    arguments_raw = getattr(module, "ARGUMENTS", [])
    arguments = []
    for arg in arguments_raw:
        if isinstance(arg, dict):
            arguments.append(
                ArgumentSpec(
                    name=arg.get("name", ""),
                    type=arg.get("type", "str"),
                    required=arg.get("required", False),
                    default=arg.get("default"),
                    description=arg.get("description", ""),
                )
            )

    return CommandMeta(
        command=command_name,
        slash_command=f"/{command_name}",
        description=description,
        arguments=arguments,
        has_handler=True,
    )


def discover_handlers() -> Dict[str, HandlerFunc]:
    """
    Auto-discover and load all handlers from this package.

    Scans the package directory for .py files (not starting with _),
    imports each module, and if the module has a 'handle' function,
    registers it. Command name is derived from filename with underscores
    converted to dashes.

    Also extracts metadata (DESCRIPTION, ARGUMENTS) for slash command support.

    Returns:
        Dict mapping command names to handler functions.
        The result is cached for subsequent calls.

    Example:
        >>> handlers = discover_handlers()
        >>> "stop-broca-audio-play" in handlers
        True
    """
    global _handlers_cache, _metadata_cache

    if _handlers_cache is not None:
        return _handlers_cache

    handlers: Dict[str, HandlerFunc] = {}
    metadata: Dict[str, CommandMeta] = {}

    # Get the directory containing this __init__.py
    package_dir = Path(__file__).parent

    # Scan for handler modules
    for py_file in package_dir.glob("*.py"):
        # Skip private modules (starting with _)
        if py_file.name.startswith("_"):
            continue

        # Get module name without extension
        module_name = py_file.stem

        # Import the module
        spec = importlib.util.spec_from_file_location(
            f"libs.control_handlers.{module_name}", py_file
        )
        if spec is None or spec.loader is None:
            continue

        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception:
            # Skip modules that fail to import
            continue

        # Check if module has a handle function
        if hasattr(module, "handle") and callable(module.handle):
            command_name = _filename_to_command(module_name)
            handlers[command_name] = module.handle
            # Extract metadata
            metadata[command_name] = _extract_metadata(module, command_name)

    _handlers_cache = handlers
    _metadata_cache = metadata
    return handlers


def get_handler(command: str) -> HandlerFunc:
    """
    Get handler for a command.

    Args:
        command: The command name (e.g., "stop-broca-audio-play")

    Returns:
        The handler function for the command

    Raises:
        KeyError: If command is not found
    """
    handlers = discover_handlers()
    if command not in handlers:
        raise KeyError(f"Unknown command: {command}")
    return handlers[command]


def list_commands() -> List[str]:
    """
    Return list of available command names.

    Returns:
        Sorted list of command names
    """
    handlers = discover_handlers()
    return sorted(handlers.keys())


def get_command_meta(command: str) -> CommandMeta:
    """
    Get metadata for a specific command.

    Args:
        command: The command name (e.g., "stop-broca-audio-play")

    Returns:
        CommandMeta instance for the command

    Raises:
        KeyError: If command is not found
    """
    # Ensure discovery has run
    discover_handlers()

    if _metadata_cache is None or command not in _metadata_cache:
        raise KeyError(f"Unknown command: {command}")
    return _metadata_cache[command]


def get_all_commands_meta() -> Dict[str, CommandMeta]:
    """
    Get metadata for all discovered commands.

    Returns:
        Dict mapping command names to CommandMeta instances
    """
    # Ensure discovery has run
    discover_handlers()
    return _metadata_cache.copy() if _metadata_cache else {}


def clear_cache() -> None:
    """
    Clear the handlers and metadata cache.

    This is primarily useful for testing to force re-discovery.
    """
    global _handlers_cache, _metadata_cache
    _handlers_cache = None
    _metadata_cache = None
