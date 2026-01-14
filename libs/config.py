"""
Unified Configuration Library for Olorin Project

Provides a Config class with type-safe getters and optional hot-reload support.
"""

import os
from pathlib import Path
from typing import Any, Optional, Union


def _find_project_root() -> Path:
    """Find the project root by looking for the .env file or known directories."""
    current = Path(__file__).resolve().parent

    # Go up from libs/ to project root
    if current.name == 'libs':
        project_root = current.parent
        if (project_root / '.env').exists():
            return project_root

    # Fallback: search upward for .env
    search = current
    for _ in range(5):  # Limit search depth
        if (search / '.env').exists():
            return search
        if search.parent == search:
            break
        search = search.parent

    # Last resort: return current working directory
    return Path.cwd()


class Config:
    """
    Configuration manager that loads from a .env file.

    Provides type-safe getters and optional hot-reload support for runtime
    configuration changes.

    Usage:
        config = Config()
        host = config.get('CHROMADB_HOST', 'localhost')
        port = config.get_int('CHROMADB_PORT', 8000)
        enabled = config.get_bool('FEATURE_ENABLED', False)
        path = config.get_path('INPUT_DIR', '~/Documents')

        # With hot-reload support
        config = Config(watch=True)
        if config.reload():
            print("Configuration was updated")
    """

    def __init__(self, env_path: Optional[Union[str, Path]] = None, watch: bool = False):
        """
        Initialize the configuration manager.

        Args:
            env_path: Path to the .env file. Defaults to project root .env
            watch: If True, enables hot-reload support via reload() method
        """
        if env_path is None:
            self._env_path = _find_project_root() / '.env'
        else:
            self._env_path = Path(env_path).resolve()

        self._watch = watch
        self._mtime: Optional[float] = None
        self._overrides: dict[str, str] = {}

        self._load()

    def _load(self) -> None:
        """Load the .env file into environment variables."""
        if not self._env_path.exists():
            return

        self._mtime = self._env_path.stat().st_mtime

        with open(self._env_path, 'r') as f:
            for line in f:
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue

                # Parse KEY=value format
                if '=' in line:
                    key, _, value = line.partition('=')
                    key = key.strip()
                    value = value.strip()

                    # Remove surrounding quotes if present
                    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                        value = value[1:-1]

                    os.environ[key] = value

    def reload(self) -> bool:
        """
        Reload configuration if the .env file has changed.

        Returns:
            True if configuration was reloaded, False otherwise
        """
        if not self._watch or not self._env_path.exists():
            return False

        current_mtime = self._env_path.stat().st_mtime

        if current_mtime != self._mtime:
            self._load()
            return True

        return False

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get a configuration value as a string.

        Args:
            key: The environment variable name
            default: Default value if not set

        Returns:
            The configuration value or default
        """
        # Check overrides first
        if key in self._overrides:
            return self._overrides[key]

        return os.environ.get(key, default)

    def get_int(self, key: str, default: Optional[int] = None) -> Optional[int]:
        """
        Get a configuration value as an integer.

        Args:
            key: The environment variable name
            default: Default value if not set or invalid

        Returns:
            The configuration value as int or default
        """
        value = self.get(key)
        if value is None:
            return default

        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    def get_float(self, key: str, default: Optional[float] = None) -> Optional[float]:
        """
        Get a configuration value as a float.

        Args:
            key: The environment variable name
            default: Default value if not set or invalid

        Returns:
            The configuration value as float or default
        """
        value = self.get(key)
        if value is None:
            return default

        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def get_bool(self, key: str, default: bool = False) -> bool:
        """
        Get a configuration value as a boolean.

        Recognizes: true, yes, 1, on (case-insensitive) as True
        Everything else (including empty string) is False

        Args:
            key: The environment variable name
            default: Default value if not set

        Returns:
            The configuration value as bool or default
        """
        value = self.get(key)
        if value is None:
            return default

        return value.lower() in ('true', 'yes', '1', 'on')

    def get_path(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get a configuration value as an expanded path.

        Expands ~ to the user's home directory.

        Args:
            key: The environment variable name
            default: Default value if not set

        Returns:
            The expanded path or default
        """
        value = self.get(key, default)
        if value is None:
            return None

        return os.path.expanduser(value)

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value (in-memory only).

        This override takes precedence over environment variables.
        Does not modify the .env file.

        Args:
            key: The environment variable name
            value: The value to set
        """
        self._overrides[key] = str(value)

    def clear_override(self, key: str) -> None:
        """
        Clear an in-memory override for a key.

        After clearing, get() will return the environment variable value.

        Args:
            key: The environment variable name
        """
        self._overrides.pop(key, None)

    def clear_all_overrides(self) -> None:
        """Clear all in-memory overrides."""
        self._overrides.clear()

    @property
    def env_path(self) -> Path:
        """Return the path to the .env file."""
        return self._env_path


# Convenience singleton for simple usage
_default_config: Optional[Config] = None


def get_config(watch: bool = False) -> Config:
    """
    Get the default Config singleton.

    Args:
        watch: Enable hot-reload support (only applies on first call)

    Returns:
        The default Config instance
    """
    global _default_config
    if _default_config is None:
        _default_config = Config(watch=watch)
    return _default_config
