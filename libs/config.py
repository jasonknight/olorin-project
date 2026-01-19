"""
Unified Configuration Library for Olorin Project

Provides a Config class with type-safe getters and optional hot-reload support.
Reads from settings.json (preferred) or falls back to .env for backward compatibility.
"""

import json
import os
from pathlib import Path
from typing import Any, List, Optional, Union


# Mapping from flat keys to JSON paths for backward compatibility
_KEY_TO_PATH = {
    # Global
    "LOG_LEVEL": "global.log_level",
    "LOG_DIR": "global.log_dir",
    # Kafka
    "KAFKA_BOOTSTRAP_SERVERS": "kafka.bootstrap_servers",
    "KAFKA_SEND_TIMEOUT": "kafka.send_timeout",
    "KAFKA_MAX_RETRIES": "kafka.max_retries",
    # Exo
    "EXO_BASE_URL": "exo.base_url",
    "EXO_API_KEY": "exo.api_key",
    "MODEL_NAME": "exo.model_name",
    "TEMPERATURE": "exo.temperature",
    "MAX_TOKENS": "exo.max_tokens",
    # Broca
    "BROCA_KAFKA_TOPIC": "broca.kafka_topic",
    "BROCA_CONSUMER_GROUP": "broca.consumer_group",
    "BROCA_AUTO_OFFSET_RESET": "broca.auto_offset_reset",
    "TTS_MODEL_NAME": "broca.tts.model_name",
    "TTS_SPEAKER": "broca.tts.speaker",
    "TTS_OUTPUT_DIR": "broca.tts.output_dir",
    # Cortex
    "CORTEX_INPUT_TOPIC": "cortex.input_topic",
    "CORTEX_OUTPUT_TOPIC": "cortex.output_topic",
    "CORTEX_CONSUMER_GROUP": "cortex.consumer_group",
    "CORTEX_AUTO_OFFSET_RESET": "cortex.auto_offset_reset",
    # Hippocampus
    "INPUT_DIR": "hippocampus.input_dir",
    "CHROMADB_HOST": "hippocampus.chromadb.host",
    "CHROMADB_PORT": "hippocampus.chromadb.port",
    "CHROMADB_COLLECTION": "hippocampus.chromadb.collection",
    "CHUNK_SIZE": "hippocampus.chunking.size",
    "CHUNK_OVERLAP": "hippocampus.chunking.overlap",
    "CHUNK_MIN_SIZE": "hippocampus.chunking.min_size",
    "POLL_INTERVAL": "hippocampus.poll_interval",
    "TRACKING_DB": "hippocampus.tracking_db",
    "PDF_TRACKING_DB": "hippocampus.pdf_tracking_db",
    "EBOOK_TRACKING_DB": "hippocampus.ebook_tracking_db",
    "TXT_TRACKING_DB": "hippocampus.txt_tracking_db",
    "OFFICE_TRACKING_DB": "hippocampus.office_tracking_db",
    "HIPPOCAMPUS_CONTEXT_DB": "hippocampus.context_db",
    "REPROCESS_ON_CHANGE": "hippocampus.reprocess_on_change",
    "DELETE_AFTER_PROCESSING": "hippocampus.delete_after_processing",
    "LOG_FILE": "hippocampus.log_file",
    # Enrichener
    "ENRICHENER_INPUT_TOPIC": "enrichener.input_topic",
    "ENRICHENER_OUTPUT_TOPIC": "enrichener.output_topic",
    "ENRICHENER_CONSUMER_GROUP": "enrichener.consumer_group",
    "ENRICHENER_AUTO_OFFSET_RESET": "enrichener.auto_offset_reset",
    "ENRICHENER_THREAD_POOL_SIZE": "enrichener.thread_pool_size",
    "LLM_TIMEOUT_SECONDS": "enrichener.llm_timeout_seconds",
    "DECISION_TEMPERATURE": "enrichener.decision_temperature",
    "CHROMADB_QUERY_N_RESULTS": "enrichener.chromadb_query_n_results",
    "CONTEXT_DB_PATH": "enrichener.context_db_path",
    "CLEANUP_CONTEXT_AFTER_USE": "enrichener.cleanup_context_after_use",
    # Chat
    "CHAT_HISTORY_ENABLED": "chat.history_enabled",
    "CHAT_DB_PATH": "chat.db_path",
    "CHAT_RESET_PATTERNS": "chat.reset_patterns",
    # State
    "STATE_DB_PATH": "state.db_path",
    # Control API
    "CONTROL_API_ENABLED": "control.api.enabled",
    "CONTROL_API_PORT": "control.api.port",
    "CONTROL_API_HOST": "control.api.host",
    # Tracker common settings
    "TRACKER_OLLAMA_ENABLED": "hippocampus.trackers.ollama.enabled",
    "TRACKER_OLLAMA_MODEL": "hippocampus.trackers.ollama.model",
    "TRACKER_OLLAMA_THRESHOLD": "hippocampus.trackers.ollama.threshold",
    "TRACKER_OLLAMA_PATHS": "hippocampus.trackers.ollama.paths",
    "TRACKER_MIN_CONTENT_CHARS": "hippocampus.trackers.min_content_chars",
    "TRACKER_MIN_CONTENT_DENSITY": "hippocampus.trackers.min_content_density",
    "TRACKER_MIN_WORD_COUNT": "hippocampus.trackers.min_word_count",
    "TRACKER_NOTIFY_BROCA": "hippocampus.trackers.notify_broca",
    # Tools (AI tool use)
    "TOOLS_WRITE_ENABLED": "tools.write.enabled",
    "TOOLS_WRITE_PORT": "tools.write.port",
    "EMBEDDINGS_TOOL_ENABLED": "tools.embeddings.enabled",
    "EMBEDDINGS_TOOL_PORT": "tools.embeddings.port",
    "EMBEDDINGS_TOOL_HOST": "tools.embeddings.host",
    "EMBEDDINGS_TOOL_MODEL": "tools.embeddings.model",
    "EMBEDDINGS_DOCUMENT_PREFIX": "tools.embeddings.document_prefix",
    "EMBEDDINGS_QUERY_PREFIX": "tools.embeddings.query_prefix",
    # Inference (centralized LLM inference settings)
    "INFERENCE_BACKEND": "inference.backend",
    "INFERENCE_TIMEOUT": "inference.timeout",
    "INFERENCE_RETRY_COUNT": "inference.retry_count",
    "INFERENCE_RETRY_DELAY": "inference.retry_delay",
    # Ollama (alternative inference backend)
    "OLLAMA_BASE_URL": "ollama.base_url",
    "OLLAMA_MODEL_NAME": "ollama.model_name",
    "OLLAMA_TEMPERATURE": "ollama.temperature",
    "OLLAMA_MAX_TOKENS": "ollama.max_tokens",
    # Temporal (voice-activated STT)
    "TEMPORAL_OUTPUT_TOPIC": "temporal.output_topic",
    "TEMPORAL_FEEDBACK_TOPIC": "temporal.feedback_topic",
    "TEMPORAL_FEEDBACK_MESSAGE": "temporal.feedback_message",
    "TEMPORAL_COMPLETION_MESSAGE": "temporal.completion_message",
    "TEMPORAL_SAMPLE_RATE": "temporal.audio.sample_rate",
    "TEMPORAL_CHANNELS": "temporal.audio.channels",
    "TEMPORAL_AUDIO_DEVICE": "temporal.audio.device",
    "TEMPORAL_CHUNK_DURATION": "temporal.audio.chunk_duration",
    "TEMPORAL_WAKE_PHRASE": "temporal.wake_word.phrase",
    "TEMPORAL_WAKE_BUFFER_SECONDS": "temporal.wake_word.buffer_seconds",
    "TEMPORAL_STT_MODEL": "temporal.stt.model",
    "TEMPORAL_STT_DEVICE": "temporal.stt.device",
    "TEMPORAL_STT_COMPUTE_TYPE": "temporal.stt.compute_type",
    "TEMPORAL_STT_LANGUAGE": "temporal.stt.language",
    "TEMPORAL_STT_MODEL_DIR": "temporal.stt.model_dir",
    "TEMPORAL_VAD_THRESHOLD": "temporal.silence.vad_threshold",
    "TEMPORAL_SILENCE_TIMEOUT": "temporal.silence.timeout_seconds",
    "TEMPORAL_STOP_PHRASES": "temporal.silence.stop_phrases",
    "TEMPORAL_VAD_SMOOTHING": "temporal.silence.vad_smoothing",
    "TEMPORAL_VAD_SPEECH_START_THRESHOLD": "temporal.silence.vad_speech_start_threshold",
    "TEMPORAL_VAD_SPEECH_END_THRESHOLD": "temporal.silence.vad_speech_end_threshold",
    "TEMPORAL_VAD_MIN_SPEECH_CHUNKS": "temporal.silence.vad_min_speech_chunks",
    "TEMPORAL_VAD_ENERGY_THRESHOLD_RATIO": "temporal.silence.vad_energy_threshold_ratio",
    "TEMPORAL_VAD_ENERGY_SMOOTHING": "temporal.silence.vad_energy_smoothing",
    "TEMPORAL_VAD_USE_ENERGY": "temporal.silence.vad_use_energy",
    "TEMPORAL_VAD_DEBUG": "temporal.silence.vad_debug",
    "TEMPORAL_PAUSE_DURING_TTS": "temporal.behavior.pause_during_tts",
    # Temporal Porcupine wake word detection
    "TEMPORAL_PORCUPINE_ACCESS_KEY": "temporal.porcupine.access_key",
    "TEMPORAL_PORCUPINE_KEYWORD_PATH": "temporal.porcupine.keyword_path",
    "TEMPORAL_PORCUPINE_SENSITIVITY": "temporal.porcupine.sensitivity",
    # Temporal STT engine selection
    "TEMPORAL_STT_ENGINE": "temporal.stt.engine",
    "TEMPORAL_LEOPARD_PUNCTUATION": "temporal.stt.leopard_punctuation",
}


def _find_project_root() -> Path:
    """Find the project root by looking for settings.json or .env file."""
    current = Path(__file__).resolve().parent

    # Go up from libs/ to project root
    if current.name == "libs":
        project_root = current.parent
        if (project_root / "settings.json").exists() or (
            project_root / ".env"
        ).exists():
            return project_root

    # Fallback: search upward for settings.json or .env
    search = current
    for _ in range(5):  # Limit search depth
        if (search / "settings.json").exists() or (search / ".env").exists():
            return search
        if search.parent == search:
            break
        search = search.parent

    # Last resort: return current working directory
    return Path.cwd()


class Config:
    """
    Configuration manager that loads from settings.json or .env file.

    Provides type-safe getters and optional hot-reload support for runtime
    configuration changes. Prefers settings.json if available, falls back to .env.

    Usage:
        config = Config()
        host = config.get('CHROMADB_HOST', 'localhost')
        port = config.get_int('CHROMADB_PORT', 8000)
        enabled = config.get_bool('FEATURE_ENABLED', False)
        path = config.get_path('INPUT_DIR', '~/Documents')
        patterns = config.get_list('CHAT_RESET_PATTERNS', [])

        # With hot-reload support
        config = Config(watch=True)
        if config.reload():
            print("Configuration was updated")
    """

    def __init__(
        self, config_path: Optional[Union[str, Path]] = None, watch: bool = False
    ):
        """
        Initialize the configuration manager.

        Args:
            config_path: Path to settings.json or .env file. Defaults to project root.
            watch: If True, enables hot-reload support via reload() method
        """
        self._project_root = _find_project_root()

        if config_path is None:
            self._json_path = self._project_root / "settings.json"
            self._env_path = self._project_root / ".env"
        else:
            path = Path(config_path).resolve()
            if path.suffix == ".json":
                self._json_path = path
                self._env_path = path.parent / ".env"
                # Project root is the directory containing the config
                self._project_root = path.parent
            elif path.suffix == ".env" or path.name == ".env":
                self._json_path = path.parent / "settings.json"
                self._env_path = path
                self._project_root = path.parent
            else:
                # Assume it's a directory
                self._json_path = path / "settings.json"
                self._env_path = path / ".env"
                self._project_root = path

        self._watch = watch
        self._mtime: Optional[float] = None
        self._overrides: dict[str, Any] = {}
        self._data: dict = {}
        self._using_json = False

        self._load()

    def _load(self) -> None:
        """Load configuration from settings.json or fall back to .env."""
        # Prefer settings.json
        if self._json_path.exists():
            self._mtime = self._json_path.stat().st_mtime
            with open(self._json_path, "r") as f:
                self._data = json.load(f)
            self._using_json = True
            return

        # Fall back to .env
        if self._env_path.exists():
            self._mtime = self._env_path.stat().st_mtime
            self._data = self._parse_env_to_nested()
            self._using_json = False

    def _parse_env_to_nested(self) -> dict:
        """Parse .env file into nested dictionary structure."""
        flat_data = {}

        with open(self._env_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip()
                    if (
                        len(value) >= 2
                        and value[0] == value[-1]
                        and value[0] in ('"', "'")
                    ):
                        value = value[1:-1]
                    flat_data[key] = value

        # Convert flat keys to nested structure
        result: dict = {}
        for flat_key, value in flat_data.items():
            if flat_key in _KEY_TO_PATH:
                self._set_nested(result, _KEY_TO_PATH[flat_key], value)
            else:
                # Store unknown keys at root level
                result[flat_key] = value

        return result

    def _set_nested(self, d: dict, path: str, value: Any) -> None:
        """Set a value in a nested dictionary using dot notation path."""
        keys = path.split(".")
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value

    def _get_nested(self, path: str) -> Any:
        """Get a value from nested dictionary using dot notation path."""
        keys = path.split(".")
        value = self._data
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value

    def reload(self) -> bool:
        """
        Reload configuration if the config file has changed.

        Returns:
            True if configuration was reloaded, False otherwise
        """
        if not self._watch:
            return False

        # Check the appropriate config file
        config_path = self._json_path if self._json_path.exists() else self._env_path
        if not config_path.exists():
            return False

        current_mtime = config_path.stat().st_mtime
        if current_mtime != self._mtime:
            self._load()
            return True

        return False

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get a configuration value as a string.

        Args:
            key: The configuration key (flat like 'CHROMADB_PORT' or nested like 'hippocampus.chromadb.port')
            default: Default value if not set

        Returns:
            The configuration value or default
        """
        # Check overrides first
        if key in self._overrides:
            val = self._overrides[key]
            if val is None:
                return default
            if isinstance(val, list):
                return ",".join(str(v) for v in val)
            return str(val)

        # Map flat key to nested path if known
        path = _KEY_TO_PATH.get(key, key)
        value = self._get_nested(path)

        if value is None:
            return default

        # Handle arrays - convert to comma-separated string for backward compat
        if isinstance(value, list):
            return ",".join(str(v) for v in value)

        return str(value)

    def get_int(self, key: str, default: Optional[int] = None) -> Optional[int]:
        """
        Get a configuration value as an integer.

        Args:
            key: The configuration key
            default: Default value if not set or invalid

        Returns:
            The configuration value as int or default
        """
        # Check overrides first for native int
        if key in self._overrides:
            val = self._overrides[key]
            if isinstance(val, int):
                return val

        # Try nested path for native JSON int
        path = _KEY_TO_PATH.get(key, key)
        value = self._get_nested(path)
        if isinstance(value, int):
            return value

        # Fall back to string parsing
        str_value = self.get(key)
        if str_value is None:
            return default

        try:
            return int(str_value)
        except (ValueError, TypeError):
            return default

    def get_float(self, key: str, default: Optional[float] = None) -> Optional[float]:
        """
        Get a configuration value as a float.

        Args:
            key: The configuration key
            default: Default value if not set or invalid

        Returns:
            The configuration value as float or default
        """
        # Check overrides first for native float
        if key in self._overrides:
            val = self._overrides[key]
            if isinstance(val, (int, float)):
                return float(val)

        # Try nested path for native JSON number
        path = _KEY_TO_PATH.get(key, key)
        value = self._get_nested(path)
        if isinstance(value, (int, float)):
            return float(value)

        # Fall back to string parsing
        str_value = self.get(key)
        if str_value is None:
            return default

        try:
            return float(str_value)
        except (ValueError, TypeError):
            return default

    def get_bool(self, key: str, default: bool = False) -> bool:
        """
        Get a configuration value as a boolean.

        Recognizes: true, yes, 1, on (case-insensitive) as True.
        Native JSON booleans are handled directly.
        Everything else (including empty string) is False.

        Args:
            key: The configuration key
            default: Default value if not set

        Returns:
            The configuration value as bool or default
        """
        # Check overrides first
        if key in self._overrides:
            val = self._overrides[key]
            if isinstance(val, bool):
                return val
            if val is None:
                return default
            return str(val).lower() in ("true", "yes", "1", "on")

        # Try nested path for native JSON bool
        path = _KEY_TO_PATH.get(key, key)
        value = self._get_nested(path)

        if value is None:
            return default

        if isinstance(value, bool):
            return value

        return str(value).lower() in ("true", "yes", "1", "on")

    def get_path(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get a configuration value as an expanded, absolute path.

        - Expands ~ to the user's home directory
        - Resolves relative paths (starting with ./ or not starting with /)
          against the project root (where settings.json lives)

        Args:
            key: The configuration key
            default: Default value if not set

        Returns:
            The absolute path or default
        """
        value = self.get(key, default)
        if value is None:
            return None

        # Expand ~ to home directory
        expanded = os.path.expanduser(value)

        # If it's already absolute, return as-is
        if os.path.isabs(expanded):
            return expanded

        # Resolve relative paths against project root
        return str(self._project_root / expanded)

    @property
    def project_root(self) -> Path:
        """Return the project root directory (where settings.json lives)."""
        return self._project_root

    def get_list(
        self, key: str, default: Optional[List[str]] = None
    ) -> Optional[List[str]]:
        """
        Get a configuration value as a list of strings.

        Native JSON arrays are returned directly.
        Comma-separated strings are split into lists.

        Args:
            key: The configuration key
            default: Default value if not set

        Returns:
            The configuration value as list or default
        """
        # Check overrides first
        if key in self._overrides:
            val = self._overrides[key]
            if isinstance(val, list):
                return [str(v) for v in val]
            if val is None:
                return default
            return [v.strip() for v in str(val).split(",")]

        # Try nested path for native JSON array
        path = _KEY_TO_PATH.get(key, key)
        value = self._get_nested(path)

        if value is None:
            return default

        if isinstance(value, list):
            return [str(v) for v in value]

        # Parse comma-separated string
        return [v.strip() for v in str(value).split(",")]

    def get_tools(self) -> dict[str, dict[str, Any]]:
        """
        Get all enabled AI tools with their configuration.

        Returns a dict mapping tool names to their configuration.
        Only includes tools where 'enabled' is True.

        Returns:
            Dict like {'write': {'enabled': True, 'port': 8770}}
        """
        tools_section = self._get_nested("tools")
        if not isinstance(tools_section, dict):
            return {}

        result = {}
        for tool_name, tool_config in tools_section.items():
            if isinstance(tool_config, dict):
                enabled = tool_config.get("enabled", False)
                if enabled:
                    result[tool_name] = tool_config
        return result

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value (in-memory only).

        This override takes precedence over file values.
        Does not modify the config file.

        Args:
            key: The configuration key
            value: The value to set
        """
        self._overrides[key] = value

    def clear_override(self, key: str) -> None:
        """
        Clear an in-memory override for a key.

        After clearing, get() will return the file value.

        Args:
            key: The configuration key
        """
        self._overrides.pop(key, None)

    def clear_all_overrides(self) -> None:
        """Clear all in-memory overrides."""
        self._overrides.clear()

    @property
    def config_path(self) -> Path:
        """Return the path to the active config file."""
        return self._json_path if self._using_json else self._env_path

    @property
    def env_path(self) -> Path:
        """Return the path to the config file (deprecated, use config_path)."""
        return self.config_path


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
