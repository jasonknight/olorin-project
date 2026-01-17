"""
Centralized Inference Library for Olorin Project

Provides a unified interface for LLM inference that abstracts backend selection
(EXO, Ollama, etc.) from consumer components. Follows the patterns established
by libs/config.py and libs/state.py.

Usage:
    from libs.inference import get_inference_client

    client = get_inference_client()

    # Non-streaming completion
    response = client.complete(messages=[{"role": "user", "content": "Hello"}])
    print(response.content)

    # Streaming completion
    for chunk in client.complete_stream(messages=[...]):
        print(chunk.content, end="", flush=True)
        if chunk.tool_calls:
            # Handle tool calls
            pass

    # Model detection
    model = client.get_running_model()

    # Tool support
    if client.supports_tools():
        # Enable tools
        pass
"""

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator, Optional

import requests
from openai import OpenAI

from libs.config import Config

logger = logging.getLogger(__name__)


class BackendType(Enum):
    """Supported inference backends."""

    EXO = "exo"
    OLLAMA = "ollama"


@dataclass
class CompletionChunk:
    """A single chunk from a streaming completion."""

    content: str = ""
    finish_reason: Optional[str] = None
    tool_calls: Optional[list[dict]] = None


@dataclass
class CompletionResponse:
    """Response from a non-streaming completion."""

    content: str
    finish_reason: str
    tool_calls: Optional[list[dict]] = None
    usage: Optional[dict] = None


@dataclass
class ModelInfo:
    """Information about the active model."""

    model_id: str
    ready: bool
    backend: BackendType


class InferenceBackend(ABC):
    """
    Abstract base class for inference backends.

    Each backend (EXO, Ollama, etc.) implements this interface to provide
    consistent inference capabilities regardless of the underlying service.
    """

    @abstractmethod
    def get_running_model(self) -> Optional[str]:
        """
        Get the currently running model ID.

        Returns:
            Model ID string or None if no model is running
        """
        pass

    @abstractmethod
    def check_health(self) -> bool:
        """
        Check if the backend is healthy and reachable.

        Returns:
            True if backend is healthy, False otherwise
        """
        pass

    @abstractmethod
    def supports_tools(self, model: Optional[str] = None) -> bool:
        """
        Check if the backend/model supports tool calling.

        Args:
            model: Specific model to check (uses default if None)

        Returns:
            True if tool calls are supported
        """
        pass

    @abstractmethod
    def complete(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[list[dict]] = None,
        timeout: Optional[float] = None,
    ) -> CompletionResponse:
        """
        Non-streaming completion.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Override model (None = use config/auto-detect)
            temperature: Override temperature
            max_tokens: Override max tokens
            tools: List of OpenAI-format tool definitions
            timeout: Request timeout in seconds

        Returns:
            CompletionResponse with content, finish_reason, and optional tool_calls
        """
        pass

    @abstractmethod
    def complete_stream(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[list[dict]] = None,
        timeout: Optional[float] = None,
    ) -> Iterator[CompletionChunk]:
        """
        Streaming completion - yields chunks.

        Args:
            Same as complete()

        Yields:
            CompletionChunk objects with content and optional tool_calls
        """
        pass

    @property
    @abstractmethod
    def backend_type(self) -> BackendType:
        """Return the backend type."""
        pass


class ExoBackend(InferenceBackend):
    """
    EXO-specific inference backend using OpenAI-compatible API.

    EXO is a distributed AI inference server that exposes an OpenAI-compatible
    API. This backend handles EXO-specific features like model auto-detection
    from the /state endpoint.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "dummy-key",
        default_model: Optional[str] = None,
        default_temperature: float = 0.7,
        default_max_tokens: Optional[int] = None,
    ):
        """
        Initialize the EXO backend.

        Args:
            base_url: EXO API base URL (e.g., http://localhost:52415/v1)
            api_key: API key (EXO accepts any key)
            default_model: Default model to use (None = auto-detect)
            default_temperature: Default temperature for completions
            default_max_tokens: Default max tokens (None = no limit)
        """
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._default_model = default_model
        self._default_temperature = default_temperature
        self._default_max_tokens = default_max_tokens

        # Initialize OpenAI client
        self._client = OpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
        )

        # Tool support cache
        self._tools_supported: Optional[bool] = None
        self._tools_model: Optional[str] = None

        logger.info(f"ExoBackend initialized with base_url={self._base_url}")

    @property
    def backend_type(self) -> BackendType:
        return BackendType.EXO

    @property
    def client(self) -> OpenAI:
        """Return the underlying OpenAI client for advanced use cases."""
        return self._client

    def get_running_model(self) -> Optional[str]:
        """
        Query EXO /state endpoint for currently running model.

        EXO tracks model instances and their runner states. A model is considered
        ready when all its runners are in RunnerReady or RunnerRunning state.

        Returns:
            Model ID string or None if no model is ready
        """
        thread_name = threading.current_thread().name
        logger.debug(f"[{thread_name}] Querying EXO for running model...")

        try:
            # Remove /v1 from base_url to get the state endpoint
            state_url = f"{self._base_url.rstrip('/v1')}/state"

            response = requests.get(state_url, timeout=5)
            response.raise_for_status()
            state = response.json()

            instances = state.get("instances", {})
            runners = state.get("runners", {})

            if not instances:
                logger.debug(f"[{thread_name}] No instances in EXO state")
                return None

            # Check each instance for a ready model
            for instance_id, instance_wrapper in instances.items():
                if isinstance(instance_wrapper, dict):
                    for variant_name, instance_data in instance_wrapper.items():
                        if isinstance(instance_data, dict):
                            shard_assignments = instance_data.get(
                                "shardAssignments", {}
                            )
                            model_id = shard_assignments.get("modelId")
                            runner_to_shard = shard_assignments.get("runnerToShard", {})

                            if not model_id:
                                continue

                            # Check if all runners are ready
                            all_ready = True
                            for runner_id in runner_to_shard.keys():
                                runner_status = runners.get(runner_id, {})
                                if isinstance(runner_status, dict):
                                    status_type = (
                                        next(iter(runner_status.keys()))
                                        if runner_status
                                        else None
                                    )
                                    if status_type not in [
                                        "RunnerReady",
                                        "RunnerRunning",
                                    ]:
                                        all_ready = False
                                        break

                            if all_ready and runner_to_shard:
                                logger.info(
                                    f"[{thread_name}] Auto-detected model: {model_id}"
                                )
                                return model_id

            logger.debug(f"[{thread_name}] No ready model found in EXO")
            return None

        except requests.exceptions.Timeout:
            logger.warning(f"[{thread_name}] Timeout querying EXO state")
            return None
        except requests.exceptions.ConnectionError:
            logger.warning(f"[{thread_name}] Connection error - is EXO running?")
            return None
        except Exception as e:
            logger.warning(f"[{thread_name}] Failed to query EXO state: {e}")
            return None

    def check_health(self) -> bool:
        """Check EXO health via /state endpoint."""
        try:
            state_url = f"{self._base_url.rstrip('/v1')}/state"
            response = requests.get(state_url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def supports_tools(self, model: Optional[str] = None) -> bool:
        """
        Detect tool support via test API call.

        Uses a minimal test call with a dummy tool definition to probe support.
        Results are cached per model.

        Args:
            model: Model to test (uses default/auto-detect if None)

        Returns:
            True if tool calls are supported
        """
        model_to_test = model or self._default_model or self.get_running_model()

        # Use cache if same model
        if self._tools_supported is not None and self._tools_model == model_to_test:
            return self._tools_supported

        if not model_to_test:
            return False

        thread_name = threading.current_thread().name
        logger.info(f"[{thread_name}] Detecting tool support for model {model_to_test}")

        try:
            test_tool = {
                "type": "function",
                "function": {
                    "name": "test_tool_support",
                    "description": "Test function to detect tool support",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            }

            self._client.chat.completions.create(
                model=model_to_test,
                messages=[{"role": "user", "content": "test"}],
                tools=[test_tool],
                max_tokens=1,
                timeout=10,
            )

            # If we get here without error, tools are supported
            self._tools_supported = True
            self._tools_model = model_to_test
            logger.info(f"[{thread_name}] Tool calls SUPPORTED for {model_to_test}")
            return True

        except Exception as e:
            error_msg = str(e).lower()
            if any(
                indicator in error_msg
                for indicator in [
                    "tools",
                    "function",
                    "not supported",
                    "invalid",
                    "unknown parameter",
                ]
            ):
                logger.info(f"[{thread_name}] Tool calls NOT supported: {e}")
            else:
                logger.warning(f"[{thread_name}] Tool detection failed: {e}")

            self._tools_supported = False
            self._tools_model = model_to_test
            return False

    def complete(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[list[dict]] = None,
        timeout: Optional[float] = None,
    ) -> CompletionResponse:
        """Non-streaming completion."""
        model_to_use = model or self._default_model or self.get_running_model()
        if not model_to_use:
            raise ValueError("No model available for inference")

        params: dict[str, Any] = {
            "model": model_to_use,
            "messages": messages,
            "temperature": temperature
            if temperature is not None
            else self._default_temperature,
        }

        effective_max_tokens = (
            max_tokens if max_tokens is not None else self._default_max_tokens
        )
        if effective_max_tokens is not None:
            params["max_tokens"] = effective_max_tokens

        if tools:
            params["tools"] = tools

        if timeout is not None:
            params["timeout"] = timeout

        response = self._client.chat.completions.create(**params)
        choice = response.choices[0]

        tool_calls = None
        if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in choice.message.tool_calls
            ]

        return CompletionResponse(
            content=choice.message.content or "",
            finish_reason=choice.finish_reason or "stop",
            tool_calls=tool_calls,
            usage=dict(response.usage) if response.usage else None,
        )

    def complete_stream(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[list[dict]] = None,
        timeout: Optional[float] = None,
    ) -> Iterator[CompletionChunk]:
        """Streaming completion - yields chunks."""
        model_to_use = model or self._default_model or self.get_running_model()
        if not model_to_use:
            raise ValueError("No model available for inference")

        params: dict[str, Any] = {
            "model": model_to_use,
            "messages": messages,
            "temperature": temperature
            if temperature is not None
            else self._default_temperature,
            "stream": True,
        }

        effective_max_tokens = (
            max_tokens if max_tokens is not None else self._default_max_tokens
        )
        if effective_max_tokens is not None:
            params["max_tokens"] = effective_max_tokens

        if tools:
            params["tools"] = tools

        if timeout is not None:
            params["timeout"] = timeout

        stream = self._client.chat.completions.create(**params)

        # Track accumulated tool calls across chunks
        accumulated_tool_calls: list[dict] = []

        for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                choice = chunk.choices[0]
                delta = choice.delta

                # Accumulate tool calls from streaming chunks
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    for tc in delta.tool_calls:
                        tc_index = tc.index if hasattr(tc, "index") else 0

                        # Expand list if needed
                        while len(accumulated_tool_calls) <= tc_index:
                            accumulated_tool_calls.append(
                                {
                                    "id": "",
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                }
                            )

                        # Accumulate tool call data
                        if hasattr(tc, "id") and tc.id:
                            accumulated_tool_calls[tc_index]["id"] = tc.id
                        if hasattr(tc, "function") and tc.function:
                            if hasattr(tc.function, "name") and tc.function.name:
                                accumulated_tool_calls[tc_index]["function"][
                                    "name"
                                ] += tc.function.name
                            if (
                                hasattr(tc.function, "arguments")
                                and tc.function.arguments
                            ):
                                accumulated_tool_calls[tc_index]["function"][
                                    "arguments"
                                ] += tc.function.arguments

                content = (
                    delta.content if hasattr(delta, "content") and delta.content else ""
                )
                finish_reason = choice.finish_reason

                # Only include tool_calls on final chunk with tool_calls finish_reason
                chunk_tool_calls = None
                if finish_reason == "tool_calls" and accumulated_tool_calls:
                    chunk_tool_calls = accumulated_tool_calls

                yield CompletionChunk(
                    content=content,
                    finish_reason=finish_reason,
                    tool_calls=chunk_tool_calls,
                )


class OllamaBackend(InferenceBackend):
    """
    Ollama inference backend.

    Placeholder for future Ollama support. Ollama provides a local LLM server
    with its own API format, but also supports OpenAI-compatible endpoints.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        default_model: Optional[str] = None,
        default_temperature: float = 0.7,
        default_max_tokens: Optional[int] = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._default_model = default_model
        self._default_temperature = default_temperature
        self._default_max_tokens = default_max_tokens

        logger.info(f"OllamaBackend initialized with base_url={self._base_url}")
        raise NotImplementedError(
            "Ollama backend not yet implemented. "
            "Set inference.backend to 'exo' in settings.json"
        )

    @property
    def backend_type(self) -> BackendType:
        return BackendType.OLLAMA

    def get_running_model(self) -> Optional[str]:
        # TODO: Query /api/tags for available models
        raise NotImplementedError()

    def check_health(self) -> bool:
        # TODO: Check /api/tags endpoint
        raise NotImplementedError()

    def supports_tools(self, model: Optional[str] = None) -> bool:
        # TODO: Check model capabilities
        raise NotImplementedError()

    def complete(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[list[dict]] = None,
        timeout: Optional[float] = None,
    ) -> CompletionResponse:
        # TODO: Use /api/chat endpoint
        raise NotImplementedError()

    def complete_stream(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[list[dict]] = None,
        timeout: Optional[float] = None,
    ) -> Iterator[CompletionChunk]:
        # TODO: Use /api/chat with stream=True
        raise NotImplementedError()


@dataclass
class InferenceClient:
    """
    Unified inference client that abstracts backend selection.

    Provides a simple interface for LLM inference with:
    - Automatic backend selection from configuration
    - Hot-reload support for configuration changes
    - Model auto-detection
    - Tool support detection and caching
    - Streaming and non-streaming completions

    Usage:
        from libs.inference import get_inference_client

        # Get singleton instance (recommended)
        client = get_inference_client()

        # Non-streaming completion
        response = client.complete(messages=[{"role": "user", "content": "Hello"}])
        print(response.content)

        # Streaming completion
        for chunk in client.complete_stream(messages=[...]):
            print(chunk.content, end="", flush=True)

        # With tools
        response = client.complete(messages=[...], tools=my_tools)
        if response.tool_calls:
            # Handle tool calls
            pass
    """

    _config: Config = field(repr=False)
    _backend: Optional[InferenceBackend] = field(default=None, repr=False)
    _mtime: Optional[float] = field(default=None, repr=False)

    # Class-level singleton management
    _instance: Optional["InferenceClient"] = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self):
        """Initialize the backend after dataclass initialization."""
        self._init_backend()

    def _init_backend(self) -> None:
        """Initialize the appropriate backend based on configuration."""
        backend_type = self._config.get("INFERENCE_BACKEND", "exo").lower()

        if backend_type == "exo":
            self._backend = ExoBackend(
                base_url=self._config.get("EXO_BASE_URL", "http://localhost:52415/v1"),
                api_key=self._config.get("EXO_API_KEY", "dummy-key"),
                default_model=self._config.get("MODEL_NAME") or None,
                default_temperature=self._config.get_float("TEMPERATURE", 0.7) or 0.7,
                default_max_tokens=self._config.get_int("MAX_TOKENS"),
            )
        elif backend_type == "ollama":
            self._backend = OllamaBackend(
                base_url=self._config.get("OLLAMA_BASE_URL", "http://localhost:11434"),
                default_model=self._config.get("OLLAMA_MODEL_NAME") or None,
                default_temperature=self._config.get_float("OLLAMA_TEMPERATURE", 0.7)
                or 0.7,
                default_max_tokens=self._config.get_int("OLLAMA_MAX_TOKENS"),
            )
        else:
            raise ValueError(f"Unknown inference backend: {backend_type}")

        # Track config mtime for reload detection
        if self._config.config_path.exists():
            self._mtime = self._config.config_path.stat().st_mtime

    def reload(self) -> bool:
        """
        Reload configuration if changed.

        Returns:
            True if configuration was reloaded, False otherwise
        """
        if self._config.reload():
            logger.info("Inference configuration changed, reinitializing backend...")
            self._init_backend()
            return True
        return False

    @property
    def backend_type(self) -> BackendType:
        """Return the active backend type."""
        if self._backend is None:
            raise RuntimeError("Backend not initialized")
        return self._backend.backend_type

    @property
    def backend(self) -> InferenceBackend:
        """Return the active backend for advanced use cases."""
        if self._backend is None:
            raise RuntimeError("Backend not initialized")
        return self._backend

    def get_running_model(self) -> Optional[str]:
        """Get the currently running model."""
        if self._backend is None:
            raise RuntimeError("Backend not initialized")
        return self._backend.get_running_model()

    def check_health(self) -> bool:
        """Check if the backend is healthy."""
        if self._backend is None:
            return False
        return self._backend.check_health()

    def supports_tools(self, model: Optional[str] = None) -> bool:
        """Check if tool calling is supported."""
        if self._backend is None:
            return False
        return self._backend.supports_tools(model)

    def complete(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[list[dict]] = None,
        timeout: Optional[float] = None,
    ) -> CompletionResponse:
        """
        Non-streaming completion.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Override model (None = use config/auto-detect)
            temperature: Override temperature
            max_tokens: Override max tokens
            tools: List of OpenAI-format tool definitions
            timeout: Request timeout in seconds

        Returns:
            CompletionResponse with content, finish_reason, and optional tool_calls
        """
        if self._backend is None:
            raise RuntimeError("Backend not initialized")
        return self._backend.complete(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            timeout=timeout,
        )

    def complete_stream(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[list[dict]] = None,
        timeout: Optional[float] = None,
    ) -> Iterator[CompletionChunk]:
        """
        Streaming completion.

        Args:
            Same as complete()

        Yields:
            CompletionChunk objects with content and optional tool_calls
        """
        if self._backend is None:
            raise RuntimeError("Backend not initialized")
        return self._backend.complete_stream(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            timeout=timeout,
        )


# Module-level singleton instance
_inference_client: Optional[InferenceClient] = None
_inference_lock = threading.Lock()


def get_inference_client(config: Optional[Config] = None) -> InferenceClient:
    """
    Get the singleton InferenceClient instance.

    Args:
        config: Optional Config instance. If not provided, creates a new one
                with hot-reload enabled.

    Returns:
        The singleton InferenceClient instance
    """
    global _inference_client

    if _inference_client is None:
        with _inference_lock:
            if _inference_client is None:
                cfg = config or Config(watch=True)
                _inference_client = InferenceClient(_config=cfg)

    return _inference_client


def reset_inference_client() -> None:
    """Reset the singleton instance (useful for testing)."""
    global _inference_client
    with _inference_lock:
        _inference_client = None
