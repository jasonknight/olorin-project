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

RAG Context Injection Notes:
    When injecting RAG context into messages, the format matters significantly
    for model comprehension. The most robust approach is combining context and
    question in a SINGLE message.

    BEST (single combined message - works across all tested models):
        messages = [
            {"role": "user", "content": "Use this context:\\n<context>...</context>\\n\\nQuestion: X?"}
        ]

    BROKEN (assistant ack causes context loss with Deepseek R1 and others):
        messages = [
            {"role": "user", "content": "Here is context: ..."},
            {"role": "assistant", "content": "I understand..."},  # BAD!
            {"role": "user", "content": "What does the context say about X?"}
        ]

    The assistant acknowledgment message causes models like Deepseek R1 to
    "forget" the context, responding with "no context was provided" even
    when 200K+ characters of context exist in the conversation. Even splitting
    into separate user messages can cause issues with some models.

    The single-message approach works reliably across Ollama, EXO, and various
    model architectures.

Model Capabilities Caching:
    When caching model capabilities (context length, sliding window), always
    verify the cached model_id matches the current model before using cached
    values. Stale data from a previous model can cause incorrect behavior.
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


@dataclass
class ModelCapabilities:
    """
    Context window and attention capabilities of a model.

    Used to detect potential context overflow issues before they cause
    silent failures (like sliding window attention truncation).

    IMPORTANT - Sliding Window Attention:
        Some models (e.g., Gemma, some Mistral variants) use sliding window
        attention, which means they can only "see" the last N tokens during
        generation, even if more tokens fit in the context window.

        For RAG use cases, this is CRITICAL: if you inject 50K tokens of
        context at the START of the conversation, a model with 1024-token
        sliding window will NOT be able to see any of it when generating
        a response. The model will claim "no context was provided" even
        though the context is technically in the conversation.

        Always check has_sliding_window before using a model for RAG with
        large context. Models with sliding window are unsuitable for RAG
        unless the context fits within the sliding window size.

    Caching Warning:
        When caching ModelCapabilities, always verify model_id matches the
        current model before using cached values. Different models have
        vastly different capabilities (e.g., gemma3 has 1024 sliding window,
        deepseek-r1 has 128K full context). Using stale cached data causes
        incorrect warnings or silent failures.
    """

    model_id: str
    context_length: int  # Maximum context window in tokens
    sliding_window: Optional[int] = None  # If set, limits effective attention span
    rope_scaling_original_context: Optional[int] = (
        None  # Original context before scaling
    )

    @property
    def effective_context(self) -> int:
        """
        Return the effective context length considering sliding window.

        For models with sliding window attention, the effective context is
        limited by the sliding window size, not the full context length.
        """
        if self.sliding_window is not None and self.sliding_window > 0:
            return self.sliding_window
        return self.context_length

    @property
    def has_sliding_window(self) -> bool:
        """Check if model uses sliding window attention."""
        return self.sliding_window is not None and self.sliding_window > 0

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Uses a simple heuristic of ~4 characters per token for English text.
        This is approximate but sufficient for warning purposes.
        """
        return len(text) // 4

    def estimate_messages_tokens(self, messages: list[dict]) -> int:
        """
        Estimate total token count for a messages array.

        Accounts for message overhead (role tags, separators, etc.)
        """
        total = 0
        for msg in messages:
            # Add overhead for message structure (~10 tokens per message)
            total += 10
            content = msg.get("content", "")
            if content:
                total += self.estimate_tokens(content)
            # Tool calls add extra tokens
            if msg.get("tool_calls"):
                total += 50  # Rough estimate for tool call overhead
        return total

    def check_context_fit(self, messages: list[dict]) -> tuple[bool, Optional[str]]:
        """
        Check if messages fit within the model's effective context.

        Returns:
            Tuple of (fits: bool, warning_message: Optional[str])
            - If fits is True, warning_message is None
            - If fits is False, warning_message explains the issue
        """
        estimated_tokens = self.estimate_messages_tokens(messages)

        # Check against sliding window first (most restrictive)
        if self.has_sliding_window:
            if estimated_tokens > self.sliding_window:
                return (
                    False,
                    f"⚠️ CONTEXT OVERFLOW: Estimated {estimated_tokens:,} tokens exceeds "
                    f"model's sliding window attention of {self.sliding_window:,} tokens. "
                    f"The model can only 'see' the last {self.sliding_window:,} tokens when generating. "
                    f"Earlier context will be effectively invisible. "
                    f"Consider using a model without sliding window attention for RAG/long context.",
                )

        # Check against full context window
        if estimated_tokens > self.context_length:
            return (
                False,
                f"⚠️ CONTEXT OVERFLOW: Estimated {estimated_tokens:,} tokens exceeds "
                f"model's context window of {self.context_length:,} tokens. "
                f"Messages may be truncated.",
            )

        # Warn if close to limit (>80%)
        if estimated_tokens > self.context_length * 0.8:
            return (
                True,
                f"⚠️ Context usage high: {estimated_tokens:,} tokens "
                f"({estimated_tokens * 100 // self.context_length}% of {self.context_length:,} limit)",
            )

        return (True, None)


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

    @abstractmethod
    def get_model_capabilities(
        self, model: Optional[str] = None
    ) -> Optional[ModelCapabilities]:
        """
        Get context window and attention capabilities for a model.

        Args:
            model: Model to query (uses default if None)

        Returns:
            ModelCapabilities with context window info, or None if unavailable
        """
        pass

    def validate_context(
        self, messages: list[dict], model: Optional[str] = None
    ) -> tuple[bool, Optional[str]]:
        """
        Validate that messages fit within the model's context window.

        This is a convenience method that combines get_model_capabilities
        and check_context_fit.

        Args:
            messages: List of message dicts to validate
            model: Model to check against (uses default if None)

        Returns:
            Tuple of (fits: bool, warning_message: Optional[str])
        """
        capabilities = self.get_model_capabilities(model)
        if capabilities is None:
            return (True, None)  # Can't validate, assume OK
        return capabilities.check_context_fit(messages)


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

    def get_model_capabilities(
        self, model: Optional[str] = None
    ) -> Optional[ModelCapabilities]:
        """
        Get model capabilities for EXO.

        EXO doesn't expose detailed model info, so we return a conservative
        default based on common model configurations. Users should monitor
        for context-related errors.

        Returns:
            ModelCapabilities with conservative defaults, or None if no model
        """
        model_id = model or self._default_model or self.get_running_model()
        if not model_id:
            return None

        # EXO doesn't expose model metadata, so we use conservative defaults
        # Most models support at least 4K context, many support 8K-32K
        # We default to 8K as a reasonable middle ground
        return ModelCapabilities(
            model_id=model_id,
            context_length=8192,  # Conservative default
            sliding_window=None,  # Assume full attention
            rope_scaling_original_context=None,
        )

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

    Ollama provides a local LLM server with its own REST API at /api/chat.
    This backend handles Ollama-specific features including:
    - Model listing via /api/tags
    - Streaming and non-streaming chat completions
    - Tool/function calling support (for compatible models)

    API Reference: https://github.com/ollama/ollama/blob/main/docs/api.md
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        default_model: Optional[str] = None,
        default_temperature: float = 0.7,
        default_max_tokens: Optional[int] = None,
    ):
        """
        Initialize the Ollama backend.

        Args:
            base_url: Ollama API base URL (e.g., http://localhost:11434)
            default_model: Default model to use (None = auto-detect first available)
            default_temperature: Default temperature for completions
            default_max_tokens: Default max tokens (None = no limit)
        """
        self._base_url = base_url.rstrip("/")
        self._default_model = default_model
        self._default_temperature = default_temperature
        self._default_max_tokens = default_max_tokens

        # Initialize OpenAI client for Ollama's OpenAI-compatible endpoint
        # Ollama exposes OpenAI-compatible API at /v1
        self._client = OpenAI(
            base_url=f"{self._base_url}/v1",
            api_key="ollama",  # Ollama doesn't require a real key
        )

        # Tool support cache
        self._tools_supported: Optional[bool] = None
        self._tools_model: Optional[str] = None

        logger.info(f"OllamaBackend initialized with base_url={self._base_url}")

    @property
    def backend_type(self) -> BackendType:
        return BackendType.OLLAMA

    @property
    def client(self) -> OpenAI:
        """Return the underlying OpenAI client for advanced use cases."""
        return self._client

    def get_running_model(self) -> Optional[str]:
        """
        Query Ollama /api/tags for available models.

        Returns the configured default model if set, otherwise returns the
        first available model from the local Ollama instance.

        Returns:
            Model name string or None if no models available
        """
        thread_name = threading.current_thread().name

        # Return configured model if set
        if self._default_model:
            logger.debug(
                f"[{thread_name}] Using configured model: {self._default_model}"
            )
            return self._default_model

        logger.debug(f"[{thread_name}] Querying Ollama for available models...")

        try:
            response = requests.get(f"{self._base_url}/api/tags", timeout=5)
            response.raise_for_status()
            data = response.json()

            models = data.get("models", [])
            if not models:
                logger.debug(f"[{thread_name}] No models found in Ollama")
                return None

            # Return the first available model
            model_name = models[0].get("name") or models[0].get("model")
            if model_name:
                logger.info(f"[{thread_name}] Auto-detected Ollama model: {model_name}")
                return model_name

            return None

        except requests.exceptions.Timeout:
            logger.warning(f"[{thread_name}] Timeout querying Ollama tags")
            return None
        except requests.exceptions.ConnectionError:
            logger.warning(f"[{thread_name}] Connection error - is Ollama running?")
            return None
        except Exception as e:
            logger.warning(f"[{thread_name}] Failed to query Ollama tags: {e}")
            return None

    def check_health(self) -> bool:
        """Check Ollama health via /api/tags endpoint."""
        try:
            response = requests.get(f"{self._base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def get_model_capabilities(
        self, model: Optional[str] = None
    ) -> Optional[ModelCapabilities]:
        """
        Get model capabilities from Ollama's /api/show endpoint.

        Queries Ollama for model metadata including context length and
        attention configuration (including sliding window if present).

        Args:
            model: Model to query (uses default if None)

        Returns:
            ModelCapabilities with context window info, or None if unavailable
        """
        model_id = model or self._default_model or self.get_running_model()
        if not model_id:
            return None

        thread_name = threading.current_thread().name

        try:
            response = requests.post(
                f"{self._base_url}/api/show",
                json={"name": model_id},
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            model_info = data.get("model_info", {})

            # Look for context length - try various possible key patterns
            context_length = None
            sliding_window = None
            rope_original = None

            for key, value in model_info.items():
                key_lower = key.lower()

                # Context length (e.g., "llama.context_length", "gptoss.context_length")
                if "context_length" in key_lower and context_length is None:
                    if isinstance(value, (int, float)):
                        context_length = int(value)

                # Sliding window attention (e.g., "gptoss.attention.sliding_window")
                if "sliding_window" in key_lower and sliding_window is None:
                    if isinstance(value, (int, float)) and value > 0:
                        sliding_window = int(value)

                # RoPE scaling original context
                if "original_context" in key_lower and rope_original is None:
                    if isinstance(value, (int, float)):
                        rope_original = int(value)

            # Default context length if not found
            if context_length is None:
                context_length = 4096  # Conservative default

            capabilities = ModelCapabilities(
                model_id=model_id,
                context_length=context_length,
                sliding_window=sliding_window,
                rope_scaling_original_context=rope_original,
            )

            # Log important capability info
            if capabilities.has_sliding_window:
                logger.warning(
                    f"[{thread_name}] ⚠️ Model {model_id} uses sliding window attention "
                    f"({sliding_window} tokens). Long context may not be fully visible!"
                )
            else:
                logger.info(
                    f"[{thread_name}] Model {model_id} capabilities: "
                    f"context={context_length:,} tokens, full attention"
                )

            return capabilities

        except requests.exceptions.RequestException as e:
            logger.warning(
                f"[{thread_name}] Failed to get model capabilities for {model_id}: {e}"
            )
            return None
        except Exception as e:
            logger.warning(
                f"[{thread_name}] Error parsing model capabilities for {model_id}: {e}"
            )
            return None

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
        logger.info(
            f"[{thread_name}] Detecting tool support for Ollama model {model_to_test}"
        )

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

            payload = {
                "model": model_to_test,
                "messages": [{"role": "user", "content": "test"}],
                "tools": [test_tool],
                "stream": False,
            }

            response = requests.post(
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=30,
            )
            response.raise_for_status()

            # If we get here without error, tools are likely supported
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
                    "unknown",
                ]
            ):
                logger.info(f"[{thread_name}] Tool calls NOT supported: {e}")
            else:
                logger.warning(f"[{thread_name}] Tool detection failed: {e}")

            self._tools_supported = False
            self._tools_model = model_to_test
            return False

    def _convert_tool_calls_to_openai_format(
        self, tool_calls: list[dict]
    ) -> list[dict]:
        """
        Convert Ollama tool calls to OpenAI format.

        Ollama returns tool_calls with arguments as parsed objects,
        while OpenAI format expects arguments as JSON strings.
        """
        import json

        converted = []
        for i, tc in enumerate(tool_calls):
            func = tc.get("function", {})
            arguments = func.get("arguments", {})

            # Convert arguments to JSON string if it's a dict
            if isinstance(arguments, dict):
                arguments_str = json.dumps(arguments)
            else:
                arguments_str = str(arguments)

            converted.append(
                {
                    "id": f"call_{i}",  # Ollama doesn't provide IDs, generate them
                    "type": "function",
                    "function": {
                        "name": func.get("name", ""),
                        "arguments": arguments_str,
                    },
                }
            )

        return converted

    def _normalize_messages_for_ollama(self, messages: list[dict]) -> list[dict]:
        """
        Normalize messages for Ollama's native /api/chat endpoint.

        Handles differences between OpenAI and Ollama message formats:
        - Converts content: None to content: "" for assistant messages
        - Ensures tool messages have the correct format
        """
        normalized = []
        for msg in messages:
            msg_copy = msg.copy()

            # Ollama doesn't like content: None, use empty string instead
            if msg_copy.get("content") is None:
                msg_copy["content"] = ""

            normalized.append(msg_copy)
        return normalized

    def complete(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[list[dict]] = None,
        timeout: Optional[float] = None,
    ) -> CompletionResponse:
        """Non-streaming completion via Ollama /api/chat endpoint."""
        model_to_use = model or self._default_model or self.get_running_model()
        if not model_to_use:
            raise ValueError("No model available for inference")

        # Normalize messages for Ollama's native API
        normalized_messages = self._normalize_messages_for_ollama(messages)

        payload: dict[str, Any] = {
            "model": model_to_use,
            "messages": normalized_messages,
            "stream": False,
        }

        # Add options for temperature and max_tokens
        options: dict[str, Any] = {}

        effective_temp = (
            temperature if temperature is not None else self._default_temperature
        )
        if effective_temp is not None:
            options["temperature"] = effective_temp

        effective_max_tokens = (
            max_tokens if max_tokens is not None else self._default_max_tokens
        )
        if effective_max_tokens is not None:
            options["num_predict"] = effective_max_tokens

        if options:
            payload["options"] = options

        if tools:
            payload["tools"] = tools

        request_timeout = timeout if timeout is not None else 120

        thread_name = threading.current_thread().name
        logger.debug(f"[{thread_name}] Ollama complete request to model {model_to_use}")

        try:
            response = requests.post(
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=request_timeout,
            )
            response.raise_for_status()
        except requests.exceptions.HTTPError:
            # Log the actual error response body for debugging
            error_body = ""
            try:
                error_body = response.text
            except Exception:
                pass
            logger.error(
                f"[{thread_name}] Ollama /api/chat returned {response.status_code}: {error_body}"
            )
            # Log the payload for debugging (truncate large messages)
            debug_payload = payload.copy()
            if "messages" in debug_payload:
                debug_messages = []
                for msg in debug_payload["messages"]:
                    msg_debug = {
                        "role": msg.get("role"),
                        "content_len": len(msg.get("content", "") or ""),
                        "has_tool_calls": "tool_calls" in msg,
                        "has_tool_call_id": "tool_call_id" in msg,
                    }
                    debug_messages.append(msg_debug)
                debug_payload["messages"] = debug_messages
            logger.error(f"[{thread_name}] Request payload summary: {debug_payload}")
            raise
        data = response.json()

        message = data.get("message", {})
        content = message.get("content", "")
        done_reason = data.get("done_reason", "stop")

        # Handle tool calls
        tool_calls = None
        raw_tool_calls = message.get("tool_calls")
        if raw_tool_calls:
            tool_calls = self._convert_tool_calls_to_openai_format(raw_tool_calls)

        # Build usage dict from Ollama stats
        usage = None
        if data.get("prompt_eval_count") or data.get("eval_count"):
            usage = {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": (
                    data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
                ),
            }

        return CompletionResponse(
            content=content,
            finish_reason=done_reason,
            tool_calls=tool_calls,
            usage=usage,
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
        """Streaming completion via Ollama /api/chat endpoint."""
        import json

        model_to_use = model or self._default_model or self.get_running_model()
        if not model_to_use:
            raise ValueError("No model available for inference")

        payload: dict[str, Any] = {
            "model": model_to_use,
            "messages": messages,
            "stream": True,
        }

        # Add options for temperature and max_tokens
        options: dict[str, Any] = {}

        effective_temp = (
            temperature if temperature is not None else self._default_temperature
        )
        if effective_temp is not None:
            options["temperature"] = effective_temp

        effective_max_tokens = (
            max_tokens if max_tokens is not None else self._default_max_tokens
        )
        if effective_max_tokens is not None:
            options["num_predict"] = effective_max_tokens

        if options:
            payload["options"] = options

        if tools:
            payload["tools"] = tools

        request_timeout = timeout if timeout is not None else 120

        thread_name = threading.current_thread().name
        logger.debug(f"[{thread_name}] Ollama stream request to model {model_to_use}")

        # Track accumulated tool calls across chunks
        accumulated_tool_calls: list[dict] = []

        with requests.post(
            f"{self._base_url}/api/chat",
            json=payload,
            timeout=request_timeout,
            stream=True,
        ) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                message = data.get("message", {})
                content = message.get("content", "")
                done = data.get("done", False)
                done_reason = data.get("done_reason")

                # Accumulate tool calls if present
                raw_tool_calls = message.get("tool_calls")
                if raw_tool_calls:
                    accumulated_tool_calls.extend(raw_tool_calls)

                # Determine finish reason
                finish_reason = None
                if done:
                    if accumulated_tool_calls:
                        finish_reason = "tool_calls"
                    else:
                        finish_reason = done_reason or "stop"

                # Only include tool_calls on final chunk
                chunk_tool_calls = None
                if done and accumulated_tool_calls:
                    chunk_tool_calls = self._convert_tool_calls_to_openai_format(
                        accumulated_tool_calls
                    )

                yield CompletionChunk(
                    content=content,
                    finish_reason=finish_reason,
                    tool_calls=chunk_tool_calls,
                )


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

    def get_model_capabilities(
        self, model: Optional[str] = None
    ) -> Optional[ModelCapabilities]:
        """
        Get context window and attention capabilities for a model.

        Args:
            model: Model to query (uses default if None)

        Returns:
            ModelCapabilities with context window info, or None if unavailable
        """
        if self._backend is None:
            return None
        return self._backend.get_model_capabilities(model)

    def validate_context(
        self, messages: list[dict], model: Optional[str] = None
    ) -> tuple[bool, Optional[str]]:
        """
        Validate that messages fit within the model's context window.

        Args:
            messages: List of message dicts to validate
            model: Model to check against (uses default if None)

        Returns:
            Tuple of (fits: bool, warning_message: Optional[str])
        """
        if self._backend is None:
            return (True, None)
        return self._backend.validate_context(messages, model)

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
