"""
Centralized Model Capabilities Library for Olorin Project

Provides unified capability detection for LLMs including:
- Context window size and attention type (sliding window vs full)
- Tool/function calling support
- Code generation ability
- Other capabilities as needed for advanced features (e.g., RLM)

This library abstracts capability detection from specific backends (Ollama, EXO)
and provides a single source of truth for model capabilities across all components.

Usage:
    from libs.capabilities import get_capabilities_detector, ModelCapabilities

    detector = get_capabilities_detector()

    # Get all capabilities for current model
    caps = detector.detect(model="deepseek-coder:6.7b")
    print(f"Tool support: {caps.tool_calling}")
    print(f"Code generation: {caps.code_generation}")
    print(f"Context length: {caps.context_length}")

    # Quick checks
    if caps.supports_rlm:
        # Model can handle recursive LLM patterns
        pass
"""

import logging
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from libs.inference import InferenceClient

logger = logging.getLogger(__name__)


class CodeCapability(Enum):
    """Code generation capability levels."""

    NONE = "none"  # No code generation ability
    BASIC = "basic"  # Can generate simple code but not reliable
    STRONG = "strong"  # Strong code generation (coder models)
    UNKNOWN = "unknown"  # Not yet determined


class ToolCapability(Enum):
    """Tool/function calling capability levels."""

    NONE = "none"  # No tool support
    SUPPORTED = "supported"  # Tools work
    UNKNOWN = "unknown"  # Not yet determined


# Model family patterns for heuristic capability detection
# These are regex patterns matched against model names (case-insensitive)
CODER_MODEL_PATTERNS = [
    r"deepseek[-_]?coder",
    r"codellama",
    r"code[-_]?llama",
    r"starcoder",
    r"wizardcoder",
    r"phind[-_]?codellama",
    r"codestral",
    r"qwen.*coder",
    r"codeqwen",
    r"granite[-_]?code",
    r"stable[-_]?code",
    r"opencodeinterpreter",
    r"magicoder",
    r"codebooga",
    r"code[-_]?gemma",
]

# Models known to have strong coding abilities even without "code" in name
STRONG_CODE_MODELS = [
    r"gpt[-_]?4",
    r"gpt[-_]?3\.5",
    r"claude",
    r"deepseek[-_]?v[23]",
    r"deepseek[-_]?r1",
    r"qwen2\.5",  # Qwen 2.5 series has strong coding
    r"llama[-_]?3\.1",  # Llama 3.1 has improved coding
    r"llama[-_]?3\.2",
    r"llama[-_]?3\.3",
    r"gemini",
    r"mistral[-_]?large",
    r"mixtral",
]

# Models known to have basic/limited coding ability
BASIC_CODE_MODELS = [
    r"llama[-_]?2",
    r"llama[-_]?3(?!\.)",  # llama-3 but not llama-3.1/3.2
    r"mistral(?![-_]large)",  # mistral but not mistral-large
    r"vicuna",
    r"alpaca",
    r"orca",
    r"phi[-_]?[123]",
    r"gemma(?![-_]code)",  # gemma but not code-gemma
    r"tinyllama",
    r"openhermes",
    r"neural[-_]?chat",
    r"zephyr",
]

# Models known to NOT support code generation well
NO_CODE_MODELS = [
    r"ggml",  # Raw GGML models often lack instruction tuning
    r"raw",
    r"base(?!line)",  # Base models without instruction tuning
]


@dataclass
class ModelCapabilities:
    """
    Comprehensive capability profile for a model.

    This class consolidates all detected capabilities for a model,
    providing a single source of truth for feature decisions.

    Attributes:
        model_id: The model identifier
        backend: Which backend was used for detection (ollama, exo)
        detected_at: When capabilities were detected

        # Context capabilities
        context_length: Maximum context window in tokens
        sliding_window: Sliding window size if applicable (None = full attention)
        effective_context: Actual usable context considering sliding window

        # Feature capabilities
        tool_calling: Tool/function calling capability level
        code_generation: Code generation capability level

        # Metadata
        confidence: Overall confidence in detection ("high", "medium", "low")
        detection_method: How capabilities were determined
    """

    model_id: str
    backend: str = "unknown"
    detected_at: datetime = field(default_factory=datetime.now)

    # Context capabilities
    context_length: int = 4096  # Conservative default
    sliding_window: Optional[int] = None
    rope_scaling_original_context: Optional[int] = None

    # Feature capabilities
    tool_calling: ToolCapability = ToolCapability.UNKNOWN
    code_generation: CodeCapability = CodeCapability.UNKNOWN

    # Metadata
    confidence: str = "low"
    detection_method: str = "default"
    _raw_model_info: dict = field(default_factory=dict, repr=False)

    @property
    def effective_context(self) -> int:
        """Return effective context length considering sliding window."""
        if self.sliding_window is not None and self.sliding_window > 0:
            return self.sliding_window
        return self.context_length

    @property
    def has_sliding_window(self) -> bool:
        """Check if model uses sliding window attention."""
        return self.sliding_window is not None and self.sliding_window > 0

    @property
    def supports_tools(self) -> bool:
        """Check if model supports tool calling."""
        return self.tool_calling == ToolCapability.SUPPORTED

    @property
    def supports_code(self) -> bool:
        """Check if model has at least basic code generation."""
        return self.code_generation in (CodeCapability.BASIC, CodeCapability.STRONG)

    @property
    def strong_code(self) -> bool:
        """Check if model has strong code generation ability."""
        return self.code_generation == CodeCapability.STRONG

    @property
    def supports_rlm(self) -> bool:
        """
        Check if model can support Recursive LLM patterns.

        Requirements:
        - Strong code generation (must write correct decomposition code)
        - No sliding window attention (would lose recursive context)
        - Tool support preferred but not strictly required
        """
        if self.has_sliding_window:
            return False
        if self.code_generation != CodeCapability.STRONG:
            return False
        return True

    @property
    def rlm_confidence(self) -> str:
        """
        Confidence level for RLM support.

        Returns "high" if all indicators are positive,
        "medium" if code generation is strong but tools unknown,
        "low" otherwise.
        """
        if not self.supports_rlm:
            return "none"
        if self.supports_tools:
            return "high"
        if self.tool_calling == ToolCapability.UNKNOWN:
            return "medium"
        return "low"

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (~4 chars per token for English)."""
        return len(text) // 4

    def estimate_messages_tokens(self, messages: list[dict]) -> int:
        """Estimate total token count for a messages array."""
        total = 0
        for msg in messages:
            total += 10  # Message overhead
            content = msg.get("content", "")
            if content:
                total += self.estimate_tokens(content)
            if msg.get("tool_calls"):
                total += 50  # Tool call overhead
        return total

    def check_context_fit(self, messages: list[dict]) -> tuple[bool, Optional[str]]:
        """Check if messages fit within the model's effective context."""
        estimated_tokens = self.estimate_messages_tokens(messages)

        if self.has_sliding_window:
            if estimated_tokens > self.sliding_window:
                return (
                    False,
                    f"Context overflow: ~{estimated_tokens:,} tokens exceeds "
                    f"sliding window of {self.sliding_window:,} tokens. "
                    f"Earlier context will be invisible to the model.",
                )

        if estimated_tokens > self.context_length:
            return (
                False,
                f"Context overflow: ~{estimated_tokens:,} tokens exceeds "
                f"context window of {self.context_length:,} tokens.",
            )

        if estimated_tokens > self.context_length * 0.8:
            return (
                True,
                f"Context usage high: ~{estimated_tokens:,} tokens "
                f"({estimated_tokens * 100 // self.context_length}% of limit)",
            )

        return (True, None)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "model_id": self.model_id,
            "backend": self.backend,
            "detected_at": self.detected_at.isoformat(),
            "context_length": self.context_length,
            "sliding_window": self.sliding_window,
            "effective_context": self.effective_context,
            "tool_calling": self.tool_calling.value,
            "code_generation": self.code_generation.value,
            "supports_tools": self.supports_tools,
            "supports_code": self.supports_code,
            "strong_code": self.strong_code,
            "supports_rlm": self.supports_rlm,
            "rlm_confidence": self.rlm_confidence,
            "has_sliding_window": self.has_sliding_window,
            "confidence": self.confidence,
            "detection_method": self.detection_method,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelCapabilities":
        """Create from dictionary."""
        caps = cls(
            model_id=data["model_id"],
            backend=data.get("backend", "unknown"),
            context_length=data.get("context_length", 4096),
            sliding_window=data.get("sliding_window"),
            rope_scaling_original_context=data.get("rope_scaling_original_context"),
            confidence=data.get("confidence", "low"),
            detection_method=data.get("detection_method", "deserialized"),
        )

        # Parse enums
        if "tool_calling" in data:
            try:
                caps.tool_calling = ToolCapability(data["tool_calling"])
            except ValueError:
                caps.tool_calling = ToolCapability.UNKNOWN

        if "code_generation" in data:
            try:
                caps.code_generation = CodeCapability(data["code_generation"])
            except ValueError:
                caps.code_generation = CodeCapability.UNKNOWN

        if "detected_at" in data:
            try:
                caps.detected_at = datetime.fromisoformat(data["detected_at"])
            except (ValueError, TypeError):
                pass

        return caps


class CapabilitiesDetector:
    """
    Detects and caches model capabilities.

    This class coordinates capability detection across different backends
    and caching layers (memory, state.db).

    Usage:
        detector = CapabilitiesDetector(inference_client)
        caps = detector.detect("deepseek-coder:6.7b")
    """

    def __init__(self, inference_client: Optional["InferenceClient"] = None):
        """
        Initialize the capabilities detector.

        Args:
            inference_client: Optional inference client for probing capabilities.
                              If not provided, only heuristic detection is available.
        """
        self._inference = inference_client
        self._cache: dict[str, ModelCapabilities] = {}
        self._lock = threading.Lock()

        # State storage for persistence (lazy init)
        self._state = None

    def _get_state(self):
        """Lazy-load state storage."""
        if self._state is None:
            try:
                from libs.state import get_state

                self._state = get_state()
            except ImportError:
                logger.warning("State library not available, caching in memory only")
        return self._state

    def detect(
        self,
        model: Optional[str] = None,
        force_refresh: bool = False,
        probe_tools: bool = True,
        probe_code: bool = False,
    ) -> ModelCapabilities:
        """
        Detect capabilities for a model.

        Args:
            model: Model to detect (uses inference client's current model if None)
            force_refresh: Bypass cache and re-detect
            probe_tools: Whether to probe tool support (makes API call)
            probe_code: Whether to probe code generation (makes API call)

        Returns:
            ModelCapabilities with detected values
        """
        thread_name = threading.current_thread().name

        # Resolve model name
        model_id = model
        if model_id is None and self._inference:
            model_id = self._inference.get_running_model()

        if not model_id:
            logger.warning(f"[{thread_name}] No model specified and none running")
            return ModelCapabilities(
                model_id="unknown",
                confidence="none",
                detection_method="no_model",
            )

        # Check cache
        if not force_refresh:
            with self._lock:
                if model_id in self._cache:
                    logger.debug(
                        f"[{thread_name}] Using cached capabilities for {model_id}"
                    )
                    return self._cache[model_id]

            # Check state.db cache
            cached = self._load_from_state(model_id)
            if cached is not None:
                with self._lock:
                    self._cache[model_id] = cached
                logger.debug(
                    f"[{thread_name}] Loaded capabilities from state.db for {model_id}"
                )
                return cached

        # Build capabilities through detection layers
        logger.info(f"[{thread_name}] Detecting capabilities for {model_id}")

        # Start with defaults
        caps = ModelCapabilities(
            model_id=model_id,
            detection_method="heuristic",
        )

        # Layer 1: Heuristic detection from model name
        self._detect_from_model_name(caps)

        # Layer 2: Backend-specific detection (context, sliding window)
        if self._inference:
            self._detect_from_backend(caps)

        # Layer 3: Probe tool support if requested
        if probe_tools and self._inference:
            self._probe_tool_support(caps)

        # Layer 4: Probe code generation if requested
        if probe_code and self._inference:
            self._probe_code_generation(caps)

        # Determine overall confidence
        self._compute_confidence(caps)

        # Cache results
        with self._lock:
            self._cache[model_id] = caps
        self._save_to_state(caps)

        logger.info(
            f"[{thread_name}] Capabilities for {model_id}: "
            f"context={caps.context_length:,}, "
            f"sliding_window={caps.sliding_window}, "
            f"tools={caps.tool_calling.value}, "
            f"code={caps.code_generation.value}, "
            f"rlm={caps.supports_rlm} ({caps.rlm_confidence})"
        )

        return caps

    def _detect_from_model_name(self, caps: ModelCapabilities) -> None:
        """Detect capabilities from model name patterns."""
        model_lower = caps.model_id.lower()

        # Check for coder models (strongest code generation)
        for pattern in CODER_MODEL_PATTERNS:
            if re.search(pattern, model_lower, re.IGNORECASE):
                caps.code_generation = CodeCapability.STRONG
                caps.detection_method = "heuristic_coder_pattern"
                logger.debug(f"Model {caps.model_id} matched coder pattern: {pattern}")
                return

        # Check for models with strong code ability
        for pattern in STRONG_CODE_MODELS:
            if re.search(pattern, model_lower, re.IGNORECASE):
                caps.code_generation = CodeCapability.STRONG
                caps.detection_method = "heuristic_strong_code"
                logger.debug(
                    f"Model {caps.model_id} matched strong code pattern: {pattern}"
                )
                return

        # Check for models with basic code ability
        for pattern in BASIC_CODE_MODELS:
            if re.search(pattern, model_lower, re.IGNORECASE):
                caps.code_generation = CodeCapability.BASIC
                caps.detection_method = "heuristic_basic_code"
                logger.debug(
                    f"Model {caps.model_id} matched basic code pattern: {pattern}"
                )
                return

        # Check for models with no code ability
        for pattern in NO_CODE_MODELS:
            if re.search(pattern, model_lower, re.IGNORECASE):
                caps.code_generation = CodeCapability.NONE
                caps.detection_method = "heuristic_no_code"
                logger.debug(
                    f"Model {caps.model_id} matched no-code pattern: {pattern}"
                )
                return

        # Default: unknown, needs probing
        caps.code_generation = CodeCapability.UNKNOWN
        caps.detection_method = "heuristic_unknown"

    def _detect_from_backend(self, caps: ModelCapabilities) -> None:
        """Detect capabilities from inference backend."""
        if not self._inference:
            return

        thread_name = threading.current_thread().name

        try:
            # Get backend type
            backend_type = self._inference.backend_type
            caps.backend = backend_type.value

            # Get context/attention capabilities from backend
            backend_caps = self._inference.get_model_capabilities(caps.model_id)
            if backend_caps:
                caps.context_length = backend_caps.context_length
                caps.sliding_window = backend_caps.sliding_window
                caps.rope_scaling_original_context = (
                    backend_caps.rope_scaling_original_context
                )
                caps.detection_method = f"{caps.detection_method}+backend"

                logger.debug(
                    f"[{thread_name}] Backend detection: "
                    f"context={caps.context_length}, sliding={caps.sliding_window}"
                )

        except Exception as e:
            logger.warning(f"[{thread_name}] Backend detection failed: {e}")

    def _probe_tool_support(self, caps: ModelCapabilities) -> None:
        """Probe tool calling support via API."""
        if not self._inference:
            return

        thread_name = threading.current_thread().name

        try:
            supported = self._inference.supports_tools(caps.model_id)
            caps.tool_calling = (
                ToolCapability.SUPPORTED if supported else ToolCapability.NONE
            )
            caps.detection_method = f"{caps.detection_method}+tool_probe"

            logger.debug(
                f"[{thread_name}] Tool probe: "
                f"{'supported' if supported else 'not supported'}"
            )

        except Exception as e:
            logger.warning(f"[{thread_name}] Tool probe failed: {e}")
            caps.tool_calling = ToolCapability.UNKNOWN

    def _probe_code_generation(self, caps: ModelCapabilities) -> None:
        """
        Probe code generation ability via simple test.

        Only runs if heuristic detection resulted in UNKNOWN.
        """
        if not self._inference:
            return

        # Skip if already determined by heuristics
        if caps.code_generation != CodeCapability.UNKNOWN:
            return

        thread_name = threading.current_thread().name
        logger.info(f"[{thread_name}] Probing code generation for {caps.model_id}")

        try:
            # Simple code generation test
            test_prompt = (
                "Write a Python function called `add_numbers` that takes two "
                "arguments `a` and `b` and returns their sum. Only output the "
                "code, no explanation."
            )

            response = self._inference.complete(
                messages=[{"role": "user", "content": test_prompt}],
                model=caps.model_id,
                temperature=0.0,  # Deterministic
                max_tokens=150,
            )

            content = response.content.strip()

            # Check if response contains valid Python function
            has_def = "def add_numbers" in content or "def add_numbers(" in content
            has_return = "return" in content
            has_addition = "a + b" in content or "a+b" in content

            if has_def and has_return and has_addition:
                # Looks like valid code
                caps.code_generation = CodeCapability.STRONG
                logger.info(
                    f"[{thread_name}] Code probe: STRONG (valid function generated)"
                )
            elif has_def or "def " in content:
                # Has function definition but maybe incomplete
                caps.code_generation = CodeCapability.BASIC
                logger.info(
                    f"[{thread_name}] Code probe: BASIC (partial function generated)"
                )
            else:
                # Couldn't generate valid code
                caps.code_generation = CodeCapability.NONE
                logger.info(
                    f"[{thread_name}] Code probe: NONE (no valid code in response)"
                )

            caps.detection_method = f"{caps.detection_method}+code_probe"

        except Exception as e:
            logger.warning(f"[{thread_name}] Code probe failed: {e}")
            # Leave as UNKNOWN on failure

    def _compute_confidence(self, caps: ModelCapabilities) -> None:
        """Compute overall confidence based on detection methods used."""
        methods = caps.detection_method.split("+")

        # High confidence if we have backend data and probes
        if "backend" in methods and (
            "tool_probe" in methods or "code_probe" in methods
        ):
            caps.confidence = "high"
        # Medium confidence with backend or probes
        elif "backend" in methods or "tool_probe" in methods or "code_probe" in methods:
            caps.confidence = "medium"
        # Low confidence with heuristics only
        elif any("heuristic" in m for m in methods):
            caps.confidence = (
                "medium" if "coder_pattern" in caps.detection_method else "low"
            )
        else:
            caps.confidence = "low"

    def _load_from_state(self, model_id: str) -> Optional[ModelCapabilities]:
        """Load cached capabilities from state.db."""
        state = self._get_state()
        if state is None:
            return None

        try:
            key = f"capabilities.{model_id.replace(':', '_').replace('/', '_')}"
            data = state.get_json(key)
            if data is None:
                return None

            # Check age - invalidate after 24 hours
            caps = ModelCapabilities.from_dict(data)
            age = datetime.now() - caps.detected_at
            if age.total_seconds() > 86400:  # 24 hours
                logger.debug(f"Cached capabilities for {model_id} expired")
                return None

            return caps

        except Exception as e:
            logger.warning(f"Failed to load capabilities from state: {e}")
            return None

    def _save_to_state(self, caps: ModelCapabilities) -> None:
        """Save capabilities to state.db for persistence."""
        state = self._get_state()
        if state is None:
            return

        try:
            key = f"capabilities.{caps.model_id.replace(':', '_').replace('/', '_')}"
            state.set_json(key, caps.to_dict())
            logger.debug(f"Saved capabilities to state.db: {key}")

        except Exception as e:
            logger.warning(f"Failed to save capabilities to state: {e}")

    def invalidate(self, model: Optional[str] = None) -> None:
        """
        Invalidate cached capabilities.

        Args:
            model: Specific model to invalidate, or None for all models
        """
        with self._lock:
            if model:
                self._cache.pop(model, None)
            else:
                self._cache.clear()

        # Also clear from state.db
        state = self._get_state()
        if state:
            try:
                if model:
                    key = f"capabilities.{model.replace(':', '_').replace('/', '_')}"
                    state.delete(key)
                else:
                    state.delete_prefix("capabilities.")
            except Exception as e:
                logger.warning(f"Failed to clear capabilities from state: {e}")


# Singleton instance
_detector: Optional[CapabilitiesDetector] = None
_detector_lock = threading.Lock()


def get_capabilities_detector(
    inference_client: Optional["InferenceClient"] = None,
) -> CapabilitiesDetector:
    """
    Get the singleton capabilities detector.

    Args:
        inference_client: Optional inference client. If provided on first call,
                          will be used for probing. Subsequent calls ignore this.

    Returns:
        The singleton CapabilitiesDetector instance
    """
    global _detector

    with _detector_lock:
        if _detector is None:
            _detector = CapabilitiesDetector(inference_client)
        elif inference_client is not None and _detector._inference is None:
            # Update inference client if not set
            _detector._inference = inference_client

    return _detector


def detect_capabilities(
    model: Optional[str] = None,
    inference_client: Optional["InferenceClient"] = None,
    **kwargs,
) -> ModelCapabilities:
    """
    Convenience function to detect capabilities for a model.

    Args:
        model: Model to detect (uses current model if None)
        inference_client: Inference client to use (uses singleton if None)
        **kwargs: Additional arguments passed to detect()

    Returns:
        ModelCapabilities for the model
    """
    detector = get_capabilities_detector(inference_client)
    return detector.detect(model=model, **kwargs)
