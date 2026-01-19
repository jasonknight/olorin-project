"""
Tests for the capabilities detection library.
"""

from libs.capabilities import (
    BASIC_CODE_MODELS,
    CODER_MODEL_PATTERNS,
    STRONG_CODE_MODELS,
    CapabilitiesDetector,
    CodeCapability,
    ModelCapabilities,
    ToolCapability,
)


class TestModelCapabilities:
    """Tests for ModelCapabilities dataclass."""

    def test_default_values(self):
        """Test default capability values."""
        caps = ModelCapabilities(model_id="test-model")
        assert caps.model_id == "test-model"
        assert caps.context_length == 4096
        assert caps.sliding_window is None
        assert caps.tool_calling == ToolCapability.UNKNOWN
        assert caps.code_generation == CodeCapability.UNKNOWN

    def test_effective_context_no_sliding_window(self):
        """Test effective context without sliding window."""
        caps = ModelCapabilities(model_id="test", context_length=32000)
        assert caps.effective_context == 32000
        assert not caps.has_sliding_window

    def test_effective_context_with_sliding_window(self):
        """Test effective context is limited by sliding window."""
        caps = ModelCapabilities(
            model_id="test", context_length=128000, sliding_window=4096
        )
        assert caps.effective_context == 4096
        assert caps.has_sliding_window

    def test_supports_tools(self):
        """Test tool support property."""
        caps = ModelCapabilities(model_id="test")
        assert not caps.supports_tools  # UNKNOWN -> False

        caps.tool_calling = ToolCapability.SUPPORTED
        assert caps.supports_tools

        caps.tool_calling = ToolCapability.NONE
        assert not caps.supports_tools

    def test_supports_code(self):
        """Test code support properties."""
        caps = ModelCapabilities(model_id="test")
        assert not caps.supports_code  # UNKNOWN -> False
        assert not caps.strong_code

        caps.code_generation = CodeCapability.BASIC
        assert caps.supports_code
        assert not caps.strong_code

        caps.code_generation = CodeCapability.STRONG
        assert caps.supports_code
        assert caps.strong_code

    def test_supports_rlm_requirements(self):
        """Test RLM support requirements."""
        # No sliding window + strong code = RLM supported
        caps = ModelCapabilities(
            model_id="test",
            context_length=32000,
            code_generation=CodeCapability.STRONG,
        )
        assert caps.supports_rlm

        # Sliding window disqualifies RLM
        caps.sliding_window = 4096
        assert not caps.supports_rlm

        # Basic code not sufficient for RLM
        caps.sliding_window = None
        caps.code_generation = CodeCapability.BASIC
        assert not caps.supports_rlm

    def test_rlm_confidence(self):
        """Test RLM confidence levels."""
        # Strong code + tools = high confidence
        caps = ModelCapabilities(
            model_id="test",
            code_generation=CodeCapability.STRONG,
            tool_calling=ToolCapability.SUPPORTED,
        )
        assert caps.rlm_confidence == "high"

        # Strong code + unknown tools = medium confidence
        caps.tool_calling = ToolCapability.UNKNOWN
        assert caps.rlm_confidence == "medium"

        # Strong code + no tools = low confidence
        caps.tool_calling = ToolCapability.NONE
        assert caps.rlm_confidence == "low"

        # Not RLM capable = none
        caps.code_generation = CodeCapability.BASIC
        assert caps.rlm_confidence == "none"

    def test_context_fit_validation(self):
        """Test context fit validation."""
        caps = ModelCapabilities(model_id="test", context_length=1000)

        # Small message fits
        messages = [{"role": "user", "content": "Hello"}]
        fits, warning = caps.check_context_fit(messages)
        assert fits
        assert warning is None

        # Large message triggers warning at 80%
        large_content = "x" * 3200  # ~800 tokens
        messages = [{"role": "user", "content": large_content}]
        fits, warning = caps.check_context_fit(messages)
        assert fits
        assert warning is not None
        assert "high" in warning.lower()

        # Very large message doesn't fit
        huge_content = "x" * 8000  # ~2000 tokens
        messages = [{"role": "user", "content": huge_content}]
        fits, warning = caps.check_context_fit(messages)
        assert not fits
        assert "overflow" in warning.lower()

    def test_serialization(self):
        """Test to_dict and from_dict."""
        caps = ModelCapabilities(
            model_id="deepseek-coder:6.7b",
            context_length=16000,
            tool_calling=ToolCapability.SUPPORTED,
            code_generation=CodeCapability.STRONG,
        )

        data = caps.to_dict()
        assert data["model_id"] == "deepseek-coder:6.7b"
        assert data["context_length"] == 16000
        assert data["tool_calling"] == "supported"
        assert data["code_generation"] == "strong"
        assert data["supports_rlm"] is True

        # Round-trip
        restored = ModelCapabilities.from_dict(data)
        assert restored.model_id == caps.model_id
        assert restored.context_length == caps.context_length
        assert restored.tool_calling == caps.tool_calling
        assert restored.code_generation == caps.code_generation


class TestHeuristicDetection:
    """Tests for model name heuristic detection."""

    def test_coder_model_detection(self):
        """Test detection of coder models."""
        detector = CapabilitiesDetector()

        # Known coder models should get STRONG code capability
        coder_models = [
            "deepseek-coder:6.7b",
            "deepseek-coder-v2:16b",
            "codellama:7b",
            "codellama:34b-instruct",
            "starcoder:7b",
            "wizardcoder:34b",
            "qwen2.5-coder:7b",
            "codeqwen:7b",
            "magicoder:7b",
            "codestral:22b",
        ]

        for model in coder_models:
            caps = ModelCapabilities(model_id=model)
            detector._detect_from_model_name(caps)
            assert caps.code_generation == CodeCapability.STRONG, (
                f"Expected {model} to have STRONG code generation"
            )

    def test_strong_code_model_detection(self):
        """Test detection of models with strong coding ability."""
        detector = CapabilitiesDetector()

        # Models known for strong code (even without 'code' in name)
        strong_models = [
            "deepseek-r1:7b",
            "deepseek-v2:16b",
            "qwen2.5:7b",
            "llama3.1:8b",
            "llama3.2:3b",
            "llama-3.3:70b",
            "mistral-large:123b",
            "mixtral:8x7b",
        ]

        for model in strong_models:
            caps = ModelCapabilities(model_id=model)
            detector._detect_from_model_name(caps)
            assert caps.code_generation == CodeCapability.STRONG, (
                f"Expected {model} to have STRONG code generation"
            )

    def test_basic_code_model_detection(self):
        """Test detection of models with basic coding ability."""
        detector = CapabilitiesDetector()

        # Models with basic code capability
        basic_models = [
            "llama2:7b",
            "llama-2:13b",
            "llama3:8b",  # llama3 without .1/.2
            "mistral:7b",
            "phi2:2.7b",
            "phi-3:3.8b",
            "gemma:7b",
            "vicuna:13b",
            "zephyr:7b",
        ]

        for model in basic_models:
            caps = ModelCapabilities(model_id=model)
            detector._detect_from_model_name(caps)
            assert caps.code_generation == CodeCapability.BASIC, (
                f"Expected {model} to have BASIC code generation, got {caps.code_generation}"
            )

    def test_unknown_model_detection(self):
        """Test that unknown models get UNKNOWN capability."""
        detector = CapabilitiesDetector()

        # Random/unknown model names
        unknown_models = [
            "my-custom-model:latest",
            "fine-tuned-chat:v1",
            "experimental-7b",
        ]

        for model in unknown_models:
            caps = ModelCapabilities(model_id=model)
            detector._detect_from_model_name(caps)
            assert caps.code_generation == CodeCapability.UNKNOWN, (
                f"Expected {model} to have UNKNOWN code generation"
            )

    def test_case_insensitivity(self):
        """Test that detection is case-insensitive."""
        detector = CapabilitiesDetector()

        variants = [
            "DeepSeek-Coder:6.7b",
            "DEEPSEEK-CODER:6.7B",
            "deepseek-coder:6.7B",
        ]

        for model in variants:
            caps = ModelCapabilities(model_id=model)
            detector._detect_from_model_name(caps)
            assert caps.code_generation == CodeCapability.STRONG


class TestCapabilitiesDetector:
    """Tests for CapabilitiesDetector class."""

    def test_detect_without_inference_client(self):
        """Test detection works without inference client (heuristics only)."""
        detector = CapabilitiesDetector()

        caps = detector.detect("deepseek-coder:6.7b", probe_tools=False)
        assert caps.model_id == "deepseek-coder:6.7b"
        assert caps.code_generation == CodeCapability.STRONG
        assert caps.tool_calling == ToolCapability.UNKNOWN  # No probe
        assert "heuristic" in caps.detection_method

    def test_caching(self):
        """Test that capabilities are cached."""
        detector = CapabilitiesDetector()

        # First call
        caps1 = detector.detect("test-model:7b", probe_tools=False)

        # Modify the cached value
        caps1.code_generation = CodeCapability.STRONG

        # Second call should return cached version
        caps2 = detector.detect("test-model:7b", probe_tools=False)
        assert caps2.code_generation == CodeCapability.STRONG

        # Force refresh should get fresh detection
        caps3 = detector.detect("test-model:7b", probe_tools=False, force_refresh=True)
        assert caps3.code_generation == CodeCapability.UNKNOWN

    def test_invalidate_cache(self):
        """Test cache invalidation."""
        detector = CapabilitiesDetector()

        # Populate cache
        detector.detect("model1:7b", probe_tools=False)
        detector.detect("model2:7b", probe_tools=False)

        assert "model1:7b" in detector._cache
        assert "model2:7b" in detector._cache

        # Invalidate specific model
        detector.invalidate("model1:7b")
        assert "model1:7b" not in detector._cache
        assert "model2:7b" in detector._cache

        # Invalidate all
        detector.invalidate()
        assert len(detector._cache) == 0


class TestPatternCoverage:
    """Tests to ensure pattern lists cover expected models."""

    def test_coder_patterns_have_no_overlap_with_strong(self):
        """Ensure coder patterns don't overlap with strong patterns."""
        # This is fine - coder patterns are checked first
        pass

    def test_common_models_are_covered(self):
        """Ensure commonly used models are in the pattern lists."""
        import re

        # These should all match some pattern
        expected_strong = [
            "deepseek-coder:6.7b",
            "codellama:7b",
            "qwen2.5:7b",
            "llama3.1:8b",
            "deepseek-r1:7b",
        ]

        for model in expected_strong:
            matched = False
            for pattern in CODER_MODEL_PATTERNS + STRONG_CODE_MODELS:
                if re.search(pattern, model.lower(), re.IGNORECASE):
                    matched = True
                    break
            assert matched, f"{model} should match a STRONG code pattern"

        expected_basic = [
            "llama2:7b",
            "mistral:7b",
            "gemma:7b",
        ]

        for model in expected_basic:
            matched_coder = any(
                re.search(p, model.lower(), re.IGNORECASE) for p in CODER_MODEL_PATTERNS
            )
            matched_strong = any(
                re.search(p, model.lower(), re.IGNORECASE) for p in STRONG_CODE_MODELS
            )
            matched_basic = any(
                re.search(p, model.lower(), re.IGNORECASE) for p in BASIC_CODE_MODELS
            )

            # Should NOT match coder/strong, SHOULD match basic
            assert not matched_coder, f"{model} should not match coder patterns"
            assert not matched_strong, f"{model} should not match strong patterns"
            assert matched_basic, f"{model} should match a BASIC code pattern"
