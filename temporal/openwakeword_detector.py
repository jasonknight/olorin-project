"""
OpenWakeWord detector for Temporal voice pipeline.

This module provides wake word detection using OpenWakeWord models,
including support for custom trained models and verifier models.
"""

import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


class OpenWakeWordDetector:
    """
    Wake word detector using OpenWakeWord.

    Supports both pre-trained models (hey_jarvis, alexa, etc.) and
    custom trained models (.onnx or .tflite files).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        verifier_path: Optional[str] = None,
        threshold: float = 0.5,
        verifier_threshold: float = 0.3,
        callback: Optional[Callable[[], None]] = None,
    ):
        """
        Initialize the OpenWakeWord detector.

        Args:
            model_path: Path to custom model file (.onnx or .tflite).
                       If None, uses default pre-trained models.
            verifier_path: Path to custom verifier model (.pkl).
                          Improves accuracy for specific voices.
            threshold: Detection threshold (0.0-1.0). Higher = fewer false positives.
            verifier_threshold: Verifier threshold. Only used if verifier_path is set.
            callback: Optional function to call when wake word is detected.
        """
        self.model_path = model_path
        self.verifier_path = verifier_path
        self.threshold = threshold
        self.verifier_threshold = verifier_threshold
        self.callback = callback
        self._model = None
        self._model_name = None

    def _init_model(self):
        """Lazy initialization of the OpenWakeWord model."""
        if self._model is not None:
            return

        try:
            from openwakeword.model import Model
        except ImportError:
            raise ImportError(
                "openwakeword not installed. Run: pip install openwakeword"
            )

        # Determine model configuration
        wakeword_models = None
        custom_verifier_models = None

        if self.model_path:
            model_file = Path(self.model_path)
            if not model_file.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            wakeword_models = [str(model_file)]
            self._model_name = model_file.stem

            logger.info(f"Loading custom wake word model: {self.model_path}")
        else:
            logger.info("Using default OpenWakeWord models")

        # Load verifier if provided
        if self.verifier_path:
            verifier_file = Path(self.verifier_path)
            if verifier_file.exists():
                if self._model_name:
                    custom_verifier_models = {self._model_name: str(verifier_file)}
                    logger.info(f"Loading verifier model: {self.verifier_path}")
                else:
                    logger.warning(
                        "Verifier specified but no custom model - skipping verifier"
                    )
            else:
                logger.warning(f"Verifier file not found: {self.verifier_path}")

        # Initialize the model
        self._model = Model(
            wakeword_models=wakeword_models,
            custom_verifier_models=custom_verifier_models,
            custom_verifier_threshold=self.verifier_threshold
            if custom_verifier_models
            else None,
        )

        logger.info(f"OpenWakeWord initialized with threshold={self.threshold}")

    @property
    def sample_rate(self) -> int:
        """Required sample rate for audio input."""
        return 16000

    @property
    def frame_length(self) -> int:
        """Recommended frame length in samples (80ms at 16kHz)."""
        return 1280

    def process_audio(self, audio: np.ndarray) -> bool:
        """
        Process an audio chunk and check for wake word.

        Args:
            audio: Audio samples as numpy array. Can be:
                   - float32 in range [-1.0, 1.0]
                   - int16 in range [-32768, 32767]

        Returns:
            True if wake word detected, False otherwise.
        """
        self._init_model()

        # Ensure 1D array
        if audio.ndim > 1:
            audio = audio.flatten()

        # Convert float32 to int16 if needed (OpenWakeWord expects int16)
        if audio.dtype == np.float32:
            audio = (audio * 32767).astype(np.int16)
        elif audio.dtype != np.int16:
            audio = audio.astype(np.int16)

        # Get predictions
        predictions = self._model.predict(audio)

        # Check all predictions against threshold
        for model_name, score in predictions.items():
            if score > self.threshold:
                logger.info(f"Wake word detected: {model_name} (score={score:.3f})")
                if self.callback:
                    self.callback()
                return True

        return False

    def get_scores(self, audio: np.ndarray) -> dict[str, float]:
        """
        Get raw prediction scores without threshold check.

        Args:
            audio: Audio samples as numpy array.

        Returns:
            Dictionary mapping model names to scores.
        """
        self._init_model()

        if audio.ndim > 1:
            audio = audio.flatten()

        if audio.dtype == np.float32:
            audio = (audio * 32767).astype(np.int16)
        elif audio.dtype != np.int16:
            audio = audio.astype(np.int16)

        return self._model.predict(audio)

    def reset(self):
        """Reset the model's internal state."""
        if self._model is not None:
            self._model.reset()

    def cleanup(self):
        """Release resources."""
        self._model = None
        self._model_name = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
        return False
