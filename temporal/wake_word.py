"""
Wake word detection using Picovoice Porcupine.

Provides low-latency, accurate wake word detection using a custom trained model.
"""

from typing import Callable, Optional

import numpy as np
import pvporcupine


class WakeWordDetector:
    """Wake word detector using Picovoice Porcupine."""

    def __init__(
        self,
        access_key: str,
        keyword_path: str,
        sensitivity: float = 0.5,
        on_detection: Optional[Callable[[], None]] = None,
    ):
        """
        Initialize wake word detector.

        Args:
            access_key: Picovoice access key (get from console.picovoice.ai)
            keyword_path: Path to .ppn wake word model file
            sensitivity: Detection sensitivity (0.0-1.0), higher = more sensitive
            on_detection: Optional callback when wake word is detected
        """
        self.access_key = access_key
        self.keyword_path = keyword_path
        self.sensitivity = sensitivity
        self.on_detection = on_detection

        self._porcupine: Optional[pvporcupine.Porcupine] = None
        self._frame_buffer: list[int] = []

    @property
    def frame_length(self) -> int:
        """Get the required frame length for Porcupine."""
        if self._porcupine is None:
            self._init_porcupine()
        return self._porcupine.frame_length

    @property
    def sample_rate(self) -> int:
        """Get the required sample rate for Porcupine."""
        if self._porcupine is None:
            self._init_porcupine()
        return self._porcupine.sample_rate

    def _init_porcupine(self):
        """Initialize Porcupine engine."""
        if self._porcupine is not None:
            return

        self._porcupine = pvporcupine.create(
            access_key=self.access_key,
            keyword_paths=[self.keyword_path],
            sensitivities=[self.sensitivity],
        )

    def process_audio(self, audio: np.ndarray) -> bool:
        """
        Process audio chunk and check for wake word.

        Args:
            audio: Audio samples as float32 numpy array (16kHz mono)

        Returns:
            True if wake word was detected in this chunk
        """
        self._init_porcupine()

        # Convert float32 [-1.0, 1.0] to int16
        if audio.dtype == np.float32:
            audio_int16 = (audio * 32767).astype(np.int16)
        else:
            audio_int16 = audio.astype(np.int16)

        # Flatten if needed
        if len(audio_int16.shape) > 1:
            audio_int16 = audio_int16.flatten()

        # Add samples to frame buffer
        self._frame_buffer.extend(audio_int16.tolist())

        detected = False

        # Process complete frames
        while len(self._frame_buffer) >= self._porcupine.frame_length:
            frame = self._frame_buffer[: self._porcupine.frame_length]
            self._frame_buffer = self._frame_buffer[self._porcupine.frame_length :]

            keyword_index = self._porcupine.process(frame)

            if keyword_index >= 0:
                detected = True
                if self.on_detection:
                    self.on_detection()

        return detected

    def reset(self):
        """Reset the frame buffer."""
        self._frame_buffer.clear()

    def cleanup(self):
        """Release Porcupine resources."""
        if self._porcupine is not None:
            self._porcupine.delete()
            self._porcupine = None
        self._frame_buffer.clear()

    def __enter__(self):
        """Context manager entry."""
        self._init_porcupine()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
        return False
