"""
Speech-to-Text engine using Picovoice Leopard.

Provides accurate, fast transcription for recorded audio.
"""

from typing import Optional

import numpy as np
import pvleopard


class LeopardSTT:
    """Speech-to-text engine using Picovoice Leopard."""

    def __init__(
        self,
        access_key: str,
        model_path: Optional[str] = None,
        enable_automatic_punctuation: bool = True,
    ):
        """
        Initialize Leopard STT engine.

        Args:
            access_key: Picovoice access key
            model_path: Path to custom Leopard model (None for default)
            enable_automatic_punctuation: Add punctuation to transcriptions
        """
        self.access_key = access_key
        self.model_path = model_path
        self.enable_automatic_punctuation = enable_automatic_punctuation

        self._leopard: Optional[pvleopard.Leopard] = None

    def _init_leopard(self):
        """Initialize Leopard engine."""
        if self._leopard is not None:
            return

        kwargs = {
            "access_key": self.access_key,
            "enable_automatic_punctuation": self.enable_automatic_punctuation,
        }
        if self.model_path:
            kwargs["model_path"] = self.model_path

        self._leopard = pvleopard.create(**kwargs)

    @property
    def sample_rate(self) -> int:
        """Get the required sample rate for Leopard."""
        self._init_leopard()
        return self._leopard.sample_rate

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe audio to text.

        Args:
            audio: Audio samples as float32 or int16 numpy array
            sample_rate: Audio sample rate in Hz (should be 16000)

        Returns:
            Transcribed text
        """
        self._init_leopard()

        # Convert float32 [-1.0, 1.0] to int16
        if audio.dtype == np.float32:
            audio_int16 = (audio * 32767).astype(np.int16)
        else:
            audio_int16 = audio.astype(np.int16)

        # Flatten if needed
        if len(audio_int16.shape) > 1:
            audio_int16 = audio_int16.flatten()

        # Leopard expects a list of int16 samples
        transcript, words = self._leopard.process(audio_int16.tolist())

        return transcript

    def transcribe_with_words(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> tuple[str, list[dict]]:
        """
        Transcribe audio and return word-level timestamps.

        Args:
            audio: Audio samples as float32 or int16 numpy array
            sample_rate: Audio sample rate in Hz

        Returns:
            Tuple of (transcript, words) where words is a list of dicts
            with 'word', 'start_sec', 'end_sec', 'confidence' keys
        """
        self._init_leopard()

        # Convert float32 [-1.0, 1.0] to int16
        if audio.dtype == np.float32:
            audio_int16 = (audio * 32767).astype(np.int16)
        else:
            audio_int16 = audio.astype(np.int16)

        # Flatten if needed
        if len(audio_int16.shape) > 1:
            audio_int16 = audio_int16.flatten()

        transcript, words = self._leopard.process(audio_int16.tolist())

        word_list = [
            {
                "word": w.word,
                "start_sec": w.start_sec,
                "end_sec": w.end_sec,
                "confidence": w.confidence,
            }
            for w in words
        ]

        return transcript, word_list

    def cleanup(self):
        """Release Leopard resources."""
        if self._leopard is not None:
            self._leopard.delete()
            self._leopard = None

    def __enter__(self):
        """Context manager entry."""
        self._init_leopard()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
        return False
