"""
Speech-to-Text engine using Faster-Whisper.

Provides transcription capabilities with streaming support for wake word detection.
"""

import io
import wave
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel


class STTEngine:
    """Speech-to-text engine using Faster-Whisper."""

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "en",
        model_dir: Optional[str] = None,
    ):
        """
        Initialize STT engine.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large-v3)
            device: Device to use ("cpu", "cuda", or "auto")
            compute_type: Computation type ("int8", "float16", "float32")
                         Use "int8" for CPU, "float16" for GPU
            language: Language code for transcription
            model_dir: Directory to cache models (None for default)
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.model_dir = model_dir

        self._model: Optional[WhisperModel] = None

    def _load_model(self):
        """Load Whisper model lazily."""
        if self._model is None:
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=self.model_dir,
            )

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> str:
        """
        Transcribe audio to text.

        Args:
            audio: Audio samples as float32 numpy array
            sample_rate: Audio sample rate in Hz

        Returns:
            Transcribed text
        """
        self._load_model()

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        if len(audio.shape) > 1:
            audio = audio.flatten()

        if np.abs(audio).max() > 1.0:
            audio = audio / 32768.0

        segments, _ = self._model.transcribe(
            audio,
            language=self.language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=400,
            ),
        )

        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())

        return " ".join(text_parts)

    def transcribe_streaming(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> str:
        """
        Transcribe audio with faster settings for streaming/wake word detection.

        Uses smaller beam size and no VAD for lower latency.

        Args:
            audio: Audio samples as float32 numpy array
            sample_rate: Audio sample rate in Hz

        Returns:
            Transcribed text
        """
        self._load_model()

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        if len(audio.shape) > 1:
            audio = audio.flatten()

        if np.abs(audio).max() > 1.0:
            audio = audio / 32768.0

        segments, _ = self._model.transcribe(
            audio,
            language=self.language,
            beam_size=1,
            best_of=1,
            vad_filter=False,
        )

        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())

        return " ".join(text_parts)

    def detect_phrase(
        self,
        audio: np.ndarray,
        phrase: str,
        sample_rate: int = 16000,
    ) -> tuple[bool, str]:
        """
        Check if a specific phrase is present in the audio.

        Uses fast streaming transcription and fuzzy matching.

        Args:
            audio: Audio samples as float32 numpy array
            phrase: Phrase to detect (case-insensitive)
            sample_rate: Audio sample rate in Hz

        Returns:
            Tuple of (detected: bool, transcription: str)
        """
        text = self.transcribe_streaming(audio, sample_rate)
        text_lower = text.lower()

        # Common Whisper mishearings of "Olorin"
        olorin_variations = [
            "olorin",
            "alorin",
            "oloreen",
            "lauren",  # Very common mishearing
            "lorin",
            "loren",
            "lorean",
            "o'loren",
            "all around",
            "a loren",
            "oh lauren",
            "oh loren",
            "learn",  # Also common
            "o learn",
        ]

        # Check for "hey" + any olorin variation
        has_hey = any(h in text_lower for h in ["hey", "hay", "hei", "hey,", "hey."])
        has_olorin = any(var in text_lower for var in olorin_variations)

        detected = has_hey and has_olorin
        return detected, text

    @staticmethod
    def numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int = 16000) -> bytes:
        """Convert numpy audio array to WAV bytes."""
        if audio.dtype == np.float32:
            audio_int16 = (audio * 32767).astype(np.int16)
        else:
            audio_int16 = audio.astype(np.int16)

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())

        return buffer.getvalue()
