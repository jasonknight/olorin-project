"""
TTS Engine abstraction for Broca.

Provides a unified interface for text-to-speech synthesis with support for
multiple backends: Coqui TTS and Picovoice Orca.
"""

from abc import ABC, abstractmethod
from typing import Optional

import logging

logger = logging.getLogger(__name__)


class TTSEngine(ABC):
    """Abstract base class for TTS engines."""

    @abstractmethod
    def synthesize(self, text: str, output_path: str) -> None:
        """
        Synthesize speech from text and save to file.

        Args:
            text: Text to convert to speech
            output_path: Path to save the audio file
        """
        pass

    @abstractmethod
    def get_available_speakers(self) -> list[str]:
        """
        Get list of available speaker/voice names.

        Returns:
            List of speaker identifiers
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Release any resources held by the engine."""
        pass


class CoquiTTSEngine(TTSEngine):
    """TTS engine using Coqui TTS."""

    def __init__(self, model_name: str, speaker: Optional[str] = None):
        """
        Initialize Coqui TTS engine.

        Args:
            model_name: Coqui TTS model name (e.g., "tts_models/en/vctk/vits")
            speaker: Speaker ID for multi-speaker models
        """
        from TTS.api import TTS

        self.model_name = model_name
        self.speaker = speaker

        logger.info(f"Loading Coqui TTS model: {model_name}...")
        self._tts = TTS(model_name=model_name, progress_bar=False)
        logger.info("Coqui TTS model loaded successfully")

        # Log available speakers if multi-speaker model
        if hasattr(self._tts, "speakers") and self._tts.speakers:
            logger.info(f"Available speakers: {self._tts.speakers}")
            logger.info(f"Using speaker: {speaker}")

    def synthesize(self, text: str, output_path: str) -> None:
        """Synthesize speech using Coqui TTS."""
        if hasattr(self._tts, "speakers") and self._tts.speakers:
            self._tts.tts_to_file(
                text=text, file_path=output_path, speaker=self.speaker
            )
        else:
            self._tts.tts_to_file(text=text, file_path=output_path)

    def get_available_speakers(self) -> list[str]:
        """Get available Coqui TTS speakers."""
        if hasattr(self._tts, "speakers") and self._tts.speakers:
            return list(self._tts.speakers)
        return []

    def cleanup(self) -> None:
        """Cleanup Coqui TTS resources."""
        # Coqui TTS doesn't require explicit cleanup
        pass


class KokoroTTSEngine(TTSEngine):
    """TTS engine using Kokoro (hexgrad/Kokoro-82M)."""

    # Language code mapping
    LANG_CODES = {
        "american": "a",
        "british": "b",
        "spanish": "e",
        "french": "f",
        "hindi": "h",
        "italian": "i",
        "japanese": "j",
        "portuguese": "p",
        "chinese": "z",
    }

    # Sample voices by language/gender
    AVAILABLE_VOICES = [
        "af_bella",
        "af_heart",
        "af_nicole",
        "af_sarah",
        "af_sky",
        "am_adam",
        "am_michael",
        "bf_emma",
        "bf_isabella",
        "bm_george",
        "bm_lewis",
    ]

    def __init__(
        self,
        voice: Optional[str] = "af_bella",
        lang_code: Optional[str] = "a",
        speed: float = 1.0,
    ):
        """
        Initialize Kokoro TTS engine.

        Args:
            voice: Voice name (e.g., "af_bella", "af_heart", "am_adam").
                   Prefix indicates language (a=American, b=British) and
                   gender (f=female, m=male).
            lang_code: Language code for phoneme conversion:
                      'a' (American), 'b' (British), 'e' (Spanish),
                      'f' (French), 'h' (Hindi), 'i' (Italian),
                      'j' (Japanese), 'p' (Portuguese), 'z' (Chinese)
            speed: Speech speed multiplier (default 1.0)
        """
        from kokoro import KPipeline

        self.voice = voice or "af_bella"
        self.lang_code = lang_code or "a"
        self.speed = speed
        self._sample_rate = 24000  # Kokoro outputs at 24kHz

        logger.info(
            f"Initializing Kokoro TTS with voice={self.voice}, lang={self.lang_code}"
        )
        self._pipeline = KPipeline(lang_code=self.lang_code)
        logger.info("Kokoro TTS initialized successfully")

    def synthesize(self, text: str, output_path: str) -> None:
        """Synthesize speech using Kokoro."""
        import soundfile as sf
        import numpy as np

        # Generate audio using the pipeline
        audio_segments = []
        for _gs, _ps, audio in self._pipeline(
            text, voice=self.voice, speed=self.speed, split_pattern=r"\n+"
        ):
            if audio is not None:
                audio_segments.append(audio)

        if not audio_segments:
            raise ValueError("Kokoro produced no audio output")

        # Concatenate all segments
        full_audio = np.concatenate(audio_segments)

        # Write to file
        sf.write(output_path, full_audio, self._sample_rate)
        logger.debug(f"Synthesized {len(full_audio)} samples to {output_path}")

    def get_available_speakers(self) -> list[str]:
        """Get available Kokoro voices."""
        return self.AVAILABLE_VOICES.copy()

    def cleanup(self) -> None:
        """Cleanup Kokoro resources."""
        # Kokoro doesn't require explicit cleanup
        self._pipeline = None


class OrcaTTSEngine(TTSEngine):
    """TTS engine using Picovoice Orca."""

    # Orca has a 2000 character limit, use 1800 to be safe
    MAX_CHUNK_CHARS = 1800

    # Available built-in voice models (language_gender format)
    AVAILABLE_VOICES = [
        "en_female",
        "en_male",
        "de_female",
        "de_male",
        "es_female",
        "es_male",
        "fr_female",
        "fr_male",
        "it_female",
        "it_male",
        "ja_female",
        "ko_female",
        "pt_female",
        "pt_male",
    ]

    def __init__(
        self,
        access_key: str,
        voice: Optional[str] = None,
        model_path: Optional[str] = None,
    ):
        """
        Initialize Picovoice Orca TTS engine.

        Args:
            access_key: Picovoice access key (from console.picovoice.ai)
            voice: Voice model to use (e.g., "en_female", "en_male", "de_female").
                   See AVAILABLE_VOICES for full list.
            model_path: Optional path to custom .pv model file. If provided,
                       this overrides the voice parameter.
        """
        import pvorca

        self.access_key = access_key
        self.voice = voice
        self.model_path = model_path

        logger.info("Initializing Picovoice Orca TTS...")

        # Determine model path
        effective_model_path = self._resolve_model_path(voice, model_path)

        # Initialize Orca
        if effective_model_path:
            logger.info(f"Using Orca model: {effective_model_path}")
            self._orca = pvorca.create(
                access_key=access_key, model_path=effective_model_path
            )
        else:
            logger.info("Using default Orca model")
            self._orca = pvorca.create(access_key=access_key)

        logger.info("Picovoice Orca TTS initialized successfully")
        logger.info(f"Orca sample rate: {self._orca.sample_rate}")
        logger.info(f"Orca version: {self._orca.version}")
        if voice:
            logger.info(f"Using voice: {voice}")

    def _resolve_model_path(
        self, voice: Optional[str], model_path: Optional[str]
    ) -> Optional[str]:
        """
        Resolve the model path from voice name or explicit path.

        Args:
            voice: Voice name like "en_female"
            model_path: Explicit path to .pv file

        Returns:
            Path to model file, or None to use default
        """
        # Explicit model path takes precedence
        if model_path:
            return model_path

        # No voice specified, use default
        if not voice:
            return None

        # Try to find the built-in model for the specified voice
        import pvorca
        import os

        # Get the pvorca package directory
        pvorca_dir = os.path.dirname(pvorca.__file__)
        model_filename = f"orca_params_{voice}.pv"

        # Check common locations for model files
        search_paths = [
            os.path.join(pvorca_dir, "lib", "common", model_filename),
            os.path.join(pvorca_dir, "resources", "models", model_filename),
            os.path.join(pvorca_dir, model_filename),
        ]

        for path in search_paths:
            if os.path.exists(path):
                return path

        # Voice not found in package, log warning and use default
        logger.warning(
            f"Voice model '{voice}' not found in pvorca package. "
            f"Available voices: {', '.join(self.AVAILABLE_VOICES)}. "
            "Using default model."
        )
        return None

    def _split_text_into_chunks(self, text: str) -> list[str]:
        """
        Split text into chunks that fit within Orca's character limit.

        Attempts to split at sentence boundaries (., !, ?) for natural speech.
        Falls back to splitting at word boundaries if sentences are too long.

        Args:
            text: Text to split

        Returns:
            List of text chunks, each under MAX_CHUNK_CHARS
        """
        import re

        if len(text) <= self.MAX_CHUNK_CHARS:
            return [text]

        chunks = []
        remaining = text.strip()

        while remaining:
            if len(remaining) <= self.MAX_CHUNK_CHARS:
                chunks.append(remaining)
                break

            # Find the best split point within the limit
            chunk_text = remaining[: self.MAX_CHUNK_CHARS]

            # Try to split at sentence boundary (., !, ?)
            # Look for the last sentence ending within the chunk
            sentence_match = None
            for match in re.finditer(r"[.!?]+\s*", chunk_text):
                sentence_match = match

            if sentence_match and sentence_match.end() > self.MAX_CHUNK_CHARS // 2:
                # Found a good sentence boundary in the latter half
                split_pos = sentence_match.end()
            else:
                # Fall back to word boundary
                last_space = chunk_text.rfind(" ")
                if last_space > self.MAX_CHUNK_CHARS // 2:
                    split_pos = last_space + 1
                else:
                    # Worst case: hard split at limit
                    split_pos = self.MAX_CHUNK_CHARS

            chunks.append(remaining[:split_pos].strip())
            remaining = remaining[split_pos:].strip()

        logger.info(f"Split text ({len(text)} chars) into {len(chunks)} chunks")
        return chunks

    def synthesize(self, text: str, output_path: str) -> None:
        """Synthesize speech using Picovoice Orca."""
        import struct
        import wave

        # Split text into chunks if needed (Orca has 2000 char limit)
        chunks = self._split_text_into_chunks(text)

        # Synthesize each chunk and collect PCM data
        all_pcm = []
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            logger.debug(
                f"Synthesizing chunk {i + 1}/{len(chunks)} ({len(chunk)} chars)"
            )
            pcm, alignments = self._orca.synthesize(chunk)
            all_pcm.extend(pcm)

        # Write combined PCM to WAV file
        with wave.open(output_path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self._orca.sample_rate)
            wav_file.writeframes(struct.pack(f"{len(all_pcm)}h", *all_pcm))

        logger.debug(f"Synthesized {len(all_pcm)} samples to {output_path}")

    def get_available_speakers(self) -> list[str]:
        """Get available Orca voices."""
        return self.AVAILABLE_VOICES.copy()

    def cleanup(self) -> None:
        """Release Orca resources."""
        if self._orca is not None:
            self._orca.delete()
            self._orca = None


def create_tts_engine(
    engine: str,
    # Coqui options
    coqui_model_name: Optional[str] = None,
    coqui_speaker: Optional[str] = None,
    # Orca options
    orca_access_key: Optional[str] = None,
    orca_voice: Optional[str] = None,
    orca_model_path: Optional[str] = None,
    # Kokoro options
    kokoro_voice: Optional[str] = None,
    kokoro_lang_code: Optional[str] = None,
    kokoro_speed: float = 1.0,
) -> TTSEngine:
    """
    Factory function to create TTS engine based on configuration.

    Args:
        engine: Engine type - "coqui", "orca", or "kokoro"
        coqui_model_name: Coqui TTS model name
        coqui_speaker: Coqui speaker ID
        orca_access_key: Picovoice access key for Orca
        orca_voice: Orca voice model (e.g., "en_female", "en_male")
        orca_model_path: Optional path to custom Orca .pv model file
        kokoro_voice: Kokoro voice (e.g., "af_bella", "af_heart", "am_adam")
        kokoro_lang_code: Kokoro language code ('a'=American, 'b'=British, etc.)
        kokoro_speed: Kokoro speech speed multiplier (default 1.0)

    Returns:
        Configured TTSEngine instance

    Raises:
        ValueError: If engine type is unknown or required options are missing
    """
    engine = engine.lower()

    if engine == "coqui":
        if not coqui_model_name:
            raise ValueError("coqui_model_name is required for Coqui TTS engine")
        return CoquiTTSEngine(model_name=coqui_model_name, speaker=coqui_speaker)

    elif engine == "orca":
        if not orca_access_key:
            raise ValueError("orca_access_key is required for Orca TTS engine")
        return OrcaTTSEngine(
            access_key=orca_access_key,
            voice=orca_voice,
            model_path=orca_model_path,
        )

    elif engine == "kokoro":
        return KokoroTTSEngine(
            voice=kokoro_voice,
            lang_code=kokoro_lang_code,
            speed=kokoro_speed,
        )

    else:
        raise ValueError(
            f"Unknown TTS engine: {engine}. Supported: coqui, orca, kokoro"
        )
