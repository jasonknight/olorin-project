"""
Voice Activity Detection with noise floor calibration.

Provides speech/silence detection for determining when the user has finished speaking.
Key insight: Calibrate to background noise first, then detect speech as energy
significantly above noise floor, and silence as return to noise floor level.
"""

import time
from typing import Optional

import numpy as np
import torch


class VoiceActivityDetector:
    """Detects speech and silence with noise floor calibration.

    Designed for noisy environments:
    1. First few chunks calibrate the noise floor
    2. Speech = energy significantly above noise floor
    3. Silence = energy returns close to noise floor
    4. Hard timeout fallback if silence detection fails
    """

    WINDOW_SIZE_16K = 512  # 32ms at 16kHz
    WINDOW_SIZE_8K = 256  # 32ms at 8kHz

    def __init__(
        self,
        threshold: float = 0.5,
        sample_rate: int = 16000,
        silence_timeout: float = 3.0,
        max_duration: float = 30.0,
        calibration_chunks: int = 4,  # Chunks to calibrate noise floor
        speech_threshold_mult: float = 2.0,  # Speech = noise * this
        silence_threshold_mult: float = 1.3,  # Silence = noise * this
    ):
        """
        Initialize VAD.

        Args:
            threshold: VAD probability threshold (fallback)
            sample_rate: Audio sample rate in Hz
            silence_timeout: Seconds of silence before stopping
            max_duration: Maximum recording duration (hard stop)
            calibration_chunks: Number of chunks to measure noise floor
            speech_threshold_mult: Energy must exceed noise_floor * this to be speech
            silence_threshold_mult: Energy below noise_floor * this is silence
        """
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.silence_timeout = silence_timeout
        self.max_duration = max_duration
        self.calibration_chunks = calibration_chunks
        self.speech_threshold_mult = speech_threshold_mult
        self.silence_threshold_mult = silence_threshold_mult

        self._window_size = (
            self.WINDOW_SIZE_16K if sample_rate == 16000 else self.WINDOW_SIZE_8K
        )
        self._model = None
        self._reset_state()

    def _reset_state(self):
        """Reset all internal state."""
        self._start_time: Optional[float] = None
        self._calibration_end_time: Optional[float] = None  # When calibration finished
        self._last_speech_time: Optional[float] = None
        self._has_detected_real_speech = False

        # Energy tracking
        self._calibration_energies: list = []
        self._noise_floor: Optional[float] = None
        self._current_energy: float = 0.0
        self._peak_speech_energy: float = 0.0

    def _load_model(self):
        """Load Silero VAD model lazily."""
        if self._model is None:
            self._model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
            )
            self._model.eval()

    def reset(self, assume_speaking: bool = False):
        """Reset VAD state for a new utterance."""
        current_time = time.time()
        self._start_time = current_time
        self._calibration_end_time = None
        self._last_speech_time = current_time if assume_speaking else None
        self._has_detected_real_speech = False

        # Reset energy tracking
        self._calibration_energies = []
        self._noise_floor = None
        self._current_energy = 0.0
        self._peak_speech_energy = 0.0

        if self._model is not None:
            self._model.reset_states()

    def _calculate_energy(self, audio_chunk: np.ndarray) -> float:
        """Calculate RMS energy of audio chunk."""
        return float(np.sqrt(np.mean(audio_chunk**2)))

    def _get_vad_probability(self, audio_chunk: np.ndarray) -> float:
        """Get VAD probability by processing audio in windows."""
        self._load_model()

        if len(audio_chunk) < self._window_size:
            padded = np.zeros(self._window_size, dtype=np.float32)
            padded[: len(audio_chunk)] = audio_chunk
            audio_chunk = padded

        probabilities = []
        for start in range(
            0, len(audio_chunk) - self._window_size + 1, self._window_size
        ):
            window = audio_chunk[start : start + self._window_size]
            tensor = torch.from_numpy(window)

            with torch.no_grad():
                prob = self._model(tensor, self.sample_rate).item()
                probabilities.append(prob)

        return max(probabilities) if probabilities else 0.0

    def process_chunk(self, audio_chunk: np.ndarray) -> dict:
        """Process an audio chunk and return VAD results."""
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        if len(audio_chunk.shape) > 1:
            audio_chunk = audio_chunk.flatten()

        if np.abs(audio_chunk).max() > 1.0:
            audio_chunk = audio_chunk / 32768.0

        current_time = time.time()
        self._current_energy = self._calculate_energy(audio_chunk)

        # Get VAD probability for logging
        raw_prob = self._get_vad_probability(audio_chunk)

        # Phase 1: Calibration - measure noise floor
        if len(self._calibration_energies) < self.calibration_chunks:
            self._calibration_energies.append(self._current_energy)

            if len(self._calibration_energies) == self.calibration_chunks:
                # Use minimum of calibration samples as noise floor (true baseline)
                self._noise_floor = min(self._calibration_energies)
                self._calibration_end_time = current_time
                # Also reset last_speech_time to now so silence timer starts fresh
                self._last_speech_time = current_time

            # During calibration, consider it "speech" to prevent early timeout
            return {
                "is_speech": True,
                "probability": raw_prob,
                "silence_duration": 0.0,
                "should_stop": False,
                "stop_reason": None,
                "energy": self._current_energy,
                "noise_floor": 0,
                "speech_threshold": 0,
                "silence_threshold": 0,
                "total_duration": current_time - self._start_time
                if self._start_time
                else 0,
                "phase": "calibrating",
            }

        # Phase 2: Detection - use noise floor to detect speech/silence
        speech_threshold = self._noise_floor * self.speech_threshold_mult
        silence_threshold = self._noise_floor * self.silence_threshold_mult

        # Detect speech: energy significantly above noise floor
        is_above_speech_threshold = self._current_energy > speech_threshold
        is_above_silence_threshold = self._current_energy > silence_threshold

        # Once we detect real speech, track peak energy
        if is_above_speech_threshold:
            self._has_detected_real_speech = True
            self._last_speech_time = current_time
            if self._current_energy > self._peak_speech_energy:
                self._peak_speech_energy = self._current_energy

        # Determine is_speech:
        # - If we haven't detected real speech yet, be generous (above silence threshold)
        # - If we have detected real speech, require above silence threshold to continue
        if self._has_detected_real_speech:
            is_speech = is_above_silence_threshold
        else:
            # Before first real speech, anything above silence threshold keeps us listening
            is_speech = is_above_silence_threshold
            if is_speech:
                self._last_speech_time = current_time

        # Calculate silence duration
        silence_duration = 0.0
        if self._last_speech_time is not None:
            silence_duration = current_time - self._last_speech_time

        # Calculate total duration
        total_duration = current_time - self._start_time if self._start_time else 0.0

        # Determine if we should stop
        silence_stop = (
            self._has_detected_real_speech and silence_duration >= self.silence_timeout
        )
        duration_stop = total_duration >= self.max_duration

        # Also stop if we've been waiting too long without detecting real speech
        # Use time since calibration ended, not total duration
        time_since_calibration = (
            current_time - self._calibration_end_time
            if self._calibration_end_time
            else 0
        )
        no_speech_timeout = (
            not self._has_detected_real_speech
            and time_since_calibration >= self.silence_timeout
        )

        should_stop = silence_stop or duration_stop or no_speech_timeout

        stop_reason = None
        if silence_stop:
            stop_reason = "silence_timeout"
        elif no_speech_timeout:
            stop_reason = "no_speech_detected"
        elif duration_stop:
            stop_reason = "max_duration"

        return {
            "is_speech": is_speech,
            "probability": raw_prob,
            "silence_duration": silence_duration,
            "should_stop": should_stop,
            "stop_reason": stop_reason,
            "energy": self._current_energy,
            "noise_floor": self._noise_floor,
            "speech_threshold": speech_threshold,
            "silence_threshold": silence_threshold,
            "peak_speech_energy": self._peak_speech_energy,
            "total_duration": total_duration,
            "phase": "detecting",
            "has_real_speech": self._has_detected_real_speech,
        }

    def is_silence_timeout(self) -> bool:
        """Check if silence timeout has been reached."""
        if self._last_speech_time is None:
            return False
        return time.time() - self._last_speech_time >= self.silence_timeout

    @property
    def has_detected_speech(self) -> bool:
        """Check if any speech has been detected since last reset."""
        return self._has_detected_real_speech
