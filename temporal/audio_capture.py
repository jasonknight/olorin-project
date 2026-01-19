"""
Audio capture module for microphone input.

Uses sounddevice for cross-platform audio capture with callback-based streaming.
"""

import queue
import threading
from typing import Callable, Optional

import numpy as np
import sounddevice as sd


class AudioCapture:
    """Captures audio from microphone in a streaming fashion."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        device: Optional[str] = None,
        chunk_duration: float = 0.5,
    ):
        """
        Initialize audio capture.

        Args:
            sample_rate: Audio sample rate in Hz (16000 for Whisper)
            channels: Number of audio channels (1 for mono)
            device: Audio device name or index (None for default)
            chunk_duration: Duration of each audio chunk in seconds
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)

        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: Optional[sd.InputStream] = None
        self._running = False
        self._callbacks: list[Callable[[np.ndarray], None]] = []

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """Callback for sounddevice stream."""
        if status:
            pass
        audio_chunk = indata.copy().flatten()
        self._audio_queue.put(audio_chunk)

    def add_callback(self, callback: Callable[[np.ndarray], None]):
        """Add a callback to be called with each audio chunk."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[np.ndarray], None]):
        """Remove a previously added callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def start(self):
        """Start capturing audio."""
        if self._running:
            return

        self._running = True
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            device=self.device,
            blocksize=self.chunk_size,
            dtype=np.float32,
            callback=self._audio_callback,
        )
        self._stream.start()

        self._processor_thread = threading.Thread(
            target=self._process_audio, daemon=True
        )
        self._processor_thread.start()

    def _process_audio(self):
        """Process audio chunks from the queue."""
        while self._running:
            try:
                chunk = self._audio_queue.get(timeout=0.1)
                for callback in self._callbacks:
                    try:
                        callback(chunk)
                    except Exception:
                        pass
            except queue.Empty:
                continue

    def stop(self):
        """Stop capturing audio."""
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def get_chunk(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Get the next audio chunk from the queue."""
        try:
            return self._audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def clear_queue(self):
        """Clear any pending audio chunks."""
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

    @staticmethod
    def list_devices() -> list[dict]:
        """List available audio input devices."""
        devices = []
        for i, device in enumerate(sd.query_devices()):
            if device["max_input_channels"] > 0:
                devices.append(
                    {
                        "index": i,
                        "name": device["name"],
                        "channels": device["max_input_channels"],
                        "sample_rate": device["default_samplerate"],
                    }
                )
        return devices

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
