#!/usr/bin/env python3
"""
Temporal - Voice-activated speech-to-text consumer.

Listens for "Hey Olorin" wake word, then transcribes speech until silence
or stop phrase, sending the result to the ai_in Kafka topic.
"""

import json
import os
import signal
import sys
import time
from datetime import datetime

import numpy as np
from kafka import KafkaProducer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from libs.config import Config
from libs.olorin_logging import OlorinLogger
from libs.state import get_state

from audio_capture import AudioCapture
from leopard_stt import LeopardSTT
from stt_engine import STTEngine
from vad import VoiceActivityDetector
from wake_word import WakeWordDetector


class TemporalConfig:
    """Configuration for the Temporal component."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._load()

    def _load(self):
        """Load configuration values."""
        self.bootstrap_servers = self.cfg.get(
            "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"
        )
        self.output_topic = self.cfg.get("TEMPORAL_OUTPUT_TOPIC", "ai_in")
        self.feedback_topic = self.cfg.get("TEMPORAL_FEEDBACK_TOPIC", "ai_out")

        self.sample_rate = self.cfg.get_int("TEMPORAL_SAMPLE_RATE", 16000)
        self.channels = self.cfg.get_int("TEMPORAL_CHANNELS", 1)
        self.audio_device = self.cfg.get("TEMPORAL_AUDIO_DEVICE", None)
        self.chunk_duration = self.cfg.get_float("TEMPORAL_CHUNK_DURATION", 0.5)

        self.wake_phrase = self.cfg.get("TEMPORAL_WAKE_PHRASE", "hey olorin")
        self.wake_buffer_seconds = self.cfg.get_float(
            "TEMPORAL_WAKE_BUFFER_SECONDS", 3.0
        )
        self.feedback_message = self.cfg.get("TEMPORAL_FEEDBACK_MESSAGE", None)
        self.completion_message = self.cfg.get("TEMPORAL_COMPLETION_MESSAGE", None)

        # Porcupine wake word detection
        self.porcupine_access_key = self.cfg.get("TEMPORAL_PORCUPINE_ACCESS_KEY", None)
        self.porcupine_keyword_path = self.cfg.get_path(
            "TEMPORAL_PORCUPINE_KEYWORD_PATH", None
        )
        self.porcupine_sensitivity = self.cfg.get_float(
            "TEMPORAL_PORCUPINE_SENSITIVITY", 0.5
        )
        self.use_porcupine = (
            self.porcupine_access_key is not None
            and self.porcupine_keyword_path is not None
        )

        # STT engine selection: "leopard" or "whisper"
        self.stt_engine = self.cfg.get("TEMPORAL_STT_ENGINE", "whisper")
        self.leopard_punctuation = self.cfg.get_bool(
            "TEMPORAL_LEOPARD_PUNCTUATION", True
        )

        # Whisper settings (used when stt_engine="whisper" or as fallback)
        self.stt_model = self.cfg.get("TEMPORAL_STT_MODEL", "small")
        self.stt_device = self.cfg.get("TEMPORAL_STT_DEVICE", "cpu")
        self.stt_compute_type = self.cfg.get("TEMPORAL_STT_COMPUTE_TYPE", "int8")
        self.stt_language = self.cfg.get("TEMPORAL_STT_LANGUAGE", "en")
        self.stt_model_dir = self.cfg.get_path(
            "TEMPORAL_STT_MODEL_DIR", "./temporal/data/models"
        )

        self.vad_threshold = self.cfg.get_float("TEMPORAL_VAD_THRESHOLD", 0.5)
        self.silence_timeout = self.cfg.get_float("TEMPORAL_SILENCE_TIMEOUT", 3.0)
        self.stop_phrases = self.cfg.get_list(
            "TEMPORAL_STOP_PHRASES",
            ["that's all", "thats all", "that is all", "end message"],
        )

        # VAD settings
        self.vad_max_duration = self.cfg.get_float("TEMPORAL_VAD_MAX_DURATION", 30.0)
        self.vad_calibration_chunks = self.cfg.get_int(
            "TEMPORAL_VAD_CALIBRATION_CHUNKS", 4
        )
        self.vad_speech_mult = self.cfg.get_float("TEMPORAL_VAD_SPEECH_MULT", 2.0)
        self.vad_silence_mult = self.cfg.get_float("TEMPORAL_VAD_SILENCE_MULT", 1.3)
        self.vad_debug = self.cfg.get_bool("TEMPORAL_VAD_DEBUG", False)

        self.pause_during_tts = self.cfg.get_bool("TEMPORAL_PAUSE_DURING_TTS", True)
        self.log_level = self.cfg.get("LOG_LEVEL", "INFO")

    def reload(self) -> bool:
        """Reload configuration if changed."""
        if self.cfg.reload():
            self._load()
            return True
        return False


class TemporalConsumer:
    """Voice-activated STT consumer with wake word detection."""

    def __init__(self, config: TemporalConfig):
        self.config = config
        self.state = get_state()

        log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
        os.makedirs(log_dir, exist_ok=True)
        self.logger = OlorinLogger(
            log_file=os.path.join(log_dir, "temporal-consumer.log"),
            log_level=config.log_level,
            name=__name__,
        )

        self._init_kafka()
        self._init_audio()
        self._init_stt()
        self._init_wake_word()

        self._running = False
        self._listening_mode = "wake_word"
        self._audio_buffer: list[np.ndarray] = []
        self._transcription_buffer: list[str] = []

    def _init_kafka(self):
        """Initialize Kafka producer."""
        self.producer = KafkaProducer(
            bootstrap_servers=self.config.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            retries=5,
            max_in_flight_requests_per_connection=1,
        )
        self.logger.info(f"Kafka producer initialized: {self.config.bootstrap_servers}")

    def _init_audio(self):
        """Initialize audio capture."""
        self.audio_capture = AudioCapture(
            sample_rate=self.config.sample_rate,
            channels=self.config.channels,
            device=self.config.audio_device,
            chunk_duration=self.config.chunk_duration,
        )
        self.logger.info(
            f"Audio capture initialized: {self.config.sample_rate}Hz, "
            f"chunk={self.config.chunk_duration}s"
        )

    def _init_stt(self):
        """Initialize STT engine and VAD."""
        self.leopard_stt = None
        self.whisper_stt = None

        if self.config.stt_engine == "leopard" and self.config.porcupine_access_key:
            try:
                self.leopard_stt = LeopardSTT(
                    access_key=self.config.porcupine_access_key,
                    enable_automatic_punctuation=self.config.leopard_punctuation,
                )
                self.logger.info(
                    f"Leopard STT engine initialized: punctuation={self.config.leopard_punctuation}"
                )
            except Exception as e:
                self.logger.error(f"Failed to initialize Leopard STT: {e}")
                self.logger.warning("Falling back to Whisper STT")

        # Initialize Whisper as primary or fallback
        if self.leopard_stt is None or self.config.stt_engine == "whisper":
            self.whisper_stt = STTEngine(
                model_size=self.config.stt_model,
                device=self.config.stt_device,
                compute_type=self.config.stt_compute_type,
                language=self.config.stt_language,
                model_dir=self.config.stt_model_dir,
            )
            self.logger.info(
                f"Whisper STT engine initialized: model={self.config.stt_model}"
            )

        # Keep reference to primary STT for backward compat
        self.stt = self.whisper_stt

        self.vad = VoiceActivityDetector(
            threshold=self.config.vad_threshold,
            sample_rate=self.config.sample_rate,
            silence_timeout=self.config.silence_timeout,
            max_duration=self.config.vad_max_duration,
            calibration_chunks=self.config.vad_calibration_chunks,
            speech_threshold_mult=self.config.vad_speech_mult,
            silence_threshold_mult=self.config.vad_silence_mult,
        )
        self.logger.info(
            f"VAD initialized: silence_timeout={self.config.silence_timeout}s, "
            f"max_duration={self.config.vad_max_duration}s, "
            f"speech_mult={self.config.vad_speech_mult}, "
            f"silence_mult={self.config.vad_silence_mult}"
        )

    def _init_wake_word(self):
        """Initialize wake word detector (Porcupine or fallback to Whisper)."""
        self.wake_word_detector = None

        if self.config.use_porcupine:
            try:
                self.wake_word_detector = WakeWordDetector(
                    access_key=self.config.porcupine_access_key,
                    keyword_path=self.config.porcupine_keyword_path,
                    sensitivity=self.config.porcupine_sensitivity,
                )
                self.logger.info(
                    f"Porcupine wake word detector initialized: "
                    f"model={self.config.porcupine_keyword_path}, "
                    f"sensitivity={self.config.porcupine_sensitivity}"
                )
            except Exception as e:
                self.logger.error(f"Failed to initialize Porcupine: {e}")
                self.logger.warning("Falling back to Whisper-based wake word detection")
                self.wake_word_detector = None
        else:
            self.logger.info(
                "Porcupine not configured, using Whisper-based wake word detection. "
                "Set temporal.porcupine.access_key and temporal.porcupine.keyword_path "
                "in settings.json to use Porcupine."
            )

    def _should_pause(self) -> bool:
        """Check if we should pause listening (e.g., during TTS playback)."""
        if not self.config.pause_during_tts:
            return False
        return self.state.get_bool("broca.is_playing", default=False)

    def _send_feedback(self, message: str):
        """Send feedback message to Broca via Kafka."""
        msg = {
            "text": message,
            "id": datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
            "source": "temporal",
        }
        try:
            future = self.producer.send(self.config.feedback_topic, value=msg)
            future.get(timeout=10)
            self.logger.info(f"Sent feedback to Broca: {message}")
        except Exception as e:
            self.logger.error(f"Failed to send feedback: {e}")

    def _send_transcription(self, text: str):
        """Send transcribed text to ai_in topic."""
        msg = {
            "text": text,
            "id": datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
            "source": "temporal",
            "type": "voice_input",
        }
        try:
            future = self.producer.send(self.config.output_topic, value=msg)
            future.get(timeout=10)
            self.logger.info(
                f"Sent transcription to {self.config.output_topic}: {text[:100]}..."
            )
        except Exception as e:
            self.logger.error(f"Failed to send transcription: {e}")

    def _check_stop_phrase(self, text: str) -> bool:
        """Check if any stop phrase is in the text."""
        text_lower = text.lower()
        return any(phrase in text_lower for phrase in self.config.stop_phrases)

    def _process_wake_word_mode(self, audio_chunk: np.ndarray):
        """Process audio in wake word detection mode."""
        if self.wake_word_detector is not None:
            # Use Porcupine for wake word detection
            detected = self.wake_word_detector.process_audio(audio_chunk)

            if detected:
                self.logger.info("Wake word detected (Porcupine)!")
                self._on_wake_word_detected()
        else:
            # Fallback to Whisper-based detection
            self._process_wake_word_whisper(audio_chunk)

    def _process_wake_word_whisper(self, audio_chunk: np.ndarray):
        """Process audio using Whisper-based wake word detection (fallback)."""
        buffer_samples = int(self.config.wake_buffer_seconds * self.config.sample_rate)
        self._audio_buffer.append(audio_chunk)

        total_samples = sum(len(chunk) for chunk in self._audio_buffer)
        while total_samples > buffer_samples and len(self._audio_buffer) > 1:
            removed = self._audio_buffer.pop(0)
            total_samples -= len(removed)

        if total_samples < self.config.sample_rate:
            return

        combined_audio = np.concatenate(self._audio_buffer)
        detected, transcription = self.stt.detect_phrase(
            combined_audio, self.config.wake_phrase, self.config.sample_rate
        )

        if detected:
            self.logger.info(
                f"Wake word detected (Whisper)! Transcription: {transcription}"
            )
            self._on_wake_word_detected()

    def _on_wake_word_detected(self):
        """Handle wake word detection - common logic for both detectors."""
        # Send feedback if configured (can be disabled by setting to null/empty)
        if self.config.feedback_message:
            self._send_feedback(self.config.feedback_message)

        self._listening_mode = "transcribing"
        self._audio_buffer.clear()
        self._transcription_buffer.clear()
        # Reset VAD and assume speech is happening - this ensures silence timeout
        # will trigger even if the user pauses right after the wake word
        self.vad.reset(assume_speaking=True)

        if self.wake_word_detector is not None:
            self.wake_word_detector.reset()

        self.state.set_bool("temporal.is_listening", True)
        self.state.set_string("temporal.status", "transcribing")
        self.logger.info(
            f"Now listening for speech (will stop after {self.config.silence_timeout}s of silence)"
        )

    def _process_transcribing_mode(self, audio_chunk: np.ndarray):
        """Process audio in transcription mode."""
        self._audio_buffer.append(audio_chunk)

        vad_result = self.vad.process_chunk(audio_chunk)

        # Debug logging to diagnose VAD behavior
        if self.config.vad_debug:
            phase = vad_result.get("phase", "unknown")
            has_speech = vad_result.get("has_real_speech", False)
            self.logger.info(
                f"VAD[{phase}]: energy={vad_result.get('energy', 0):.5f} "
                f"floor={vad_result.get('noise_floor', 0):.5f} "
                f"speech_th={vad_result.get('speech_threshold', 0):.5f} "
                f"is_speech={vad_result['is_speech']} "
                f"real_speech={has_speech} "
                f"silence={vad_result['silence_duration']:.1f}s"
            )

        should_stop = False
        stop_reason = None

        # Check silence timeout
        if (
            vad_result["should_stop"]
            and vad_result["silence_duration"] >= self.config.silence_timeout
        ):
            should_stop = True
            stop_reason = f"silence timeout ({vad_result['silence_duration']:.1f}s)"

        # Check for stop phrases using Whisper streaming (only if Whisper is available)
        # Leopard doesn't support streaming, so we skip this check when using Leopard
        if (
            not should_stop
            and self.whisper_stt is not None
            and len(self._audio_buffer) >= 4
        ):
            recent_audio = np.concatenate(self._audio_buffer[-4:])
            text = self.whisper_stt.transcribe_streaming(
                recent_audio, self.config.sample_rate
            )
            if text.strip():
                self.logger.debug(f"Partial transcription: {text}")
            if self._check_stop_phrase(text):
                should_stop = True
                stop_reason = f"stop phrase detected in: {text}"

        if should_stop:
            self.logger.info(f"Stopping transcription: {stop_reason}")
            self._finalize_transcription()

    def _finalize_transcription(self):
        """Finalize and send the complete transcription."""
        if not self._audio_buffer:
            self.logger.warning("No audio to transcribe")
            self._reset_to_wake_word_mode()
            return

        combined_audio = np.concatenate(self._audio_buffer)
        duration_secs = len(combined_audio) / self.config.sample_rate
        self.logger.info(f"Finalizing transcription: {duration_secs:.1f}s of audio")

        # Use Leopard if available, otherwise Whisper
        if self.leopard_stt is not None:
            full_text = self.leopard_stt.transcribe(
                combined_audio, self.config.sample_rate
            )
            self.logger.info(f"Leopard raw transcription: {full_text}")
        else:
            full_text = self.whisper_stt.transcribe(
                combined_audio, self.config.sample_rate
            )
            self.logger.info(f"Whisper raw transcription: {full_text}")

        # Clean up stop phrases
        for phrase in self.config.stop_phrases:
            full_text = full_text.lower().replace(phrase, "").strip()

        # Remove wake phrase if present at start
        wake_phrase_lower = self.config.wake_phrase.lower()
        if full_text.lower().startswith(wake_phrase_lower):
            full_text = full_text[len(wake_phrase_lower) :].strip()
        if full_text.lower().startswith(","):
            full_text = full_text[1:].strip()

        if full_text.strip():
            self.logger.info(f"Final transcription: {full_text}")
            self._send_transcription(full_text)
            # Send completion feedback if configured
            if self.config.completion_message:
                self._send_feedback(self.config.completion_message)
        else:
            self.logger.info("Empty transcription after cleanup, not sending")

        self._reset_to_wake_word_mode()

    def _reset_to_wake_word_mode(self):
        """Reset to wake word detection mode."""
        self._listening_mode = "wake_word"
        self._audio_buffer.clear()
        self._transcription_buffer.clear()
        self.vad.reset()

        # Reset wake word detector to clear any buffered frames
        if self.wake_word_detector is not None:
            self.wake_word_detector.reset()

        self.state.set_bool("temporal.is_listening", False)
        self.state.set_string("temporal.status", "listening_for_wake_word")
        self.logger.debug("Reset to wake word mode")

    def _process_audio(self, audio_chunk: np.ndarray):
        """Process an audio chunk based on current mode."""
        if self._should_pause():
            self.logger.debug("Audio processing paused (TTS playing)")
            return

        if self._listening_mode == "wake_word":
            self._process_wake_word_mode(audio_chunk)
        elif self._listening_mode == "transcribing":
            self._process_transcribing_mode(audio_chunk)

    def start(self):
        """Start the temporal consumer."""
        self._running = True

        self.state.set_string("temporal.status", "starting")
        self.logger.info("Starting Temporal consumer...")

        # Pre-load STT engines
        if self.leopard_stt is not None:
            self.logger.info("Pre-loading Leopard STT...")
            self.leopard_stt._init_leopard()
            self.logger.info("Leopard STT loaded")

        if self.whisper_stt is not None:
            self.logger.info("Pre-loading Whisper STT model...")
            self.whisper_stt._load_model()
            self.logger.info("Whisper STT model loaded")

        self.audio_capture.add_callback(self._process_audio)

        self.state.set_string("temporal.status", "listening_for_wake_word")
        self.state.set_bool("temporal.is_listening", False)

        detector_type = "Porcupine" if self.wake_word_detector else "Whisper"
        stt_type = "Leopard" if self.leopard_stt else "Whisper"
        self.logger.info(
            f"Temporal consumer started. Wake word: {detector_type}, STT: {stt_type}"
        )
        self.logger.info(
            f"Output topic: {self.config.output_topic}, "
            f"Silence timeout: {self.config.silence_timeout}s"
        )

        with self.audio_capture:
            try:
                while self._running:
                    if self.config.reload():
                        self.logger.info("Configuration reloaded")

                        self.vad.silence_timeout = self.config.silence_timeout
                        self.vad.threshold = self.config.vad_threshold
                        self.vad.max_duration = self.config.vad_max_duration
                        self.vad.calibration_chunks = self.config.vad_calibration_chunks
                        self.vad.speech_threshold_mult = self.config.vad_speech_mult
                        self.vad.silence_threshold_mult = self.config.vad_silence_mult

                    time.sleep(0.1)

            except KeyboardInterrupt:
                self.logger.info("Received keyboard interrupt")
            finally:
                self.stop()

    def stop(self):
        """Stop the temporal consumer."""
        self._running = False
        self.state.set_string("temporal.status", "stopped")
        self.state.delete("temporal.is_listening")

        if self.wake_word_detector is not None:
            self.wake_word_detector.cleanup()

        if self.leopard_stt is not None:
            self.leopard_stt.cleanup()

        if self.producer:
            self.producer.close()

        self.logger.info("Temporal consumer stopped")


def main():
    """Main entry point."""
    config = TemporalConfig(Config(watch=True))
    consumer = TemporalConsumer(config)

    def signal_handler(signum, frame):
        consumer.logger.info(f"Received signal {signum}")
        consumer.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    consumer.start()


if __name__ == "__main__":
    main()
