# consumer.py
import json
import os
import signal
import subprocess
import sys
from datetime import datetime

from kafka import KafkaConsumer

# Add parent directory to path for libs import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from libs.config import Config
from libs.olorin_logging import OlorinLogger
from libs.state import get_state

from tts_engine import TTSEngine, create_tts_engine

# Initialize config with hot-reload support
config = Config(watch=True)

# Set up logging
default_log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
log_dir = config.get("LOG_DIR", default_log_dir)
log_file = os.path.join(log_dir, "broca-consumer.log")
env_log_level = config.get("LOG_LEVEL", "INFO")

# Initialize logger
logger = OlorinLogger(log_file=log_file, log_level=env_log_level, name=__name__)


class BrocaConfig:
    """Configuration wrapper for Broca consumer"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._load()

    def _load(self):
        """Load configuration values from Config"""
        self.bootstrap_servers = self.cfg.get(
            "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"
        )
        self.topic = self.cfg.get("BROCA_KAFKA_TOPIC", "ai_out")
        self.consumer_group = self.cfg.get("BROCA_CONSUMER_GROUP", "tts-consumer-group")
        self.auto_offset_reset = self.cfg.get("BROCA_AUTO_OFFSET_RESET", "earliest")
        self.output_dir = self.cfg.get("TTS_OUTPUT_DIR", "output")
        self.log_level = self.cfg.get("LOG_LEVEL", "INFO")

        # TTS engine selection (coqui or orca)
        self.tts_engine = self.cfg.get("TTS_ENGINE", "coqui")

        # Coqui TTS settings
        self.coqui_model_name = self.cfg.get(
            "TTS_MODEL_NAME", "tts_models/en/vctk/vits"
        )
        self.coqui_speaker = self.cfg.get("TTS_SPEAKER", "p225")

        # Orca TTS settings
        self.orca_access_key = self.cfg.get("ORCA_ACCESS_KEY", None)
        self.orca_voice = self.cfg.get("ORCA_VOICE", None)
        self.orca_model_path = self.cfg.get_path("ORCA_MODEL_PATH", None)

    def reload(self) -> bool:
        """Check for config changes and reload if needed"""
        if self.cfg.reload():
            self._load()
            return True
        return False


def load_config() -> BrocaConfig:
    """Load configuration from environment variables"""
    broca_cfg = BrocaConfig(config)

    # Update logging level
    import logging

    logger.setLevel(getattr(logging, broca_cfg.log_level.upper(), logging.INFO))

    return broca_cfg


class TTSConsumer:
    def __init__(self, config: BrocaConfig):
        self.config = config

        # Track last message to detect duplicates
        self.last_message_text = None
        self.is_playing = False
        self._audio_process = None

        # Initialize state for cross-component communication
        self.state = get_state()

        # Initialize playback state on startup (clear any stale state from previous run)
        self._init_playback_state()

        self.consumer = KafkaConsumer(
            config.topic,
            bootstrap_servers=config.bootstrap_servers,
            value_deserializer=lambda m: m.decode("utf-8"),
            auto_offset_reset=config.auto_offset_reset,
            enable_auto_commit=True,
            group_id=config.consumer_group,
        )

        # Initialize TTS engine
        self.tts: TTSEngine = self._init_tts_engine(config)

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

    def _init_tts_engine(self, config: BrocaConfig) -> TTSEngine:
        """Initialize or reinitialize the TTS engine"""
        logger.info(f"Initializing TTS engine: {config.tts_engine}...")

        engine = create_tts_engine(
            engine=config.tts_engine,
            coqui_model_name=config.coqui_model_name,
            coqui_speaker=config.coqui_speaker,
            orca_access_key=config.orca_access_key,
            orca_voice=config.orca_voice,
            orca_model_path=config.orca_model_path,
        )

        logger.info(f"TTS engine '{config.tts_engine}' initialized successfully")
        return engine

    def _init_playback_state(self):
        """Initialize playback state on startup, clearing any stale state."""
        # Check if there's a stale audio PID from a previous run
        old_pid = self.state.get_int("broca.audio_pid")
        if old_pid is not None:
            logger.warning(
                f"Found stale audio PID {old_pid} from previous run, clearing state"
            )

        self.state.set_bool("broca.is_playing", False)
        self.state.delete("broca.audio_pid")
        logger.info("Initialized playback state: is_playing=False")

    def _cleanup_playback_state(self):
        """Clean up playback state on shutdown."""
        # Kill any running audio process
        if self._audio_process is not None:
            try:
                self._audio_process.terminate()
                self._audio_process.wait(timeout=2)
            except Exception:
                try:
                    self._audio_process.kill()
                except Exception:
                    pass
            self._audio_process = None

        # Clear state
        self.is_playing = False
        self.state.set_bool("broca.is_playing", False)
        self.state.delete("broca.audio_pid")
        logger.info("Cleaned up playback state")

    def _check_config_reload(self):
        """Check if config has changed and reload configuration if needed"""
        old_tts_engine = self.config.tts_engine
        old_coqui_model = self.config.coqui_model_name
        old_coqui_speaker = self.config.coqui_speaker
        old_orca_access_key = self.config.orca_access_key
        old_orca_voice = self.config.orca_voice
        old_orca_model_path = self.config.orca_model_path
        old_output_dir = self.config.output_dir

        if self.config.reload():
            logger.info("Detected config file change, reloading configuration...")

            # Check if TTS settings changed
            engine_changed = self.config.tts_engine != old_tts_engine
            coqui_changed = (
                self.config.coqui_model_name != old_coqui_model
                or self.config.coqui_speaker != old_coqui_speaker
            )
            orca_changed = (
                self.config.orca_access_key != old_orca_access_key
                or self.config.orca_voice != old_orca_voice
                or self.config.orca_model_path != old_orca_model_path
            )

            # Reinitialize if engine changed or current engine's settings changed
            should_reinit = engine_changed
            if self.config.tts_engine == "coqui" and coqui_changed:
                should_reinit = True
            if self.config.tts_engine == "orca" and orca_changed:
                should_reinit = True

            if should_reinit:
                logger.info("TTS settings changed, reinitializing engine...")
                # Cleanup old engine
                self.tts.cleanup()
                self.tts = self._init_tts_engine(self.config)

            # Update output directory
            if self.config.output_dir != old_output_dir:
                os.makedirs(self.config.output_dir, exist_ok=True)

            logger.info("Configuration reloaded successfully")

    def process_message(self, message):
        """Process a single message and convert to speech"""
        try:
            # Check if audio is muted - if so, skip processing but still consume
            if self.state.get_bool("broca.audio_muted", default=False):
                logger.info(
                    "Audio muted - skipping message (still consuming to prevent backlog)"
                )
                return

            # Try to parse as JSON first, fallback to plain text
            if isinstance(message, str):
                try:
                    parsed = json.loads(message)
                    text = parsed.get("text", "")
                    message_id = parsed.get(
                        "id", datetime.now().strftime("%Y%m%d_%H%M%S")
                    )
                    is_processing_notice = parsed.get("is_processing_notice", False)
                    no_skip = parsed.get("no_skip", False)
                except json.JSONDecodeError:
                    # Treat as plain text
                    logger.info("Received plain text message, treating as text")
                    text = message
                    message_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                    is_processing_notice = False
                    no_skip = False
            else:
                # Already a dict
                text = message.get("text", "")
                message_id = message.get("id", datetime.now().strftime("%Y%m%d_%H%M%S"))
                is_processing_notice = message.get("is_processing_notice", False)
                no_skip = message.get("no_skip", False)

            # Log processing notices specifically
            if is_processing_notice:
                logger.info(f"Received processing notice: {message_id} - '{text}'")

            if not text:
                logger.warning(f"Empty text in message: {message}")
                return

            # Check for duplicate message (skip check for processing notices or no_skip flag)
            skip_duplicate_check = is_processing_notice or no_skip
            if not skip_duplicate_check and text == self.last_message_text:
                logger.info(f"Duplicate message skipped: '{text[:30]}...'")
                return

            logger.info(f"Processing message {message_id}: {text[:50]}...")

            # Set playing state (both local and in shared state)
            self.is_playing = True
            self.state.set_bool("broca.is_playing", True)

            # Generate speech
            output_path = f"{self.config.output_dir}/{message_id}.wav"

            # Use unified TTS engine interface
            self.tts.synthesize(text=text, output_path=output_path)

            logger.info(f"Speech generated: {output_path}")

            # Play the audio using subprocess to track PID
            logger.info("Playing audio...")
            self._audio_process = subprocess.Popen(
                ["afplay", output_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # Store PID in state for cross-component control
            self.state.set_int("broca.audio_pid", self._audio_process.pid)
            logger.info(f"Audio playback started (PID: {self._audio_process.pid})")

            # Wait for playback to complete
            self._audio_process.wait()
            logger.info("Audio playback completed")

            # Clear audio state
            self._audio_process = None
            self.state.delete("broca.audio_pid")

            # Delete the file after playing
            os.remove(output_path)
            logger.info(f"Audio file deleted: {output_path}")

            # Update last message and playing state
            self.last_message_text = text
            self.is_playing = False
            self.state.set_bool("broca.is_playing", False)

        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            self.is_playing = False
            self.state.set_bool("broca.is_playing", False)
            self._audio_process = None
            self.state.delete("broca.audio_pid")

    def start(self):
        """Start consuming messages"""
        logger.info(f"Starting consumer for topic: {self.config.topic}")
        logger.info("Waiting for messages...")

        # Set up signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            self._cleanup_playback_state()
            self.consumer.close()
            sys.exit(0)

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        try:
            for message in self.consumer:
                # Check if configuration has changed
                self._check_config_reload()

                # Process the message
                self.process_message(message.value)
        except KeyboardInterrupt:
            logger.info("Shutting down consumer...")
        finally:
            self._cleanup_playback_state()
            self.consumer.close()


if __name__ == "__main__":
    config = load_config()
    consumer = TTSConsumer(config)
    consumer.start()
