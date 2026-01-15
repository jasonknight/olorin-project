# consumer.py
from kafka import KafkaConsumer
import json
import os
import sys
from datetime import datetime

# Add parent directory to path for libs import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from libs.config import Config
from libs.olorin_logging import OlorinLogger

from TTS.api import TTS

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
        self.tts_model_name = self.cfg.get("TTS_MODEL_NAME", "tts_models/en/vctk/vits")
        self.tts_speaker = self.cfg.get("TTS_SPEAKER", "p225")
        self.consumer_group = self.cfg.get("BROCA_CONSUMER_GROUP", "tts-consumer-group")
        self.auto_offset_reset = self.cfg.get("BROCA_AUTO_OFFSET_RESET", "earliest")
        self.output_dir = self.cfg.get("TTS_OUTPUT_DIR", "output")
        self.log_level = self.cfg.get("LOG_LEVEL", "INFO")

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

        self.consumer = KafkaConsumer(
            config.topic,
            bootstrap_servers=config.bootstrap_servers,
            value_deserializer=lambda m: m.decode("utf-8"),
            auto_offset_reset=config.auto_offset_reset,
            enable_auto_commit=True,
            group_id=config.consumer_group,
        )

        # Initialize TTS model
        self.tts = self._init_tts_model(config)

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

    def _init_tts_model(self, config: BrocaConfig):
        """Initialize or reinitialize the TTS model"""
        logger.info(f"Loading TTS model: {config.tts_model_name}...")
        tts = TTS(model_name=config.tts_model_name, progress_bar=False)
        logger.info("TTS model loaded successfully")

        # Log available speakers if multi-speaker model
        if hasattr(tts, "speakers") and tts.speakers:
            logger.info(f"Available speakers: {tts.speakers}")
            logger.info(f"Using speaker: {config.tts_speaker}")

        return tts

    def _check_config_reload(self):
        """Check if .env has changed and reload configuration if needed"""
        old_tts_model = self.config.tts_model_name
        old_tts_speaker = self.config.tts_speaker
        old_output_dir = self.config.output_dir

        if self.config.reload():
            logger.info("Detected .env file change, reloading configuration...")

            # Check if TTS settings changed
            if (
                self.config.tts_model_name != old_tts_model
                or self.config.tts_speaker != old_tts_speaker
            ):
                logger.info("TTS settings changed, reinitializing model...")
                self.tts = self._init_tts_model(self.config)

            # Update output directory
            if self.config.output_dir != old_output_dir:
                os.makedirs(self.config.output_dir, exist_ok=True)

            logger.info("Configuration reloaded successfully")

    def process_message(self, message):
        """Process a single message and convert to speech"""
        try:
            # Try to parse as JSON first, fallback to plain text
            if isinstance(message, str):
                try:
                    parsed = json.loads(message)
                    text = parsed.get("text", "")
                    message_id = parsed.get(
                        "id", datetime.now().strftime("%Y%m%d_%H%M%S")
                    )
                except json.JSONDecodeError:
                    # Treat as plain text
                    logger.info("Received plain text message, treating as text")
                    text = message
                    message_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            else:
                # Already a dict
                text = message.get("text", "")
                message_id = message.get("id", datetime.now().strftime("%Y%m%d_%H%M%S"))

            if not text:
                logger.warning(f"Empty text in message: {message}")
                return

            # Check for duplicate message
            if text == self.last_message_text:
                logger.info("I received a duplicate message")
                return

            logger.info(f"Processing message {message_id}: {text[:50]}...")

            # Set playing state
            self.is_playing = True

            # Generate speech
            output_path = f"{self.config.output_dir}/{message_id}.wav"

            # Use speaker parameter if model supports it
            if hasattr(self.tts, "speakers") and self.tts.speakers:
                self.tts.tts_to_file(
                    text=text, file_path=output_path, speaker=self.config.tts_speaker
                )
            else:
                self.tts.tts_to_file(text=text, file_path=output_path)

            logger.info(f"Speech generated: {output_path}")

            # Play the audio
            logger.info("Playing audio...")
            os.system(f"afplay {output_path}")

            # Delete the file after playing
            os.remove(output_path)
            logger.info(f"Audio played and file deleted: {output_path}")

            # Update last message and playing state
            self.last_message_text = text
            self.is_playing = False

        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            self.is_playing = False

    def start(self):
        """Start consuming messages"""
        logger.info(f"Starting consumer for topic: {self.config.topic}")
        logger.info("Waiting for messages...")

        try:
            for message in self.consumer:
                # Check if configuration has changed
                self._check_config_reload()

                # Process the message
                self.process_message(message.value)
        except KeyboardInterrupt:
            logger.info("Shutting down consumer...")
        finally:
            self.consumer.close()


if __name__ == "__main__":
    config = load_config()
    consumer = TTSConsumer(config)
    consumer.start()
