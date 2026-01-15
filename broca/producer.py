#!/usr/bin/env python3
# producer.py
from kafka import KafkaProducer
from kafka.errors import KafkaError, InvalidTimestampError
import json
from datetime import datetime
import sys
import os
import time
import readline

# Add parent directory to path for libs import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from libs.config import Config
from libs.olorin_logging import OlorinLogger

# Initialize config
config = Config()

# Set up logging
default_log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
log_dir = config.get("LOG_DIR", default_log_dir)
log_file = os.path.join(log_dir, "broca-producer.log")
env_log_level = config.get("LOG_LEVEL", "INFO")

logger = OlorinLogger(log_file=log_file, log_level=env_log_level, name=__name__)


class TTSProducer:
    def __init__(
        self, bootstrap_servers=None, topic=None, send_timeout=None, max_retries=None
    ):
        # Load configuration from config or use parameter overrides
        self.bootstrap_servers = bootstrap_servers or config.get(
            "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"
        )
        self.topic = topic or config.get("BROCA_KAFKA_TOPIC", "ai_out")
        self.send_timeout = int(
            send_timeout or config.get_int("KAFKA_SEND_TIMEOUT", 10)
        )
        self.max_retries = int(max_retries or config.get_int("KAFKA_MAX_RETRIES", 3))

        self.producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            # Let Kafka broker assign the timestamp to avoid clock skew issues
            api_version_auto_timeout_ms=5000,
            retries=5,
            max_in_flight_requests_per_connection=1,
        )
        logger.info(f"Connected to Kafka at {self.bootstrap_servers}")
        logger.info(f"Sending messages to topic: {self.topic}")

    def send_message(self, text):
        """Send a text message to Kafka in JSON format with automatic retry on timestamp errors"""
        message = {"text": text, "id": datetime.now().strftime("%Y%m%d_%H%M%S_%f")}

        for attempt in range(self.max_retries):
            try:
                # Send without explicit timestamp - let Kafka broker assign it
                future = self.producer.send(self.topic, value=message)
                # Wait for message to be sent
                record_metadata = future.get(timeout=self.send_timeout)
                logger.info(
                    f"Message sent: {text[:50]}... (offset: {record_metadata.offset})"
                )
                return True

            except InvalidTimestampError as e:
                logger.warning(
                    f"Timestamp error on attempt {attempt + 1}/{self.max_retries}: {e}"
                )
                if attempt < self.max_retries - 1:
                    wait_time = 2**attempt  # Exponential backoff: 1s, 2s, 4s
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    # Regenerate message ID for retry
                    message["id"] = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                else:
                    logger.error(
                        f"Failed to send message after {self.max_retries} attempts due to timestamp issues"
                    )
                    return False

            except KafkaError as e:
                logger.error(
                    f"Kafka error on attempt {attempt + 1}/{self.max_retries}: {e}"
                )
                if attempt < self.max_retries - 1:
                    wait_time = 2**attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"Failed to send message after {self.max_retries} attempts"
                    )
                    return False

            except Exception as e:
                logger.error(f"Unexpected error sending message: {e}")
                return False

        return False

    def start_repl(self):
        """Start the REPL interface"""
        # Set up readline history
        history_file = os.path.expanduser("~/.tts_producer_history")
        try:
            readline.read_history_file(history_file)
        except FileNotFoundError:
            pass

        print("\n" + "=" * 60)
        print("TTS Producer REPL")
        print("=" * 60)
        print("Type your text and press Enter to send to Kafka")
        print("Commands: 'quit' or 'exit' to exit, Ctrl+C to interrupt")
        print("=" * 60 + "\n")

        try:
            while True:
                try:
                    text = input(">>> ").strip()

                    if not text:
                        continue

                    if text.lower() in ["quit", "exit"]:
                        print("Exiting...")
                        break

                    self.send_message(text)

                except EOFError:
                    print("\nExiting...")
                    break

        except KeyboardInterrupt:
            print("\n\nInterrupted. Exiting...")
        finally:
            # Save command history
            try:
                readline.write_history_file(history_file)
            except Exception as e:
                logger.warning(f"Could not save history: {e}")

            self.producer.flush()
            self.producer.close()
            logger.info("Producer closed")


if __name__ == "__main__":
    producer = TTSProducer()
    producer.start_repl()
