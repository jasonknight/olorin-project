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
log_file = os.path.join(log_dir, "cortex-producer.log")
env_log_level = config.get("LOG_LEVEL", "INFO")

logger = OlorinLogger(log_file=log_file, log_level=env_log_level, name=__name__)


class PromptProducer:
    def __init__(
        self, bootstrap_servers=None, topic=None, send_timeout=None, max_retries=None
    ):
        logger.info("=" * 60)
        logger.info("Initializing PromptProducer...")
        logger.info("=" * 60)

        # Load configuration from config or use parameter overrides
        logger.info("Loading configuration...")
        self.bootstrap_servers = bootstrap_servers or config.get(
            "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"
        )
        self.topic = topic or config.get("CORTEX_INPUT_TOPIC", "prompts")
        self.send_timeout = int(
            send_timeout or config.get_int("KAFKA_SEND_TIMEOUT", 10)
        )
        self.max_retries = int(max_retries or config.get_int("KAFKA_MAX_RETRIES", 3))

        logger.info("Configuration loaded:")
        logger.info(f"  Bootstrap servers: {self.bootstrap_servers}")
        logger.info(f"  Topic: {self.topic}")
        logger.info(f"  Send timeout: {self.send_timeout}s")
        logger.info(f"  Max retries: {self.max_retries}")

        logger.info("Creating Kafka producer...")
        logger.debug("Producer settings:")
        logger.debug("  api_version_auto_timeout_ms: 5000")
        logger.debug("  retries: 5")
        logger.debug("  max_in_flight_requests_per_connection: 1")

        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                # Let Kafka broker assign the timestamp to avoid clock skew issues
                api_version_auto_timeout_ms=5000,
                retries=5,
                max_in_flight_requests_per_connection=1,
            )
            logger.info(f"Successfully connected to Kafka at {self.bootstrap_servers}")
            logger.info(f"Ready to send messages to topic: {self.topic}")
            logger.info("=" * 60)
        except Exception as e:
            logger.error(f"Failed to create Kafka producer: {e}", exc_info=True)
            raise

    def send_prompt(self, text):
        """Send a prompt message to Kafka in JSON format with automatic retry on timestamp errors"""
        logger.info("\n" + "=" * 60)
        logger.info("SENDING PROMPT")
        logger.info("=" * 60)
        logger.info(f"Prompt text: {text[:100]}{'...' if len(text) > 100 else ''}")
        logger.info(f"  Full text length: {len(text)} characters")
        logger.debug(f"  Full text: {text}")

        message = {"prompt": text, "id": datetime.now().strftime("%Y%m%d_%H%M%S_%f")}
        logger.info(f"Message ID: {message['id']}")
        logger.debug(f"Complete message: {message}")

        for attempt in range(self.max_retries):
            logger.info(f"\nAttempt {attempt + 1}/{self.max_retries}")
            logger.debug(f"Sending to topic: {self.topic}")
            logger.debug(f"Bootstrap servers: {self.bootstrap_servers}")

            try:
                # Send without explicit timestamp - let Kafka broker assign it
                send_start_time = datetime.now()
                logger.debug("Calling producer.send()...")
                future = self.producer.send(self.topic, value=message)

                # Wait for message to be sent
                logger.debug(
                    f"Waiting for confirmation (timeout: {self.send_timeout}s)..."
                )
                record_metadata = future.get(timeout=self.send_timeout)
                send_end_time = datetime.now()
                send_duration = (send_end_time - send_start_time).total_seconds()

                logger.info("SUCCESS - Message sent to Kafka")
                logger.info(f"  Topic: {record_metadata.topic}")
                logger.info(f"  Partition: {record_metadata.partition}")
                logger.info(f"  Offset: {record_metadata.offset}")
                logger.info(f"  Timestamp: {record_metadata.timestamp}")
                logger.info(f"  Send duration: {send_duration:.2f}s")
                logger.info(f"  Preview: {text[:50]}{'...' if len(text) > 50 else ''}")
                logger.info("=" * 60)
                return True

            except InvalidTimestampError as e:
                logger.warning("=" * 60)
                logger.warning(
                    f"TIMESTAMP ERROR on attempt {attempt + 1}/{self.max_retries}"
                )
                logger.warning("=" * 60)
                logger.warning(f"Error details: {e}")
                logger.warning(f"Error type: {type(e).__name__}")

                if attempt < self.max_retries - 1:
                    wait_time = 2**attempt  # Exponential backoff: 1s, 2s, 4s
                    logger.info(f"Will retry in {wait_time} seconds...")
                    logger.info("Regenerating message ID for retry...")
                    time.sleep(wait_time)
                    # Regenerate message ID for retry
                    old_id = message["id"]
                    message["id"] = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    logger.info(f"  Old ID: {old_id}")
                    logger.info(f"  New ID: {message['id']}")
                else:
                    logger.error("=" * 60)
                    logger.error(
                        f"FAILED after {self.max_retries} attempts (timestamp issues)"
                    )
                    logger.error("=" * 60)
                    return False

            except KafkaError as e:
                logger.error("=" * 60)
                logger.error(f"KAFKA ERROR on attempt {attempt + 1}/{self.max_retries}")
                logger.error("=" * 60)
                logger.error(f"Error details: {e}", exc_info=True)
                logger.error(f"Error type: {type(e).__name__}")

                if attempt < self.max_retries - 1:
                    wait_time = 2**attempt
                    logger.info(f"Will retry in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error("=" * 60)
                    logger.error(
                        f"FAILED after {self.max_retries} attempts (Kafka error)"
                    )
                    logger.error("=" * 60)
                    return False

            except Exception as e:
                logger.error("=" * 60)
                logger.error("UNEXPECTED ERROR sending prompt")
                logger.error("=" * 60)
                logger.error(f"Error: {e}", exc_info=True)
                logger.error(f"Error type: {type(e).__name__}")
                return False

        logger.error("=" * 60)
        logger.error("FAILED - All retry attempts exhausted")
        logger.error("=" * 60)
        return False

    def start_repl(self):
        """Start the REPL interface"""
        logger.info("=" * 60)
        logger.info("STARTING REPL INTERFACE")
        logger.info("=" * 60)

        # Set up readline history
        history_file = os.path.expanduser("~/.prompt_producer_history")
        logger.info(f"History file: {history_file}")
        try:
            readline.read_history_file(history_file)
            history_length = readline.get_current_history_length()
            logger.info(f"Loaded {history_length} history entries")
        except FileNotFoundError:
            logger.info("No history file found, starting fresh")

        print("\n" + "=" * 60)
        print("Prompt Producer REPL")
        print("=" * 60)
        print("Type your prompt and press Enter to send to Kafka")
        print("Commands: 'quit' or 'exit' to exit, Ctrl+C to interrupt")
        print("=" * 60 + "\n")

        prompt_count = 0

        try:
            logger.info("Entering REPL loop...")
            while True:
                try:
                    logger.debug("Waiting for user input...")
                    text = input(">>> ").strip()
                    logger.debug(f"User input received: '{text}'")

                    if not text:
                        logger.debug("Empty input, skipping...")
                        continue

                    if text.lower() in ["quit", "exit"]:
                        logger.info("User requested exit")
                        print("Exiting...")
                        break

                    prompt_count += 1
                    logger.info(f"Processing prompt #{prompt_count}")
                    self.send_prompt(text)

                except EOFError:
                    logger.info("EOF received")
                    print("\nExiting...")
                    break

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            print("\n\nInterrupted. Exiting...")
        finally:
            logger.info("=" * 60)
            logger.info("REPL SHUTDOWN")
            logger.info("=" * 60)
            logger.info(f"Total prompts sent: {prompt_count}")

            # Save command history
            logger.info("Saving command history...")
            try:
                readline.write_history_file(history_file)
                logger.info(f"History saved to {history_file}")
            except Exception as e:
                logger.warning(f"Could not save history: {e}")

            logger.info("Flushing producer...")
            self.producer.flush()
            logger.info("Closing producer...")
            self.producer.close()
            logger.info("Producer closed successfully")
            logger.info("=" * 60)


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("PRODUCER APPLICATION STARTING")
    logger.info("=" * 60)
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info("Configuration values:")
    logger.info(
        f"  KAFKA_BOOTSTRAP_SERVERS: {config.get('KAFKA_BOOTSTRAP_SERVERS', 'not set')}"
    )
    logger.info(f"  CORTEX_INPUT_TOPIC: {config.get('CORTEX_INPUT_TOPIC', 'not set')}")
    logger.info(f"  KAFKA_SEND_TIMEOUT: {config.get('KAFKA_SEND_TIMEOUT', 'not set')}")
    logger.info(f"  KAFKA_MAX_RETRIES: {config.get('KAFKA_MAX_RETRIES', 'not set')}")
    logger.info(f"  LOG_LEVEL: {config.get('LOG_LEVEL', 'not set')}")

    try:
        producer = PromptProducer()
        producer.start_repl()
    except Exception as e:
        logger.error("=" * 60)
        logger.error("FATAL ERROR DURING STARTUP OR EXECUTION")
        logger.error("=" * 60)
        logger.error(f"Error: {e}", exc_info=True)
        raise

    logger.info("=" * 60)
    logger.info("PRODUCER APPLICATION EXITING")
    logger.info("=" * 60)
