#!/usr/bin/env python3
"""
Control Controller Service for Olorin Project

Central controller service that listens on the 'control_in' Kafka topic,
executes handlers, and responds on 'control_out' (or custom reply_to topic).

Message Formats:
    Request (from control_in):
        {
            "id": "ctrl_20260115_123456_abc123",
            "command": "stop-broca-audio-play",
            "payload": {"force": false},
            "timestamp": "2026-01-15T10:30:45.123456",
            "expects_response": true,
            "reply_to": "control_out",
            "source": "client-id"
        }

    Response (to control_out or reply_to):
        {
            "id": "resp_20260115_123456_xyz789",
            "request_id": "ctrl_20260115_123456_abc123",
            "command": "stop-broca-audio-play",
            "success": true,
            "result": {"pid_killed": 12345},
            "error": null,
            "timestamp": "2026-01-15T10:30:45.345678",
            "handler_duration_ms": 45.2
        }
"""

import json
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, Optional

from kafka import KafkaConsumer, KafkaProducer

# Add parent directory to path for libs import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from libs.config import Config
from libs.control_handlers import discover_handlers, get_handler
from libs.control_server import ControlServer
from libs.olorin_logging import OlorinLogger

# Initialize config with hot-reload support
config = Config(watch=True)

# Set up logging
default_log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
log_dir = config.get("LOG_DIR", default_log_dir)
log_file = os.path.join(log_dir, "cortex-controller.log")
env_log_level = config.get("LOG_LEVEL", "INFO")

# Initialize logger
logger = OlorinLogger(log_file=log_file, log_level=env_log_level, name=__name__)


class ControllerConfig:
    """Configuration wrapper for Control Controller"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._load()

    def _load(self):
        """Load configuration values from Config"""
        self.kafka_bootstrap_servers = self.cfg.get(
            "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"
        )
        self.input_topic = self.cfg.get("CONTROL_INPUT_TOPIC", "control_in")
        self.output_topic = self.cfg.get("CONTROL_OUTPUT_TOPIC", "control_out")
        self.consumer_group = self.cfg.get(
            "CONTROL_CONSUMER_GROUP", "control-controller-group"
        )
        self.handler_timeout = self.cfg.get_int("CONTROL_HANDLER_TIMEOUT", 30)
        self.callback_pool_size = self.cfg.get_int("CONTROL_CALLBACK_POOL_SIZE", 4)
        self.log_level = self.cfg.get("LOG_LEVEL", "INFO")
        # API server configuration
        self.api_port = self.cfg.get_int("CONTROL_API_PORT", 8765)
        self.api_enabled = self.cfg.get_bool("CONTROL_API_ENABLED", True)

    def reload(self) -> bool:
        """Check for config changes and reload if needed"""
        if self.cfg.reload():
            self._load()
            return True
        return False


def load_config() -> ControllerConfig:
    """Load configuration from settings.json"""
    logger.info("Loading configuration...")

    ctrl_cfg = ControllerConfig(config)

    logger.info("Configuration loaded successfully:")
    logger.info(f"  Kafka Bootstrap Servers: {ctrl_cfg.kafka_bootstrap_servers}")
    logger.info(f"  Input Topic: {ctrl_cfg.input_topic}")
    logger.info(f"  Output Topic: {ctrl_cfg.output_topic}")
    logger.info(f"  Consumer Group: {ctrl_cfg.consumer_group}")
    logger.info(f"  Handler Timeout: {ctrl_cfg.handler_timeout}s")
    logger.info(f"  Callback Pool Size: {ctrl_cfg.callback_pool_size}")
    logger.info(f"  API Server Enabled: {ctrl_cfg.api_enabled}")
    logger.info(f"  API Server Port: {ctrl_cfg.api_port}")

    return ctrl_cfg


def generate_response_id() -> str:
    """Generate a unique response ID"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    unique = uuid.uuid4().hex[:6]
    return f"resp_{timestamp}_{unique}"


class ControlController:
    """
    Central controller service that executes control commands.

    Listens on 'control_in' topic, executes handlers, responds on 'control_out'.
    """

    def __init__(self, cfg: Optional[ControllerConfig] = None):
        """
        Initialize the Control Controller.

        Args:
            cfg: ControllerConfig instance. If None, loads from settings.json
        """
        logger.info("Initializing ControlController...")
        self.config = cfg if cfg is not None else load_config()

        # Discover handlers
        logger.info("Discovering control handlers...")
        self.handlers = discover_handlers()
        logger.info(f"Discovered {len(self.handlers)} handler(s):")
        for cmd in sorted(self.handlers.keys()):
            logger.info(f"  - {cmd}")

        # Initialize Kafka consumer
        logger.info(f"Creating Kafka consumer for topic '{self.config.input_topic}'...")
        try:
            self.consumer = KafkaConsumer(
                self.config.input_topic,
                bootstrap_servers=self.config.kafka_bootstrap_servers,
                value_deserializer=lambda m: m.decode("utf-8"),
                auto_offset_reset="earliest",
                enable_auto_commit=True,
                group_id=self.config.consumer_group,
                # Reasonable timeouts for control messages
                max_poll_interval_ms=300000,  # 5 minutes
                session_timeout_ms=30000,  # 30 seconds
                heartbeat_interval_ms=10000,  # 10 seconds
            )
            logger.info("Kafka consumer created successfully")
        except Exception as e:
            logger.error(f"Failed to create Kafka consumer: {e}", exc_info=True)
            raise

        # Initialize Kafka producer for responses
        logger.info(
            f"Creating Kafka producer for output topic '{self.config.output_topic}'..."
        )
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.config.kafka_bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
            logger.info("Kafka producer created successfully")
        except Exception as e:
            logger.error(f"Failed to create Kafka producer: {e}", exc_info=True)
            raise

        # Initialize thread pool for handler execution
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.callback_pool_size,
            thread_name_prefix="ctrl-handler",
        )
        logger.info(
            f"Thread pool executor initialized with {self.config.callback_pool_size} workers"
        )

        # Initialize API server (if enabled)
        self.api_server: Optional[ControlServer] = None
        if self.config.api_enabled:
            api_host = self.config.cfg.get("CONTROL_API_HOST", "0.0.0.0")
            logger.info(
                f"Creating Control API server on {api_host}:{self.config.api_port}..."
            )
            self.api_server = ControlServer(port=self.config.api_port, host=api_host)
            logger.info("Control API server created")

        self._running = False
        logger.info("ControlController initialized successfully")

    def _parse_request(self, message_value: str) -> Dict[str, Any]:
        """
        Parse a control request message.

        Args:
            message_value: Raw message string from Kafka

        Returns:
            Parsed request dictionary

        Raises:
            ValueError: If message is not valid JSON
        """
        try:
            return json.loads(message_value)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in control message: {e}")

    def _build_response(
        self,
        request_id: str,
        command: str,
        success: bool,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        handler_duration_ms: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Build a response message.

        Args:
            request_id: ID of the original request
            command: Command that was executed
            success: Whether the command succeeded
            result: Result data from handler (if success)
            error: Error message (if not success)
            handler_duration_ms: Time taken to execute handler

        Returns:
            Response dictionary
        """
        return {
            "id": generate_response_id(),
            "request_id": request_id,
            "command": command,
            "success": success,
            "result": result,
            "error": error,
            "timestamp": datetime.now().isoformat(),
            "handler_duration_ms": handler_duration_ms,
        }

    def _send_response(self, response: Dict[str, Any], topic: str) -> None:
        """
        Send a response to a Kafka topic.

        Args:
            response: Response dictionary to send
            topic: Target topic
        """
        try:
            self.producer.send(topic, value=response)
            logger.info(f"Sent response to topic '{topic}': {response['id']}")
        except Exception as e:
            logger.error(f"Failed to send response to '{topic}': {e}", exc_info=True)

    def _execute_command(self, request: Dict[str, Any]) -> None:
        """
        Execute a control command and send response if needed.

        Args:
            request: The control request dictionary
        """
        # Extract request fields
        request_id = request.get("id", "unknown")
        command = request.get("command", "")
        payload = request.get("payload", {})
        expects_response = request.get("expects_response", True)
        reply_to = request.get("reply_to", self.config.output_topic)

        logger.info(f"Executing command '{command}' (request_id={request_id})")
        logger.debug(f"Payload: {payload}")

        start_time = time.time()
        success = False
        result = None
        error = None

        try:
            # Get handler for command
            handler = get_handler(command)

            # Execute handler
            result = handler(payload)
            success = True
            logger.info(f"Handler '{command}' completed successfully")
            logger.debug(f"Result: {result}")

        except KeyError as e:
            error = f"Unknown command: {command}"
            logger.error(f"Handler not found: {e}")

        except Exception as e:
            error = f"Handler error: {type(e).__name__}: {str(e)}"
            logger.error(f"Handler '{command}' failed: {e}", exc_info=True)

        end_time = time.time()
        handler_duration_ms = (end_time - start_time) * 1000

        # Send response if expected
        if expects_response:
            response = self._build_response(
                request_id=request_id,
                command=command,
                success=success,
                result=result,
                error=error,
                handler_duration_ms=handler_duration_ms,
            )
            self._send_response(response, reply_to)
        else:
            logger.debug(f"Response not expected for request '{request_id}'")

    def start(self) -> None:
        """
        Start the controller main loop.

        Consumes messages from control_in topic and executes handlers.
        Also starts the API server in a background thread if enabled.
        """
        logger.info("=" * 60)
        logger.info("STARTING CONTROL CONTROLLER")
        logger.info("=" * 60)
        logger.info(f"Input topic: {self.config.input_topic}")
        logger.info(f"Output topic: {self.config.output_topic}")
        logger.info(f"Consumer group: {self.config.consumer_group}")

        # Start API server in background thread
        if self.api_server is not None:
            logger.info(
                f"Starting Control API server on port {self.config.api_port}..."
            )
            self.api_server.start_background()
            logger.info(f"Control API server running at {self.api_server.url}")
            logger.info(f"  GET  {self.api_server.url}/commands")
            logger.info(f"  POST {self.api_server.url}/execute")

        logger.info("Waiting for control messages...")
        logger.info("=" * 60)

        self._running = True
        message_count = 0

        try:
            for message in self.consumer:
                if not self._running:
                    logger.info("Controller stop requested, exiting loop")
                    break

                message_count += 1
                logger.info(f"Received message #{message_count} from {message.topic}")
                logger.debug(f"Message offset: {message.offset}")
                logger.debug(f"Message value: {message.value[:200]}...")

                try:
                    # Parse the request
                    request = self._parse_request(message.value)

                    # Submit to thread pool
                    self.executor.submit(self._execute_command, request)
                    logger.info(
                        f"Command submitted to thread pool (message #{message_count})"
                    )

                except ValueError as e:
                    logger.error(f"Failed to parse message #{message_count}: {e}")
                    # Send error response for malformed messages
                    error_response = self._build_response(
                        request_id="unknown",
                        command="unknown",
                        success=False,
                        error=str(e),
                    )
                    self._send_response(error_response, self.config.output_topic)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.error(f"Fatal error in controller loop: {e}", exc_info=True)
            raise
        finally:
            logger.info("Controller loop ended")
            logger.info(f"Total messages processed: {message_count}")

    def stop(self) -> None:
        """
        Stop the controller gracefully.

        Shuts down API server, thread pool, and closes Kafka connections.
        """
        logger.info("=" * 60)
        logger.info("STOPPING CONTROL CONTROLLER")
        logger.info("=" * 60)

        self._running = False

        # Stop API server
        if self.api_server is not None and self.api_server.is_running:
            logger.info("Stopping Control API server...")
            self.api_server.stop()
            logger.info("Control API server stopped")

        # Shutdown thread pool
        logger.info("Shutting down thread pool...")
        self.executor.shutdown(wait=True, cancel_futures=False)
        logger.info("Thread pool shut down")

        # Close Kafka connections
        logger.info("Closing Kafka consumer...")
        self.consumer.close()
        logger.info("Consumer closed")

        logger.info("Flushing Kafka producer...")
        self.producer.flush()
        logger.info("Producer flushed")

        logger.info("Closing Kafka producer...")
        self.producer.close()
        logger.info("Producer closed")

        logger.info("=" * 60)
        logger.info("CONTROL CONTROLLER STOPPED")
        logger.info("=" * 60)


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("CONTROL CONTROLLER APPLICATION STARTING")
    logger.info("=" * 60)
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")

    try:
        cfg = load_config()
        controller = ControlController(cfg)
        controller.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise
    finally:
        if "controller" in locals():
            controller.stop()

    logger.info("=" * 60)
    logger.info("CONTROL CONTROLLER APPLICATION EXITING")
    logger.info("=" * 60)
