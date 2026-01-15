"""
Control Client Library for Olorin Project

Provides a ControlClient class for cross-component communication via Kafka.
Components use this library to send commands and receive responses through
the control_in and control_out Kafka topics.

Message Formats:
    Request (sent to control_in topic):
        {
            "id": "ctrl_20260115_123456_abc123",
            "command": "stop-broca-audio-play",
            "payload": {"force": false},
            "timestamp": "2026-01-15T10:30:45.123456",
            "expects_response": true,
            "source": "client-id"
        }

    Response (received from control_out topic):
        {
            "id": "resp_...",
            "request_id": "ctrl_20260115_123456_abc123",
            "command": "stop-broca-audio-play",
            "success": true,
            "result": {"pid_killed": 12345},
            "error": null,
            "handler_duration_ms": 45
        }
"""

import json
import logging
import os
import secrets
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from libs.config import Config  # noqa: E402

logger = logging.getLogger(__name__)


class ControlError(Exception):
    """Base exception for control client errors."""

    pass


@dataclass
class PendingRequest:
    """Tracks a pending request waiting for a response."""

    request_id: str
    command: str
    created_at: datetime
    timeout_ms: int
    expects_response: bool
    response_event: threading.Event = field(default_factory=threading.Event)
    response: Optional[Dict[str, Any]] = None
    error: Optional[Exception] = None
    callback: Optional[Callable[[Dict[str, Any]], None]] = None
    error_callback: Optional[Callable[[Exception], None]] = None
    _timer: Optional[threading.Timer] = field(default=None, repr=False)


class ControlClient:
    """
    Client for sending control commands via Kafka.

    Provides synchronous and asynchronous methods for sending commands
    to the control_in topic and optionally waiting for responses from
    the control_out topic.

    Usage:
        client = ControlClient()

        # Synchronous call (blocks until response)
        result = client.call_sync("stop-broca-audio-play", {"force": True})

        # Asynchronous call (non-blocking with callbacks)
        request_id = client.call_async(
            "stop-broca-audio-play",
            {"force": True},
            callback=on_success,
            error_callback=on_error
        )

        # Fire-and-forget (no response expected)
        request_id = client.send("log-message", {"level": "INFO", "msg": "test"})

        # Clean shutdown
        client.close()

    Threading Design:
        - Background listener thread: Lazy init on first call requiring response
        - Register pending request BEFORE Kafka send (handles race condition)
        - Use threading.Event for sync blocking
        - Use threading.Timer for async timeouts
        - Callbacks execute in ThreadPoolExecutor(max_workers=4)
        - Lock hierarchy: _producer_lock -> _pending_lock (prevent deadlocks)
        - Never hold locks during Kafka I/O
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        client_id: Optional[str] = None,
        bootstrap_servers: Optional[str] = None,
        input_topic: Optional[str] = None,
        output_topic: Optional[str] = None,
    ):
        """
        Initialize the control client.

        Args:
            config: Config instance for reading settings. Uses default if None.
            client_id: Unique identifier for this client. Auto-generated if None.
            bootstrap_servers: Kafka bootstrap servers. Uses config if None.
            input_topic: Topic for sending commands. Uses config if None.
            output_topic: Topic for receiving responses. Uses config if None.
        """
        self._config = config or Config()

        # Client identification
        self._client_id = client_id or f"control-client-{uuid.uuid4().hex[:8]}"

        # Kafka settings
        self._bootstrap_servers = bootstrap_servers or self._config.get(
            "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"
        )
        self._input_topic = input_topic or self._config.get(
            "CONTROL_INPUT_TOPIC", "control_in"
        )
        self._output_topic = output_topic or self._config.get(
            "CONTROL_OUTPUT_TOPIC", "control_out"
        )

        # Threading primitives - follow lock hierarchy: _producer_lock -> _pending_lock
        self._producer_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._listener_lock = threading.Lock()

        # Kafka producer (lazy init)
        self._producer: Optional[Any] = None

        # Kafka consumer and listener thread (lazy init)
        self._consumer: Optional[Any] = None
        self._listener_thread: Optional[threading.Thread] = None
        self._listener_running = threading.Event()
        self._shutdown_event = threading.Event()

        # Pending requests map: request_id -> PendingRequest
        self._pending: Dict[str, PendingRequest] = {}

        # Thread pool for executing callbacks
        self._executor: Optional[ThreadPoolExecutor] = None

        # Track if closed
        self._closed = False

    def _get_producer(self) -> Any:
        """Get or create the Kafka producer (thread-safe, lazy init)."""
        if self._producer is not None:
            return self._producer

        with self._producer_lock:
            if self._producer is not None:
                return self._producer

            try:
                from kafka import KafkaProducer

                self._producer = KafkaProducer(
                    bootstrap_servers=self._bootstrap_servers,
                    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                    key_serializer=lambda k: k.encode("utf-8") if k else None,
                )
                logger.debug(
                    f"Created Kafka producer connected to {self._bootstrap_servers}"
                )
            except ImportError:
                raise ControlError(
                    "kafka-python package is required. Install with: pip install kafka-python"
                )
            except Exception as e:
                raise ControlError(f"Failed to create Kafka producer: {e}")

            return self._producer

    def _get_executor(self) -> ThreadPoolExecutor:
        """Get or create the thread pool executor for callbacks."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=4)
        return self._executor

    def _ensure_listener(self) -> None:
        """Ensure the listener thread is running (lazy init)."""
        if self._listener_running.is_set():
            return

        with self._listener_lock:
            if self._listener_running.is_set():
                return

            if self._closed:
                raise ControlError("Client is closed")

            try:
                from kafka import KafkaConsumer

                # Create consumer with unique group to receive all messages
                consumer_group = f"{self._client_id}-{uuid.uuid4().hex[:8]}"
                self._consumer = KafkaConsumer(
                    self._output_topic,
                    bootstrap_servers=self._bootstrap_servers,
                    group_id=consumer_group,
                    auto_offset_reset="latest",
                    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                    consumer_timeout_ms=1000,  # Poll timeout for checking shutdown
                )
                logger.debug(
                    f"Created Kafka consumer for {self._output_topic} with group {consumer_group}"
                )
            except ImportError:
                raise ControlError(
                    "kafka-python package is required. Install with: pip install kafka-python"
                )
            except Exception as e:
                raise ControlError(f"Failed to create Kafka consumer: {e}")

            # Start listener thread
            self._listener_thread = threading.Thread(
                target=self._listener_loop, daemon=True, name="control-listener"
            )
            self._listener_thread.start()
            self._listener_running.set()
            logger.debug("Started control listener thread")

    def _listener_loop(self) -> None:
        """Background thread that listens for responses."""
        while not self._shutdown_event.is_set():
            try:
                # Poll for messages (non-blocking due to consumer_timeout_ms)
                for message in self._consumer:
                    if self._shutdown_event.is_set():
                        break

                    try:
                        response = message.value
                        request_id = response.get("request_id")

                        if not request_id:
                            logger.warning(
                                f"Received response without request_id: {response}"
                            )
                            continue

                        # Find and process pending request
                        self._process_response(request_id, response)

                    except Exception as e:
                        logger.error(f"Error processing response: {e}")

            except Exception as e:
                if not self._shutdown_event.is_set():
                    logger.error(f"Listener error: {e}")

        logger.debug("Listener thread exiting")

    def _process_response(self, request_id: str, response: Dict[str, Any]) -> None:
        """Process a response for a pending request."""
        pending: Optional[PendingRequest] = None

        with self._pending_lock:
            pending = self._pending.pop(request_id, None)

        if pending is None:
            logger.debug(f"Received response for unknown request: {request_id}")
            return

        # Cancel timeout timer if set
        if pending._timer is not None:
            pending._timer.cancel()

        # Store response
        pending.response = response

        # Signal sync waiters
        pending.response_event.set()

        # Execute callbacks in thread pool (not listener thread)
        if pending.callback is not None or pending.error_callback is not None:
            self._get_executor().submit(self._execute_callback, pending, response)

    def _execute_callback(
        self, pending: PendingRequest, response: Dict[str, Any]
    ) -> None:
        """Execute callback for a response (runs in thread pool)."""
        try:
            if response.get("success", False):
                if pending.callback:
                    pending.callback(response)
            else:
                error_msg = response.get("error") or "Unknown error"
                error = ControlError(error_msg)
                if pending.error_callback:
                    pending.error_callback(error)
                elif pending.callback:
                    # If no error callback, call regular callback anyway
                    pending.callback(response)
        except Exception as e:
            logger.error(f"Error executing callback for {pending.request_id}: {e}")

    def _generate_request_id(self) -> str:
        """Generate a unique request ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        random_suffix = secrets.token_hex(6)
        return f"ctrl_{timestamp}_{random_suffix}"

    def _build_request(
        self, command: str, payload: Optional[Dict[str, Any]], expects_response: bool
    ) -> Dict[str, Any]:
        """Build a request message."""
        request_id = self._generate_request_id()
        return {
            "id": request_id,
            "command": command,
            "payload": payload or {},
            "timestamp": datetime.utcnow().isoformat(),
            "expects_response": expects_response,
            "source": self._client_id,
        }

    def _handle_async_timeout(self, request_id: str) -> None:
        """Handle timeout for an async request."""
        pending: Optional[PendingRequest] = None

        with self._pending_lock:
            pending = self._pending.pop(request_id, None)

        if pending is None:
            return  # Already processed

        # Set error
        timeout_error = TimeoutError(
            f"Request {request_id} timed out after {pending.timeout_ms}ms"
        )
        pending.error = timeout_error
        pending.response_event.set()

        # Execute error callback in thread pool
        if pending.error_callback is not None:
            self._get_executor().submit(pending.error_callback, timeout_error)

    def call_sync(
        self,
        command: str,
        payload: Optional[Dict[str, Any]] = None,
        timeout: float = 5.0,
    ) -> Dict[str, Any]:
        """
        Send a command and block until response is received.

        Args:
            command: The command name to execute
            payload: Optional command payload
            timeout: Timeout in seconds (default 5.0)

        Returns:
            The response dictionary

        Raises:
            TimeoutError: If no response received within timeout
            ControlError: If there was an error sending the command
        """
        if self._closed:
            raise ControlError("Client is closed")

        # Build request
        request = self._build_request(command, payload, expects_response=True)
        request_id = request["id"]
        timeout_ms = int(timeout * 1000)

        # Create pending request entry
        pending = PendingRequest(
            request_id=request_id,
            command=command,
            created_at=datetime.utcnow(),
            timeout_ms=timeout_ms,
            expects_response=True,
        )

        # Register pending request BEFORE sending (handles race condition)
        with self._pending_lock:
            self._pending[request_id] = pending

        # Ensure listener is running
        self._ensure_listener()

        # Send request (without holding any locks)
        try:
            producer = self._get_producer()
            producer.send(self._input_topic, key=request_id, value=request)
            producer.flush()
            logger.debug(f"Sent sync request {request_id}: {command}")
        except Exception as e:
            # Clean up pending on send failure
            with self._pending_lock:
                self._pending.pop(request_id, None)
            raise ControlError(f"Failed to send command: {e}")

        # Wait for response
        received = pending.response_event.wait(timeout=timeout)

        if not received:
            # Clean up on timeout
            with self._pending_lock:
                self._pending.pop(request_id, None)
            raise TimeoutError(f"Request {request_id} timed out after {timeout}s")

        if pending.error is not None:
            raise pending.error

        if pending.response is None:
            raise ControlError(f"No response received for request {request_id}")

        return pending.response

    def call_async(
        self,
        command: str,
        payload: Optional[Dict[str, Any]] = None,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        error_callback: Optional[Callable[[Exception], None]] = None,
        timeout: float = 10.0,
    ) -> str:
        """
        Send a command asynchronously with optional callbacks.

        Callbacks are executed in a thread pool, not the listener thread.

        Args:
            command: The command name to execute
            payload: Optional command payload
            callback: Called with response dict on success
            error_callback: Called with exception on error or timeout
            timeout: Timeout in seconds (default 10.0)

        Returns:
            The request ID for tracking

        Raises:
            ControlError: If there was an error sending the command
        """
        if self._closed:
            raise ControlError("Client is closed")

        # Build request
        request = self._build_request(command, payload, expects_response=True)
        request_id = request["id"]
        timeout_ms = int(timeout * 1000)

        # Create pending request entry with callbacks
        pending = PendingRequest(
            request_id=request_id,
            command=command,
            created_at=datetime.utcnow(),
            timeout_ms=timeout_ms,
            expects_response=True,
            callback=callback,
            error_callback=error_callback,
        )

        # Set up timeout timer
        timer = threading.Timer(timeout, self._handle_async_timeout, args=[request_id])
        pending._timer = timer

        # Register pending request BEFORE sending (handles race condition)
        with self._pending_lock:
            self._pending[request_id] = pending

        # Ensure listener is running
        self._ensure_listener()

        # Start timeout timer
        timer.start()

        # Send request (without holding any locks)
        try:
            producer = self._get_producer()
            producer.send(self._input_topic, key=request_id, value=request)
            producer.flush()
            logger.debug(f"Sent async request {request_id}: {command}")
        except Exception as e:
            # Clean up on send failure
            timer.cancel()
            with self._pending_lock:
                self._pending.pop(request_id, None)
            raise ControlError(f"Failed to send command: {e}")

        return request_id

    def send(
        self,
        command: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Send a fire-and-forget command (no response expected).

        Args:
            command: The command name to execute
            payload: Optional command payload

        Returns:
            The request ID for logging/tracking

        Raises:
            ControlError: If there was an error sending the command
        """
        if self._closed:
            raise ControlError("Client is closed")

        # Build request with expects_response=False
        request = self._build_request(command, payload, expects_response=False)
        request_id = request["id"]

        # Send request (no need to register pending or start listener)
        try:
            producer = self._get_producer()
            producer.send(self._input_topic, key=request_id, value=request)
            producer.flush()
            logger.debug(f"Sent fire-and-forget request {request_id}: {command}")
        except Exception as e:
            raise ControlError(f"Failed to send command: {e}")

        return request_id

    def close(self, timeout: float = 5.0) -> None:
        """
        Gracefully shut down the client.

        Args:
            timeout: Maximum time to wait for cleanup in seconds
        """
        if self._closed:
            return

        self._closed = True
        logger.debug("Closing control client")

        # Signal listener to stop
        self._shutdown_event.set()

        # Cancel all pending timeouts
        with self._pending_lock:
            for pending in self._pending.values():
                if pending._timer is not None:
                    pending._timer.cancel()
                pending.response_event.set()  # Unblock any waiters
            self._pending.clear()

        # Wait for listener thread
        if self._listener_thread is not None and self._listener_thread.is_alive():
            self._listener_thread.join(timeout=timeout)

        # Close consumer
        if self._consumer is not None:
            try:
                self._consumer.close()
            except Exception as e:
                logger.warning(f"Error closing consumer: {e}")

        # Close producer
        if self._producer is not None:
            try:
                self._producer.close(timeout=timeout)
            except Exception as e:
                logger.warning(f"Error closing producer: {e}")

        # Shutdown executor
        if self._executor is not None:
            self._executor.shutdown(wait=False)

        logger.debug("Control client closed")

    def __enter__(self) -> "ControlClient":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - close client."""
        self.close()


# Convenience singleton for simple usage
_default_client: Optional[ControlClient] = None
_singleton_lock = threading.Lock()


def get_control_client(**kwargs: Any) -> ControlClient:
    """
    Get the default ControlClient singleton.

    Args:
        **kwargs: Arguments passed to ControlClient constructor (only on first call)

    Returns:
        The default ControlClient instance
    """
    global _default_client
    if _default_client is None:
        with _singleton_lock:
            if _default_client is None:
                _default_client = ControlClient(**kwargs)
    return _default_client


def reset_default_client() -> None:
    """
    Reset the default client singleton (mainly for testing).

    Closes the existing client if any.
    """
    global _default_client
    with _singleton_lock:
        if _default_client is not None:
            _default_client.close()
            _default_client = None
