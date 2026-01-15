"""
Comprehensive unit tests for the Control Client Library.

Tests use mocked Kafka to avoid requiring actual Kafka infrastructure.
"""

import json
import sys
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add parent paths for imports
sys.path.insert(0, "/Users/olorin/olorin-project")

from libs.control import (
    ControlClient,
    ControlError,
    PendingRequest,
    get_control_client,
    reset_default_client,
)


@dataclass
class MockMessage:
    """Mock Kafka message."""

    value: Dict[str, Any]
    key: Optional[bytes] = None
    topic: str = "control_out"
    partition: int = 0
    offset: int = 0


class MockKafkaProducer:
    """Mock KafkaProducer for testing."""

    def __init__(self, **kwargs: Any):
        self.messages: List[Dict[str, Any]] = []
        self.flushed = False
        self.closed = False
        self._value_serializer = kwargs.get(
            "value_serializer", lambda v: json.dumps(v).encode("utf-8")
        )
        self._key_serializer = kwargs.get(
            "key_serializer", lambda k: k.encode("utf-8") if k else None
        )

    def send(self, topic: str, key: Optional[str] = None, value: Any = None) -> Future:
        """Mock send that stores messages."""
        self.messages.append({"topic": topic, "key": key, "value": value})
        future = Future()
        future.set_result(MagicMock())
        return future

    def flush(self, timeout: Optional[float] = None) -> None:
        """Mock flush."""
        self.flushed = True

    def close(self, timeout: Optional[float] = None) -> None:
        """Mock close."""
        self.closed = True


class MockKafkaConsumer:
    """Mock KafkaConsumer for testing."""

    def __init__(self, *topics: str, **kwargs: Any):
        self.topics = topics
        self.kwargs = kwargs
        self.closed = False
        self._messages: List[MockMessage] = []
        self._message_index = 0
        self._lock = threading.Lock()
        self._value_deserializer = kwargs.get(
            "value_deserializer", lambda m: json.loads(m.decode("utf-8"))
        )

    def inject_message(self, message: Dict[str, Any]) -> None:
        """Inject a message to be consumed."""
        with self._lock:
            self._messages.append(MockMessage(value=message))

    def __iter__(self) -> "MockKafkaConsumer":
        return self

    def __next__(self) -> MockMessage:
        """Return next message or raise StopIteration."""
        with self._lock:
            if self._message_index < len(self._messages):
                msg = self._messages[self._message_index]
                self._message_index += 1
                return msg
        # Raise StopIteration to simulate consumer_timeout_ms behavior
        raise StopIteration

    def close(self) -> None:
        """Mock close."""
        self.closed = True


@pytest.fixture
def mock_kafka():
    """Fixture that patches Kafka classes."""
    with (
        patch("libs.control.KafkaProducer", MockKafkaProducer),
        patch("libs.control.KafkaConsumer", MockKafkaConsumer),
    ):
        yield


@pytest.fixture
def mock_config():
    """Fixture that provides a mock Config."""
    config = MagicMock()
    config.get.side_effect = lambda key, default=None: {
        "KAFKA_BOOTSTRAP_SERVERS": "localhost:9092",
        "CONTROL_INPUT_TOPIC": "control_in",
        "CONTROL_OUTPUT_TOPIC": "control_out",
    }.get(key, default)
    return config


@pytest.fixture
def client(mock_config):
    """Fixture that provides a ControlClient with mocked Kafka."""
    # Create mock producer and consumer
    mock_producer = MockKafkaProducer()
    mock_consumer = MockKafkaConsumer("control_out")

    with (
        patch("kafka.KafkaProducer", return_value=mock_producer),
        patch("kafka.KafkaConsumer", return_value=mock_consumer),
    ):
        client = ControlClient(
            config=mock_config,
            client_id="test-client",
        )
        # Store mocks for test access
        client._test_mock_producer = mock_producer
        client._test_mock_consumer = mock_consumer
        yield client
        client.close()


class TestControlClientInit:
    """Tests for ControlClient initialization."""

    def test_init_with_defaults(self, mock_config):
        """Test initialization with default values."""
        with (
            patch("kafka.KafkaProducer", MockKafkaProducer),
            patch("kafka.KafkaConsumer", MockKafkaConsumer),
        ):
            client = ControlClient(config=mock_config)
            assert client._bootstrap_servers == "localhost:9092"
            assert client._input_topic == "control_in"
            assert client._output_topic == "control_out"
            assert not client._closed
            client.close()

    def test_init_with_custom_values(self, mock_config):
        """Test initialization with custom values."""
        with (
            patch("kafka.KafkaProducer", MockKafkaProducer),
            patch("kafka.KafkaConsumer", MockKafkaConsumer),
        ):
            client = ControlClient(
                config=mock_config,
                client_id="custom-client",
                bootstrap_servers="kafka:9093",
                input_topic="my_input",
                output_topic="my_output",
            )
            assert client._client_id == "custom-client"
            assert client._bootstrap_servers == "kafka:9093"
            assert client._input_topic == "my_input"
            assert client._output_topic == "my_output"
            client.close()


class TestPendingRequest:
    """Tests for PendingRequest dataclass."""

    def test_pending_request_creation(self):
        """Test creating a PendingRequest."""
        pending = PendingRequest(
            request_id="ctrl_123",
            command="test-command",
            created_at=datetime.utcnow(),
            timeout_ms=5000,
            expects_response=True,
        )
        assert pending.request_id == "ctrl_123"
        assert pending.command == "test-command"
        assert pending.timeout_ms == 5000
        assert pending.expects_response is True
        assert pending.response is None
        assert pending.error is None
        assert not pending.response_event.is_set()

    def test_pending_request_with_callbacks(self):
        """Test PendingRequest with callbacks."""
        callback = Mock()
        error_callback = Mock()
        pending = PendingRequest(
            request_id="ctrl_456",
            command="test-command",
            created_at=datetime.utcnow(),
            timeout_ms=5000,
            expects_response=True,
            callback=callback,
            error_callback=error_callback,
        )
        assert pending.callback is callback
        assert pending.error_callback is error_callback


class TestSendMethod:
    """Tests for the send() fire-and-forget method."""

    def test_send_basic(self, mock_config):
        """Test basic send functionality."""
        mock_producer = MockKafkaProducer()

        with patch("kafka.KafkaProducer", return_value=mock_producer):
            client = ControlClient(config=mock_config, client_id="test-send")
            request_id = client.send("test-command", {"key": "value"})

            # Check request was sent
            assert len(mock_producer.messages) == 1
            msg = mock_producer.messages[0]
            assert msg["topic"] == "control_in"
            assert msg["value"]["command"] == "test-command"
            assert msg["value"]["payload"] == {"key": "value"}
            assert msg["value"]["expects_response"] is False
            assert request_id.startswith("ctrl_")
            assert mock_producer.flushed
            client.close()

    def test_send_without_payload(self, mock_config):
        """Test send with no payload."""
        mock_producer = MockKafkaProducer()

        with patch("kafka.KafkaProducer", return_value=mock_producer):
            client = ControlClient(config=mock_config)
            request_id = client.send("simple-command")

            assert len(mock_producer.messages) == 1
            assert mock_producer.messages[0]["value"]["payload"] == {}
            assert request_id.startswith("ctrl_")
            client.close()

    def test_send_when_closed_raises_error(self, mock_config):
        """Test that send raises ControlError when client is closed."""
        mock_producer = MockKafkaProducer()

        with patch("kafka.KafkaProducer", return_value=mock_producer):
            client = ControlClient(config=mock_config)
            client.close()

            with pytest.raises(ControlError, match="Client is closed"):
                client.send("test-command")


class TestCallSyncMethod:
    """Tests for the call_sync() synchronous method."""

    def test_call_sync_with_response(self, mock_config):
        """Test call_sync receives response correctly."""
        mock_producer = MockKafkaProducer()
        mock_consumer = MockKafkaConsumer("control_out")

        with (
            patch("kafka.KafkaProducer", return_value=mock_producer),
            patch("kafka.KafkaConsumer", return_value=mock_consumer),
        ):
            client = ControlClient(config=mock_config, client_id="test-sync")

            # Simulate response injection in a separate thread
            def inject_response():
                # Wait for message to be sent
                time.sleep(0.1)
                if mock_producer.messages:
                    request_id = mock_producer.messages[0]["value"]["id"]
                    response = {
                        "id": "resp_123",
                        "request_id": request_id,
                        "command": "test-command",
                        "success": True,
                        "result": {"data": "test"},
                        "error": None,
                        "handler_duration_ms": 10,
                    }
                    # Directly call _process_response to simulate response
                    client._process_response(request_id, response)

            thread = threading.Thread(target=inject_response, daemon=True)
            thread.start()

            result = client.call_sync("test-command", {"param": 1}, timeout=2.0)

            assert result["success"] is True
            assert result["result"] == {"data": "test"}
            assert result["command"] == "test-command"

            thread.join(timeout=1.0)
            client.close()

    def test_call_sync_timeout(self, mock_config):
        """Test call_sync raises TimeoutError when no response."""
        mock_producer = MockKafkaProducer()
        mock_consumer = MockKafkaConsumer("control_out")

        with (
            patch("kafka.KafkaProducer", return_value=mock_producer),
            patch("kafka.KafkaConsumer", return_value=mock_consumer),
        ):
            client = ControlClient(config=mock_config)

            # No response injected, should timeout
            with pytest.raises(TimeoutError):
                client.call_sync("test-command", timeout=0.1)

            client.close()

    def test_call_sync_when_closed_raises_error(self, mock_config):
        """Test that call_sync raises ControlError when client is closed."""
        mock_producer = MockKafkaProducer()
        mock_consumer = MockKafkaConsumer("control_out")

        with (
            patch("kafka.KafkaProducer", return_value=mock_producer),
            patch("kafka.KafkaConsumer", return_value=mock_consumer),
        ):
            client = ControlClient(config=mock_config)
            client.close()

            with pytest.raises(ControlError, match="Client is closed"):
                client.call_sync("test-command")

    def test_call_sync_registers_pending_before_send(self, mock_config):
        """Test that pending request is registered before Kafka send."""
        mock_producer = MockKafkaProducer()
        mock_consumer = MockKafkaConsumer("control_out")
        pending_registered_before_send = []

        original_send = mock_producer.send

        def capture_send(*args, **kwargs):
            # Check if pending is registered at time of send
            pending_count = len(client._pending)
            pending_registered_before_send.append(pending_count > 0)
            return original_send(*args, **kwargs)

        mock_producer.send = capture_send

        with (
            patch("kafka.KafkaProducer", return_value=mock_producer),
            patch("kafka.KafkaConsumer", return_value=mock_consumer),
        ):
            client = ControlClient(config=mock_config)

            # Inject immediate response
            def inject_immediate():
                time.sleep(0.05)
                if mock_producer.messages:
                    request_id = mock_producer.messages[0]["value"]["id"]
                    client._process_response(
                        request_id,
                        {
                            "request_id": request_id,
                            "success": True,
                            "result": {},
                        },
                    )

            thread = threading.Thread(target=inject_immediate, daemon=True)
            thread.start()

            client.call_sync("test", timeout=1.0)

            # Verify pending was registered before send
            assert len(pending_registered_before_send) > 0
            assert pending_registered_before_send[0] is True

            thread.join(timeout=1.0)
            client.close()


class TestCallAsyncMethod:
    """Tests for the call_async() asynchronous method."""

    def test_call_async_returns_request_id(self, mock_config):
        """Test that call_async returns request ID immediately."""
        mock_producer = MockKafkaProducer()
        mock_consumer = MockKafkaConsumer("control_out")

        with (
            patch("kafka.KafkaProducer", return_value=mock_producer),
            patch("kafka.KafkaConsumer", return_value=mock_consumer),
        ):
            client = ControlClient(config=mock_config)

            request_id = client.call_async("test-command")

            assert request_id.startswith("ctrl_")
            assert len(mock_producer.messages) == 1
            client.close()

    def test_call_async_callback_on_success(self, mock_config):
        """Test that callback is called on successful response."""
        mock_producer = MockKafkaProducer()
        mock_consumer = MockKafkaConsumer("control_out")

        with (
            patch("kafka.KafkaProducer", return_value=mock_producer),
            patch("kafka.KafkaConsumer", return_value=mock_consumer),
        ):
            client = ControlClient(config=mock_config)

            callback_result = []
            callback_event = threading.Event()

            def on_success(response: Dict[str, Any]) -> None:
                callback_result.append(response)
                callback_event.set()

            request_id = client.call_async(
                "test-command", callback=on_success, timeout=2.0
            )

            # Inject response
            response = {
                "request_id": request_id,
                "command": "test-command",
                "success": True,
                "result": {"data": "async-test"},
            }
            client._process_response(request_id, response)

            # Wait for callback
            callback_event.wait(timeout=2.0)

            assert len(callback_result) == 1
            assert callback_result[0]["success"] is True
            assert callback_result[0]["result"] == {"data": "async-test"}
            client.close()

    def test_call_async_callback_runs_in_thread_pool(self, mock_config):
        """Test that callback executes in thread pool, not main thread."""
        mock_producer = MockKafkaProducer()
        mock_consumer = MockKafkaConsumer("control_out")

        with (
            patch("kafka.KafkaProducer", return_value=mock_producer),
            patch("kafka.KafkaConsumer", return_value=mock_consumer),
        ):
            client = ControlClient(config=mock_config)

            callback_thread_name = []
            callback_event = threading.Event()
            main_thread = threading.current_thread()

            def on_success(response: Dict[str, Any]) -> None:
                callback_thread_name.append(threading.current_thread().name)
                callback_event.set()

            request_id = client.call_async(
                "test-command", callback=on_success, timeout=2.0
            )

            # Inject response
            client._process_response(
                request_id,
                {"request_id": request_id, "success": True, "result": {}},
            )

            callback_event.wait(timeout=2.0)

            # Verify callback ran in a different thread
            assert len(callback_thread_name) == 1
            assert callback_thread_name[0] != main_thread.name
            # Thread pool threads have names like "ThreadPoolExecutor-0_0"
            assert (
                "ThreadPoolExecutor" in callback_thread_name[0]
                or callback_thread_name[0] != main_thread.name
            )
            client.close()

    def test_call_async_error_callback_on_timeout(self, mock_config):
        """Test that error_callback is called on timeout."""
        mock_producer = MockKafkaProducer()
        mock_consumer = MockKafkaConsumer("control_out")

        with (
            patch("kafka.KafkaProducer", return_value=mock_producer),
            patch("kafka.KafkaConsumer", return_value=mock_consumer),
        ):
            client = ControlClient(config=mock_config)

            error_result = []
            error_event = threading.Event()

            def on_error(error: Exception) -> None:
                error_result.append(error)
                error_event.set()

            client.call_async(
                "test-command",
                error_callback=on_error,
                timeout=0.1,  # Short timeout
            )

            # Wait for timeout callback
            error_event.wait(timeout=2.0)

            assert len(error_result) == 1
            assert isinstance(error_result[0], TimeoutError)
            client.close()

    def test_call_async_error_callback_on_failure_response(self, mock_config):
        """Test that error_callback is called when response indicates failure."""
        mock_producer = MockKafkaProducer()
        mock_consumer = MockKafkaConsumer("control_out")

        with (
            patch("kafka.KafkaProducer", return_value=mock_producer),
            patch("kafka.KafkaConsumer", return_value=mock_consumer),
        ):
            client = ControlClient(config=mock_config)

            error_result = []
            error_event = threading.Event()

            def on_error(error: Exception) -> None:
                error_result.append(error)
                error_event.set()

            request_id = client.call_async(
                "test-command", error_callback=on_error, timeout=5.0
            )

            # Inject failure response
            client._process_response(
                request_id,
                {
                    "request_id": request_id,
                    "success": False,
                    "error": "Command failed",
                },
            )

            error_event.wait(timeout=2.0)

            assert len(error_result) == 1
            assert isinstance(error_result[0], ControlError)
            assert "Command failed" in str(error_result[0])
            client.close()


class TestRaceConditions:
    """Tests for race condition handling."""

    def test_response_arrives_before_send_returns(self, mock_config):
        """Test that response arriving before send returns still works."""
        mock_producer = MockKafkaProducer()
        mock_consumer = MockKafkaConsumer("control_out")

        # Make send trigger immediate response
        original_send = mock_producer.send

        def send_with_immediate_response(*args, **kwargs):
            result = original_send(*args, **kwargs)
            # Immediately inject response after send
            request_value = args[2] if len(args) > 2 else kwargs.get("value", {})
            request_id = request_value.get("id")
            if request_id:
                # This simulates a very fast response
                threading.Thread(
                    target=lambda: client._process_response(
                        request_id,
                        {"request_id": request_id, "success": True, "result": {}},
                    ),
                    daemon=True,
                ).start()
            return result

        mock_producer.send = send_with_immediate_response

        with (
            patch("kafka.KafkaProducer", return_value=mock_producer),
            patch("kafka.KafkaConsumer", return_value=mock_consumer),
        ):
            client = ControlClient(config=mock_config)

            # This should work even though response might arrive
            # before call_sync returns from send
            result = client.call_sync("fast-command", timeout=2.0)

            assert result["success"] is True
            client.close()

    def test_concurrent_requests_handled_correctly(self, mock_config):
        """Test that concurrent requests with different IDs are handled correctly."""
        mock_producer = MockKafkaProducer()
        mock_consumer = MockKafkaConsumer("control_out")

        with (
            patch("kafka.KafkaProducer", return_value=mock_producer),
            patch("kafka.KafkaConsumer", return_value=mock_consumer),
        ):
            client = ControlClient(config=mock_config)

            results = {}
            events = {}
            lock = threading.Lock()

            def make_callback(name: str) -> Callable[[Dict[str, Any]], None]:
                def callback(response: Dict[str, Any]) -> None:
                    with lock:
                        results[name] = response
                    events[name].set()

                return callback

            # Start multiple async requests
            request_ids = {}
            for i in range(3):
                name = f"request_{i}"
                events[name] = threading.Event()
                request_ids[name] = client.call_async(
                    f"command-{i}",
                    payload={"index": i},
                    callback=make_callback(name),
                    timeout=5.0,
                )

            # Inject responses in random order
            for name in ["request_2", "request_0", "request_1"]:
                req_id = request_ids[name]
                idx = int(name.split("_")[1])
                client._process_response(
                    req_id,
                    {
                        "request_id": req_id,
                        "success": True,
                        "result": {"index": idx, "doubled": idx * 2},
                    },
                )

            # Wait for all callbacks
            for name, event in events.items():
                event.wait(timeout=2.0)

            # Verify each request got correct response
            assert len(results) == 3
            for i in range(3):
                name = f"request_{i}"
                assert results[name]["result"]["index"] == i
                assert results[name]["result"]["doubled"] == i * 2

            client.close()


class TestCloseMethod:
    """Tests for the close() method."""

    def test_close_cleans_up_properly(self, mock_config):
        """Test that close cleans up all resources."""
        mock_producer = MockKafkaProducer()
        mock_consumer = MockKafkaConsumer("control_out")

        with (
            patch("kafka.KafkaProducer", return_value=mock_producer),
            patch("kafka.KafkaConsumer", return_value=mock_consumer),
        ):
            client = ControlClient(config=mock_config)

            # Start listener
            client.call_async("test-command", timeout=10.0)

            # Close client
            client.close()

            assert client._closed is True
            assert client._shutdown_event.is_set()

    def test_close_is_idempotent(self, mock_config):
        """Test that close can be called multiple times safely."""
        mock_producer = MockKafkaProducer()
        mock_consumer = MockKafkaConsumer("control_out")

        with (
            patch("kafka.KafkaProducer", return_value=mock_producer),
            patch("kafka.KafkaConsumer", return_value=mock_consumer),
        ):
            client = ControlClient(config=mock_config)
            client.send("test")

            # Close multiple times
            client.close()
            client.close()
            client.close()

            assert client._closed is True

    def test_close_cancels_pending_timeouts(self, mock_config):
        """Test that close cancels all pending timeout timers."""
        mock_producer = MockKafkaProducer()
        mock_consumer = MockKafkaConsumer("control_out")

        with (
            patch("kafka.KafkaProducer", return_value=mock_producer),
            patch("kafka.KafkaConsumer", return_value=mock_consumer),
        ):
            client = ControlClient(config=mock_config)

            # Start async requests with long timeouts
            for i in range(3):
                client.call_async(f"command-{i}", timeout=60.0)

            # Verify pending requests exist
            assert len(client._pending) == 3

            # Close should cancel timers and clear pending
            client.close()

            assert len(client._pending) == 0

    def test_context_manager(self, mock_config):
        """Test that ControlClient works as context manager."""
        mock_producer = MockKafkaProducer()
        mock_consumer = MockKafkaConsumer("control_out")

        with (
            patch("kafka.KafkaProducer", return_value=mock_producer),
            patch("kafka.KafkaConsumer", return_value=mock_consumer),
        ):
            with ControlClient(config=mock_config) as client:
                client.send("test-command")
                assert not client._closed

            # After context exit, client should be closed
            assert client._closed


class TestSingletonFunction:
    """Tests for the get_control_client singleton function."""

    def test_get_control_client_returns_singleton(self, mock_config):
        """Test that get_control_client returns the same instance."""
        mock_producer = MockKafkaProducer()
        mock_consumer = MockKafkaConsumer("control_out")

        with (
            patch("kafka.KafkaProducer", return_value=mock_producer),
            patch("kafka.KafkaConsumer", return_value=mock_consumer),
        ):
            # Reset singleton first
            reset_default_client()

            client1 = get_control_client(config=mock_config)
            client2 = get_control_client(config=mock_config)

            assert client1 is client2

            reset_default_client()

    def test_reset_default_client(self, mock_config):
        """Test that reset_default_client closes and resets singleton."""
        mock_producer = MockKafkaProducer()
        mock_consumer = MockKafkaConsumer("control_out")

        with (
            patch("kafka.KafkaProducer", return_value=mock_producer),
            patch("kafka.KafkaConsumer", return_value=mock_consumer),
        ):
            reset_default_client()

            client1 = get_control_client(config=mock_config)
            reset_default_client()
            client2 = get_control_client(config=mock_config)

            assert client1 is not client2
            assert client1._closed is True

            reset_default_client()


class TestMessageFormat:
    """Tests for message format compliance."""

    def test_request_message_format(self, mock_config):
        """Test that request messages have correct format."""
        mock_producer = MockKafkaProducer()

        with patch("kafka.KafkaProducer", return_value=mock_producer):
            client = ControlClient(config=mock_config, client_id="format-test")
            client.send("test-command", {"key": "value"})

            msg = mock_producer.messages[0]["value"]

            # Check required fields
            assert "id" in msg
            assert msg["id"].startswith("ctrl_")
            assert msg["command"] == "test-command"
            assert msg["payload"] == {"key": "value"}
            assert "timestamp" in msg
            assert msg["expects_response"] is False
            assert msg["source"] == "format-test"

            # Validate timestamp format (ISO 8601)
            datetime.fromisoformat(msg["timestamp"])

            client.close()

    def test_request_id_uniqueness(self, mock_config):
        """Test that request IDs are unique."""
        mock_producer = MockKafkaProducer()

        with patch("kafka.KafkaProducer", return_value=mock_producer):
            client = ControlClient(config=mock_config)

            request_ids = set()
            for _ in range(100):
                req_id = client.send("test-command")
                request_ids.add(req_id)

            # All IDs should be unique
            assert len(request_ids) == 100

            client.close()


class TestErrorHandling:
    """Tests for error handling."""

    def test_unknown_request_id_response_ignored(self, mock_config):
        """Test that responses for unknown request IDs are ignored."""
        mock_producer = MockKafkaProducer()
        mock_consumer = MockKafkaConsumer("control_out")

        with (
            patch("kafka.KafkaProducer", return_value=mock_producer),
            patch("kafka.KafkaConsumer", return_value=mock_consumer),
        ):
            client = ControlClient(config=mock_config)

            # Process response for unknown request
            client._process_response(
                "unknown_request_id",
                {"request_id": "unknown_request_id", "success": True},
            )

            # Should not raise, just log debug
            client.close()

    def test_response_without_request_id_logged(self, mock_config):
        """Test that responses without request_id are logged but don't crash."""
        mock_producer = MockKafkaProducer()
        mock_consumer = MockKafkaConsumer("control_out")

        with (
            patch("kafka.KafkaProducer", return_value=mock_producer),
            patch("kafka.KafkaConsumer", return_value=mock_consumer),
        ):
            client = ControlClient(config=mock_config)

            # Process response without request_id
            # This tests the listener loop handling
            client._process_response(None, {"success": True})

            client.close()


class TestControlError:
    """Tests for ControlError exception."""

    def test_control_error_inheritance(self):
        """Test that ControlError inherits from Exception."""
        error = ControlError("test error")
        assert isinstance(error, Exception)
        assert str(error) == "test error"

    def test_control_error_raised_on_send_failure(self, mock_config):
        """Test that ControlError is raised when send fails."""
        mock_producer = MockKafkaProducer()

        def failing_send(*args, **kwargs):
            raise Exception("Kafka send failed")

        mock_producer.send = failing_send

        with patch("kafka.KafkaProducer", return_value=mock_producer):
            client = ControlClient(config=mock_config)

            with pytest.raises(ControlError, match="Failed to send command"):
                client.send("test-command")

            client.close()
