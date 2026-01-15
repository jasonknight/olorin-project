#!/usr/bin/env python3
"""
Unit tests for ControlController class.

Tests controller initialization, command execution, message handling,
and error scenarios with mocked Kafka dependencies.
"""

import json
import os
import sys
import time
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_controller_config():
    """Mock ControllerConfig with test values."""
    config = MagicMock()
    config.kafka_bootstrap_servers = "localhost:9092"
    config.input_topic = "test-control-in"
    config.output_topic = "test-control-out"
    config.consumer_group = "test-control-group"
    config.handler_timeout = 30
    config.callback_pool_size = 2
    config.log_level = "INFO"
    config.reload = Mock(return_value=False)
    return config


@pytest.fixture
def mock_kafka_consumer():
    """Mock KafkaConsumer that can yield test messages."""
    consumer = MagicMock()
    consumer.__iter__ = Mock(return_value=iter([]))
    consumer.close = Mock()
    return consumer


@pytest.fixture
def mock_kafka_producer():
    """Mock KafkaProducer that captures sent messages."""
    producer = MagicMock()
    producer.sent_messages = []

    def capture_send(topic, value=None):
        producer.sent_messages.append({"topic": topic, "value": value})
        future = MagicMock()
        future.get.return_value = MagicMock(
            topic=topic, partition=0, offset=len(producer.sent_messages)
        )
        return future

    producer.send = Mock(side_effect=capture_send)
    producer.flush = Mock()
    producer.close = Mock()
    return producer


@pytest.fixture
def sample_control_request():
    """Sample control request message."""
    return {
        "id": "ctrl_20260115_123456_abc123",
        "command": "stop-broca-audio-play",
        "payload": {"force": False},
        "timestamp": "2026-01-15T10:30:45.123456",
        "expects_response": True,
        "reply_to": "control_out",
        "source": "test-client",
    }


@pytest.fixture
def sample_kafka_message(sample_control_request):
    """Sample Kafka message wrapping a control request."""
    message = MagicMock()
    message.topic = "control_in"
    message.partition = 0
    message.offset = 1
    message.timestamp = int(time.time() * 1000)
    message.value = json.dumps(sample_control_request)
    return message


# =============================================================================
# TestControlControllerInit
# =============================================================================


class TestControlControllerInit:
    """Tests for ControlController initialization."""

    def test_init_with_defaults(self, mock_controller_config):
        """Should initialize with default configuration values."""
        with (
            patch("controller.KafkaConsumer") as mock_consumer_cls,
            patch("controller.KafkaProducer") as mock_producer_cls,
            patch("controller.discover_handlers") as mock_discover,
        ):
            mock_discover.return_value = {}
            mock_consumer_cls.return_value = MagicMock()
            mock_producer_cls.return_value = MagicMock()

            from controller import ControlController

            controller = ControlController(mock_controller_config)

            # Verify consumer created with correct topic
            mock_consumer_cls.assert_called_once()
            call_args = mock_consumer_cls.call_args
            assert call_args[0][0] == "test-control-in"

            # Verify producer created
            mock_producer_cls.assert_called_once()

            # Verify config stored
            assert controller.config == mock_controller_config

    def test_discovers_handlers(self, mock_controller_config):
        """Should discover handlers during initialization."""
        with (
            patch("controller.KafkaConsumer"),
            patch("controller.KafkaProducer"),
            patch("controller.discover_handlers") as mock_discover,
        ):
            mock_discover.return_value = {
                "stop-broca-audio-play": Mock(),
                "reload-config": Mock(),
            }

            from controller import ControlController

            controller = ControlController(mock_controller_config)

            mock_discover.assert_called_once()
            assert len(controller.handlers) == 2
            assert "stop-broca-audio-play" in controller.handlers
            assert "reload-config" in controller.handlers

    def test_init_creates_thread_pool(self, mock_controller_config):
        """Should create thread pool with configured size."""
        with (
            patch("controller.KafkaConsumer"),
            patch("controller.KafkaProducer"),
            patch("controller.discover_handlers", return_value={}),
        ):
            from controller import ControlController

            controller = ControlController(mock_controller_config)

            # ThreadPoolExecutor should be created
            assert controller.executor is not None
            assert controller.executor._max_workers == 2  # callback_pool_size


# =============================================================================
# TestExecuteCommand
# =============================================================================


class TestExecuteCommand:
    """Tests for _execute_command method."""

    def test_dispatches_to_correct_handler(
        self, mock_controller_config, mock_kafka_producer, sample_control_request
    ):
        """Should dispatch command to the correct handler."""
        mock_handler = Mock(return_value={"pid_killed": 12345, "was_playing": True})

        with (
            patch("controller.KafkaConsumer"),
            patch("controller.KafkaProducer", return_value=mock_kafka_producer),
            patch("controller.discover_handlers", return_value={}),
            patch("controller.get_handler", return_value=mock_handler),
        ):
            from controller import ControlController

            controller = ControlController(mock_controller_config)
            controller.producer = mock_kafka_producer

            controller._execute_command(sample_control_request)

            mock_handler.assert_called_once_with({"force": False})

    def test_success_response_format(
        self, mock_controller_config, mock_kafka_producer, sample_control_request
    ):
        """Should produce response with all required fields."""
        mock_handler = Mock(return_value={"result_key": "result_value"})

        with (
            patch("controller.KafkaConsumer"),
            patch("controller.KafkaProducer", return_value=mock_kafka_producer),
            patch("controller.discover_handlers", return_value={}),
            patch("controller.get_handler", return_value=mock_handler),
        ):
            from controller import ControlController

            controller = ControlController(mock_controller_config)
            controller.producer = mock_kafka_producer

            controller._execute_command(sample_control_request)

            # Verify response was sent
            assert len(mock_kafka_producer.sent_messages) == 1
            response = mock_kafka_producer.sent_messages[0]["value"]

            # Check all required fields
            assert "id" in response
            assert response["id"].startswith("resp_")
            assert response["request_id"] == "ctrl_20260115_123456_abc123"
            assert response["command"] == "stop-broca-audio-play"
            assert response["success"] is True
            assert response["result"] == {"result_key": "result_value"}
            assert response["error"] is None
            assert "timestamp" in response
            assert "handler_duration_ms" in response
            assert isinstance(response["handler_duration_ms"], float)

    def test_error_response_for_unknown_command(
        self, mock_controller_config, mock_kafka_producer
    ):
        """Should produce error response for unknown command."""
        with (
            patch("controller.KafkaConsumer"),
            patch("controller.KafkaProducer", return_value=mock_kafka_producer),
            patch("controller.discover_handlers", return_value={}),
        ):
            from controller import ControlController

            controller = ControlController(mock_controller_config)
            controller.producer = mock_kafka_producer

            request = {
                "id": "req_001",
                "command": "unknown-command",
                "payload": {},
                "expects_response": True,
                "reply_to": "control_out",
            }

            controller._execute_command(request)

            assert len(mock_kafka_producer.sent_messages) == 1
            response = mock_kafka_producer.sent_messages[0]["value"]

            assert response["success"] is False
            assert response["result"] is None
            assert "Unknown command" in response["error"]

    def test_error_response_for_handler_exception(
        self, mock_controller_config, mock_kafka_producer
    ):
        """Should produce error response when handler raises exception."""
        mock_handler = Mock(side_effect=RuntimeError("Handler crashed!"))

        with (
            patch("controller.KafkaConsumer"),
            patch("controller.KafkaProducer", return_value=mock_kafka_producer),
            patch("controller.discover_handlers", return_value={}),
            patch("controller.get_handler", return_value=mock_handler),
        ):
            from controller import ControlController

            controller = ControlController(mock_controller_config)
            controller.producer = mock_kafka_producer

            request = {
                "id": "req_002",
                "command": "test-command",
                "payload": {},
                "expects_response": True,
                "reply_to": "control_out",
            }

            controller._execute_command(request)

            assert len(mock_kafka_producer.sent_messages) == 1
            response = mock_kafka_producer.sent_messages[0]["value"]

            assert response["success"] is False
            assert response["result"] is None
            assert "RuntimeError" in response["error"]
            assert "Handler crashed" in response["error"]

    def test_no_response_when_expects_response_false(
        self, mock_controller_config, mock_kafka_producer
    ):
        """Should not send response when expects_response is False."""
        mock_handler = Mock(return_value={"done": True})

        with (
            patch("controller.KafkaConsumer"),
            patch("controller.KafkaProducer", return_value=mock_kafka_producer),
            patch("controller.discover_handlers", return_value={}),
            patch("controller.get_handler", return_value=mock_handler),
        ):
            from controller import ControlController

            controller = ControlController(mock_controller_config)
            controller.producer = mock_kafka_producer

            request = {
                "id": "req_003",
                "command": "test-command",
                "payload": {},
                "expects_response": False,  # No response expected
                "reply_to": "control_out",
            }

            controller._execute_command(request)

            # Handler should still be called
            mock_handler.assert_called_once()

            # But no response sent
            assert len(mock_kafka_producer.sent_messages) == 0

    def test_uses_reply_to_topic(self, mock_controller_config, mock_kafka_producer):
        """Should send response to reply_to topic if specified."""
        mock_handler = Mock(return_value={"status": "ok"})

        with (
            patch("controller.KafkaConsumer"),
            patch("controller.KafkaProducer", return_value=mock_kafka_producer),
            patch("controller.discover_handlers", return_value={}),
            patch("controller.get_handler", return_value=mock_handler),
        ):
            from controller import ControlController

            controller = ControlController(mock_controller_config)
            controller.producer = mock_kafka_producer

            request = {
                "id": "req_004",
                "command": "test-command",
                "payload": {},
                "expects_response": True,
                "reply_to": "custom-reply-topic",  # Custom reply topic
            }

            controller._execute_command(request)

            assert len(mock_kafka_producer.sent_messages) == 1
            assert mock_kafka_producer.sent_messages[0]["topic"] == "custom-reply-topic"

    def test_handler_duration_ms_calculated(
        self, mock_controller_config, mock_kafka_producer
    ):
        """Should calculate handler duration in milliseconds."""

        def slow_handler(payload):
            time.sleep(0.1)  # Sleep 100ms
            return {"done": True}

        with (
            patch("controller.KafkaConsumer"),
            patch("controller.KafkaProducer", return_value=mock_kafka_producer),
            patch("controller.discover_handlers", return_value={}),
            patch("controller.get_handler", return_value=slow_handler),
        ):
            from controller import ControlController

            controller = ControlController(mock_controller_config)
            controller.producer = mock_kafka_producer

            request = {
                "id": "req_005",
                "command": "slow-command",
                "payload": {},
                "expects_response": True,
                "reply_to": "control_out",
            }

            controller._execute_command(request)

            response = mock_kafka_producer.sent_messages[0]["value"]

            # Duration should be at least 100ms
            assert response["handler_duration_ms"] >= 100

    def test_defaults_reply_to_output_topic(
        self, mock_controller_config, mock_kafka_producer
    ):
        """Should default reply_to to output_topic if not specified."""
        mock_handler = Mock(return_value={})

        with (
            patch("controller.KafkaConsumer"),
            patch("controller.KafkaProducer", return_value=mock_kafka_producer),
            patch("controller.discover_handlers", return_value={}),
            patch("controller.get_handler", return_value=mock_handler),
        ):
            from controller import ControlController

            controller = ControlController(mock_controller_config)
            controller.producer = mock_kafka_producer

            # Request without reply_to
            request = {
                "id": "req_006",
                "command": "test-command",
                "payload": {},
                "expects_response": True,
                # No reply_to specified
            }

            controller._execute_command(request)

            # Should use default output topic
            assert mock_kafka_producer.sent_messages[0]["topic"] == "test-control-out"


# =============================================================================
# TestParseRequest
# =============================================================================


class TestParseRequest:
    """Tests for _parse_request method."""

    def test_parses_valid_json(self, mock_controller_config):
        """Should parse valid JSON request."""
        with (
            patch("controller.KafkaConsumer"),
            patch("controller.KafkaProducer"),
            patch("controller.discover_handlers", return_value={}),
        ):
            from controller import ControlController

            controller = ControlController(mock_controller_config)

            message = '{"id": "test", "command": "test-cmd", "payload": {"key": "val"}}'
            result = controller._parse_request(message)

            assert result["id"] == "test"
            assert result["command"] == "test-cmd"
            assert result["payload"] == {"key": "val"}

    def test_raises_on_invalid_json(self, mock_controller_config):
        """Should raise ValueError on invalid JSON."""
        with (
            patch("controller.KafkaConsumer"),
            patch("controller.KafkaProducer"),
            patch("controller.discover_handlers", return_value={}),
        ):
            from controller import ControlController

            controller = ControlController(mock_controller_config)

            with pytest.raises(ValueError) as exc_info:
                controller._parse_request("not valid json {")

            assert "Invalid JSON" in str(exc_info.value)


# =============================================================================
# TestBuildResponse
# =============================================================================


class TestBuildResponse:
    """Tests for _build_response method."""

    def test_builds_success_response(self, mock_controller_config):
        """Should build success response correctly."""
        with (
            patch("controller.KafkaConsumer"),
            patch("controller.KafkaProducer"),
            patch("controller.discover_handlers", return_value={}),
        ):
            from controller import ControlController

            controller = ControlController(mock_controller_config)

            response = controller._build_response(
                request_id="req_123",
                command="test-cmd",
                success=True,
                result={"data": "value"},
                error=None,
                handler_duration_ms=42.5,
            )

            assert response["request_id"] == "req_123"
            assert response["command"] == "test-cmd"
            assert response["success"] is True
            assert response["result"] == {"data": "value"}
            assert response["error"] is None
            assert response["handler_duration_ms"] == 42.5
            assert response["id"].startswith("resp_")
            assert "timestamp" in response

    def test_builds_error_response(self, mock_controller_config):
        """Should build error response correctly."""
        with (
            patch("controller.KafkaConsumer"),
            patch("controller.KafkaProducer"),
            patch("controller.discover_handlers", return_value={}),
        ):
            from controller import ControlController

            controller = ControlController(mock_controller_config)

            response = controller._build_response(
                request_id="req_456",
                command="fail-cmd",
                success=False,
                result=None,
                error="Something went wrong",
                handler_duration_ms=5.0,
            )

            assert response["success"] is False
            assert response["result"] is None
            assert response["error"] == "Something went wrong"


# =============================================================================
# TestStart
# =============================================================================


class TestStart:
    """Tests for start method."""

    def test_consumes_from_control_in(
        self, mock_controller_config, mock_kafka_consumer, sample_kafka_message
    ):
        """Should consume messages from control_in topic."""
        # Make consumer yield one message then stop
        mock_kafka_consumer.__iter__ = Mock(return_value=iter([sample_kafka_message]))

        with (
            patch("controller.KafkaConsumer", return_value=mock_kafka_consumer),
            patch("controller.KafkaProducer"),
            patch(
                "controller.discover_handlers",
                return_value={"stop-broca-audio-play": Mock(return_value={})},
            ),
        ):
            from controller import ControlController

            controller = ControlController(mock_controller_config)
            controller._running = False  # Stop after first iteration

            # Make consumer stop after yielding one message
            def stop_after_one():
                yield sample_kafka_message
                controller._running = False

            mock_kafka_consumer.__iter__ = Mock(side_effect=stop_after_one)

            # Start will process one message then stop
            try:
                controller.start()
            except StopIteration:
                pass  # Expected when mock runs out

    def test_submits_to_thread_pool(
        self, mock_controller_config, mock_kafka_consumer, sample_kafka_message
    ):
        """Should submit messages to thread pool."""
        with (
            patch("controller.KafkaConsumer", return_value=mock_kafka_consumer),
            patch("controller.KafkaProducer"),
            patch(
                "controller.discover_handlers",
                return_value={"stop-broca-audio-play": Mock(return_value={})},
            ),
        ):
            from controller import ControlController

            controller = ControlController(mock_controller_config)

            # Mock executor
            mock_executor = MagicMock()
            controller.executor = mock_executor

            # Set up consumer to yield one message then stop
            messages_yielded = [False]

            def single_message_iter():
                if not messages_yielded[0]:
                    messages_yielded[0] = True
                    yield sample_kafka_message
                controller._running = False

            mock_kafka_consumer.__iter__ = Mock(side_effect=single_message_iter)

            try:
                controller.start()
            except StopIteration:
                pass

            # Verify executor.submit was called
            mock_executor.submit.assert_called()


# =============================================================================
# TestStop
# =============================================================================


class TestStop:
    """Tests for stop method."""

    def test_shuts_down_executor(self, mock_controller_config):
        """Should shut down thread pool executor."""
        with (
            patch("controller.KafkaConsumer"),
            patch("controller.KafkaProducer"),
            patch("controller.discover_handlers", return_value={}),
        ):
            from controller import ControlController

            controller = ControlController(mock_controller_config)

            # Mock executor
            mock_executor = MagicMock()
            controller.executor = mock_executor

            controller.stop()

            mock_executor.shutdown.assert_called_once_with(
                wait=True, cancel_futures=False
            )

    def test_closes_consumer_and_producer(
        self, mock_controller_config, mock_kafka_consumer, mock_kafka_producer
    ):
        """Should close both consumer and producer."""
        with (
            patch("controller.KafkaConsumer", return_value=mock_kafka_consumer),
            patch("controller.KafkaProducer", return_value=mock_kafka_producer),
            patch("controller.discover_handlers", return_value={}),
        ):
            from controller import ControlController

            controller = ControlController(mock_controller_config)

            controller.stop()

            mock_kafka_consumer.close.assert_called_once()
            mock_kafka_producer.flush.assert_called_once()
            mock_kafka_producer.close.assert_called_once()

    def test_sets_running_flag_false(self, mock_controller_config):
        """Should set _running flag to False."""
        with (
            patch("controller.KafkaConsumer"),
            patch("controller.KafkaProducer"),
            patch("controller.discover_handlers", return_value={}),
        ):
            from controller import ControlController

            controller = ControlController(mock_controller_config)
            controller._running = True

            controller.stop()

            assert controller._running is False


# =============================================================================
# TestIntegration
# =============================================================================


class TestIntegration:
    """Integration tests with realistic scenarios."""

    def test_full_command_execution_flow(
        self, mock_controller_config, mock_kafka_producer
    ):
        """Test complete flow: receive request, execute handler, send response."""
        handler_calls = []

        def test_handler(payload):
            handler_calls.append(payload)
            return {"processed": True, "payload_received": payload}

        with (
            patch("controller.KafkaConsumer"),
            patch("controller.KafkaProducer", return_value=mock_kafka_producer),
            patch("controller.discover_handlers", return_value={}),
            patch("controller.get_handler", return_value=test_handler),
        ):
            from controller import ControlController

            controller = ControlController(mock_controller_config)
            controller.producer = mock_kafka_producer

            request = {
                "id": "integration_001",
                "command": "test-handler",
                "payload": {"data": "test-data"},
                "expects_response": True,
                "reply_to": "test-reply",
            }

            controller._execute_command(request)

            # Verify handler was called with correct payload
            assert len(handler_calls) == 1
            assert handler_calls[0] == {"data": "test-data"}

            # Verify response was sent
            assert len(mock_kafka_producer.sent_messages) == 1
            response = mock_kafka_producer.sent_messages[0]["value"]
            assert response["success"] is True
            assert response["result"]["processed"] is True
            assert response["result"]["payload_received"] == {"data": "test-data"}

    def test_multiple_commands_in_sequence(
        self, mock_controller_config, mock_kafka_producer
    ):
        """Test multiple commands executed in sequence."""
        handler1_calls = []
        handler2_calls = []

        def handler1(payload):
            handler1_calls.append(payload)
            return {"handler": 1}

        def handler2(payload):
            handler2_calls.append(payload)
            return {"handler": 2}

        # Use side_effect to return different handlers for different commands
        def get_handler_side_effect(command):
            if command == "cmd-one":
                return handler1
            elif command == "cmd-two":
                return handler2
            raise KeyError(f"Unknown command: {command}")

        with (
            patch("controller.KafkaConsumer"),
            patch("controller.KafkaProducer", return_value=mock_kafka_producer),
            patch("controller.discover_handlers", return_value={}),
            patch("controller.get_handler", side_effect=get_handler_side_effect),
        ):
            from controller import ControlController

            controller = ControlController(mock_controller_config)
            controller.producer = mock_kafka_producer

            # Execute first command
            controller._execute_command(
                {
                    "id": "seq_001",
                    "command": "cmd-one",
                    "payload": {"seq": 1},
                    "expects_response": True,
                    "reply_to": "out",
                }
            )

            # Execute second command
            controller._execute_command(
                {
                    "id": "seq_002",
                    "command": "cmd-two",
                    "payload": {"seq": 2},
                    "expects_response": True,
                    "reply_to": "out",
                }
            )

            # Both handlers should be called
            assert len(handler1_calls) == 1
            assert len(handler2_calls) == 1

            # Both responses should be sent
            assert len(mock_kafka_producer.sent_messages) == 2

    def test_mixed_success_and_failure(
        self, mock_controller_config, mock_kafka_producer
    ):
        """Test handling mix of successful and failing commands."""

        def success_handler(payload):
            return {"status": "success"}

        def fail_handler(payload):
            raise ValueError("Intentional failure")

        # Use side_effect to return different handlers for different commands
        def get_handler_side_effect(command):
            if command == "success-cmd":
                return success_handler
            elif command == "fail-cmd":
                return fail_handler
            raise KeyError(f"Unknown command: {command}")

        with (
            patch("controller.KafkaConsumer"),
            patch("controller.KafkaProducer", return_value=mock_kafka_producer),
            patch("controller.discover_handlers", return_value={}),
            patch("controller.get_handler", side_effect=get_handler_side_effect),
        ):
            from controller import ControlController

            controller = ControlController(mock_controller_config)
            controller.producer = mock_kafka_producer

            # Successful command
            controller._execute_command(
                {
                    "id": "mix_001",
                    "command": "success-cmd",
                    "payload": {},
                    "expects_response": True,
                    "reply_to": "out",
                }
            )

            # Failing command
            controller._execute_command(
                {
                    "id": "mix_002",
                    "command": "fail-cmd",
                    "payload": {},
                    "expects_response": True,
                    "reply_to": "out",
                }
            )

            assert len(mock_kafka_producer.sent_messages) == 2

            # First should be success
            assert mock_kafka_producer.sent_messages[0]["value"]["success"] is True
            # Second should be failure
            assert mock_kafka_producer.sent_messages[1]["value"]["success"] is False


# =============================================================================
# TestHelperFunctions
# =============================================================================


class TestHelperFunctions:
    """Tests for module-level helper functions."""

    def test_generate_response_id_format(self):
        """Should generate response ID in expected format."""
        with (
            patch("controller.KafkaConsumer"),
            patch("controller.KafkaProducer"),
            patch("controller.discover_handlers", return_value={}),
        ):
            from controller import generate_response_id

            response_id = generate_response_id()

            assert response_id.startswith("resp_")
            # Format: resp_YYYYMMDD_HHMMSS_ffffff_hexhex
            parts = response_id.split("_")
            assert len(parts) == 5

    def test_generate_response_id_unique(self):
        """Should generate unique response IDs."""
        with (
            patch("controller.KafkaConsumer"),
            patch("controller.KafkaProducer"),
            patch("controller.discover_handlers", return_value={}),
        ):
            from controller import generate_response_id

            ids = [generate_response_id() for _ in range(100)]
            unique_ids = set(ids)

            # All IDs should be unique
            assert len(unique_ids) == 100


# =============================================================================
# TestControllerConfig
# =============================================================================


class TestControllerConfig:
    """Tests for ControllerConfig class."""

    def test_loads_defaults(self):
        """Should load default values when config not set."""
        with patch("controller.config") as mock_cfg:
            mock_cfg.get.return_value = None
            mock_cfg.get_int.return_value = None

            # Need to set up get to return defaults
            def get_side_effect(key, default=None):
                return default

            def get_int_side_effect(key, default=None):
                return default

            mock_cfg.get.side_effect = get_side_effect
            mock_cfg.get_int.side_effect = get_int_side_effect
            mock_cfg.reload.return_value = False

            from controller import ControllerConfig

            cfg = ControllerConfig(mock_cfg)

            assert cfg.input_topic == "control_in"
            assert cfg.output_topic == "control_out"
            assert cfg.consumer_group == "control-controller-group"
            assert cfg.handler_timeout == 30
            assert cfg.callback_pool_size == 4

    def test_reload_returns_true_on_change(self):
        """Should return True when config file changed."""
        mock_cfg = MagicMock()
        mock_cfg.get.return_value = "test"
        mock_cfg.get_int.return_value = 10
        mock_cfg.reload.return_value = True

        from controller import ControllerConfig

        cfg = ControllerConfig(mock_cfg)
        result = cfg.reload()

        assert result is True

    def test_reload_returns_false_when_unchanged(self):
        """Should return False when config file unchanged."""
        mock_cfg = MagicMock()
        mock_cfg.get.return_value = "test"
        mock_cfg.get_int.return_value = 10
        mock_cfg.reload.return_value = False

        from controller import ControllerConfig

        cfg = ControllerConfig(mock_cfg)
        result = cfg.reload()

        assert result is False
