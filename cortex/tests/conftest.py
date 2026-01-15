#!/usr/bin/env python3
"""
Pytest configuration and fixtures for cortex tests.

Provides mocks for external dependencies:
- Kafka (consumer/producer)
- OpenAI client (Exo API)
- Context store
"""

import pytest
import tempfile
import os
import sys
from unittest.mock import Mock, MagicMock
from datetime import datetime

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from libs.chat_store import ChatStore
from libs.context_store import ContextStore


# =============================================================================
# Database Fixtures
# =============================================================================


@pytest.fixture
def temp_db_path():
    """Fixture providing a temporary database path that gets cleaned up."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield os.path.join(tmpdir, "test.db")


@pytest.fixture
def chat_store(temp_db_path):
    """Fixture providing a ChatStore with temporary database."""
    return ChatStore(temp_db_path)


@pytest.fixture
def context_store():
    """Fixture providing a ContextStore with temporary database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "context.db")
        yield ContextStore(db_path)


# =============================================================================
# Kafka Mocks
# =============================================================================


@pytest.fixture
def mock_kafka_consumer():
    """Mock KafkaConsumer that yields test messages."""
    consumer = MagicMock()
    consumer.__iter__ = Mock(return_value=iter([]))
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


# =============================================================================
# OpenAI/Exo API Mocks
# =============================================================================


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client that returns controlled responses."""
    client = MagicMock()

    def create_completion(**kwargs):
        """Simulates streaming response from Exo."""
        response_text = kwargs.get(
            "_test_response", "This is a test response from the AI."
        )

        # Create mock streaming response
        chunks = []
        words = response_text.split()
        for i, word in enumerate(words):
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta = MagicMock()
            chunk.choices[0].delta.content = word + (" " if i < len(words) - 1 else "")
            chunks.append(chunk)

        return iter(chunks)

    client.chat.completions.create = Mock(side_effect=create_completion)
    return client


@pytest.fixture
def mock_openai_client_with_thinking():
    """Mock OpenAI client that returns response with thinking blocks."""
    client = MagicMock()

    def create_completion(**kwargs):
        response_text = "<think>Let me think about this...</think>The answer is 42."
        chunks = []

        # Simulate streaming character by character for thinking block detection
        for char in response_text:
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta = MagicMock()
            chunk.choices[0].delta.content = char
            chunks.append(chunk)

        return iter(chunks)

    client.chat.completions.create = Mock(side_effect=create_completion)
    return client


# =============================================================================
# Config Fixtures
# =============================================================================


@pytest.fixture
def mock_config():
    """Mock CortexConfig with test values."""
    config = MagicMock()
    config.kafka_bootstrap_servers = "localhost:9092"
    config.kafka_input_topic = "test-prompts"
    config.kafka_output_topic = "test-ai-out"
    config.kafka_consumer_group = "test-consumer-group"
    config.kafka_auto_offset_reset = "earliest"
    config.exo_base_url = "http://localhost:52415/v1"
    config.exo_api_key = "test-key"
    config.model_name = "test-model"
    config.temperature = 0.7
    config.max_tokens = None
    config.log_level = "INFO"
    config.context_db_path = "/tmp/test_context.db"
    config.cleanup_context_after_use = True
    config.chat_history_enabled = True
    config.chat_db_path = "/tmp/test_chat.db"
    config.chat_reset_patterns = ["/reset", "reset conversation", "new conversation"]
    config.reload = Mock(return_value=False)
    return config


# =============================================================================
# Message Fixtures
# =============================================================================


@pytest.fixture
def sample_kafka_message():
    """Sample Kafka message as received by consumer."""
    message = MagicMock()
    message.topic = "prompts"
    message.partition = 0
    message.offset = 1
    message.timestamp = int(datetime.now().timestamp() * 1000)
    message.timestamp_type = 0
    message.key = None
    message.value = '{"prompt": "Hello, how are you?", "id": "test_msg_001"}'
    return message


@pytest.fixture
def sample_message_with_context():
    """Sample message that has context available from enrichener."""
    return {
        "prompt": "What is Python?",
        "id": "test_msg_002",
        "context_available": True,
        "contexts_stored": 3,
        "timestamp": datetime.now().isoformat(),
    }


@pytest.fixture
def sample_context_chunks():
    """Sample context chunks as returned by ContextStore."""
    return [
        {
            "id": "ctx_001",
            "prompt_id": "test_msg_002",
            "content": "Python is a high-level programming language.",
            "source": "python_docs.md",
            "h1": "Introduction",
            "h2": "Overview",
            "h3": None,
            "distance": 0.15,
        },
        {
            "id": "ctx_002",
            "prompt_id": "test_msg_002",
            "content": "Python was created by Guido van Rossum.",
            "source": "python_history.md",
            "h1": "History",
            "h2": None,
            "h3": None,
            "distance": 0.22,
        },
    ]


# =============================================================================
# Integration Test Helpers
# =============================================================================


@pytest.fixture
def isolated_consumer_env(
    temp_db_path,
    mock_kafka_consumer,
    mock_kafka_producer,
    mock_openai_client,
    mock_config,
):
    """
    Provides an isolated environment for testing ExoConsumer.
    Returns a dict with all mocked components.
    """
    # Create separate temp dirs for chat and context DBs
    with tempfile.TemporaryDirectory() as tmpdir:
        chat_db = os.path.join(tmpdir, "chat.db")
        context_db = os.path.join(tmpdir, "context.db")

        mock_config.chat_db_path = chat_db
        mock_config.context_db_path = context_db

        yield {
            "config": mock_config,
            "kafka_consumer": mock_kafka_consumer,
            "kafka_producer": mock_kafka_producer,
            "openai_client": mock_openai_client,
            "chat_db_path": chat_db,
            "context_db_path": context_db,
            "tmpdir": tmpdir,
        }
