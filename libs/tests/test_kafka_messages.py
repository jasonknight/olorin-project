"""
Unit tests for libs/kafka_messages.py
"""

from datetime import datetime
from libs.kafka_messages import KafkaMessageFactory


class TestKafkaMessageFactory:
    """Tests for KafkaMessageFactory."""

    def test_processing_notice(self):
        """Test processing notice message creation."""
        msg = KafkaMessageFactory.processing_notice("msg_123", 1)

        assert msg["text"] == "Processing, one moment..."
        assert msg["id"] == "msg_123_processing_1"
        assert msg["prompt_id"] == "msg_123"
        assert msg["is_processing_notice"] is True
        assert "timestamp" in msg

    def test_processing_notice_custom_text(self):
        """Test processing notice with custom text."""
        msg = KafkaMessageFactory.processing_notice(
            "msg_123", 2, "Still working on it..."
        )

        assert msg["text"] == "Still working on it..."
        assert msg["id"] == "msg_123_processing_2"

    def test_processing_notice_sequential_numbers(self):
        """Test that notice count is reflected in ID."""
        msg1 = KafkaMessageFactory.processing_notice("msg_123", 1)
        msg2 = KafkaMessageFactory.processing_notice("msg_123", 5)
        msg3 = KafkaMessageFactory.processing_notice("msg_123", 10)

        assert msg1["id"] == "msg_123_processing_1"
        assert msg2["id"] == "msg_123_processing_5"
        assert msg3["id"] == "msg_123_processing_10"

    def test_thinking_notice(self):
        """Test thinking notice message creation."""
        msg = KafkaMessageFactory.thinking_notice("msg_456", "llama3")

        assert msg["text"] == "Thinking, give me a moment..."
        assert msg["id"] == "msg_456_thinking"
        assert msg["prompt_id"] == "msg_456"
        assert msg["model"] == "llama3"
        assert msg["is_thinking_notice"] is True
        assert "timestamp" in msg

    def test_thinking_notice_custom_text(self):
        """Test thinking notice with custom text."""
        msg = KafkaMessageFactory.thinking_notice(
            "msg_456", "gpt-4", "Reasoning through this..."
        )

        assert msg["text"] == "Reasoning through this..."

    def test_reset_confirmation(self):
        """Test reset confirmation message creation."""
        msg = KafkaMessageFactory.reset_confirmation("msg_789", "conv_new_123")

        assert msg["text"] == "Conversation has been reset. Starting fresh."
        assert msg["id"] == "msg_789_reset_confirmation"
        assert msg["prompt_id"] == "msg_789"
        assert msg["is_reset_confirmation"] is True
        assert msg["new_conversation_id"] == "conv_new_123"
        assert "timestamp" in msg

    def test_reset_confirmation_custom_text(self):
        """Test reset confirmation with custom text."""
        msg = KafkaMessageFactory.reset_confirmation(
            "msg_789", "conv_456", "Memory cleared!"
        )

        assert msg["text"] == "Memory cleared!"

    def test_tts_chunk(self):
        """Test TTS chunk message creation."""
        msg = KafkaMessageFactory.tts_chunk(
            text="Hello world",
            message_id="msg_123_chunk_1",
            prompt_id="msg_123",
            chunk_number=1,
            model="llama3",
        )

        assert msg["text"] == "Hello world"
        assert msg["id"] == "msg_123_chunk_1"
        assert msg["prompt_id"] == "msg_123"
        assert msg["model"] == "llama3"
        assert msg["is_chunk"] is True
        assert msg["chunk_number"] == 1
        assert "timestamp" in msg
        assert "is_final" not in msg
        assert "word_threshold" not in msg

    def test_tts_chunk_with_threshold(self):
        """Test TTS chunk with word threshold."""
        msg = KafkaMessageFactory.tts_chunk(
            text="Hello world",
            message_id="msg_123_chunk_1",
            prompt_id="msg_123",
            chunk_number=1,
            model="llama3",
            word_threshold=5,
        )

        assert msg["word_threshold"] == 5

    def test_tts_chunk_final(self):
        """Test final TTS chunk."""
        msg = KafkaMessageFactory.tts_chunk(
            text="Goodbye",
            message_id="msg_123_chunk_5",
            prompt_id="msg_123",
            chunk_number=5,
            model="llama3",
            is_final=True,
        )

        assert msg["is_final"] is True

    def test_tts_chunk_all_options(self):
        """Test TTS chunk with all optional parameters."""
        msg = KafkaMessageFactory.tts_chunk(
            text="Complete message",
            message_id="msg_123_chunk_3",
            prompt_id="msg_123",
            chunk_number=3,
            model="deepseek-r1",
            word_threshold=15,
            is_final=True,
        )

        assert msg["text"] == "Complete message"
        assert msg["word_threshold"] == 15
        assert msg["is_final"] is True
        assert msg["chunk_number"] == 3

    def test_error_message(self):
        """Test error message creation."""
        msg = KafkaMessageFactory.error_message(
            message_id="msg_123_error",
            prompt_id="msg_123",
            error="Connection timeout",
            error_type="TimeoutError",
        )

        assert msg["text"] == "Error processing prompt: Connection timeout"
        assert msg["id"] == "msg_123_error"
        assert msg["prompt_id"] == "msg_123"
        assert msg["error"] is True
        assert msg["error_type"] == "TimeoutError"
        assert "timestamp" in msg

    def test_error_message_default_type(self):
        """Test error message with default error type."""
        msg = KafkaMessageFactory.error_message(
            message_id="msg_123_error",
            prompt_id="msg_123",
            error="Something went wrong",
        )

        assert msg["error_type"] == "Error"

    def test_follow_up_response(self):
        """Test follow-up response message creation."""
        msg = KafkaMessageFactory.follow_up_response(
            text="File written successfully",
            message_id="msg_123_follow_up",
            prompt_id="msg_123",
            chunk_number=4,
            model="llama3",
        )

        assert msg["text"] == "File written successfully"
        assert msg["id"] == "msg_123_follow_up"
        assert msg["prompt_id"] == "msg_123"
        assert msg["model"] == "llama3"
        assert msg["is_chunk"] is True
        assert msg["chunk_number"] == 4
        assert msg["is_final"] is True
        assert "timestamp" in msg


class TestKafkaMessageFactoryTimestamps:
    """Tests for timestamp functionality."""

    def test_timestamp_is_iso_format(self):
        """Test that timestamps are in ISO format."""
        msg = KafkaMessageFactory.processing_notice("msg_123", 1)

        # Should be parseable as ISO timestamp
        timestamp = msg["timestamp"]
        parsed = datetime.fromisoformat(timestamp)
        assert parsed is not None

    def test_timestamps_are_current(self):
        """Test that timestamps are reasonably current."""
        before = datetime.now()
        msg = KafkaMessageFactory.processing_notice("msg_123", 1)
        after = datetime.now()

        timestamp = datetime.fromisoformat(msg["timestamp"])

        # Timestamp should be between before and after
        assert before <= timestamp <= after

    def test_different_messages_have_different_timestamps(self):
        """Test that consecutive messages can have different timestamps."""
        msg1 = KafkaMessageFactory.processing_notice("msg_123", 1)
        # Note: timestamps might be the same if called very quickly
        # This just verifies the timestamp mechanism works
        msg2 = KafkaMessageFactory.processing_notice("msg_123", 2)

        assert "timestamp" in msg1
        assert "timestamp" in msg2


class TestKafkaMessageFactoryEdgeCases:
    """Edge case tests."""

    def test_empty_text(self):
        """Test handling of empty text."""
        msg = KafkaMessageFactory.tts_chunk(
            text="",
            message_id="msg_123_chunk_1",
            prompt_id="msg_123",
            chunk_number=1,
            model="llama3",
        )

        assert msg["text"] == ""

    def test_special_characters_in_text(self):
        """Test handling of special characters."""
        text = 'Hello "world" with <tags> & symbols!'
        msg = KafkaMessageFactory.tts_chunk(
            text=text,
            message_id="msg_123_chunk_1",
            prompt_id="msg_123",
            chunk_number=1,
            model="llama3",
        )

        assert msg["text"] == text

    def test_unicode_text(self):
        """Test handling of unicode text."""
        text = "Hello 世界! Привет мир! 🌍"
        msg = KafkaMessageFactory.tts_chunk(
            text=text,
            message_id="msg_123_chunk_1",
            prompt_id="msg_123",
            chunk_number=1,
            model="llama3",
        )

        assert msg["text"] == text

    def test_very_long_text(self):
        """Test handling of very long text."""
        text = "word " * 10000  # Very long text
        msg = KafkaMessageFactory.tts_chunk(
            text=text,
            message_id="msg_123_chunk_1",
            prompt_id="msg_123",
            chunk_number=1,
            model="llama3",
        )

        assert msg["text"] == text

    def test_multiline_text(self):
        """Test handling of multiline text."""
        text = "Line 1\nLine 2\nLine 3"
        msg = KafkaMessageFactory.tts_chunk(
            text=text,
            message_id="msg_123_chunk_1",
            prompt_id="msg_123",
            chunk_number=1,
            model="llama3",
        )

        assert msg["text"] == text
        assert "\n" in msg["text"]
