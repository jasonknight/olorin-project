"""
Kafka Message Factory for Olorin Project

Provides standardized message creation for Kafka topics to ensure
consistent message format across components.

Usage:
    from libs.kafka_messages import KafkaMessageFactory

    factory = KafkaMessageFactory()

    # Create a TTS chunk message
    msg = factory.tts_chunk(
        text="Hello, world!",
        message_id="abc123",
        prompt_id="prompt_456",
        chunk_number=1,
        model="llama3",
    )
    producer.send(topic, value=msg)
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class KafkaMessageFactory:
    """Factory for creating standardized Kafka messages."""

    @staticmethod
    def _timestamp() -> str:
        """Get current ISO timestamp."""
        return datetime.now().isoformat()

    @staticmethod
    def processing_notice(
        message_id: str,
        notice_count: int,
        text: str = "Processing, one moment...",
    ) -> dict:
        """
        Create a processing notice message.

        Sent while waiting for AI inference to complete.

        Args:
            message_id: The original message ID
            notice_count: Sequential notice number
            text: Notice text to display/speak

        Returns:
            Kafka message dict
        """
        return {
            "text": text,
            "id": f"{message_id}_processing_{notice_count}",
            "prompt_id": message_id,
            "is_processing_notice": True,
            "timestamp": KafkaMessageFactory._timestamp(),
        }

    @staticmethod
    def thinking_notice(
        message_id: str,
        model: str,
        text: str = "Thinking, give me a moment...",
    ) -> dict:
        """
        Create a thinking notice message.

        Sent when the model enters a <think>/<thinking> block.

        Args:
            message_id: The original message ID
            model: Model name that is thinking
            text: Notice text to display/speak

        Returns:
            Kafka message dict
        """
        return {
            "text": text,
            "id": f"{message_id}_thinking",
            "prompt_id": message_id,
            "model": model,
            "is_thinking_notice": True,
            "timestamp": KafkaMessageFactory._timestamp(),
        }

    @staticmethod
    def reset_confirmation(
        message_id: str,
        new_conversation_id: str,
        text: str = "Conversation has been reset. Starting fresh.",
    ) -> dict:
        """
        Create a conversation reset confirmation message.

        Args:
            message_id: The original message ID
            new_conversation_id: ID of the new conversation
            text: Confirmation text

        Returns:
            Kafka message dict
        """
        return {
            "text": text,
            "id": f"{message_id}_reset_confirmation",
            "prompt_id": message_id,
            "is_reset_confirmation": True,
            "new_conversation_id": new_conversation_id,
            "timestamp": KafkaMessageFactory._timestamp(),
        }

    @staticmethod
    def tts_chunk(
        text: str,
        message_id: str,
        prompt_id: str,
        chunk_number: int,
        model: str,
        word_threshold: int | None = None,
        is_final: bool = False,
    ) -> dict:
        """
        Create a TTS chunk message for streaming responses.

        Args:
            text: Text content for TTS
            message_id: Unique ID for this chunk
            prompt_id: Original prompt message ID
            chunk_number: Sequential chunk number
            model: Model name that generated the response
            word_threshold: Current word threshold (for logging)
            is_final: Whether this is the final chunk

        Returns:
            Kafka message dict
        """
        msg = {
            "text": text,
            "id": message_id,
            "prompt_id": prompt_id,
            "model": model,
            "is_chunk": True,
            "chunk_number": chunk_number,
            "timestamp": KafkaMessageFactory._timestamp(),
        }

        if word_threshold is not None:
            msg["word_threshold"] = word_threshold

        if is_final:
            msg["is_final"] = True

        return msg

    @staticmethod
    def error_message(
        message_id: str,
        prompt_id: str,
        error: str,
        error_type: str = "Error",
    ) -> dict:
        """
        Create an error message.

        Args:
            message_id: Unique ID for this error message
            prompt_id: Original prompt message ID
            error: Error description
            error_type: Type of error (exception class name)

        Returns:
            Kafka message dict
        """
        return {
            "text": f"Error processing prompt: {error}",
            "id": message_id,
            "prompt_id": prompt_id,
            "error": True,
            "error_type": error_type,
            "timestamp": KafkaMessageFactory._timestamp(),
        }

    @staticmethod
    def follow_up_response(
        text: str,
        message_id: str,
        prompt_id: str,
        chunk_number: int,
        model: str,
    ) -> dict:
        """
        Create a follow-up response message (after tool execution).

        Args:
            text: Response text
            message_id: Unique ID for this message
            prompt_id: Original prompt message ID
            chunk_number: Sequential chunk number
            model: Model name

        Returns:
            Kafka message dict
        """
        return {
            "text": text,
            "id": message_id,
            "prompt_id": prompt_id,
            "model": model,
            "is_chunk": True,
            "chunk_number": chunk_number,
            "is_final": True,
            "timestamp": KafkaMessageFactory._timestamp(),
        }

    @staticmethod
    def context_overflow_notice(
        message_id: str,
        query: str,
        text: str = "I found relevant information, but it's too large to process. Please try a more specific search query.",
    ) -> dict:
        """
        Create a context overflow notification message.

        Sent when search results exceed the context window limit and
        cannot be trimmed (even single result is too large).

        Args:
            message_id: Unique ID for this message
            query: The search query that caused the overflow
            text: Notification text to display/speak

        Returns:
            Kafka message dict
        """
        return {
            "text": text,
            "id": message_id,
            "is_context_overflow": True,
            "query": query,
            "timestamp": KafkaMessageFactory._timestamp(),
        }
