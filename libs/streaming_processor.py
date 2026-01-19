"""
Streaming Response Processor for Olorin Project

Handles streaming API responses with support for:
- Thinking block detection (pauses output during <think>/<thinking> tags)
- Sentence boundary detection for TTS-friendly chunking
- Growing word threshold for progressive chunk sizes
- Tool call accumulation from streaming chunks

Usage:
    from libs.streaming_processor import StreamingProcessor

    processor = StreamingProcessor(
        initial_word_threshold=5,
        growth_factor=1.5,
        sentence_punctuation={".", "!", "?"},
    )

    for chunk in api_stream:
        content = chunk.choices[0].delta.content
        if content:
            for tts_chunk in processor.process_content(content):
                send_to_tts(tts_chunk)

    # Get any remaining content
    final_chunk = processor.flush()
    if final_chunk:
        send_to_tts(final_chunk)
"""

from dataclasses import dataclass, field
from libs.text_processing import strip_markdown


@dataclass
class StreamingState:
    """State tracking for streaming response processing."""

    full_response: str = ""
    display_text: str = ""  # Clean text excluding thinking blocks
    word_buffer: str = ""
    sentence_buffer: str = ""
    in_thinking: bool = False
    chunk_count: int = 0
    chunks_sent: int = 0
    current_word_threshold: float = 5.0


@dataclass
class TTSChunk:
    """A chunk of text ready for TTS processing."""

    text: str
    chunk_number: int
    word_count: int
    word_threshold: int
    is_final: bool = False


@dataclass
class ToolCallAccumulator:
    """Accumulates tool call data from streaming chunks."""

    tool_calls: list[dict] = field(default_factory=list)

    def process_delta(self, tool_calls_delta) -> None:
        """
        Process tool call deltas from a streaming chunk.

        Args:
            tool_calls_delta: List of tool call delta objects from the API
        """
        if not tool_calls_delta:
            return

        for tc in tool_calls_delta:
            tc_index = tc.index if hasattr(tc, "index") else 0

            # Expand list if needed
            while len(self.tool_calls) <= tc_index:
                self.tool_calls.append(
                    {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                )

            # Accumulate tool call data
            if hasattr(tc, "id") and tc.id:
                self.tool_calls[tc_index]["id"] = tc.id
            if hasattr(tc, "function") and tc.function:
                if hasattr(tc.function, "name") and tc.function.name:
                    self.tool_calls[tc_index]["function"]["name"] += tc.function.name
                if hasattr(tc.function, "arguments") and tc.function.arguments:
                    self.tool_calls[tc_index]["function"]["arguments"] += (
                        tc.function.arguments
                    )

    def get_tool_calls(self) -> list[dict]:
        """Get accumulated tool calls."""
        return self.tool_calls

    def has_tool_calls(self) -> bool:
        """Check if any tool calls were accumulated."""
        return len(self.tool_calls) > 0


class StreamingProcessor:
    """
    Processes streaming API responses for TTS output.

    Handles thinking block detection, sentence boundary detection,
    and progressive chunk sizing for natural TTS output.
    """

    def __init__(
        self,
        initial_word_threshold: int = 5,
        growth_factor: float = 1.5,
        sentence_punctuation: set[str] | None = None,
    ):
        """
        Initialize the streaming processor.

        Args:
            initial_word_threshold: Starting word count before sending first chunk
            growth_factor: Multiplier for word threshold after each chunk sent
            sentence_punctuation: Set of characters that end sentences (default: {".", "!", "?"})
        """
        self.initial_word_threshold = initial_word_threshold
        self.growth_factor = growth_factor
        self.sentence_punctuation = sentence_punctuation or {".", "!", "?"}

        self._state = StreamingState(
            current_word_threshold=float(initial_word_threshold)
        )

    @property
    def state(self) -> StreamingState:
        """Get current streaming state (read-only access)."""
        return self._state

    @property
    def full_response(self) -> str:
        """Get the full accumulated response text."""
        return self._state.full_response

    @property
    def display_text(self) -> str:
        """Get display text (excludes thinking blocks)."""
        return self._state.display_text

    @property
    def in_thinking(self) -> bool:
        """Check if currently inside a thinking block."""
        return self._state.in_thinking

    @property
    def chunks_sent(self) -> int:
        """Get count of TTS chunks sent."""
        return self._state.chunks_sent

    def reset(self) -> None:
        """Reset processor state for a new response."""
        self._state = StreamingState(
            current_word_threshold=float(self.initial_word_threshold)
        )

    def process_content(self, content: str) -> list[TTSChunk]:
        """
        Process a streaming content chunk.

        Args:
            content: Text content from a streaming API chunk

        Returns:
            List of TTSChunk objects ready for TTS (may be empty if buffering)
        """
        chunks_to_send: list[TTSChunk] = []

        self._state.chunk_count += 1
        self._state.full_response += content

        # Handle thinking block tags - need to process text before/after tags
        remaining_content = content

        # Check for thinking block entry
        for tag in ("<think>", "<thinking>"):
            if tag in remaining_content and not self._state.in_thinking:
                # Split on the tag - process text before it
                before, _, after = remaining_content.partition(tag)
                if before:
                    # Process text before the thinking tag
                    self._state.display_text += before
                    self._state.word_buffer += before
                    chunks_to_send.extend(self._process_sentences())
                self._state.in_thinking = True
                remaining_content = after
                break

        # Check for thinking block exit
        for tag in ("</think>", "</thinking>"):
            if tag in remaining_content and self._state.in_thinking:
                # Split on the tag - process text after it
                _, _, after = remaining_content.partition(tag)
                self._state.in_thinking = False
                # Clear buffers to avoid sending any accumulated thinking content
                self._state.word_buffer = ""
                self._state.sentence_buffer = ""
                remaining_content = after
                break

        # Skip remaining content if in thinking block
        if self._state.in_thinking:
            return chunks_to_send

        # Process any remaining content after tag handling
        if remaining_content:
            self._state.display_text += remaining_content
            self._state.word_buffer += remaining_content
            chunks_to_send.extend(self._process_sentences())

        return chunks_to_send

    def _process_sentences(self) -> list[TTSChunk]:
        """Process sentence boundaries and return chunks ready for TTS."""
        chunks: list[TTSChunk] = []

        while True:
            # Find the earliest sentence-ending punctuation
            earliest_pos = -1
            for punct in self.sentence_punctuation:
                pos = self._state.word_buffer.find(punct)
                if pos != -1 and (earliest_pos == -1 or pos < earliest_pos):
                    earliest_pos = pos

            if earliest_pos == -1:
                break  # No more sentence boundaries

            # Extract sentence up to and including the punctuation
            complete_sentence = self._state.word_buffer[: earliest_pos + 1]
            self._state.word_buffer = self._state.word_buffer[earliest_pos + 1 :]

            # Add to sentence buffer
            self._state.sentence_buffer += complete_sentence

            # Check if we've accumulated enough words to send
            current_word_count = len(self._state.sentence_buffer.split())
            if current_word_count >= self._state.current_word_threshold:
                chunk = self._create_chunk(self._state.sentence_buffer, is_final=False)
                if chunk:
                    chunks.append(chunk)
                    # Grow threshold for next chunk
                    self._state.current_word_threshold *= self.growth_factor
                self._state.sentence_buffer = ""

        return chunks

    def _create_chunk(self, text: str, is_final: bool = False) -> TTSChunk | None:
        """Create a TTS chunk from text, stripping markdown."""
        tts_text = strip_markdown(text)
        if not tts_text.strip():
            return None

        self._state.chunks_sent += 1
        return TTSChunk(
            text=tts_text,
            chunk_number=self._state.chunks_sent,
            word_count=len(text.split()),
            word_threshold=int(self._state.current_word_threshold),
            is_final=is_final,
        )

    def flush(self) -> TTSChunk | None:
        """
        Flush any remaining buffered content.

        Call this after the stream ends to get the final chunk.

        Returns:
            Final TTSChunk if there's remaining content, None otherwise
        """
        if self._state.in_thinking:
            return None

        remaining = self._state.sentence_buffer + self._state.word_buffer
        if not remaining.strip():
            return None

        return self._create_chunk(remaining, is_final=True)

    def entered_thinking(self) -> bool:
        """
        Check if we just entered a thinking block.

        This is useful for sending a "Thinking..." notice to the user.
        The flag is reset after being read.
        """
        # This is tracked externally by checking in_thinking state changes
        return self._state.in_thinking

    def get_threshold_info(self) -> tuple[int, float]:
        """
        Get current threshold info for logging.

        Returns:
            Tuple of (initial_threshold, current_threshold)
        """
        return (self.initial_word_threshold, self._state.current_word_threshold)
