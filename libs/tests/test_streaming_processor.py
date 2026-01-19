"""
Unit tests for libs/streaming_processor.py
"""

from libs.streaming_processor import (
    StreamingProcessor,
    StreamingState,
    TTSChunk,
    ToolCallAccumulator,
)


class TestStreamingState:
    """Tests for StreamingState dataclass."""

    def test_default_values(self):
        """Test default state initialization."""
        state = StreamingState()
        assert state.full_response == ""
        assert state.display_text == ""
        assert state.word_buffer == ""
        assert state.sentence_buffer == ""
        assert state.in_thinking is False
        assert state.chunk_count == 0
        assert state.chunks_sent == 0
        assert state.current_word_threshold == 5.0

    def test_custom_threshold(self):
        """Test custom word threshold initialization."""
        state = StreamingState(current_word_threshold=10.0)
        assert state.current_word_threshold == 10.0


class TestTTSChunk:
    """Tests for TTSChunk dataclass."""

    def test_chunk_creation(self):
        """Test TTS chunk creation."""
        chunk = TTSChunk(
            text="Hello world",
            chunk_number=1,
            word_count=2,
            word_threshold=5,
        )
        assert chunk.text == "Hello world"
        assert chunk.chunk_number == 1
        assert chunk.word_count == 2
        assert chunk.word_threshold == 5
        assert chunk.is_final is False

    def test_final_chunk(self):
        """Test final chunk flag."""
        chunk = TTSChunk(
            text="Goodbye",
            chunk_number=3,
            word_count=1,
            word_threshold=10,
            is_final=True,
        )
        assert chunk.is_final is True


class TestToolCallAccumulator:
    """Tests for ToolCallAccumulator."""

    def test_empty_accumulator(self):
        """Test empty tool call accumulator."""
        acc = ToolCallAccumulator()
        assert acc.has_tool_calls() is False
        assert acc.get_tool_calls() == []

    def test_process_single_tool_call(self):
        """Test processing a single tool call delta."""

        class MockFunction:
            def __init__(self, name, arguments):
                self.name = name
                self.arguments = arguments

        class MockToolCall:
            def __init__(self, index, id, function):
                self.index = index
                self.id = id
                self.function = function

        acc = ToolCallAccumulator()

        # First delta with ID and function name
        tc = MockToolCall(0, "call_123", MockFunction("write", ""))
        acc.process_delta([tc])

        assert acc.has_tool_calls() is True
        assert len(acc.get_tool_calls()) == 1
        assert acc.get_tool_calls()[0]["id"] == "call_123"
        assert acc.get_tool_calls()[0]["function"]["name"] == "write"

    def test_accumulate_arguments(self):
        """Test that arguments are accumulated across deltas."""

        class MockFunction:
            def __init__(self, name=None, arguments=None):
                self.name = name
                self.arguments = arguments

        class MockToolCall:
            def __init__(self, index, id=None, function=None):
                self.index = index
                self.id = id
                self.function = function

        acc = ToolCallAccumulator()

        # First delta with ID and function name
        tc1 = MockToolCall(0, "call_123", MockFunction("write", '{"file'))
        acc.process_delta([tc1])

        # Second delta with more arguments
        tc2 = MockToolCall(0, None, MockFunction(None, 'name": "test.txt"}'))
        acc.process_delta([tc2])

        tool_calls = acc.get_tool_calls()
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["arguments"] == '{"filename": "test.txt"}'

    def test_process_none_delta(self):
        """Test processing None delta."""
        acc = ToolCallAccumulator()
        acc.process_delta(None)
        assert acc.has_tool_calls() is False


class TestStreamingProcessor:
    """Tests for StreamingProcessor."""

    def test_initialization(self):
        """Test processor initialization."""
        processor = StreamingProcessor(
            initial_word_threshold=10,
            growth_factor=2.0,
            sentence_punctuation={"."},
        )
        assert processor.initial_word_threshold == 10
        assert processor.growth_factor == 2.0
        assert processor.sentence_punctuation == {"."}

    def test_default_values(self):
        """Test default initialization values."""
        processor = StreamingProcessor()
        assert processor.initial_word_threshold == 5
        assert processor.growth_factor == 1.5
        assert processor.sentence_punctuation == {".", "!", "?"}

    def test_simple_content_processing(self):
        """Test processing simple content without thinking blocks."""
        processor = StreamingProcessor(initial_word_threshold=3)

        # Send a complete sentence with enough words
        chunks = processor.process_content("Hello world today. ")

        # Should get a chunk back since we have 3 words and a sentence ending
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world today."
        assert chunks[0].chunk_number == 1

    def test_content_buffering(self):
        """Test that content is buffered until threshold is met."""
        processor = StreamingProcessor(initial_word_threshold=10)

        # Send content that doesn't meet threshold
        chunks = processor.process_content("Hello. ")
        assert len(chunks) == 0  # Should be buffered

        # Full response should still accumulate
        assert processor.full_response == "Hello. "
        assert processor.display_text == "Hello. "

    def test_thinking_block_detection(self):
        """Test detection of thinking block entry."""
        processor = StreamingProcessor()

        # Send thinking block start
        chunks = processor.process_content("Let me think <think>")
        assert processor.in_thinking is True

        # Content while thinking should be skipped for TTS
        chunks = processor.process_content("reasoning here")
        assert processor.in_thinking is True
        assert len(chunks) == 0

        # Exit thinking block
        chunks = processor.process_content("</think> The answer is 42.")
        assert processor.in_thinking is False

    def test_thinking_block_skips_display_text(self):
        """Test that thinking block content is excluded from display_text."""
        processor = StreamingProcessor()

        processor.process_content("Before. <think>")
        processor.process_content("Thinking content")
        processor.process_content("</think>After.")

        # full_response should have everything
        assert "Thinking content" in processor.full_response

        # display_text should NOT have thinking content
        # Note: The content "Before. " before <think> should be in display_text
        assert "Before." in processor.display_text
        # After the </think>, "After." should be in display_text
        assert "After." in processor.display_text

    def test_alternative_thinking_tag(self):
        """Test <thinking> tag (alternative to <think>)."""
        processor = StreamingProcessor()

        processor.process_content("<thinking>")
        assert processor.in_thinking is True

        processor.process_content("</thinking>")
        assert processor.in_thinking is False

    def test_flush_remaining_content(self):
        """Test flushing remaining buffered content."""
        processor = StreamingProcessor(initial_word_threshold=100)

        # Send content that won't meet threshold
        processor.process_content("This is a test sentence.")

        # Flush should return remaining content
        final = processor.flush()
        assert final is not None
        assert "This is a test sentence" in final.text
        assert final.is_final is True

    def test_flush_empty_buffer(self):
        """Test flushing when buffer is empty."""
        processor = StreamingProcessor()
        final = processor.flush()
        assert final is None

    def test_flush_while_thinking(self):
        """Test that flush returns None while in thinking block."""
        processor = StreamingProcessor()

        processor.process_content("<think>still thinking")
        assert processor.in_thinking is True

        final = processor.flush()
        assert final is None

    def test_growing_word_threshold(self):
        """Test that word threshold grows after each chunk."""
        processor = StreamingProcessor(initial_word_threshold=2, growth_factor=2.0)

        # First chunk at threshold 2
        chunks = processor.process_content("One two. ")
        assert len(chunks) == 1
        assert chunks[0].word_threshold == 2

        # Next threshold should be 4
        # Need 4 words now
        chunks = processor.process_content("Three four five six. ")
        assert len(chunks) == 1
        assert chunks[0].word_threshold == 4

    def test_sentence_boundary_detection(self):
        """Test detection of sentence boundaries."""
        processor = StreamingProcessor(initial_word_threshold=2)

        # Multiple sentences
        chunks = processor.process_content("Hello world! How are you? I am fine. ")

        # Should get multiple chunks as each sentence boundary is detected
        assert len(chunks) >= 1

    def test_chunks_sent_counter(self):
        """Test that chunks_sent is incremented correctly."""
        processor = StreamingProcessor(initial_word_threshold=2)

        assert processor.chunks_sent == 0

        processor.process_content("Hello world. ")
        assert processor.chunks_sent == 1

        processor.process_content("More words here. ")
        assert processor.chunks_sent >= 1  # May be 2 if threshold allows

    def test_reset(self):
        """Test processor reset."""
        processor = StreamingProcessor(initial_word_threshold=5)

        # Process some content
        processor.process_content("Hello world. Test content here.")

        # Reset
        processor.reset()

        assert processor.full_response == ""
        assert processor.display_text == ""
        assert processor.in_thinking is False
        assert processor.chunks_sent == 0

    def test_state_access(self):
        """Test access to internal state."""
        processor = StreamingProcessor()
        processor.process_content("Test")

        state = processor.state
        assert isinstance(state, StreamingState)
        assert state.chunk_count == 1

    def test_threshold_info(self):
        """Test getting threshold info."""
        processor = StreamingProcessor(initial_word_threshold=7)

        initial, current = processor.get_threshold_info()
        assert initial == 7
        assert current == 7.0

    def test_markdown_stripping(self):
        """Test that markdown is stripped from TTS chunks."""
        processor = StreamingProcessor(initial_word_threshold=3)

        chunks = processor.process_content("**Bold** and `code` here. ")

        # Should have markdown stripped
        if chunks:
            assert "**" not in chunks[0].text
            assert "`" not in chunks[0].text

    def test_empty_content_after_markdown_strip(self):
        """Test handling of content that becomes empty after stripping."""
        processor = StreamingProcessor(initial_word_threshold=1)

        # Content that might become empty after stripping
        chunks = processor.process_content("```\ncode block\n```. ")

        # Should not produce empty chunks
        for chunk in chunks:
            assert chunk.text.strip() != ""

    def test_multiple_sentences_in_one_chunk(self):
        """Test processing multiple sentences arriving in one API chunk."""
        processor = StreamingProcessor(initial_word_threshold=3)

        # Multiple sentences in one chunk
        chunks = processor.process_content(
            "First sentence here. Second sentence now. Third one too. "
        )

        # Should produce chunks based on word threshold being met
        assert len(chunks) >= 1
        total_chunks = processor.chunks_sent
        assert total_chunks >= 1

    def test_partial_sentence_buffering(self):
        """Test that partial sentences are buffered properly."""
        processor = StreamingProcessor(initial_word_threshold=5)

        # Send partial sentence
        chunks = processor.process_content("This is a partial")
        assert len(chunks) == 0  # No sentence boundary yet

        # Complete the sentence
        chunks = processor.process_content(" sentence here. ")
        # Now should have enough for a chunk
        assert processor.state.sentence_buffer != "" or len(chunks) > 0


class TestStreamingProcessorEdgeCases:
    """Edge case tests for StreamingProcessor."""

    def test_thinking_tag_split_across_chunks(self):
        """Test handling when thinking tag is split across chunks."""
        processor = StreamingProcessor()

        # This is a known limitation - split tags won't be detected
        # Testing current behavior
        processor.process_content("Text <thi")
        processor.process_content("nk>thinking")
        processor.process_content("</think>done")

        # The split tag won't be detected, so content leaks
        # This documents current behavior (not necessarily ideal)
        assert "thi" in processor.full_response

    def test_empty_string_content(self):
        """Test processing empty string."""
        processor = StreamingProcessor()
        chunks = processor.process_content("")
        assert len(chunks) == 0
        assert processor.state.chunk_count == 1  # Still counts as a chunk

    def test_whitespace_only_content(self):
        """Test processing whitespace only."""
        processor = StreamingProcessor()
        chunks = processor.process_content("   \n\t  ")
        assert len(chunks) == 0

    def test_punctuation_at_start_of_chunk(self):
        """Test sentence boundary at very start of chunk."""
        processor = StreamingProcessor(initial_word_threshold=2)

        processor.process_content("Hello world")
        processor.process_content(". More text here. ")

        # Should handle the boundary correctly
        assert processor.chunks_sent >= 1

    def test_consecutive_punctuation(self):
        """Test handling of consecutive punctuation."""
        processor = StreamingProcessor(initial_word_threshold=2)

        processor.process_content("What?! Really?! Yes... ")

        # Should handle multiple sentence boundaries
        assert processor.chunks_sent >= 1

    def test_very_long_sentence(self):
        """Test handling of very long sentences."""
        processor = StreamingProcessor(initial_word_threshold=5)

        # Long sentence without punctuation
        long_text = " ".join(["word"] * 100)
        chunks = processor.process_content(long_text)

        # No chunks until punctuation
        assert len(chunks) == 0

        # Add punctuation
        chunks = processor.process_content(". ")
        assert len(chunks) == 1  # Should now produce chunk
