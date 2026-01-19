"""
Unit tests for libs/context_formatter.py
"""

from libs.context_formatter import ContextFormatter


class TestContextFormatterInit:
    """Tests for ContextFormatter initialization."""

    def test_default_system_prompt(self):
        """Test default system prompt is set."""
        formatter = ContextFormatter()
        assert formatter.system_prompt is not None
        assert "AI assistant" in formatter.system_prompt

    def test_custom_system_prompt(self):
        """Test custom system prompt."""
        custom_prompt = "You are a helpful code reviewer."
        formatter = ContextFormatter(system_prompt=custom_prompt)
        assert formatter.system_prompt == custom_prompt


class TestFormatContextWithPrompt:
    """Tests for format_context_with_prompt method."""

    def test_basic_context_formatting(self):
        """Test basic context formatting."""
        formatter = ContextFormatter(system_prompt="Test prompt.")
        context_chunks = [
            {"content": "Document content here", "source": "doc.md"},
        ]

        result = formatter.format_context_with_prompt(context_chunks, "What is this?")

        assert "Test prompt." in result
        assert "<context>" in result
        assert "</context>" in result
        assert "Document content here" in result
        assert "doc.md" in result
        assert "Question: What is this?" in result

    def test_multiple_context_chunks(self):
        """Test formatting multiple context chunks."""
        formatter = ContextFormatter()
        context_chunks = [
            {"content": "First document", "source": "doc1.md"},
            {"content": "Second document", "source": "doc2.md"},
            {"content": "Third document", "source": "doc3.md"},
        ]

        result = formatter.format_context_with_prompt(
            context_chunks, "Summarize these."
        )

        assert "First document" in result
        assert "Second document" in result
        assert "Third document" in result
        assert "doc1.md" in result
        assert "doc2.md" in result
        assert "doc3.md" in result

    def test_context_with_headers(self):
        """Test context chunks with header metadata."""
        formatter = ContextFormatter()
        context_chunks = [
            {
                "content": "Section content",
                "source": "manual.md",
                "h1": "Chapter 1",
                "h2": "Section A",
                "h3": "Subsection",
            },
        ]

        result = formatter.format_context_with_prompt(context_chunks, "What is this?")

        # Should include header hierarchy in source ref
        assert "manual.md" in result
        assert "Chapter 1" in result
        assert "Section A" in result
        assert "Subsection" in result

    def test_manual_context_separation(self):
        """Test that manual context is separated from auto context."""
        formatter = ContextFormatter()
        context_chunks = [
            {"content": "Manual doc", "source": "manual.md", "is_manual": True},
            {"content": "Auto doc", "source": "auto.md", "is_manual": False},
        ]

        result = formatter.format_context_with_prompt(context_chunks, "Question?")

        # Check for section headers
        assert "Selected Context (manually chosen)" in result
        assert "Retrieved Context (automatically found)" in result
        assert "Manual doc" in result
        assert "Auto doc" in result

    def test_only_manual_context(self):
        """Test formatting with only manual context."""
        formatter = ContextFormatter()
        context_chunks = [
            {"content": "Manual doc 1", "source": "doc1.md", "is_manual": True},
            {"content": "Manual doc 2", "source": "doc2.md", "is_manual": True},
        ]

        result = formatter.format_context_with_prompt(context_chunks, "Question?")

        assert "Selected Context (manually chosen)" in result
        assert "Retrieved Context (automatically found)" not in result

    def test_only_auto_context(self):
        """Test formatting with only auto context."""
        formatter = ContextFormatter()
        context_chunks = [
            {"content": "Auto doc 1", "source": "doc1.md", "is_manual": False},
            {
                "content": "Auto doc 2",
                "source": "doc2.md",
            },  # is_manual defaults to False
        ]

        result = formatter.format_context_with_prompt(context_chunks, "Question?")

        # Auto context without manual context shouldn't have "Retrieved Context" header
        # (it's only added when both types are present)
        assert "Auto doc 1" in result
        assert "Auto doc 2" in result

    def test_context_without_source(self):
        """Test context chunks without source metadata."""
        formatter = ContextFormatter()
        context_chunks = [
            {"content": "Some content"},
            {"content": "More content"},
        ]

        result = formatter.format_context_with_prompt(context_chunks, "Question?")

        # Should use fallback source references
        assert "Source 1" in result or "Some content" in result
        assert "Source 2" in result or "More content" in result

    def test_empty_context(self):
        """Test with empty context list."""
        formatter = ContextFormatter()

        result = formatter.format_context_with_prompt([], "Question?")

        # Should still have the basic structure
        assert "<context>" in result
        assert "</context>" in result
        assert "Question: Question?" in result


class TestBuildToolSystemPrompt:
    """Tests for build_tool_system_prompt method."""

    def test_no_tools(self):
        """Test that None is returned when no tools available."""
        formatter = ContextFormatter()
        result = formatter.build_tool_system_prompt([])
        assert result is None

    def test_single_tool(self):
        """Test formatting a single tool."""
        formatter = ContextFormatter()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "write",
                    "description": "Write content to a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "The content to write",
                            },
                            "filename": {
                                "type": "string",
                                "description": "The filename",
                            },
                        },
                        "required": ["content", "filename"],
                    },
                },
            }
        ]

        result = formatter.build_tool_system_prompt(tools)

        assert result is not None
        assert "write" in result
        assert "Write content to a file" in result
        assert "content" in result
        assert "filename" in result
        assert "(required)" in result

    def test_multiple_tools(self):
        """Test formatting multiple tools."""
        formatter = ContextFormatter()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write",
                    "description": "Write a file",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
        ]

        result = formatter.build_tool_system_prompt(tools)

        assert "read" in result
        assert "write" in result
        assert "Read a file" in result
        assert "Write a file" in result

    def test_tool_with_optional_params(self):
        """Test tool with optional parameters."""
        formatter = ContextFormatter()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search for content",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Max results",
                            },
                        },
                        "required": ["query"],  # limit is optional
                    },
                },
            }
        ]

        result = formatter.build_tool_system_prompt(tools)

        assert "query" in result
        assert "(required)" in result
        assert "limit" in result
        assert "(optional)" in result

    def test_tool_system_prompt_structure(self):
        """Test that tool system prompt has proper structure."""
        formatter = ContextFormatter()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "test",
                    "description": "Test tool",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
        ]

        result = formatter.build_tool_system_prompt(tools)

        assert "Available Tools" in result
        assert "When to Use Tools" in result
        assert "How to Use Tools" in result


class TestShouldSkipToolsForRag:
    """Tests for should_skip_tools_for_rag method."""

    def test_none_context(self):
        """Test with None context."""
        formatter = ContextFormatter()
        assert formatter.should_skip_tools_for_rag(None) is False

    def test_empty_context(self):
        """Test with empty context."""
        formatter = ContextFormatter()
        assert formatter.should_skip_tools_for_rag([]) is False

    def test_small_context(self):
        """Test with small context (under threshold)."""
        formatter = ContextFormatter()
        context = [{"content": "Small content"}]
        assert formatter.should_skip_tools_for_rag(context) is False

    def test_large_context(self):
        """Test with large context (over threshold)."""
        formatter = ContextFormatter()
        # Create context over 10KB (default threshold)
        large_content = "x" * 15000
        context = [{"content": large_content}]
        assert formatter.should_skip_tools_for_rag(context) is True

    def test_multiple_chunks_over_threshold(self):
        """Test with multiple chunks that combined exceed threshold."""
        formatter = ContextFormatter()
        # Create chunks that together exceed 10KB
        context = [
            {"content": "x" * 4000},
            {"content": "y" * 4000},
            {"content": "z" * 4000},
        ]
        assert formatter.should_skip_tools_for_rag(context) is True

    def test_custom_threshold(self):
        """Test with custom threshold."""
        formatter = ContextFormatter()
        context = [{"content": "x" * 5000}]

        # Default threshold (10000) - should not skip
        assert formatter.should_skip_tools_for_rag(context) is False

        # Custom lower threshold - should skip
        assert formatter.should_skip_tools_for_rag(context, threshold=3000) is True


class TestGetContextSize:
    """Tests for get_context_size method."""

    def test_none_context(self):
        """Test with None context."""
        formatter = ContextFormatter()
        assert formatter.get_context_size(None) == 0

    def test_empty_context(self):
        """Test with empty context."""
        formatter = ContextFormatter()
        assert formatter.get_context_size([]) == 0

    def test_single_chunk(self):
        """Test with single chunk."""
        formatter = ContextFormatter()
        context = [{"content": "Hello world"}]
        assert formatter.get_context_size(context) == len("Hello world")

    def test_multiple_chunks(self):
        """Test with multiple chunks."""
        formatter = ContextFormatter()
        context = [
            {"content": "First"},
            {"content": "Second"},
            {"content": "Third"},
        ]
        expected = len("First") + len("Second") + len("Third")
        assert formatter.get_context_size(context) == expected

    def test_chunk_without_content(self):
        """Test chunks without content field."""
        formatter = ContextFormatter()
        context = [
            {"content": "Has content"},
            {"source": "no_content.md"},  # Missing content
            {"content": "Also has content"},
        ]
        expected = len("Has content") + len("Also has content")
        assert formatter.get_context_size(context) == expected

    def test_empty_content(self):
        """Test chunks with empty content."""
        formatter = ContextFormatter()
        context = [
            {"content": ""},
            {"content": "Has content"},
        ]
        assert formatter.get_context_size(context) == len("Has content")


class TestContextFormatterIntegration:
    """Integration tests for ContextFormatter."""

    def test_full_rag_flow(self):
        """Test a complete RAG context formatting flow."""
        formatter = ContextFormatter(
            system_prompt="You are a helpful assistant for documentation."
        )

        # Simulate RAG context
        context_chunks = [
            {
                "content": "## Installation\n\nRun `pip install mypackage`",
                "source": "README.md",
                "h1": "Getting Started",
                "h2": "Installation",
            },
            {
                "content": "## Configuration\n\nSet the API key in .env",
                "source": "README.md",
                "h1": "Getting Started",
                "h2": "Configuration",
            },
        ]

        prompt = "How do I install the package?"
        result = formatter.format_context_with_prompt(context_chunks, prompt)

        # Verify all parts are present
        assert "helpful assistant for documentation" in result
        assert "pip install mypackage" in result
        assert "API key" in result
        assert "How do I install the package?" in result
        assert "<context>" in result
        assert "</context>" in result

    def test_tool_decision_with_rag(self):
        """Test deciding whether to use tools based on RAG context size."""
        formatter = ContextFormatter()

        # Small context - tools should be enabled
        small_context = [{"content": "Small doc"}]
        assert formatter.should_skip_tools_for_rag(small_context) is False

        # Large context - tools should be disabled
        large_context = [{"content": "x" * 20000}]
        assert formatter.should_skip_tools_for_rag(large_context) is True

        # When tools should be skipped, tool prompt should return None
        # (This is handled by the consumer, but we can verify the method behavior)
        tools = [{"type": "function", "function": {"name": "test"}}]
        prompt = formatter.build_tool_system_prompt(tools)
        assert prompt is not None  # Tools exist, so prompt is generated
