"""
Context Formatter for Olorin Project

Handles RAG context formatting and tool system prompt generation
for LLM API calls.

Usage:
    from libs.context_formatter import ContextFormatter

    formatter = ContextFormatter(system_prompt="You are a helpful assistant.")

    # Format context with prompt for RAG
    combined = formatter.format_context_with_prompt(context_chunks, user_prompt)

    # Build tool system prompt
    tool_prompt = formatter.build_tool_system_prompt(available_tools)
"""


class ContextFormatter:
    """
    Formats RAG context and tool prompts for LLM API calls.

    Combining context and prompt in one message is more robust across different
    models. Some models (e.g., Deepseek R1) lose context awareness when messages
    are split, even with consecutive user messages.
    """

    def __init__(self, system_prompt: str | None = None):
        """
        Initialize the context formatter.

        Args:
            system_prompt: Base system prompt for context injection
        """
        self.system_prompt = system_prompt or (
            "You are an AI assistant who provides help to a user based on provided "
            "context and instruction prompts. Your goal is to answer questions and "
            "complete tasks based on the user's input."
        )

    def format_context_with_prompt(
        self, context_chunks: list[dict], prompt: str
    ) -> str:
        """
        Format RAG context chunks combined with the user prompt into a single message.

        Args:
            context_chunks: List of context dicts with 'content', 'source', etc.
                           May include 'is_manual' flag to distinguish manually selected context.
            prompt: The user's question/prompt to append after context.

        Returns:
            Combined message string with context and prompt
        """
        # Separate manual and auto context for clearer presentation
        manual_chunks = [c for c in context_chunks if c.get("is_manual")]
        auto_chunks = [c for c in context_chunks if not c.get("is_manual")]

        context_parts = []

        # Format manual context first (user-selected)
        if manual_chunks:
            context_parts.append("## Selected Context (manually chosen)")
            for i, ctx in enumerate(manual_chunks):
                source_parts = []
                if ctx.get("source"):
                    source_parts.append(ctx["source"])
                source_ref = (
                    " > ".join(source_parts)
                    if source_parts
                    else f"Selected document {i + 1}"
                )
                context_parts.append(f"### {source_ref}\n{ctx['content']}")

        # Format auto context (RAG-retrieved)
        if auto_chunks:
            if manual_chunks:
                context_parts.append("\n## Retrieved Context (automatically found)")
            for i, ctx in enumerate(auto_chunks):
                source_parts = []
                if ctx.get("source"):
                    source_parts.append(ctx["source"])
                if ctx.get("h1"):
                    source_parts.append(ctx["h1"])
                if ctx.get("h2"):
                    source_parts.append(ctx["h2"])
                if ctx.get("h3"):
                    source_parts.append(ctx["h3"])

                source_ref = (
                    " > ".join(source_parts) if source_parts else f"Source {i + 1}"
                )
                context_parts.append(f"### {source_ref}\n{ctx['content']}")

        context_block = "\n\n".join(context_parts)

        # Single combined message with system prompt, context + user prompt
        return f"""{self.system_prompt}

Use the following reference context to answer my question.

<context>
{context_block}
</context>

Question: {prompt}"""

    def build_tool_system_prompt(self, available_tools: list[dict]) -> str | None:
        """
        Build a system prompt that describes available tools and how to use them.

        Args:
            available_tools: List of tool definitions in OpenAI format

        Returns:
            System prompt string if tools are available, None otherwise
        """
        if not available_tools:
            return None

        tool_descriptions = []
        for tool in available_tools:
            func = tool.get("function", {})
            name = func.get("name", "unknown")
            description = func.get("description", "No description")
            params = func.get("parameters", {}).get("properties", {})
            required = func.get("parameters", {}).get("required", [])

            # Build parameter documentation
            param_docs = []
            for param_name, param_info in params.items():
                param_type = param_info.get("type", "string")
                param_desc = param_info.get("description", "")
                req_marker = " (required)" if param_name in required else " (optional)"
                param_docs.append(
                    f"    - {param_name} ({param_type}){req_marker}: {param_desc}"
                )

            params_str = "\n".join(param_docs) if param_docs else "    (no parameters)"

            tool_descriptions.append(f"""- **{name}**: {description}
  Parameters:
{params_str}""")

        tools_block = "\n\n".join(tool_descriptions)

        return f"""You are a helpful AI assistant with access to tools. When the user asks you to perform an action that matches one of your available tools, you MUST use the appropriate tool to complete the task.

## Available Tools

{tools_block}

## When to Use Tools

- When the user asks you to write, save, or export content to a file, use the `write` tool.
- When the user mentions a filename or asks to create a file, use the `write` tool.
- Always use tools when they match the user's request - do not just describe what you would do.

## How to Use Tools

When you decide to use a tool, call it with the required parameters. After the tool executes, you will receive the result and can then respond to the user with confirmation or any follow-up information."""

    def should_skip_tools_for_rag(
        self, context_chunks: list[dict] | None, threshold: int = 10000
    ) -> bool:
        """
        Determine if tools should be skipped due to large RAG context.

        Small models get confused trying to use tools when they should
        answer from context.

        Args:
            context_chunks: List of context chunks
            threshold: Character threshold above which tools are skipped

        Returns:
            True if tools should be skipped, False otherwise
        """
        if not context_chunks:
            return False

        context_size = sum(len(c.get("content", "")) for c in context_chunks)
        return context_size > threshold

    def get_context_size(self, context_chunks: list[dict] | None) -> int:
        """
        Calculate total character size of context chunks.

        Args:
            context_chunks: List of context chunks

        Returns:
            Total character count
        """
        if not context_chunks:
            return 0
        return sum(len(c.get("content", "")) for c in context_chunks)
