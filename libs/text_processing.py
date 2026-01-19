"""
Text Processing Utilities for Olorin Project

Provides text processing functions shared across components:
- Thinking block extraction (for models like Deepseek R1)
- Markdown stripping for TTS readability
- Other text transformation utilities

Usage:
    from libs.text_processing import (
        extract_thinking_blocks,
        compute_thinking_stats,
        strip_markdown,
    )

    # Extract and analyze thinking blocks
    cleaned, blocks = extract_thinking_blocks(response_text)
    if blocks:
        stats = compute_thinking_stats(blocks)
        print(f"Model used {stats['total_words']} words of reasoning")

    # Strip markdown for TTS
    tts_text = strip_markdown(ai_response)
"""

import re


def extract_thinking_blocks(text: str) -> tuple[str, list[str]]:
    """
    Extract thinking blocks from text and return cleaned text + thinking block list.
    Handles both <think> and <thinking> tags.

    Some models (e.g., Deepseek R1) emit their reasoning process in special tags.
    This function extracts those blocks for logging/analysis while returning
    clean text for display or TTS.

    Args:
        text: The raw text containing potential thinking blocks

    Returns:
        tuple: (cleaned_text, list_of_thinking_blocks)
            - cleaned_text: Text with thinking blocks removed
            - list_of_thinking_blocks: List of extracted thinking block strings
                                       (including tags)

    Example:
        >>> text = "Let me think. <think>First I'll consider...</think> The answer is 42."
        >>> clean, blocks = extract_thinking_blocks(text)
        >>> clean
        'Let me think. The answer is 42.'
        >>> len(blocks)
        1
    """
    # Pattern to match both <think>...</think> and <thinking>...</thinking> blocks
    thinking_pattern = r"<think(?:ing)?>.*?</think(?:ing)?>"

    # Find all thinking blocks
    thinking_blocks = re.findall(thinking_pattern, text, re.DOTALL)

    # Remove thinking blocks from text
    cleaned_text = re.sub(thinking_pattern, "", text, flags=re.DOTALL)

    # Clean up any excessive whitespace left behind
    cleaned_text = re.sub(r"\n\n\n+", "\n\n", cleaned_text)
    cleaned_text = cleaned_text.strip()

    return cleaned_text, thinking_blocks


def compute_thinking_stats(thinking_blocks: list[str]) -> dict:
    """
    Compute statistics about thinking blocks.
    Handles both <think> and <thinking> tags.

    Args:
        thinking_blocks: List of thinking block strings (including tags)

    Returns:
        dict: Statistics including:
            - block_count: Number of thinking blocks
            - total_characters: Character count of thinking content
            - total_words: Word count of thinking content
            - total_lines: Line count of thinking content
            - avg_words_per_block: Average words per thinking block

    Example:
        >>> blocks = ["<think>This is reasoning...</think>"]
        >>> stats = compute_thinking_stats(blocks)
        >>> stats['block_count']
        1
    """
    if not thinking_blocks:
        return {}

    # Concatenate all thinking blocks
    full_thinking = "\n".join(thinking_blocks)

    # Strip the tags for content analysis (handles both <think> and <thinking>)
    content_only = re.sub(r"</?think(?:ing)?>", "", full_thinking)

    # Compute stats
    char_count = len(content_only)
    word_count = len(content_only.split())
    line_count = len(content_only.strip().split("\n"))
    block_count = len(thinking_blocks)

    return {
        "block_count": block_count,
        "total_characters": char_count,
        "total_words": word_count,
        "total_lines": line_count,
        "avg_words_per_block": word_count / block_count if block_count > 0 else 0,
    }


def strip_markdown(text: str) -> str:
    """
    Strip markdown formatting from text for TTS readability.

    Removes:
    - Code blocks (```...```)
    - Inline code (`...`)
    - Bold/italic markers (**, __, *, _)
    - Headers (#, ##, etc.)
    - Links [text](url) - keeps text only
    - Images ![alt](url)
    - Lists markers (-, *, 1., etc.)
    - Blockquotes (>)
    - Horizontal rules
    - HTML tags

    Args:
        text: Markdown-formatted text

    Returns:
        str: Plain text suitable for TTS

    Example:
        >>> text = "**Bold** and `code` here"
        >>> strip_markdown(text)
        'Bold and code here'
    """
    # Remove code blocks (triple backticks with optional language)
    text = re.sub(r"```[a-z]*\n.*?\n```", "", text, flags=re.DOTALL)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

    # Remove inline code
    text = re.sub(r"`([^`]+)`", r"\1", text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Convert links [text](url) to just text
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)

    # Remove images ![alt](url)
    text = re.sub(r"!\[([^\]]*)\]\([^\)]+\)", "", text)

    # Remove bold/italic markers (process longer patterns first to avoid partial matches)
    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"\1", text)  # Bold+italic ***text***
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)  # Bold **text**
    text = re.sub(r"___(.+?)___", r"\1", text)  # Bold+italic ___text___
    text = re.sub(r"__(.+?)__", r"\1", text)  # Bold __text__
    text = re.sub(
        r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", text
    )  # Italic *text* (not part of **)
    text = re.sub(
        r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)", r"\1", text
    )  # Italic _text_ (not part of __)

    # Remove any remaining stray asterisks (handles malformed markdown, standalone *, **, etc.)
    text = re.sub(r"\*+", "", text)

    # Remove headers (# Header) - keep the text
    text = re.sub(r"^#{1,6}\s+(.+)$", r"\1", text, flags=re.MULTILINE)

    # Remove blockquote markers
    text = re.sub(r"^>\s+", "", text, flags=re.MULTILINE)

    # Remove horizontal rules
    text = re.sub(r"^[-*_]{3,}$", "", text, flags=re.MULTILINE)

    # Remove list markers but keep the text
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)  # Unordered lists
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)  # Ordered lists

    # Clean up excessive whitespace
    text = re.sub(r"\n\n\n+", "\n\n", text)
    text = re.sub(r" +", " ", text)
    text = text.strip()

    return text
