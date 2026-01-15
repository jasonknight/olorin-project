"""
Write Handler

This handler writes the last assistant message to a markdown file
in the ~/Documents/AI_OUT directory.
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Add parent directory to path for libs import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from libs.chat_store import ChatStore
from libs.config import Config

# Slash command metadata for API exposure
DESCRIPTION = "Write the last assistant message to a markdown file"

ARGUMENTS: list[dict[str, Any]] = [
    {
        "name": "filename",
        "type": "str",
        "required": False,
        "default": None,
        "description": "Output filename (without .md extension). Defaults to current datetime.",
    }
]


def handle(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Write the last assistant message to a markdown file.

    Args:
        payload: Dict with optional 'filename' key

    Returns:
        Dict with:
            - success: Whether the write succeeded
            - file_path: Path to the written file
            - message: Human-readable status message
    """
    config = Config()

    # Get output directory - default to ~/Documents/AI_OUT
    output_dir = Path(os.path.expanduser("~/Documents/AI_OUT"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get database path from config
    chat_db_path = config.get_path("CHAT_DB_PATH", "./cortex/data/chat.db")

    if not chat_db_path or not os.path.exists(chat_db_path):
        return {
            "success": False,
            "file_path": None,
            "message": "Chat database not found",
        }

    # Connect to chat store and get active conversation
    chat_store = ChatStore(str(chat_db_path))
    conversation_id = chat_store.get_active_conversation_id()

    if not conversation_id:
        return {
            "success": False,
            "file_path": None,
            "message": "No active conversation found",
        }

    # Get all messages and find the last assistant message
    messages = chat_store.get_conversation_messages(conversation_id)
    assistant_messages = [m for m in messages if m["role"] == "assistant"]

    if not assistant_messages:
        return {
            "success": False,
            "file_path": None,
            "message": "No assistant messages found in current conversation",
        }

    last_assistant_message = assistant_messages[-1]
    content = last_assistant_message["content"]

    # Determine filename
    filename = payload.get("filename")
    if not filename:
        # Use current datetime as filename
        filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Ensure .md extension
    if not filename.endswith(".md"):
        filename = f"{filename}.md"

    # Write to file (overwrites if exists)
    file_path = output_dir / filename
    file_path.write_text(content, encoding="utf-8")

    return {
        "success": True,
        "file_path": str(file_path),
        "message": f"Written to {file_path}",
    }
