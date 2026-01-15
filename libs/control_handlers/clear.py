"""
Clear Handler

This handler clears all conversation history and context data,
deleting records from both chat.db and context.db.
"""

import os
import sys
from typing import Any, Dict

# Add parent directory to path for libs import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from libs.chat_store import ChatStore
from libs.config import Config
from libs.context_store import ContextStore

# Slash command metadata for API exposure
DESCRIPTION = "Clear all conversation history and context data"

ARGUMENTS: list[dict[str, Any]] = []


def handle(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clear all conversation history and context data.

    Deletes all records from both the chat database (conversations and messages)
    and the context database (retrieved context chunks).

    Args:
        payload: Dict (currently unused, no arguments required)

    Returns:
        Dict with:
            - conversations_deleted: Number of conversations deleted
            - messages_deleted: Number of messages deleted
            - contexts_deleted: Number of context chunks deleted
            - message: Human-readable status message
    """
    config = Config()

    # Get database paths from config
    chat_db_path = config.get_path("CHAT_DB_PATH", "./cortex/data/chat.db")
    context_db_path = config.get_path(
        "CONTEXT_DB_PATH", "./hippocampus/data/context.db"
    )

    conversations_deleted = 0
    messages_deleted = 0
    contexts_deleted = 0

    # Clear chat database
    if chat_db_path and os.path.exists(chat_db_path):
        chat_store = ChatStore(str(chat_db_path))
        conversations_deleted, messages_deleted = chat_store.clear_all()

    # Clear context database
    if context_db_path and os.path.exists(context_db_path):
        context_store = ContextStore(str(context_db_path))
        contexts_deleted = context_store.clear_all()

    total_deleted = conversations_deleted + messages_deleted + contexts_deleted

    return {
        "conversations_deleted": conversations_deleted,
        "messages_deleted": messages_deleted,
        "contexts_deleted": contexts_deleted,
        "message": f"Cleared {total_deleted} records ({conversations_deleted} conversations, "
        f"{messages_deleted} messages, {contexts_deleted} contexts)",
    }
