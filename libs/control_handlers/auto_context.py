"""
Auto-Context Handler

This handler toggles automatic context enrichment on or off.
When enabled, the enrichener will query hippocampus to add relevant
context to prompts before they reach the AI.
"""

import os
import sys
from typing import Any, Dict

# Add parent directory to path for libs import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from libs.state import State

# Slash command metadata for API exposure
DESCRIPTION = "Toggle automatic context enrichment on or off"

ARGUMENTS: list[dict[str, Any]] = [
    {
        "name": "enabled",
        "type": "str",
        "required": True,
        "description": "Whether to enable auto-context: 'on' or 'off'",
    }
]


def handle(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Toggle automatic context enrichment.

    Args:
        payload: Dict with:
            - enabled: "on" or "off" (required)

    Returns:
        Dict with:
            - auto_context: Boolean indicating new state
            - message: Human-readable status message
    """
    enabled_str = payload.get("enabled", "").lower().strip()

    if enabled_str not in ("on", "off"):
        return {
            "error": f"Invalid value '{enabled_str}'. Use 'on' or 'off'.",
            "auto_context": None,
        }

    enabled = enabled_str == "on"

    state = State()
    state.set_bool("enrichener.auto_context", enabled)

    status = "enabled" if enabled else "disabled"
    return {
        "auto_context": enabled,
        "message": f"Auto-context enrichment {status}",
    }
