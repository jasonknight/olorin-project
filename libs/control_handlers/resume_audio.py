"""
Resume Audio Handler

This handler unmutes Broca audio production, allowing it to generate
and play audio for incoming messages again.
"""

from typing import Any, Dict

from libs.state import get_state

# Slash command metadata for API exposure
DESCRIPTION = "Resume Broca audio production"

ARGUMENTS = []


def handle(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resume Broca audio production.

    Clears the muted state, allowing Broca to generate and play audio
    for incoming messages.

    Args:
        payload: Dict (unused, but required for handler interface)

    Returns:
        Dict with:
            - was_muted: Whether audio was muted before this command
            - muted: Always False after this command
            - message: Human-readable status message
    """
    state = get_state()

    # Get current muted state
    was_muted = state.get_bool("broca.audio_muted", default=False)

    # Clear muted state
    state.set_bool("broca.audio_muted", False)

    if was_muted:
        message = "Audio production resumed"
    else:
        message = "Audio was not muted"

    return {
        "was_muted": was_muted,
        "muted": False,
        "message": message,
    }
