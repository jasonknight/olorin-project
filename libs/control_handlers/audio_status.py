"""
Audio Status Handler

This handler returns the current state of Broca audio:
muted status, playing status, and audio process PID (if any).
"""

from typing import Any, Dict

from libs.state import get_state

# Slash command metadata for API exposure
DESCRIPTION = "Get Broca audio status"

ARGUMENTS = []


def handle(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get current Broca audio status.

    Returns the muted state, playing status, and audio process PID.

    Args:
        payload: Dict (unused, but required for handler interface)

    Returns:
        Dict with:
            - muted: Whether audio production is muted
            - is_playing: Whether audio is currently playing
            - audio_pid: PID of the audio process (or None if not playing)
            - message: Human-readable status summary
    """
    state = get_state()

    # Get all audio-related state
    muted = state.get_bool("broca.audio_muted", default=False)
    is_playing = state.get_bool("broca.is_playing", default=False)
    audio_pid = state.get_int("broca.audio_pid")

    # Build status message
    status_parts = []
    if muted:
        status_parts.append("muted")
    else:
        status_parts.append("unmuted")

    if is_playing and audio_pid:
        status_parts.append(f"playing (PID {audio_pid})")
    elif is_playing:
        status_parts.append("playing")
    else:
        status_parts.append("idle")

    message = "Audio is " + ", ".join(status_parts)

    return {
        "muted": muted,
        "is_playing": is_playing,
        "audio_pid": audio_pid,
        "message": message,
    }
