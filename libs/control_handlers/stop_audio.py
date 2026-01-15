"""
Stop Audio Handler

This handler stops currently playing Broca audio and mutes future audio
production until resumed with /resume-audio.
"""

import os
import signal
from typing import Any, Dict

from libs.state import get_state

# Slash command metadata for API exposure
DESCRIPTION = "Stop audio playback and mute Broca"

ARGUMENTS = [
    {
        "name": "force",
        "type": "bool",
        "required": False,
        "default": False,
        "description": "If True, use SIGKILL instead of SIGTERM for immediate termination",
    },
]


def handle(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stop currently playing Broca audio and mute future audio production.

    Reads the current audio playback PID from state and sends a termination
    signal to stop playback. Also sets the muted state to prevent Broca from
    producing new audio until /resume-audio is called.

    Args:
        payload: Dict with optional fields:
            - force (bool): If True, use SIGKILL instead of SIGTERM.
                           Default is False.

    Returns:
        Dict with:
            - pid_killed: PID that was killed (or None if nothing was playing)
            - was_playing: Whether audio was playing when command received
            - muted: Always True after this command
            - message: Human-readable status message
    """
    state = get_state()

    # Get current audio state
    pid = state.get_int("broca.audio_pid")
    was_playing = state.get_bool("broca.is_playing", default=False)

    # Set muted state to prevent future audio production
    state.set_bool("broca.audio_muted", True)

    # If no process to kill, just return with muted confirmation
    if pid is None:
        return {
            "pid_killed": None,
            "was_playing": was_playing,
            "muted": True,
            "message": "Audio muted (no audio was playing)",
        }

    # Determine signal to send
    force = payload.get("force", False)
    sig = signal.SIGKILL if force else signal.SIGTERM
    sig_name = "SIGKILL" if force else "SIGTERM"

    # Attempt to kill the process
    try:
        os.kill(pid, sig)
        message = f"Sent {sig_name} to process {pid}, audio muted"
    except ProcessLookupError:
        # Process already dead - that's fine
        message = f"Process {pid} was already terminated, audio muted"
    except PermissionError:
        # Can't kill process - permission denied, but still muted
        return {
            "pid_killed": None,
            "was_playing": was_playing,
            "muted": True,
            "message": f"Permission denied killing process {pid}, but audio muted",
        }

    # Clear the audio state
    state.delete("broca.audio_pid")
    state.set_bool("broca.is_playing", False)

    return {
        "pid_killed": pid,
        "was_playing": was_playing,
        "muted": True,
        "message": message,
    }
