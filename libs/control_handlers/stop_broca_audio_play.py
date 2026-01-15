"""
Stop Broca Audio Playback Handler

This handler stops currently playing Broca audio by sending a signal
to the audio process identified in the state database.
"""

import os
import signal
from typing import Any, Dict

from libs.state import get_state

# Slash command metadata for API exposure
DESCRIPTION = "Stop currently playing Broca audio"

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
    Stop currently playing Broca audio.

    Reads the current audio playback PID from state and sends a termination
    signal to stop playback. Clears the state after stopping.

    Args:
        payload: Dict with optional fields:
            - force (bool): If True, use SIGKILL instead of SIGTERM.
                           Default is False.

    Returns:
        Dict with:
            - pid_killed: PID that was killed (or None if nothing was playing)
            - was_playing: Whether audio was playing when command received
            - message: Human-readable status message
    """
    state = get_state()

    # Get current audio state
    pid = state.get_int("broca.audio_pid")
    was_playing = state.get_bool("broca.is_playing", default=False)

    # If no process to kill
    if pid is None:
        return {
            "pid_killed": None,
            "was_playing": was_playing,
            "message": "No audio process was playing",
        }

    # Determine signal to send
    force = payload.get("force", False)
    sig = signal.SIGKILL if force else signal.SIGTERM
    sig_name = "SIGKILL" if force else "SIGTERM"

    # Attempt to kill the process
    try:
        os.kill(pid, sig)
        message = f"Sent {sig_name} to process {pid}"
    except ProcessLookupError:
        # Process already dead - that's fine
        message = f"Process {pid} was already terminated"
    except PermissionError:
        # Can't kill process - permission denied
        return {
            "pid_killed": None,
            "was_playing": was_playing,
            "message": f"Permission denied killing process {pid}",
        }

    # Clear the state
    state.delete("broca.audio_pid")
    state.set_bool("broca.is_playing", False)

    return {
        "pid_killed": pid,
        "was_playing": was_playing,
        "message": message,
    }
