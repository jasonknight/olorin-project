#!/usr/bin/env python3
"""Debug script to see what Whisper is transcribing from the microphone."""

import sys
import os
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from temporal.audio_capture import AudioCapture
from temporal.stt_engine import STTEngine


def main():
    print("Initializing audio capture...")
    audio = AudioCapture(sample_rate=16000, channels=1, chunk_duration=0.5)

    print("Loading Whisper model (this may take a moment)...")
    stt = STTEngine(
        model_size="small", device="cpu", compute_type="int8", language="en"
    )
    stt._load_model()

    print("\n" + "=" * 60)
    print("LISTENING - Speak and see what Whisper transcribes")
    print("Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    audio_buffer = []
    buffer_duration = 3.0  # seconds
    sample_rate = 16000

    with audio:
        try:
            while True:
                chunk = audio.get_chunk(timeout=1.0)
                if chunk is None:
                    continue

                audio_buffer.append(chunk)

                # Keep only last N seconds
                total_samples = sum(len(c) for c in audio_buffer)
                max_samples = int(buffer_duration * sample_rate)
                while total_samples > max_samples and len(audio_buffer) > 1:
                    removed = audio_buffer.pop(0)
                    total_samples -= len(removed)

                # Transcribe every second or so
                if total_samples >= sample_rate:
                    combined = np.concatenate(audio_buffer)

                    # Check audio level
                    level = np.abs(combined).mean()
                    level_db = 20 * np.log10(level + 1e-10)

                    text = stt.transcribe_streaming(combined, sample_rate)

                    if text.strip():
                        print(f"[Level: {level_db:+.1f}dB] Heard: '{text}'")

                        # Check for wake word using same logic as consumer
                        text_lower = text.lower()
                        olorin_variations = [
                            "olorin",
                            "alorin",
                            "oloreen",
                            "lauren",
                            "lorin",
                            "loren",
                            "lorean",
                            "o'loren",
                            "all around",
                            "a loren",
                            "oh lauren",
                            "oh loren",
                            "learn",
                        ]
                        has_hey = any(
                            h in text_lower
                            for h in ["hey", "hay", "hei", "hey,", "hey."]
                        )
                        has_olorin = any(var in text_lower for var in olorin_variations)

                        if has_hey and has_olorin:
                            print("  >>> WAKE WORD DETECTED! <<<")
                        elif has_hey:
                            print("  (heard 'hey', waiting for name...)")
                        elif has_olorin:
                            print("  (heard name variant, waiting for 'hey'...)")

                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n\nStopped.")


if __name__ == "__main__":
    main()
