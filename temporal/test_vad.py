#!/usr/bin/env python3
"""
Interactive VAD test script.
Tests silence detection after simulated wake word.

Usage: cd temporal && source venv/bin/activate && python3 test_vad.py
"""

import numpy as np
import sounddevice as sd

from vad import VoiceActivityDetector


def main():
    sample_rate = 16000
    chunk_duration = 0.5
    chunk_size = int(sample_rate * chunk_duration)

    print("=" * 70)
    print("VAD Interactive Test - Noise Floor Calibration")
    print("=" * 70)
    print()

    print("Initializing VAD...")
    vad = VoiceActivityDetector(
        threshold=0.5,
        sample_rate=sample_rate,
        silence_timeout=3.0,
        max_duration=30.0,
        calibration_chunks=4,
        speech_threshold_mult=2.0,
        silence_threshold_mult=1.3,
    )
    vad._load_model()
    print("Model loaded!")
    print()

    print("HOW IT WORKS:")
    print("1. First 4 chunks (2 sec) calibrate the noise floor")
    print("2. 'Real speech' detected when energy > 2x noise floor")
    print("3. 'Silence' detected when energy < 1.3x noise floor")
    print("4. Recording stops after 3 sec of silence OR if no speech for 3 sec")
    print()
    print("INSTRUCTIONS:")
    print("- BE QUIET during calibration (first 2 seconds)")
    print("- Then SPEAK clearly")
    print("- Then STOP speaking and wait for silence detection")
    print()
    input("Press ENTER when ready...")
    print()

    vad.reset(assume_speaking=True)
    print("CALIBRATING (stay quiet)...")
    print("-" * 70)

    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype=np.float32,
        blocksize=chunk_size,
    ) as stream:
        chunk_num = 0
        while True:
            audio_chunk, overflowed = stream.read(chunk_size)
            if overflowed:
                print("WARNING: Audio buffer overflow!")

            audio_chunk = audio_chunk.flatten()
            result = vad.process_chunk(audio_chunk)
            chunk_num += 1

            phase = result.get("phase", "unknown")
            energy = result["energy"]
            noise_floor = result.get("noise_floor", 0)
            speech_th = result.get("speech_threshold", 0)
            silence_th = result.get("silence_threshold", 0)
            is_speech = result["is_speech"]
            has_real = result.get("has_real_speech", False)
            silence = result["silence_duration"]
            should_stop = result["should_stop"]
            stop_reason = result.get("stop_reason", "")

            # Visual energy bar (scaled)
            bar_len = min(50, int(energy * 2000))
            energy_bar = "█" * bar_len + "░" * (50 - bar_len)

            if phase == "calibrating":
                print(f"CAL {chunk_num}: [{energy_bar}] energy={energy:.5f}")
                if chunk_num == 4:
                    print("-" * 70)
                    print(f"Noise floor calibrated: {noise_floor:.5f}")
                    print(f"Speech threshold (2x): {speech_th:.5f}")
                    print(f"Silence threshold (1.3x): {silence_th:.5f}")
                    print("-" * 70)
                    print("NOW SPEAK! (then stop and wait for silence detection)")
                    print("-" * 70)
            else:
                marker = "🎤 SPEECH" if is_speech else "🔇 quiet "
                real = "[REAL]" if has_real else "[    ]"
                print(
                    f"{chunk_num:3d}: [{energy_bar}] "
                    f"energy={energy:.5f} {marker} {real} "
                    f"silence={silence:.1f}s"
                )

            if should_stop:
                print()
                print("=" * 70)
                print(f">>> STOPPED: {stop_reason} <<<")
                print("=" * 70)
                break

    print()
    print("Test complete.")


if __name__ == "__main__":
    main()
