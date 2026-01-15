#!/usr/bin/env python3
"""
Script to fix Kafka topic timestamp configuration.
This changes the topic to use LogAppendTime (broker time) instead of CreateTime (client time).
"""

import subprocess
import sys
import os

# Add parent directory to path for libs import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from libs.config import Config

config = Config()

KAFKA_TOPIC = config.get("BROCA_KAFKA_TOPIC", "ai_out")
BOOTSTRAP_SERVERS = config.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")


def run_command(cmd):
    """Run a shell command and return output"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.returncode == 0


def main():
    print(f"Fixing timestamp configuration for topic: {KAFKA_TOPIC}")
    print(f"Broker: {BOOTSTRAP_SERVERS}\n")

    # Option 1: Alter the topic to use LogAppendTime (broker assigns timestamps)
    print("Setting topic to use LogAppendTime (broker-assigned timestamps)...")
    cmd = f"kafka-configs --bootstrap-server {BOOTSTRAP_SERVERS} --entity-type topics --entity-name {KAFKA_TOPIC} --alter --add-config message.timestamp.type=LogAppendTime"

    if run_command(cmd):
        print("\n✓ Topic configuration updated successfully!")
        print("Messages will now use broker time instead of client time.")
        print("This prevents timestamp validation errors due to clock skew.")
    else:
        print("\n✗ Failed to update topic configuration.")
        print("\nAlternative: Manually run this command:")
        print(f"\n  {cmd}\n")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
