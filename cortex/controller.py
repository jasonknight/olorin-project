#!/usr/bin/env python3
"""
Control Controller Service for Olorin Project

HTTP API server that exposes control handlers as endpoints for cross-component
and cross-language communication.

Endpoints:
    GET  /commands          - List all available commands with metadata
    GET  /commands/{name}   - Get metadata for a specific command
    POST /execute           - Execute a command
    GET  /health            - Health check endpoint
"""

import os
import sys

# Add parent directory to path for libs import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from libs.config import Config
from libs.control_handlers import discover_handlers
from libs.control_server import ControlServer
from libs.olorin_logging import OlorinLogger

# Initialize config with hot-reload support
config = Config(watch=True)

# Set up logging
default_log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
log_dir = config.get("LOG_DIR", default_log_dir)
log_file = os.path.join(log_dir, "cortex-controller.log")
env_log_level = config.get("LOG_LEVEL", "INFO")

# Initialize logger
logger = OlorinLogger(log_file=log_file, log_level=env_log_level, name=__name__)


def main() -> None:
    """Run the control API server."""
    logger.info("=" * 60)
    logger.info("CONTROL CONTROLLER APPLICATION STARTING")
    logger.info("=" * 60)
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")

    # Load configuration
    api_port = config.get_int("CONTROL_API_PORT", 8765)
    api_host = config.get("CONTROL_API_HOST", "0.0.0.0")
    api_enabled = config.get_bool("CONTROL_API_ENABLED", True)

    if not api_enabled:
        logger.warning("Control API is disabled in configuration. Exiting.")
        return

    logger.info("Configuration loaded:")
    logger.info(f"  API Host: {api_host}")
    logger.info(f"  API Port: {api_port}")

    # Discover handlers
    logger.info("Discovering control handlers...")
    handlers = discover_handlers()
    logger.info(f"Discovered {len(handlers)} handler(s):")
    for cmd in sorted(handlers.keys()):
        logger.info(f"  - {cmd}")

    # Create and start the server
    server = ControlServer(port=api_port, host=api_host)

    logger.info("=" * 60)
    logger.info(f"Starting Control API server on {api_host}:{api_port}")
    logger.info(f"  GET  {server.url}/health")
    logger.info(f"  GET  {server.url}/commands")
    logger.info(f"  GET  {server.url}/commands/<name>")
    logger.info(f"  POST {server.url}/execute")
    logger.info("=" * 60)

    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise
    finally:
        server.stop()
        logger.info("=" * 60)
        logger.info("CONTROL CONTROLLER APPLICATION EXITING")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
