#!/usr/bin/env python3
"""
AI Tool Server: Embeddings

A simple HTTP server that implements the Olorin tool protocol for text embeddings.
Loads the embedding model once on startup and serves embedding requests via HTTP.

Endpoints:
- GET  /health   - Health check
- GET  /describe - Tool metadata
- POST /call     - Execute embedding

Usage:
    python server.py [--port 8771] [--host 127.0.0.1]
"""

import argparse
import logging
import sys
from pathlib import Path

from flask import Flask, jsonify, request

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from libs.config import Config  # noqa: E402
from libs.embeddings_local import LocalEmbedder  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global embedder instance (loaded on startup)
embedder: LocalEmbedder | None = None


def init_embedder() -> bool:
    """Initialize the local embedder (loads the model directly)."""
    global embedder
    try:
        config = Config()
        embedder = LocalEmbedder(config=config)
        logger.info(
            f"Embedder initialized: {embedder.model_name} (dim: {embedder.dimension})"
        )
        return True
    except Exception as e:
        logger.error(f"Failed to initialize embedder: {e}")
        return False


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    if embedder is None:
        return jsonify({"status": "error", "message": "Embedder not initialized"}), 503
    return jsonify({"status": "ok"})


@app.route("/describe", methods=["GET"])
def describe():
    """Return tool metadata for AI function calling."""
    return jsonify(
        {
            "name": "embeddings",
            "description": "Generate text embeddings for documents or queries. Use mode='document' for storing content and mode='query' for searching.",
            "parameters": [
                {
                    "name": "texts",
                    "type": "array",
                    "required": True,
                    "description": "List of text strings to embed",
                },
                {
                    "name": "mode",
                    "type": "string",
                    "required": False,
                    "description": "Embedding mode: 'document' (default) or 'query'. Documents use storage-optimized prefixes, queries use search-optimized prefixes.",
                },
            ],
        }
    )


@app.route("/call", methods=["POST"])
def call():
    """Execute embedding generation."""
    if embedder is None:
        return (
            jsonify(
                {
                    "success": False,
                    "error": {
                        "type": "ServiceUnavailable",
                        "message": "Embedder not initialized",
                    },
                }
            ),
            503,
        )

    try:
        data = request.get_json()
        if not data:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": {
                            "type": "ValidationError",
                            "message": "Request body must be JSON",
                        },
                    }
                ),
                400,
            )

        # Extract parameters
        texts = data.get("texts")
        mode = data.get("mode", "document")

        # Validate texts
        if texts is None:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": {
                            "type": "ValidationError",
                            "message": "Missing required parameter: texts",
                        },
                    }
                ),
                400,
            )

        if not isinstance(texts, list):
            return (
                jsonify(
                    {
                        "success": False,
                        "error": {
                            "type": "ValidationError",
                            "message": "Parameter 'texts' must be a list",
                        },
                    }
                ),
                400,
            )

        if not all(isinstance(t, str) for t in texts):
            return (
                jsonify(
                    {
                        "success": False,
                        "error": {
                            "type": "ValidationError",
                            "message": "All items in 'texts' must be strings",
                        },
                    }
                ),
                400,
            )

        # Validate mode
        if mode not in ("document", "query"):
            return (
                jsonify(
                    {
                        "success": False,
                        "error": {
                            "type": "ValidationError",
                            "message": "Parameter 'mode' must be 'document' or 'query'",
                        },
                    }
                ),
                400,
            )

        # Handle empty list
        if not texts:
            return jsonify(
                {
                    "success": True,
                    "result": {
                        "embeddings": [],
                        "model": embedder.model_name,
                        "dimension": embedder.dimension,
                    },
                }
            )

        # Generate embeddings
        embeddings = embedder.embed_batch(texts, mode=mode, as_list=True)

        return jsonify(
            {
                "success": True,
                "result": {
                    "embeddings": embeddings,
                    "model": embedder.model_name,
                    "dimension": embedder.dimension,
                },
            }
        )

    except Exception as e:
        logger.exception(f"Error generating embeddings: {e}")
        return (
            jsonify(
                {
                    "success": False,
                    "error": {"type": "InternalError", "message": str(e)},
                }
            ),
            500,
        )


@app.route("/info", methods=["GET"])
def info():
    """Return information about the loaded model."""
    if embedder is None:
        return jsonify({"error": "Embedder not initialized"}), 503
    return jsonify(
        {
            "model": embedder.model_name,
            "dimension": embedder.dimension,
            "document_prefix": embedder.document_prefix,
            "query_prefix": embedder.query_prefix,
        }
    )


def main():
    parser = argparse.ArgumentParser(
        description="AI Tool Server: Embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--port", type=int, default=8771, help="Port to listen on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    args = parser.parse_args()

    # Initialize embedder before starting server
    print("Initializing embedding model...")
    if not init_embedder():
        print("Failed to initialize embedder. Exiting.")
        sys.exit(1)

    print(f"Embedding tool server listening on http://{args.host}:{args.port}")
    print(f"  Model: {embedder.model_name}")
    print(f"  Dimension: {embedder.dimension}")
    print("  GET  /health   - Health check")
    print("  GET  /describe - Tool metadata")
    print("  GET  /info     - Model information")
    print("  POST /call     - Generate embeddings")

    # Run with waitress for production-ready WSGI server
    try:
        from waitress import serve

        serve(app, host=args.host, port=args.port)
    except ImportError:
        # Fall back to Flask development server
        logger.warning("waitress not installed, using Flask development server")
        app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
