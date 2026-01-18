#!/usr/bin/env python3
"""
AI Tool Server: ChromaDB Search

A simple HTTP server that implements the Olorin tool protocol for searching
the ChromaDB knowledge base. Uses the embeddings tool to generate query
embeddings and returns paginated results.

Endpoints:
- GET  /health   - Health check
- GET  /describe - Tool metadata
- POST /call     - Execute search

Usage:
    python server.py [--port 8772] [--host 127.0.0.1]
"""

import argparse
import logging
import math
import sys
import uuid
from pathlib import Path
from typing import Any

import chromadb
import requests
from chromadb.config import Settings
from flask import Flask, jsonify, request
from werkzeug.exceptions import UnsupportedMediaType

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from libs.config import Config  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global configuration and clients
config: Config | None = None
chroma_client: Any = None
embeddings_url: str = ""
chromadb_host: str = ""
chromadb_port: int = 0

# Constants
DEFAULT_PER_PAGE = 10
MAX_PER_PAGE = 100
DEFAULT_COLLECTION = "documents"


def init_config():
    """Initialize configuration and client settings (always succeeds)."""
    global config, chroma_client, embeddings_url, chromadb_host, chromadb_port

    config = Config()

    # Set up ChromaDB client configuration (store for lazy connection)
    chromadb_host = config.get("CHROMADB_HOST", "localhost")
    chromadb_port = config.get_int("CHROMADB_PORT", 8000)

    # Try to create ChromaDB client, but don't fail if it's unavailable
    # The client will be created lazily on first request if needed
    try:
        chroma_client = chromadb.HttpClient(
            host=chromadb_host,
            port=chromadb_port,
            settings=Settings(anonymized_telemetry=False),
        )
        logger.info(f"ChromaDB client connected to {chromadb_host}:{chromadb_port}")
    except Exception as e:
        chroma_client = None
        logger.warning(
            f"ChromaDB at {chromadb_host}:{chromadb_port} not available at startup: {e}. "
            "Will retry on requests."
        )

    # Set up embeddings tool URL
    embeddings_port = config.get_int("EMBEDDINGS_TOOL_PORT", 8771)
    embeddings_host = config.get("EMBEDDINGS_TOOL_HOST", "localhost")
    embeddings_url = f"http://{embeddings_host}:{embeddings_port}"
    logger.info(f"Embeddings tool configured at {embeddings_url}")

    # Test embeddings connection at startup (informational only)
    _check_embeddings_connection()


def _ensure_chromadb_client():
    """Ensure ChromaDB client is connected, attempting reconnection if needed.

    Returns the client if connected, None otherwise.
    """
    global chroma_client

    # If we have a client, verify it's still connected
    if chroma_client is not None:
        try:
            chroma_client.heartbeat()
            return chroma_client
        except Exception:
            # Connection lost, will try to reconnect below
            chroma_client = None

    # Try to connect/reconnect
    try:
        chroma_client = chromadb.HttpClient(
            host=chromadb_host,
            port=chromadb_port,
            settings=Settings(anonymized_telemetry=False),
        )
        logger.info(f"ChromaDB client reconnected to {chromadb_host}:{chromadb_port}")
        return chroma_client
    except Exception as e:
        logger.warning(f"ChromaDB connection failed: {e}")
        chroma_client = None
        return None


def _check_embeddings_connection() -> bool:
    """Check if embeddings tool is reachable. Returns True if healthy."""
    try:
        resp = requests.get(f"{embeddings_url}/health", timeout=5)
        if resp.status_code == 200:
            logger.info("Embeddings tool connection: healthy")
            return True
        else:
            logger.warning(
                f"Embeddings tool connection: unhealthy (status {resp.status_code})"
            )
            return False
    except requests.RequestException as e:
        logger.warning(f"Embeddings tool connection: unavailable ({e})")
        return False


def get_embedding(query: str) -> list[float] | None:
    """Get query embedding from the embeddings tool."""
    try:
        resp = requests.post(
            f"{embeddings_url}/call",
            json={"texts": [query], "mode": "query"},
            timeout=30,
        )

        if resp.status_code != 200:
            logger.error(f"Embeddings tool returned {resp.status_code}")
            return None

        data = resp.json()
        if not data.get("success"):
            logger.error(f"Embeddings tool error: {data.get('error')}")
            return None

        embeddings = data.get("result", {}).get("embeddings", [])
        if not embeddings:
            logger.error("No embeddings returned")
            return None

        return embeddings[0]

    except requests.RequestException as e:
        logger.error(f"Failed to get embedding: {e}")
        return None


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint.

    Returns 200 OK if the service is running, regardless of dependency status.
    Dependency status is included in the response for monitoring.
    """
    # Service is always "ok" if it's responding - dependencies are checked separately
    status = {"status": "ok", "dependencies": {}}

    # Check ChromaDB (attempts reconnection if previously unavailable)
    client = _ensure_chromadb_client()
    if client is not None:
        status["dependencies"]["chromadb"] = "healthy"
    else:
        status["dependencies"]["chromadb"] = (
            f"unavailable ({chromadb_host}:{chromadb_port})"
        )

    # Check embeddings tool
    try:
        resp = requests.get(f"{embeddings_url}/health", timeout=2)
        if resp.status_code == 200:
            status["dependencies"]["embeddings"] = "healthy"
        else:
            status["dependencies"]["embeddings"] = (
                f"unhealthy: status {resp.status_code}"
            )
    except requests.RequestException as e:
        status["dependencies"]["embeddings"] = f"unavailable: {e}"

    return jsonify(status)


@app.route("/describe", methods=["GET"])
def describe():
    """Return tool metadata for AI function calling."""
    return jsonify(
        {
            "name": "search",
            "description": "Search the ChromaDB knowledge base using semantic similarity or source path matching. Returns paginated results.",
            "parameters": [
                {
                    "name": "query",
                    "type": "string",
                    "required": True,
                    "description": "Search query text",
                },
                {
                    "name": "mode",
                    "type": "string",
                    "required": False,
                    "description": "Search mode: 'semantic' (default) for embedding-based similarity, 'source' for filename/path substring match",
                },
                {
                    "name": "page",
                    "type": "integer",
                    "required": False,
                    "description": "Page number (1-indexed, default: 1)",
                },
                {
                    "name": "per_page",
                    "type": "integer",
                    "required": False,
                    "description": f"Results per page (default: {DEFAULT_PER_PAGE}, max: {MAX_PER_PAGE})",
                },
                {
                    "name": "collection",
                    "type": "string",
                    "required": False,
                    "description": f"Collection name (default: {DEFAULT_COLLECTION})",
                },
                {
                    "name": "include_distances",
                    "type": "boolean",
                    "required": False,
                    "description": "Include similarity distances in results (default: true, only applies to semantic mode)",
                },
            ],
        }
    )


@app.route("/call", methods=["POST"])
def call():
    """Execute search query."""
    # Ensure ChromaDB client is connected (will attempt reconnection if needed)
    client = _ensure_chromadb_client()
    if client is None:
        return (
            jsonify(
                {
                    "success": False,
                    "error": {
                        "type": "ServiceUnavailable",
                        "message": f"ChromaDB at {chromadb_host}:{chromadb_port} is unavailable",
                    },
                }
            ),
            503,
        )

    try:
        data = request.get_json()
    except UnsupportedMediaType:
        return (
            jsonify(
                {
                    "success": False,
                    "error": {
                        "type": "ValidationError",
                        "message": "Request Content-Type must be application/json",
                    },
                }
            ),
            400,
        )

    try:
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

        # Extract and validate parameters
        query = data.get("query")
        if not query or not isinstance(query, str):
            return (
                jsonify(
                    {
                        "success": False,
                        "error": {
                            "type": "ValidationError",
                            "message": "Missing or invalid required parameter: query",
                        },
                    }
                ),
                400,
            )

        page = data.get("page", 1)
        if not isinstance(page, int) or page < 1:
            page = 1

        per_page = data.get("per_page", DEFAULT_PER_PAGE)
        if not isinstance(per_page, int) or per_page < 1:
            per_page = DEFAULT_PER_PAGE
        per_page = min(per_page, MAX_PER_PAGE)

        collection_name = data.get("collection", DEFAULT_COLLECTION)
        if not isinstance(collection_name, str):
            collection_name = DEFAULT_COLLECTION

        include_distances = data.get("include_distances", True)
        if not isinstance(include_distances, bool):
            include_distances = True

        mode = data.get("mode", "semantic")
        if mode not in ("semantic", "source"):
            mode = "semantic"

        # Get collection
        try:
            collection = client.get_collection(name=collection_name)
        except ValueError as e:
            # Collection not found
            return (
                jsonify(
                    {
                        "success": False,
                        "error": {
                            "type": "NotFoundError",
                            "message": f"Collection '{collection_name}' not found: {e}",
                        },
                    }
                ),
                404,
            )
        except Exception as e:
            # Connection or other error
            logger.warning(f"ChromaDB error getting collection: {e}")
            return (
                jsonify(
                    {
                        "success": False,
                        "error": {
                            "type": "ServiceUnavailable",
                            "message": f"ChromaDB error: {e}",
                        },
                    }
                ),
                503,
            )

        # Get total count for pagination info
        total_results = collection.count()

        if total_results == 0:
            return jsonify(
                {
                    "success": True,
                    "result": {
                        "query": query,
                        "mode": mode,
                        "total_results": 0,
                        "total_pages": 0,
                        "page": page,
                        "per_page": per_page,
                        "results": [],
                    },
                }
            )

        # Handle source search mode (substring match on source field)
        if mode == "source":
            # Get all documents with metadata
            all_docs = collection.get(include=["documents", "metadatas"])

            ids = all_docs.get("ids", [])
            documents = all_docs.get("documents", [])
            metadatas = all_docs.get("metadatas", [])

            # Filter by source containing query (case-insensitive)
            query_lower = query.lower()
            filtered_results = []
            for doc_id, doc_text, metadata in zip(ids, documents, metadatas):
                source = metadata.get("source", "") if metadata else ""
                if source and query_lower in source.lower():
                    filtered_results.append(
                        {
                            "id": doc_id,
                            "text": doc_text,
                            "metadata": metadata,
                        }
                    )

            # Calculate pagination for filtered results
            filtered_total = len(filtered_results)
            total_pages = (
                math.ceil(filtered_total / per_page) if filtered_total > 0 else 0
            )
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page

            # Slice to get just the requested page
            page_results = filtered_results[start_idx:end_idx]

            return jsonify(
                {
                    "success": True,
                    "result": {
                        "query": query,
                        "mode": mode,
                        "total_results": filtered_total,
                        "total_pages": total_pages,
                        "page": page,
                        "per_page": per_page,
                        "results": page_results,
                    },
                }
            )

        # Semantic search mode (default)
        # Get query embedding
        query_embedding = get_embedding(query)
        if query_embedding is None:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": {
                            "type": "ServiceUnavailable",
                            "message": "Failed to get query embedding from embeddings tool",
                        },
                    }
                ),
                503,
            )

        # ChromaDB doesn't support offset, so we fetch enough results to cover
        # the requested page and then slice
        n_results_needed = page * per_page
        n_results_to_fetch = min(n_results_needed, total_results)

        # Query ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results_to_fetch,
        )

        # Extract results from nested structure
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        # Calculate pagination
        total_pages = math.ceil(total_results / per_page)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page

        # Slice to get just the requested page
        page_ids = ids[start_idx:end_idx]
        page_documents = documents[start_idx:end_idx]
        page_metadatas = metadatas[start_idx:end_idx]
        page_distances = distances[start_idx:end_idx]

        # Build result list
        result_list = []
        for doc_id, doc_text, metadata, distance in zip(
            page_ids, page_documents, page_metadatas, page_distances
        ):
            result_item = {
                "id": doc_id,
                "text": doc_text,
                "metadata": metadata,
            }
            if include_distances:
                result_item["distance"] = distance
            result_list.append(result_item)

        return jsonify(
            {
                "success": True,
                "result": {
                    "query": query,
                    "mode": mode,
                    "total_results": total_results,
                    "total_pages": total_pages,
                    "page": page,
                    "per_page": per_page,
                    "results": result_list,
                },
            }
        )

    except Exception as e:
        logger.exception(f"Error executing search: {e}")
        return (
            jsonify(
                {
                    "success": False,
                    "error": {"type": "InternalError", "message": str(e)},
                }
            ),
            500,
        )


@app.route("/add", methods=["POST"])
def add():
    """Add a document to ChromaDB.

    This endpoint adds a single document to the ChromaDB collection,
    generating an embedding for semantic search. Used for manual
    user-provided context entries.
    """
    # Ensure ChromaDB client is connected
    client = _ensure_chromadb_client()
    if client is None:
        return (
            jsonify(
                {
                    "success": False,
                    "error": {
                        "type": "ServiceUnavailable",
                        "message": f"ChromaDB at {chromadb_host}:{chromadb_port} is unavailable",
                    },
                }
            ),
            503,
        )

    try:
        data = request.get_json()
    except UnsupportedMediaType:
        return (
            jsonify(
                {
                    "success": False,
                    "error": {
                        "type": "ValidationError",
                        "message": "Request Content-Type must be application/json",
                    },
                }
            ),
            400,
        )

    try:
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

        # Extract and validate parameters
        text = data.get("text")
        if not text or not isinstance(text, str):
            return (
                jsonify(
                    {
                        "success": False,
                        "error": {
                            "type": "ValidationError",
                            "message": "Missing or invalid required parameter: text",
                        },
                    }
                ),
                400,
            )

        # Source defaults to "User Context"
        source = data.get("source", "User Context")
        if not isinstance(source, str):
            source = "User Context"

        # ID is optional, generate a UUID if not provided
        doc_id = data.get("id")
        if not doc_id or not isinstance(doc_id, str):
            doc_id = f"user-context-{uuid.uuid4()}"

        collection_name = data.get("collection", DEFAULT_COLLECTION)
        if not isinstance(collection_name, str):
            collection_name = DEFAULT_COLLECTION

        # Get or create collection
        try:
            collection = client.get_or_create_collection(name=collection_name)
        except Exception as e:
            logger.warning(f"ChromaDB error getting/creating collection: {e}")
            return (
                jsonify(
                    {
                        "success": False,
                        "error": {
                            "type": "ServiceUnavailable",
                            "message": f"ChromaDB error: {e}",
                        },
                    }
                ),
                503,
            )

        # Generate embedding for the document
        embedding = get_embedding(text)
        if embedding is None:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": {
                            "type": "ServiceUnavailable",
                            "message": "Failed to get embedding from embeddings tool",
                        },
                    }
                ),
                503,
            )

        # Add to ChromaDB
        collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[{"source": source}],
        )

        logger.info(f"Added document to ChromaDB: id={doc_id}, source={source}")

        return jsonify(
            {
                "success": True,
                "result": {
                    "id": doc_id,
                    "source": source,
                    "text_length": len(text),
                    "collection": collection_name,
                },
            }
        )

    except Exception as e:
        logger.exception(f"Error adding document: {e}")
        return (
            jsonify(
                {
                    "success": False,
                    "error": {"type": "InternalError", "message": str(e)},
                }
            ),
            500,
        )


def main():
    parser = argparse.ArgumentParser(
        description="AI Tool Server: ChromaDB Search",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--port", type=int, default=8772, help="Port to listen on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    args = parser.parse_args()

    # Initialize configuration (always succeeds - dependencies checked at request time)
    print("Initializing search tool configuration...")
    init_config()

    print(f"Search tool server listening on http://{args.host}:{args.port}")
    print("  GET  /health   - Health check")
    print("  GET  /describe - Tool metadata")
    print("  POST /call     - Execute search")
    print("  POST /add      - Add document to ChromaDB")

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
