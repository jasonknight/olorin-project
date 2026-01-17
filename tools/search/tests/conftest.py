#!/usr/bin/env python3
"""
Pytest configuration and fixtures for search tool tests.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root and tool directory to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
TOOL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TOOL_DIR))


@pytest.fixture
def mock_config():
    """Fixture providing a mock Config object."""
    config = MagicMock()
    config.get.side_effect = lambda key, default=None: {
        "CHROMADB_HOST": "localhost",
        "EMBEDDINGS_TOOL_HOST": "localhost",
    }.get(key, default)
    config.get_int.side_effect = lambda key, default=None: {
        "CHROMADB_PORT": 8000,
        "EMBEDDINGS_TOOL_PORT": 8771,
    }.get(key, default)
    return config


@pytest.fixture
def mock_chroma_collection():
    """Fixture providing a mock ChromaDB collection."""
    collection = MagicMock()
    collection.count.return_value = 100
    return collection


@pytest.fixture
def mock_chroma_client(mock_chroma_collection):
    """Fixture providing a mock ChromaDB client."""
    client = MagicMock()
    client.heartbeat.return_value = True
    client.get_collection.return_value = mock_chroma_collection
    return client


@pytest.fixture
def sample_query_results():
    """Fixture providing sample ChromaDB query results."""
    return {
        "ids": [["doc1", "doc2", "doc3", "doc4", "doc5"]],
        "documents": [
            [
                "First document content",
                "Second document content",
                "Third document content",
                "Fourth document content",
                "Fifth document content",
            ]
        ],
        "metadatas": [
            [
                {"source": "/path/to/file1.md", "h1": "Title 1"},
                {"source": "/path/to/file2.md", "h1": "Title 2"},
                {"source": "/path/to/file3.md", "h1": "Title 3"},
                {"source": "/path/to/file4.md", "h1": "Title 4"},
                {"source": "/path/to/file5.md", "h1": "Title 5"},
            ]
        ],
        "distances": [[0.1, 0.2, 0.3, 0.4, 0.5]],
    }


@pytest.fixture
def large_query_results():
    """Fixture providing larger sample ChromaDB query results for pagination testing."""
    n = 25
    return {
        "ids": [[f"doc{i}" for i in range(n)]],
        "documents": [[f"Document {i} content" for i in range(n)]],
        "metadatas": [
            [{"source": f"/path/to/file{i}.md", "h1": f"Title {i}"} for i in range(n)]
        ],
        "distances": [[0.1 + (i * 0.01) for i in range(n)]],
    }


@pytest.fixture
def mock_embedding():
    """Fixture providing a mock embedding vector."""
    return [0.1] * 768  # 768-dimensional embedding


@pytest.fixture
def app(mock_chroma_client, mock_config):
    """Fixture providing Flask test app with mocked dependencies."""
    with patch("server.chromadb.HttpClient", return_value=mock_chroma_client):
        with patch("server.Config", return_value=mock_config):
            with patch("server.requests.get") as mock_get:
                # Mock embeddings health check
                mock_get.return_value.status_code = 200

                import server

                server.chroma_client = mock_chroma_client
                server.config = mock_config
                server.embeddings_url = "http://localhost:8771"

                server.app.config["TESTING"] = True
                yield server.app


@pytest.fixture
def client(app):
    """Fixture providing Flask test client."""
    return app.test_client()


@pytest.fixture
def initialized_app(mock_chroma_client, mock_config, mock_embedding):
    """Fixture providing fully initialized app with mocked embeddings."""
    with patch("server.chromadb.HttpClient", return_value=mock_chroma_client):
        with patch("server.Config", return_value=mock_config):
            with patch("server.requests.get") as mock_get:
                with patch("server.requests.post") as mock_post:
                    # Mock embeddings health check
                    mock_get.return_value.status_code = 200

                    # Mock embeddings call
                    mock_post.return_value.status_code = 200
                    mock_post.return_value.json.return_value = {
                        "success": True,
                        "result": {"embeddings": [mock_embedding]},
                    }

                    import server

                    server.chroma_client = mock_chroma_client
                    server.config = mock_config
                    server.embeddings_url = "http://localhost:8771"

                    server.app.config["TESTING"] = True
                    yield server.app, mock_chroma_client, mock_post


@pytest.fixture
def initialized_client(initialized_app):
    """Fixture providing test client with fully initialized app."""
    app, chroma_client, mock_post = initialized_app
    return app.test_client(), chroma_client, mock_post
