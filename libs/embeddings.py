"""
Embedding Client Library for Olorin Project

Provides a simple API for text embeddings via the embeddings tool server.
The server handles model loading and prefix management.

Usage:
    from libs.embeddings import Embedder, get_embedder

    # Get shared instance (recommended)
    embedder = get_embedder()

    # Embed documents for storage
    embeddings = embedder.embed_documents(["text1", "text2"])

    # Embed query for search
    embedding = embedder.embed_query("search text")

    # Access model info
    print(embedder.dimension)    # e.g., 768
    print(embedder.model_name)   # e.g., "nomic-ai/nomic-embed-text-v1.5"
"""

import logging
import threading
from typing import List, Optional, Union

import numpy as np
import requests

from libs.config import Config

logger = logging.getLogger(__name__)


class Embedder:
    """
    API-based embedding client for the Olorin project.

    Communicates with the embeddings tool server via HTTP.
    The server loads the model once and handles all embedding requests.

    Thread-safe singleton pattern - use get_instance() for shared access.
    """

    _instance: Optional["Embedder"] = None
    _lock = threading.Lock()

    def __init__(
        self,
        base_url: Optional[str] = None,
        config: Optional[Config] = None,
        timeout: float = 30.0,
    ):
        """
        Initialize the Embedder client.

        Args:
            base_url: Override the API base URL. If None, reads from config.
            config: Config instance. If None, creates a new one.
            timeout: HTTP request timeout in seconds.
        """
        self._config = config or Config()
        self._timeout = timeout

        # Get base URL from config or use provided override
        if base_url:
            self._base_url = base_url
        else:
            port = self._config.get_int("EMBEDDINGS_TOOL_PORT", 8771)
            host = self._config.get("EMBEDDINGS_TOOL_HOST", "localhost")
            self._base_url = f"http://{host}:{port}"

        # Fetch model info from the server
        self._model_name: str = "unknown"
        self._dimension: int = 0
        self._document_prefix: str = ""
        self._query_prefix: str = ""

        logger.info(f"Initializing Embedder with server: {self._base_url}")
        self._fetch_model_info()

    def _fetch_model_info(self) -> None:
        """Fetch model information from the server."""
        try:
            response = requests.get(f"{self._base_url}/info", timeout=self._timeout)
            response.raise_for_status()
            data = response.json()
            self._model_name = data.get("model", "unknown")
            self._dimension = data.get("dimension", 0)
            self._document_prefix = data.get("document_prefix", "")
            self._query_prefix = data.get("query_prefix", "")
            logger.info(
                f"Embedder connected: {self._model_name} (dim: {self._dimension})"
            )
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch model info from {self._base_url}: {e}")
            # Try health check to see if server is up
            try:
                health = requests.get(f"{self._base_url}/health", timeout=self._timeout)
                if health.status_code == 200:
                    logger.info("Server is healthy, will fetch info on first request")
            except requests.RequestException:
                logger.error(f"Embedding server at {self._base_url} is not reachable")

    @classmethod
    def get_instance(
        cls,
        base_url: Optional[str] = None,
        config: Optional[Config] = None,
        timeout: float = 30.0,
    ) -> "Embedder":
        """
        Get the singleton Embedder instance.

        Thread-safe. The first call initializes the instance with the provided
        arguments. Subsequent calls return the same instance (arguments ignored).

        Args:
            base_url: Override API base URL (only used on first call)
            config: Config instance (only used on first call)
            timeout: HTTP timeout in seconds (only used on first call)

        Returns:
            The shared Embedder instance
        """
        if cls._instance is None:
            with cls._lock:
                # Double-check locking pattern
                if cls._instance is None:
                    cls._instance = cls(
                        base_url=base_url, config=config, timeout=timeout
                    )
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """
        Reset the singleton instance.

        Useful for testing or when configuration changes require reconnection.
        """
        with cls._lock:
            cls._instance = None

    @property
    def model_name(self) -> str:
        """The name of the embedding model on the server."""
        return self._model_name

    @property
    def dimension(self) -> int:
        """The dimension of the embedding vectors."""
        return self._dimension

    @property
    def document_prefix(self) -> str:
        """The prefix applied to documents before embedding."""
        return self._document_prefix

    @property
    def query_prefix(self) -> str:
        """The prefix applied to queries before embedding."""
        return self._query_prefix

    def _call_api(self, texts: List[str], mode: str) -> List[List[float]]:
        """
        Call the embedding API.

        Args:
            texts: List of texts to embed
            mode: Either "document" or "query"

        Returns:
            List of embeddings as lists of floats

        Raises:
            RuntimeError: If API call fails
        """
        try:
            response = requests.post(
                f"{self._base_url}/call",
                json={"texts": texts, "mode": mode},
                timeout=self._timeout,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            data = response.json()

            if not data.get("success"):
                error = data.get("error", {})
                raise RuntimeError(
                    f"Embedding API error: {error.get('type', 'Unknown')}: "
                    f"{error.get('message', 'No message')}"
                )

            result = data.get("result", {})

            # Update cached model info if available
            if result.get("model"):
                self._model_name = result["model"]
            if result.get("dimension"):
                self._dimension = result["dimension"]

            return result.get("embeddings", [])

        except requests.RequestException as e:
            raise RuntimeError(f"Failed to call embedding API: {e}") from e

    def embed_documents(
        self,
        texts: List[str],
        show_progress: bool = False,
        as_list: bool = False,
    ) -> Union[List[np.ndarray], List[List[float]]]:
        """
        Embed a list of documents for storage.

        Args:
            texts: List of document texts to embed
            show_progress: Ignored (kept for API compatibility)
            as_list: Return List[List[float]] instead of List[np.ndarray]

        Returns:
            List of embeddings, one per document
        """
        if not texts:
            return []

        embeddings = self._call_api(texts, mode="document")

        if as_list:
            return embeddings
        return [np.array(emb) for emb in embeddings]

    def embed_query(
        self,
        text: str,
        as_list: bool = False,
    ) -> Union[np.ndarray, List[float]]:
        """
        Embed a single query for search.

        Args:
            text: Query text to embed
            as_list: Return List[float] instead of np.ndarray

        Returns:
            The query embedding
        """
        embeddings = self._call_api([text], mode="query")

        if not embeddings:
            raise RuntimeError("API returned no embeddings for query")

        if as_list:
            return embeddings[0]
        return np.array(embeddings[0])

    def embed_batch(
        self,
        texts: List[str],
        mode: str = "document",
        show_progress: bool = False,
        as_list: bool = False,
    ) -> Union[List[np.ndarray], List[List[float]]]:
        """
        Embed a batch of texts with explicit mode selection.

        Args:
            texts: List of texts to embed
            mode: Either "document" or "query" - determines which prefix to use
            show_progress: Ignored (kept for API compatibility)
            as_list: Return lists instead of numpy arrays

        Returns:
            List of embeddings
        """
        if mode not in ("document", "query"):
            raise ValueError(f"Invalid mode: {mode}. Must be 'document' or 'query'.")

        if not texts:
            return []

        embeddings = self._call_api(texts, mode=mode)

        if as_list:
            return embeddings
        return [np.array(emb) for emb in embeddings]


def get_embedder(config: Optional[Config] = None) -> Embedder:
    """
    Get the singleton Embedder instance.

    Args:
        config: Config instance. If None, creates a new one.

    Returns:
        The shared Embedder instance
    """
    return Embedder.get_instance(config=config)
