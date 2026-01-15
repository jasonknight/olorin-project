"""
Unified Embedding Library for Olorin Project

Provides a simple API for text embeddings with automatic handling of
model-specific prefixes and configuration.

Usage:
    from libs.embeddings import Embedder

    # Get shared instance (recommended - loads model once)
    embedder = Embedder.get_instance()

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
from sentence_transformers import SentenceTransformer

from libs.config import Config

logger = logging.getLogger(__name__)

# Registry of model-specific configurations
# Models not listed here will use no prefixes
MODEL_CONFIGS = {
    # Nomic models - require search_document/search_query prefixes
    "nomic-ai/nomic-embed-text-v1.5": {
        "document_prefix": "search_document: ",
        "query_prefix": "search_query: ",
        "trust_remote_code": True,
    },
    "nomic-ai/nomic-embed-text-v1": {
        "document_prefix": "search_document: ",
        "query_prefix": "search_query: ",
        "trust_remote_code": True,
    },
    # BGE models - optional query instruction prefix
    "BAAI/bge-base-en-v1.5": {
        "document_prefix": "",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "trust_remote_code": False,
    },
    "BAAI/bge-large-en-v1.5": {
        "document_prefix": "",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "trust_remote_code": False,
    },
    "BAAI/bge-small-en-v1.5": {
        "document_prefix": "",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "trust_remote_code": False,
    },
    # E5 models - use query/passage prefixes
    "intfloat/e5-large-v2": {
        "document_prefix": "passage: ",
        "query_prefix": "query: ",
        "trust_remote_code": False,
    },
    "intfloat/e5-base-v2": {
        "document_prefix": "passage: ",
        "query_prefix": "query: ",
        "trust_remote_code": False,
    },
    "intfloat/e5-small-v2": {
        "document_prefix": "passage: ",
        "query_prefix": "query: ",
        "trust_remote_code": False,
    },
    # GTE models - no prefix required
    "thenlper/gte-large": {
        "document_prefix": "",
        "query_prefix": "",
        "trust_remote_code": False,
    },
    "thenlper/gte-base": {
        "document_prefix": "",
        "query_prefix": "",
        "trust_remote_code": False,
    },
    # Jina models - no prefix required, long context
    "jinaai/jina-embeddings-v2-base-en": {
        "document_prefix": "",
        "query_prefix": "",
        "trust_remote_code": True,
    },
    # MiniLM models - no prefix required (default/fallback)
    "all-MiniLM-L6-v2": {
        "document_prefix": "",
        "query_prefix": "",
        "trust_remote_code": False,
    },
    "sentence-transformers/all-MiniLM-L6-v2": {
        "document_prefix": "",
        "query_prefix": "",
        "trust_remote_code": False,
    },
}

# Default config for unknown models
DEFAULT_MODEL_CONFIG = {
    "document_prefix": "",
    "query_prefix": "",
    "trust_remote_code": False,
}


class Embedder:
    """
    Unified embedding interface for the Olorin project.

    Handles model loading, prefix management, and provides a simple API
    for embedding documents and queries.

    Thread-safe singleton pattern - use get_instance() for shared access.
    """

    _instance: Optional["Embedder"] = None
    _lock = threading.Lock()

    def __init__(
        self, model_name: Optional[str] = None, config: Optional[Config] = None
    ):
        """
        Initialize the Embedder.

        Args:
            model_name: Override the model name from config. If None, reads from config.
            config: Config instance. If None, creates a new one.
        """
        self._config = config or Config()

        # Get model name from config or use provided override
        self._model_name = model_name or self._config.get(
            "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
        )

        # Get model-specific configuration
        self._model_config = MODEL_CONFIGS.get(self._model_name, DEFAULT_MODEL_CONFIG)

        # Log configuration
        logger.info(f"Initializing Embedder with model: {self._model_name}")
        if self._model_config["document_prefix"]:
            logger.debug(
                f"  Document prefix: '{self._model_config['document_prefix']}'"
            )
        if self._model_config["query_prefix"]:
            logger.debug(f"  Query prefix: '{self._model_config['query_prefix']}'")

        # Load the model
        self._model = SentenceTransformer(
            self._model_name,
            trust_remote_code=self._model_config["trust_remote_code"],
        )

        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info(f"Embedder initialized (dimension: {self._dimension})")

    @classmethod
    def get_instance(
        cls, model_name: Optional[str] = None, config: Optional[Config] = None
    ) -> "Embedder":
        """
        Get the singleton Embedder instance.

        Thread-safe. The first call initializes the instance with the provided
        arguments. Subsequent calls return the same instance (arguments ignored).

        Args:
            model_name: Override model name (only used on first call)
            config: Config instance (only used on first call)

        Returns:
            The shared Embedder instance
        """
        if cls._instance is None:
            with cls._lock:
                # Double-check locking pattern
                if cls._instance is None:
                    cls._instance = cls(model_name=model_name, config=config)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """
        Reset the singleton instance.

        Useful for testing or when configuration changes require a new model.
        """
        with cls._lock:
            cls._instance = None

    @property
    def model_name(self) -> str:
        """The name of the loaded embedding model."""
        return self._model_name

    @property
    def dimension(self) -> int:
        """The dimension of the embedding vectors."""
        return self._dimension

    @property
    def document_prefix(self) -> str:
        """The prefix applied to documents before embedding."""
        return self._model_config["document_prefix"]

    @property
    def query_prefix(self) -> str:
        """The prefix applied to queries before embedding."""
        return self._model_config["query_prefix"]

    def embed_documents(
        self,
        texts: List[str],
        show_progress: bool = False,
        as_list: bool = False,
    ) -> Union[List[np.ndarray], List[List[float]]]:
        """
        Embed a list of documents for storage.

        Automatically applies the appropriate document prefix for the model.

        Args:
            texts: List of document texts to embed
            show_progress: Show progress bar during embedding
            as_list: Return List[List[float]] instead of List[np.ndarray]
                    (useful for ChromaDB which expects lists)

        Returns:
            List of embeddings, one per document
        """
        if not texts:
            return []

        # Apply document prefix
        prefix = self._model_config["document_prefix"]
        if prefix:
            prefixed_texts = [f"{prefix}{text}" for text in texts]
        else:
            prefixed_texts = texts

        # Generate embeddings
        embeddings = self._model.encode(
            prefixed_texts,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        if as_list:
            return [emb.tolist() for emb in embeddings]
        return list(embeddings)

    def embed_query(
        self,
        text: str,
        as_list: bool = False,
    ) -> Union[np.ndarray, List[float]]:
        """
        Embed a single query for search.

        Automatically applies the appropriate query prefix for the model.

        Args:
            text: Query text to embed
            as_list: Return List[float] instead of np.ndarray
                    (useful for ChromaDB which expects lists)

        Returns:
            The query embedding
        """
        # Apply query prefix
        prefix = self._model_config["query_prefix"]
        if prefix:
            prefixed_text = f"{prefix}{text}"
        else:
            prefixed_text = text

        # Generate embedding
        embedding = self._model.encode(
            [prefixed_text],
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0]

        if as_list:
            return embedding.tolist()
        return embedding

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
            show_progress: Show progress bar during embedding
            as_list: Return lists instead of numpy arrays

        Returns:
            List of embeddings
        """
        if mode == "document":
            return self.embed_documents(
                texts, show_progress=show_progress, as_list=as_list
            )
        elif mode == "query":
            # For batch query embedding, apply query prefix to all
            if not texts:
                return []
            prefix = self._model_config["query_prefix"]
            if prefix:
                prefixed_texts = [f"{prefix}{text}" for text in texts]
            else:
                prefixed_texts = texts

            embeddings = self._model.encode(
                prefixed_texts,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            )
            if as_list:
                return [emb.tolist() for emb in embeddings]
            return list(embeddings)
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'document' or 'query'.")


# Convenience function for simple usage
def get_embedder(
    model_name: Optional[str] = None, config: Optional[Config] = None
) -> Embedder:
    """
    Get the singleton Embedder instance.

    Convenience wrapper around Embedder.get_instance().

    Args:
        model_name: Override model name (only used on first call)
        config: Config instance (only used on first call)

    Returns:
        The shared Embedder instance
    """
    return Embedder.get_instance(model_name=model_name, config=config)
