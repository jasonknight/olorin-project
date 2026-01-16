"""
Local Embedding Model Loader for Olorin Project

This module loads embedding models directly using sentence-transformers.
It is intended ONLY for use by the embeddings tool server.

All other components should use libs/embeddings.py (the API client) to
communicate with the embeddings server.

Usage (server only):
    from libs.embeddings_local import LocalEmbedder

    embedder = LocalEmbedder()
    embeddings = embedder.embed_documents(["text1", "text2"])
    embedding = embedder.embed_query("search text")
"""

import logging
from typing import List, Optional, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from libs.config import Config

logger = logging.getLogger(__name__)

# Known model prefixes for different embedding modes
# These are model-specific and affect embedding quality significantly
MODEL_PREFIXES = {
    "nomic-ai/nomic-embed-text-v1.5": {
        "document": "search_document: ",
        "query": "search_query: ",
    },
    "nomic-ai/nomic-embed-text-v1": {
        "document": "search_document: ",
        "query": "search_query: ",
    },
    # Add other models with prefixes here as needed
}


class LocalEmbedder:
    """
    Local embedding model that loads directly with sentence-transformers.

    This class is for server-side use only. Client code should use the
    Embedder class from libs/embeddings.py instead.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        config: Optional[Config] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the local embedder with a sentence-transformers model.

        Args:
            model_name: Model name/path. If None, reads from config.
            config: Config instance. If None, creates a new one.
            device: Device to use ('cpu', 'cuda', 'mps'). If None, auto-detect.
        """
        self._config = config or Config()

        # Get model name from config if not provided
        if model_name:
            self._model_name = model_name
        else:
            self._model_name = self._config.get(
                "EMBEDDINGS_TOOL_MODEL", "nomic-ai/nomic-embed-text-v1.5"
            )

        logger.info(f"Loading embedding model: {self._model_name}")

        # Determine device
        if device:
            self._device = device
        else:
            self._device = self._auto_detect_device()

        logger.info(f"Using device: {self._device}")

        # Load the model
        self._model = SentenceTransformer(
            self._model_name,
            device=self._device,
            trust_remote_code=True,  # Required for some models like nomic
        )

        # Get embedding dimension
        self._dimension = self._model.get_sentence_embedding_dimension()

        # Set up prefixes based on model
        prefixes = MODEL_PREFIXES.get(self._model_name, {})
        self._document_prefix = self._config.get(
            "EMBEDDINGS_DOCUMENT_PREFIX", prefixes.get("document", "")
        )
        self._query_prefix = self._config.get(
            "EMBEDDINGS_QUERY_PREFIX", prefixes.get("query", "")
        )

        logger.info(f"Model loaded successfully: {self._model_name}")
        logger.info(f"  Dimension: {self._dimension}")
        logger.info(f"  Document prefix: '{self._document_prefix}'")
        logger.info(f"  Query prefix: '{self._query_prefix}'")

    def _auto_detect_device(self) -> str:
        """Auto-detect the best available device."""
        import torch

        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    @property
    def model_name(self) -> str:
        """The name of the embedding model."""
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

    def _apply_prefix(self, texts: List[str], mode: str) -> List[str]:
        """Apply the appropriate prefix based on mode."""
        if mode == "query":
            prefix = self._query_prefix
        else:
            prefix = self._document_prefix

        if not prefix:
            return texts

        return [prefix + text for text in texts]

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
            show_progress: Show progress bar during encoding
            as_list: Return List[List[float]] instead of List[np.ndarray]

        Returns:
            List of embeddings, one per document
        """
        if not texts:
            return []

        # Apply document prefix
        prefixed_texts = self._apply_prefix(texts, mode="document")

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

        Args:
            text: Query text to embed
            as_list: Return List[float] instead of np.ndarray

        Returns:
            The query embedding
        """
        # Apply query prefix
        prefixed_text = self._apply_prefix([text], mode="query")[0]

        # Generate embedding
        embedding = self._model.encode(prefixed_text, convert_to_numpy=True)

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
            show_progress: Show progress bar
            as_list: Return lists instead of numpy arrays

        Returns:
            List of embeddings
        """
        if mode not in ("document", "query"):
            raise ValueError(f"Invalid mode: {mode}. Must be 'document' or 'query'.")

        if not texts:
            return []

        # Apply appropriate prefix
        prefixed_texts = self._apply_prefix(texts, mode=mode)

        # Generate embeddings
        embeddings = self._model.encode(
            prefixed_texts,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        if as_list:
            return [emb.tolist() for emb in embeddings]
        return list(embeddings)
