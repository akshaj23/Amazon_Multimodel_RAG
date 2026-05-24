"""
Vector database management for storing and retrieving embeddings
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class VectorStore:
    """
    In-memory vector database for storing and retrieving embeddings

    This is a simplified implementation. For production, use:
    - Google Vertex AI Vector Search
    - Pinecone
    - Weaviate
    - Milvus
    """

    def __init__(self, dimension: int = 512):
        """
        Initialize vector store

        Args:
            dimension: Dimension of embeddings
        """
        self.dimension = dimension
        self.embeddings = {}  # {id: embedding}
        self.metadata = {}    # {id: metadata}
        self.embedding_ids = []  # List of IDs for faster lookup

        logger.info(f"Initialized VectorStore with dimension {dimension}")

    def add_embedding(
        self,
        embedding_id: str,
        embedding: np.ndarray,
        metadata: Dict = None
    ) -> None:
        """
        Add embedding to the store

        Args:
            embedding_id: Unique identifier for the embedding
            embedding: Embedding vector (should have shape [dimension])
            metadata: Additional metadata to store with embedding
        """
        if embedding.shape[0] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embedding.shape[0]} "
                f"does not match store dimension {self.dimension}"
            )

        # Normalize embedding
        normalized = embedding / np.linalg.norm(embedding)

        self.embeddings[embedding_id] = normalized
        self.metadata[embedding_id] = metadata or {}
        self.embedding_ids.append(embedding_id)

        logger.debug(f"Added embedding: {embedding_id}")

    def add_embeddings(
        self,
        embeddings: Dict[str, np.ndarray],
        metadata: Dict[str, Dict] = None
    ) -> None:
        """
        Add multiple embeddings at once

        Args:
            embeddings: Dictionary mapping IDs to embeddings
            metadata: Dictionary mapping IDs to metadata
        """
        metadata = metadata or {}

        for embedding_id, embedding in embeddings.items():
            meta = metadata.get(embedding_id, {})
            self.add_embedding(embedding_id, embedding, meta)

        logger.info(f"Added {len(embeddings)} embeddings")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Tuple[str, float, Dict]]:
        """
        Search for similar embeddings

        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return

        Returns:
            List of tuples (id, similarity_score, metadata)
        """
        if len(self.embeddings) == 0:
            logger.warning("Vector store is empty")
            return []

        # Normalize query
        query_norm = query_embedding / np.linalg.norm(query_embedding)

        # Compute similarities
        similarities = {}
        for embedding_id, embedding in self.embeddings.items():
            similarity = np.dot(query_norm, embedding)
            similarities[embedding_id] = similarity

        # Sort by similarity (descending)
        sorted_results = sorted(
            similarities.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Return top K
        results = []
        for embedding_id, similarity in sorted_results[:top_k]:
            results.append((
                embedding_id,
                float(similarity),
                self.metadata[embedding_id]
            ))

        logger.debug(f"Search returned {len(results)} results")
        return results

    def batch_search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 5
    ) -> List[List[Tuple[str, float, Dict]]]:
        """
        Batch search for multiple query embeddings

        Args:
            query_embeddings: Array of query embeddings [batch_size, dimension]
            top_k: Number of top results per query

        Returns:
            List of search results for each query
        """
        batch_results = []

        for query_embedding in query_embeddings:
            results = self.search(query_embedding, top_k)
            batch_results.append(results)

        return batch_results

    def get_embedding(self, embedding_id: str) -> Optional[np.ndarray]:
        """
        Get embedding by ID

        Args:
            embedding_id: Embedding ID

        Returns:
            Embedding vector or None if not found
        """
        return self.embeddings.get(embedding_id)

    def get_metadata(self, embedding_id: str) -> Optional[Dict]:
        """
        Get metadata by ID

        Args:
            embedding_id: Embedding ID

        Returns:
            Metadata dictionary or None if not found
        """
        return self.metadata.get(embedding_id)

    def delete_embedding(self, embedding_id: str) -> bool:
        """
        Delete embedding from store

        Args:
            embedding_id: Embedding ID

        Returns:
            True if deleted, False if not found
        """
        if embedding_id in self.embeddings:
            del self.embeddings[embedding_id]
            del self.metadata[embedding_id]
            self.embedding_ids.remove(embedding_id)
            logger.debug(f"Deleted embedding: {embedding_id}")
            return True

        return False

    def clear(self) -> None:
        """Clear all embeddings from store"""
        self.embeddings.clear()
        self.metadata.clear()
        self.embedding_ids.clear()
        logger.info("Vector store cleared")

    def get_size(self) -> int:
        """
        Get number of embeddings in store

        Returns:
            Number of embeddings
        """
        return len(self.embeddings)

    def save(self, filepath: str) -> None:
        """
        Save vector store to disk

        Args:
            filepath: Path to save file
        """
        logger.info(f"Saving vector store to {filepath}")

        # Prepare data
        data = {
            'dimension': self.dimension,
            'embeddings': {
                k: v.tolist() for k, v in self.embeddings.items()
            },
            'metadata': self.metadata
        }

        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(data, f)

        logger.info(f"Vector store saved: {len(self.embeddings)} embeddings")

    def load(self, filepath: str) -> None:
        """
        Load vector store from disk

        Args:
            filepath: Path to load file
        """
        logger.info(f"Loading vector store from {filepath}")

        with open(filepath, 'r') as f:
            data = json.load(f)

        self.dimension = data['dimension']
        self.embeddings = {
            k: np.array(v) for k, v in data['embeddings'].items()
        }
        self.metadata = data['metadata']
        self.embedding_ids = list(self.embeddings.keys())

        logger.info(f"Vector store loaded: {len(self.embeddings)} embeddings")

    def get_stats(self) -> Dict:
        """
        Get statistics about the vector store

        Returns:
            Dictionary with statistics
        """
        return {
            'total_embeddings': len(self.embeddings),
            'dimension': self.dimension,
            'memory_usage_mb': (
                sum(v.nbytes for v in self.embeddings.values()) / (1024 ** 2)
            )
        }
