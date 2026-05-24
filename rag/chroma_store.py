"""
ChromaDB-backed vector store for product embeddings.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """
    Persistent ChromaDB vector store with the same search-facing interface as
    the in-memory VectorStore.
    """

    def __init__(
        self,
        persist_directory: str = "chroma_db",
        collection_name: str = "amazon_products",
        dimension: int = 512,
        reset: bool = False,
    ):
        import chromadb

        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.dimension = dimension

        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)

        if reset:
            try:
                self.client.delete_collection(collection_name)
            except Exception:
                pass

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            "Initialized ChromaVectorStore collection=%s path=%s",
            collection_name,
            persist_directory,
        )

    def add_embedding(
        self,
        embedding_id: str,
        embedding: np.ndarray,
        metadata: Dict = None,
    ) -> None:
        """Add or update one embedding."""
        self.add_embeddings({embedding_id: embedding}, {embedding_id: metadata or {}})

    def add_embeddings(
        self,
        embeddings: Dict[str, np.ndarray],
        metadata: Dict[str, Dict] = None,
        batch_size: int = 500,
    ) -> None:
        """Add embeddings to Chroma in batches."""
        metadata = metadata or {}
        items = list(embeddings.items())

        for start in range(0, len(items), batch_size):
            batch = items[start:start + batch_size]
            ids = [embedding_id for embedding_id, _ in batch]
            vectors = [self._normalize_vector(embedding).tolist() for _, embedding in batch]
            metadatas = [
                self._clean_metadata(metadata.get(embedding_id, {}))
                for embedding_id in ids
            ]
            documents = [self._metadata_to_document(meta) for meta in metadatas]

            self.collection.upsert(
                ids=ids,
                embeddings=vectors,
                metadatas=metadatas,
                documents=documents,
            )

        logger.info("Added %s embeddings to Chroma", len(embeddings))

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        where_document: Dict = None,
    ) -> List[Tuple[str, float, Dict]]:
        """Search Chroma for nearest products."""
        query = self._normalize_vector(query_embedding).tolist()
        kwargs = {
            "query_embeddings": [query],
            "n_results": top_k,
            "include": ["metadatas", "distances"],
        }
        if where_document:
            kwargs["where_document"] = where_document

        result = self.collection.query(**kwargs)
        return self._format_results(result)

    def search_product_terms(
        self,
        query_embedding: np.ndarray,
        terms: List[str],
        top_k: int = 5,
    ) -> List[Tuple[str, float, Dict]]:
        """
        Search a product type by querying term-filtered Chroma subsets and
        merging by score.
        """
        primary = {}
        fallback = {}
        for term in terms:
            try:
                results = self.search(
                    query_embedding,
                    top_k=max(top_k, 10),
                    where_document={"$contains": term},
                )
            except Exception as exc:
                logger.debug("Chroma term filter failed for %s: %s", term, exc)
                continue

            for product_id, similarity, metadata in results:
                title_category = (
                    f"{metadata.get('title', '')} {metadata.get('category', '')}".lower()
                )
                target = primary if term in title_category else fallback
                boosted = similarity + (0.12 if target is primary else 0.02)
                current = target.get(product_id)
                if current is None or boosted > current[0]:
                    target[product_id] = (boosted, metadata)

        merged = primary or fallback
        if not merged:
            return self.search(query_embedding, top_k=top_k)

        sorted_results = sorted(merged.items(), key=lambda item: item[1][0], reverse=True)
        return [
            (product_id, float(score), metadata)
            for product_id, (score, metadata) in sorted_results[:top_k]
        ]

    def get_size(self) -> int:
        """Return number of indexed embeddings."""
        return self.collection.count()

    def save(self, filepath: str = None) -> None:
        """Chroma persists automatically."""
        logger.info("Chroma persists automatically at %s", self.persist_directory)

    def load(self, filepath: str = None) -> None:
        """Chroma loads through PersistentClient initialization."""
        logger.info("Chroma loaded from %s", self.persist_directory)

    def get_metadata(self, embedding_id: str):
        """Fetch metadata by id."""
        result = self.collection.get(ids=[embedding_id], include=["metadatas"])
        metadatas = result.get("metadatas") or []
        return metadatas[0] if metadatas else None

    def _format_results(self, result) -> List[Tuple[str, float, Dict]]:
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]

        formatted = []
        for product_id, distance, metadata in zip(ids, distances, metadatas):
            similarity = 1 - float(distance)
            formatted.append((product_id, similarity, metadata or {}))

        return formatted

    def _normalize_vector(self, embedding: np.ndarray) -> np.ndarray:
        vector = np.asarray(embedding, dtype=np.float32).flatten()
        if vector.shape[0] != self.dimension:
            raise ValueError(
                f"Embedding dimension {vector.shape[0]} does not match {self.dimension}"
            )
        return vector / (np.linalg.norm(vector) + 1e-8)

    def _clean_metadata(self, metadata: Dict) -> Dict:
        cleaned = {}
        for key, value in (metadata or {}).items():
            if value is None:
                cleaned[key] = ""
            elif isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            else:
                cleaned[key] = str(value)
        return cleaned

    def _metadata_to_document(self, metadata: Dict) -> str:
        fields = [
            metadata.get("title", ""),
            metadata.get("brand", ""),
            metadata.get("category", ""),
            metadata.get("features", ""),
            metadata.get("description", ""),
            metadata.get("combined_description", ""),
        ]
        return " ".join(str(field) for field in fields if field).lower()
