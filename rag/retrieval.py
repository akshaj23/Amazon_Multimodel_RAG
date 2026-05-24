"""
Retrieval system for RAG (Retrieval-Augmented Generation)
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class RetrieverSystem:
    """
    Retrieval system for finding relevant products using embeddings

    Combines CLIP embeddings with vector search to retrieve
    products relevant to user queries.
    """

    def __init__(self, vector_store: VectorStore):
        """
        Initialize retriever system

        Args:
            vector_store: VectorStore instance for managing embeddings
        """
        self.vector_store = vector_store
        self.retrieval_stats = {
            'total_queries': 0,
            'total_retrieved': 0,
        }

        logger.info("Initialized RetrieverSystem")

    def index_product(
        self,
        product_id: str,
        text_embedding: np.ndarray,
        image_embedding: Optional[np.ndarray] = None,
        product_info: Dict = None
    ) -> None:
        """
        Index a product with its embeddings

        Args:
            product_id: Unique product identifier
            text_embedding: Text embedding from CLIP
            image_embedding: Image embedding from CLIP (optional)
            product_info: Product metadata (title, price, etc.)
        """
        # Use multimodal embedding (average of text and image)
        if image_embedding is not None:
            combined_embedding = (text_embedding + image_embedding) / 2
        else:
            combined_embedding = text_embedding

        # Prepare metadata
        metadata = product_info or {}
        metadata['product_id'] = product_id
        metadata['has_image'] = image_embedding is not None

        # Add to vector store
        self.vector_store.add_embedding(
            embedding_id=product_id,
            embedding=combined_embedding,
            metadata=metadata
        )

        logger.debug(f"Indexed product: {product_id}")

    def index_products(
        self,
        products: List[Dict]
    ) -> None:
        """
        Index multiple products at once

        Args:
            products: List of product dictionaries with embeddings
        """
        for product in products:
            self.index_product(
                product_id=product['product_id'],
                text_embedding=product['text_embedding'],
                image_embedding=product.get('image_embedding'),
                product_info=product.get('metadata', {})
            )

        logger.info(f"Indexed {len(products)} products")

    def retrieve_by_text(
        self,
        query: str,
        embedding_model,
        top_k: int = 5
    ) -> List[Tuple[str, float, Dict]]:
        """
        Retrieve products by text query

        Args:
            query: Text query
            embedding_model: Model to generate query embedding
            top_k: Number of results to return

        Returns:
            List of (product_id, similarity, metadata) tuples
        """
        logger.debug(f"Text retrieval for query: {query}")

        # Generate query embedding
        query_embedding = embedding_model.encode_text(query)
        if isinstance(query_embedding, np.ndarray) and query_embedding.ndim > 1:
            query_embedding = query_embedding[0]

        # Search
        results = self.vector_store.search(query_embedding, top_k=top_k)

        self.retrieval_stats['total_queries'] += 1
        self.retrieval_stats['total_retrieved'] += len(results)

        return results

    def retrieve_by_image(
        self,
        image_path: str,
        embedding_model,
        top_k: int = 5
    ) -> List[Tuple[str, float, Dict]]:
        """
        Retrieve products by image query

        Args:
            image_path: Path to query image
            embedding_model: Model to generate image embedding
            top_k: Number of results to return

        Returns:
            List of (product_id, similarity, metadata) tuples
        """
        logger.debug(f"Image retrieval for: {image_path}")

        # Generate image embedding
        image_embedding = embedding_model.encode_image(image_path)
        if isinstance(image_embedding, np.ndarray) and image_embedding.ndim > 1:
            image_embedding = image_embedding[0]

        # Search
        results = self.vector_store.search(image_embedding, top_k=top_k)

        self.retrieval_stats['total_queries'] += 1
        self.retrieval_stats['total_retrieved'] += len(results)

        return results

    def retrieve_by_multimodal(
        self,
        query_text: str,
        image_path: Optional[str],
        embedding_model,
        top_k: int = 5,
        text_weight: float = 0.5
    ) -> List[Tuple[str, float, Dict]]:
        """
        Retrieve products by combined text and image query

        Args:
            query_text: Text query
            image_path: Path to image (optional)
            embedding_model: Model to generate embeddings
            top_k: Number of results to return
            text_weight: Weight for text in combined query (0-1)

        Returns:
            List of (product_id, similarity, metadata) tuples
        """
        logger.debug(f"Multimodal retrieval: text='{query_text}', image={image_path}")

        # Generate embeddings
        text_embedding = embedding_model.encode_text(query_text)
        if isinstance(text_embedding, np.ndarray) and text_embedding.ndim > 1:
            text_embedding = text_embedding[0]

        # Combine embeddings
        if image_path:
            image_embedding = embedding_model.encode_image(image_path)
            if isinstance(image_embedding, np.ndarray) and image_embedding.ndim > 1:
                image_embedding = image_embedding[0]

            # Weighted combination
            image_weight = 1 - text_weight
            combined_embedding = (
                text_weight * text_embedding +
                image_weight * image_embedding
            )
        else:
            combined_embedding = text_embedding

        # Search
        results = self.vector_store.search(combined_embedding, top_k=top_k)

        self.retrieval_stats['total_queries'] += 1
        self.retrieval_stats['total_retrieved'] += len(results)

        return results

    def format_retrieval_results(
        self,
        results: List[Tuple[str, float, Dict]],
        include_metadata: List[str] = None
    ) -> str:
        """
        Format retrieval results as readable context

        Args:
            results: List of retrieval results
            include_metadata: Which metadata fields to include

        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant products found."

        context_parts = []

        for i, (product_id, similarity, metadata) in enumerate(results, 1):
            part = f"\n{i}. Product: {metadata.get('title', 'Unknown')}"

            if 'brand' in metadata:
                part += f"\n   Brand: {metadata['brand']}"

            if 'price' in metadata:
                part += f"\n   Price: ${metadata['price']}"

            if 'features' in metadata:
                part += f"\n   Features: {metadata['features']}"

            part += f"\n   Relevance Score: {similarity:.2%}"

            context_parts.append(part)

        return "\n".join(context_parts)

    def evaluate_retrieval(
        self,
        queries: List[str],
        ground_truth: List[List[str]],
        embedding_model,
        top_k_values: List[int] = [1, 5, 10]
    ) -> Dict:
        """
        Evaluate retrieval performance

        Args:
            queries: List of test queries
            ground_truth: List of ground truth results per query
            embedding_model: Embedding model
            top_k_values: Cutoff values for recall calculation

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating retrieval on {len(queries)} queries")

        metrics = {f'recall@{k}': [] for k in top_k_values}
        metrics['accuracy'] = []

        for query, ground_products in zip(queries, ground_truth):
            # Retrieve results
            results = self.retrieve_by_text(query, embedding_model, top_k=max(top_k_values))
            retrieved_ids = [r[0] for r in results]

            # Calculate metrics
            for k in top_k_values:
                top_k_retrieved = set(retrieved_ids[:k])
                ground_set = set(ground_products)

                if len(ground_set) > 0:
                    recall = len(top_k_retrieved & ground_set) / len(ground_set)
                else:
                    recall = 0

                metrics[f'recall@{k}'].append(recall)

            # Accuracy: whether first result is correct
            if retrieved_ids and retrieved_ids[0] in ground_products:
                metrics['accuracy'].append(1.0)
            else:
                metrics['accuracy'].append(0.0)

        # Calculate averages
        avg_metrics = {}
        for metric_name, values in metrics.items():
            avg_metrics[f'{metric_name}_avg'] = np.mean(values)

        logger.info(f"Evaluation complete: {avg_metrics}")
        return avg_metrics

    def get_stats(self) -> Dict:
        """
        Get retriever statistics

        Returns:
            Dictionary with statistics
        """
        return {
            **self.retrieval_stats,
            'vector_store_size': self.vector_store.get_size(),
            **self.vector_store.get_stats()
        }
