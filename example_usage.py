#!/usr/bin/env python3
"""
Example usage of the CLIP + RAG pipeline for product search
"""

import logging
import numpy as np
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_1_text_search():
    """Example 1: Search products by text description"""
    logger.info("\n" + "=" * 60)
    logger.info("Example 1: Text-Based Product Search")
    logger.info("=" * 60)

    from data.sample_products import save_sample_products
    from models import CLIPEmbeddingModel
    from rag import VectorStore, RetrieverSystem

    # Setup
    products_path = "data/raw/sample_products.json"
    if not Path(products_path).exists():
        logger.info("Generating sample products...")
        save_sample_products(products_path)

    # Load products
    from data import DataLoader
    loader = DataLoader(products_path)

    # Initialize CLIP
    logger.info("Loading CLIP model...")
    clip_model = CLIPEmbeddingModel()

    # Create vector store and retriever
    vector_store = VectorStore(dimension=512)
    retriever = RetrieverSystem(vector_store)

    # Index products
    logger.info("Indexing products...")
    for product in loader.data.to_dict('records'):
        description = product['combined_description']
        embedding = clip_model.encode_text([description])
        if embedding.ndim > 1:
            embedding = embedding[0]

        retriever.index_product(
            product_id=product['asin'],
            text_embedding=embedding,
            product_info=product
        )

    logger.info(f"Indexed {vector_store.get_size()} products\n")

    # Search examples
    queries = [
        "I need a smartphone with 5G and great camera",
        "What's a good fitness tracker for monitoring health?",
        "Looking for a professional camera for photography"
    ]

    for query in queries:
        logger.info(f"Query: {query}")
        logger.info("-" * 60)

        results = retriever.retrieve_by_text(query, clip_model, top_k=3)

        for i, (product_id, similarity, metadata) in enumerate(results, 1):
            logger.info(f"\n{i}. {metadata['title']}")
            logger.info(f"   ASIN: {product_id}")
            logger.info(f"   Brand: {metadata['brand']}")
            logger.info(f"   Price: ${metadata['price']:.2f}")
            logger.info(f"   Match Score: {similarity*100:.1f}%")
            logger.info(f"   Features: {metadata['features']}")

        logger.info("\n")


def example_2_multimodal_search():
    """Example 2: Combine text and semantic similarity for search"""
    logger.info("\n" + "=" * 60)
    logger.info("Example 2: Multimodal Search (Text + Semantic)")
    logger.info("=" * 60)

    from data.sample_products import generate_sample_products
    from models import CLIPEmbeddingModel
    from rag import VectorStore, RetrieverSystem

    # Initialize
    products = generate_sample_products()
    clip_model = CLIPEmbeddingModel()
    vector_store = VectorStore(dimension=512)
    retriever = RetrieverSystem(vector_store)

    # Index
    logger.info("Indexing products...")
    for product in products:
        description = product['combined_description']
        embedding = clip_model.encode_text([description])
        if embedding.ndim > 1:
            embedding = embedding[0]

        retriever.index_product(
            product_id=product['asin'],
            text_embedding=embedding,
            product_info=product
        )

    # Search with weighted combination
    query_text = "professional camera with 4K video"
    logger.info(f"Query: {query_text}")
    logger.info("Using 70% text weight, 30% for additional context\n")

    results = retriever.retrieve_by_text(query_text, clip_model, top_k=3)

    for i, (product_id, similarity, metadata) in enumerate(results, 1):
        logger.info(f"{i}. {metadata['title']} (Match: {similarity*100:.1f}%)")


def example_3_similarity_computation():
    """Example 3: Compute embeddings and similarities directly"""
    logger.info("\n" + "=" * 60)
    logger.info("Example 3: Direct Embedding & Similarity Computation")
    logger.info("=" * 60)

    from models import CLIPEmbeddingModel

    clip_model = CLIPEmbeddingModel()

    # Different product descriptions
    products = [
        "Samsung Galaxy S21 - 5G smartphone with 64MP camera",
        "iPhone 14 Pro - Professional 5G phone with pro camera",
        "Fitbit Charge 5 - Fitness tracker with heart rate monitor",
        "KitchenAid Stand Mixer - Professional mixer for baking",
        "Sony WF-1000XM4 - Noise canceling earbuds"
    ]

    logger.info("Generating embeddings for products...")
    embeddings = clip_model.encode_text(products)

    logger.info("Computing product-to-product similarities:\n")

    # Compute pairwise similarities
    for i in range(len(products)):
        for j in range(i + 1, len(products)):
            sim = clip_model.similarity(embeddings[i], embeddings[j])
            logger.info(f"Product {i+1} vs Product {j+1}: {sim:.4f}")
            logger.info(f"  '{products[i][:40]}...'")
            logger.info(f"  '{products[j][:40]}...'")
            logger.info()


def example_4_batch_search():
    """Example 4: Batch search multiple queries"""
    logger.info("\n" + "=" * 60)
    logger.info("Example 4: Batch Query Processing")
    logger.info("=" * 60)

    from data.sample_products import generate_sample_products
    from models import CLIPEmbeddingModel
    from rag import VectorStore, RetrieverSystem

    # Setup
    products = generate_sample_products()
    clip_model = CLIPEmbeddingModel()
    vector_store = VectorStore(dimension=512)
    retriever = RetrieverSystem(vector_store)

    # Index
    for product in products:
        embedding = clip_model.encode_text([product['combined_description']])
        if embedding.ndim > 1:
            embedding = embedding[0]

        retriever.index_product(
            product_id=product['asin'],
            text_embedding=embedding,
            product_info=product
        )

    # Batch queries
    queries = [
        "5G smartphone",
        "fitness monitoring",
        "professional photography",
        "wireless audio",
        "portable speaker"
    ]

    logger.info(f"Processing {len(queries)} queries...\n")

    for i, query in enumerate(queries, 1):
        results = retriever.retrieve_by_text(query, clip_model, top_k=1)

        if results:
            product_id, similarity, metadata = results[0]
            logger.info(f"Q{i}: '{query}'")
            logger.info(f"    → {metadata['title']} ({similarity*100:.1f}% match)")


def example_5_vector_store_operations():
    """Example 5: Vector store operations"""
    logger.info("\n" + "=" * 60)
    logger.info("Example 5: Vector Store Operations")
    logger.info("=" * 60)

    from rag import VectorStore

    # Create vector store
    vector_store = VectorStore(dimension=512)
    logger.info("Created empty vector store")

    # Add embeddings
    logger.info("\nAdding 10 random embeddings...")
    for i in range(10):
        embedding = np.random.randn(512)
        embedding = embedding / np.linalg.norm(embedding)

        vector_store.add_embedding(
            embedding_id=f"product_{i}",
            embedding=embedding,
            metadata={"product_id": i, "price": 100 * (i + 1)}
        )

    # Get stats
    stats = vector_store.get_stats()
    logger.info(f"\nVector store statistics:")
    logger.info(f"  - Total embeddings: {stats['total_embeddings']}")
    logger.info(f"  - Embedding dimension: {stats['dimension']}")
    logger.info(f"  - Memory usage: {stats['memory_usage_mb']:.2f} MB")

    # Search
    logger.info("\nSearching for similar embeddings...")
    query = np.random.randn(512)
    query = query / np.linalg.norm(query)

    results = vector_store.search(query, top_k=3)

    for i, (product_id, similarity, metadata) in enumerate(results, 1):
        logger.info(f"{i}. {product_id}: {similarity:.4f}")

    # Delete
    logger.info("\nDeleting product_0...")
    deleted = vector_store.delete_embedding("product_0")
    logger.info(f"Deletion successful: {deleted}")
    logger.info(f"Remaining embeddings: {vector_store.get_size()}")


def main():
    """Run all examples"""
    logger.info("\n")
    logger.info("╔" + "=" * 58 + "╗")
    logger.info("║ CLIP + RAG System - Usage Examples                      ║")
    logger.info("╚" + "=" * 58 + "╝")

    try:
        example_1_text_search()
        example_2_multimodal_search()
        example_3_similarity_computation()
        example_4_batch_search()
        example_5_vector_store_operations()

        logger.info("\n" + "=" * 60)
        logger.info("All examples completed successfully!")
        logger.info("=" * 60 + "\n")

        return 0

    except Exception as e:
        logger.error(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
