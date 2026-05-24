#!/usr/bin/env python3
"""
Test script to verify CLIP + RAG pipeline is working
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_clip_embedding():
    """Test CLIP model loading and embedding generation"""
    logger.info("=" * 50)
    logger.info("Testing CLIP Embedding Model")
    logger.info("=" * 50)

    try:
        from models import CLIPEmbeddingModel
        import numpy as np

        # Initialize CLIP
        logger.info("Initializing CLIP model...")
        clip_model = CLIPEmbeddingModel(model_name="ViT-B/32")

        # Test text embedding
        logger.info("Generating text embedding...")
        texts = ["Samsung Galaxy S21 smartphone with 5G"]
        text_embeddings = clip_model.encode_text(texts)

        logger.info(f"✅ Text embedding shape: {text_embeddings.shape}")
        logger.info(f"✅ Embedding dimension: {clip_model.get_embedding_dimension()}")

        # Test similarity
        logger.info("Computing text-text similarity...")
        text1_emb = clip_model.encode_text("smartphone")
        text2_emb = clip_model.encode_text("mobile phone")
        similarity = clip_model.similarity(text1_emb[0], text2_emb[0])

        logger.info(f"✅ Similarity between 'smartphone' and 'mobile phone': {similarity:.4f}")

        return True

    except Exception as e:
        logger.error(f"❌ CLIP test failed: {e}")
        return False


def test_vector_store():
    """Test vector store"""
    logger.info("\n" + "=" * 50)
    logger.info("Testing Vector Store")
    logger.info("=" * 50)

    try:
        from rag import VectorStore
        import numpy as np

        # Create vector store
        logger.info("Creating vector store...")
        vector_store = VectorStore(dimension=512)

        # Add embeddings
        logger.info("Adding test embeddings...")
        for i in range(5):
            embedding = np.random.randn(512)
            embedding = embedding / np.linalg.norm(embedding)

            vector_store.add_embedding(
                embedding_id=f"product_{i}",
                embedding=embedding,
                metadata={
                    "title": f"Product {i}",
                    "price": 100 * (i + 1),
                    "brand": f"Brand {i}"
                }
            )

        logger.info(f"✅ Added {vector_store.get_size()} embeddings")

        # Test search
        logger.info("Testing search...")
        query_embedding = np.random.randn(512)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        results = vector_store.search(query_embedding, top_k=3)
        logger.info(f"✅ Found {len(results)} results")

        for product_id, similarity, metadata in results:
            logger.info(f"  - {metadata['title']}: {similarity:.4f}")

        return True

    except Exception as e:
        logger.error(f"❌ Vector store test failed: {e}")
        return False


def test_retrieval_system():
    """Test retrieval system"""
    logger.info("\n" + "=" * 50)
    logger.info("Testing Retrieval System")
    logger.info("=" * 50)

    try:
        from data.sample_products import generate_sample_products
        from models import CLIPEmbeddingModel
        from rag import VectorStore, RetrieverSystem

        # Load sample products
        logger.info("Loading sample products...")
        products = generate_sample_products()
        logger.info(f"✅ Loaded {len(products)} products")

        # Initialize CLIP
        logger.info("Initializing CLIP model...")
        clip_model = CLIPEmbeddingModel()

        # Create vector store
        logger.info("Creating vector store and indexing products...")
        vector_store = VectorStore(dimension=512)
        retriever = RetrieverSystem(vector_store)

        # Index products
        for product in products:
            # Get description
            description = product['combined_description']

            # Generate embedding
            embedding = clip_model.encode_text([description])
            if embedding.ndim > 1:
                embedding = embedding[0]

            # Index
            retriever.index_product(
                product_id=product['asin'],
                text_embedding=embedding,
                product_info=product
            )

        logger.info(f"✅ Indexed {vector_store.get_size()} products")

        # Test retrieval
        logger.info("\nTesting retrieval...")
        test_queries = [
            "smartphone with 5G and great camera",
            "fitness tracker",
            "kitchen mixer"
        ]

        for query in test_queries:
            logger.info(f"\nQuery: {query}")

            results = retriever.retrieve_by_text(
                query,
                clip_model,
                top_k=3
            )

            for i, (product_id, similarity, metadata) in enumerate(results, 1):
                logger.info(f"  {i}. {metadata['title']} - {similarity*100:.1f}% match")

        logger.info("✅ Retrieval system working!")
        return True

    except Exception as e:
        logger.error(f"❌ Retrieval system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_end_to_end():
    """Test end-to-end pipeline"""
    logger.info("\n" + "=" * 50)
    logger.info("Testing End-to-End Pipeline")
    logger.info("=" * 50)

    try:
        from data.sample_products import save_sample_products
        from build_embeddings import build_embeddings

        # Generate sample products
        logger.info("Generating sample products...")
        products_path = "test_products.json"
        save_sample_products(products_path)

        # Build embeddings
        logger.info("Building embeddings...")
        vector_store = build_embeddings(products_path, "test_vector_store.json")

        logger.info(f"✅ End-to-end pipeline successful!")
        logger.info(f"   - Vector store size: {vector_store.get_size()}")

        # Cleanup
        import os
        os.remove(products_path)
        os.remove("test_vector_store.json")

        return True

    except Exception as e:
        logger.error(f"❌ End-to-end test failed: {e}")
        return False


def main():
    """Run all tests"""
    logger.info("\n")
    logger.info("╔" + "=" * 48 + "╗")
    logger.info("║ CLIP + RAG Pipeline Test Suite                  ║")
    logger.info("╚" + "=" * 48 + "╝")

    results = {
        "CLIP Embedding": test_clip_embedding(),
        "Vector Store": test_vector_store(),
        "Retrieval System": test_retrieval_system(),
        "End-to-End Pipeline": test_end_to_end(),
    }

    logger.info("\n" + "=" * 50)
    logger.info("Test Results Summary")
    logger.info("=" * 50)

    passed = 0
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1

    logger.info(f"\nTotal: {passed}/{len(results)} tests passed")

    if passed == len(results):
        logger.info("\n🎉 All tests passed! System is ready to use.")
        return 0
    else:
        logger.error(f"\n⚠️  {len(results) - passed} test(s) failed. See logs above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
