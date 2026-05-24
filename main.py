#!/usr/bin/env python3
"""
Main script to run the Multimodal E-commerce Chatbot

This script demonstrates how to:
1. Load and preprocess data
2. Generate embeddings using CLIP
3. Store embeddings in vector database
4. Retrieve products using RAG
5. Generate responses using LLM
"""

import logging
import argparse
import sys
from pathlib import Path

from config import (
    CLIP_MODEL_NAME,
    LLM_MODEL_NAME,
    VECTOR_STORE_PATH,
    LOG_LEVEL
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_logging(log_file: str = None):
    """Setup logging configuration"""
    if log_file:
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)


def load_data(data_path: str):
    """
    Load and preprocess data

    Args:
        data_path: Path to dataset
    """
    logger.info(f"Loading data from {data_path}")

    try:
        from data import DataLoader, DataPreprocessor

        # Preprocess data
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.preprocess(
            data_path,
            output_path="data/processed/amazon_products.json"
        )

        logger.info(f"Loaded and processed {len(processed_data)} products")

        # Get statistics
        stats = preprocessor.get_statistics()
        logger.info(f"Dataset statistics: {stats}")

        return processed_data

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def initialize_embedding_model():
    """
    Initialize CLIP embedding model

    Returns:
        CLIPEmbeddingModel instance
    """
    logger.info(f"Initializing CLIP model: {CLIP_MODEL_NAME}")

    try:
        from models import CLIPEmbeddingModel

        model = CLIPEmbeddingModel(model_name=CLIP_MODEL_NAME)
        embedding_dim = model.get_embedding_dimension()
        logger.info(f"CLIP model initialized (embedding dim: {embedding_dim})")

        return model

    except Exception as e:
        logger.error(f"Error initializing embedding model: {e}")
        raise


def initialize_vector_store(embedding_dim: int = 512):
    """
    Initialize vector store for embeddings

    Args:
        embedding_dim: Dimension of embeddings

    Returns:
        VectorStore instance
    """
    logger.info(f"Initializing vector store (dimension: {embedding_dim})")

    try:
        from rag import VectorStore

        vector_store = VectorStore(dimension=embedding_dim)
        logger.info("Vector store initialized")

        return vector_store

    except Exception as e:
        logger.error(f"Error initializing vector store: {e}")
        raise


def initialize_retriever(vector_store):
    """
    Initialize retrieval system

    Args:
        vector_store: VectorStore instance

    Returns:
        RetrieverSystem instance
    """
    logger.info("Initializing retriever system")

    try:
        from rag import RetrieverSystem

        retriever = RetrieverSystem(vector_store)
        logger.info("Retriever system initialized")

        return retriever

    except Exception as e:
        logger.error(f"Error initializing retriever: {e}")
        raise


def initialize_llm():
    """
    Initialize Language Model

    Returns:
        LLMInterface instance
    """
    logger.info(f"Initializing LLM: {LLM_MODEL_NAME}")

    try:
        from models import LLMInterface

        llm = LLMInterface(model_name=LLM_MODEL_NAME)
        # Note: Uncomment to load actual model
        # llm.load_model()
        logger.info("LLM interface initialized")

        return llm

    except Exception as e:
        logger.error(f"Error initializing LLM: {e}")
        raise


def demo_text_query(embedding_model, retriever, llm):
    """
    Demonstrate text-based query

    Args:
        embedding_model: CLIP model
        retriever: RetrieverSystem
        llm: LLMInterface
    """
    logger.info("=== Demo: Text Query ===")

    query = "What are the features of Samsung Galaxy S21?"
    logger.info(f"Query: {query}")

    try:
        # Retrieve products
        results = retriever.retrieve_by_text(
            query,
            embedding_model,
            top_k=5
        )

        logger.info(f"Retrieved {len(results)} products")

        # Format context
        context = retriever.format_retrieval_results(results)
        logger.info(f"Context:\n{context}")

        # Generate response
        response = llm.generate_with_strategy(
            query,
            context=context
        )

        logger.info(f"Response: {response}")

    except Exception as e:
        logger.error(f"Error in text query demo: {e}")


def demo_evaluation(embedding_model, retriever):
    """
    Demonstrate system evaluation

    Args:
        embedding_model: CLIP model
        retriever: RetrieverSystem
    """
    logger.info("=== Demo: Evaluation ===")

    # Mock test queries and ground truth
    test_queries = [
        "smartphone with good camera",
        "fitness tracker",
        "kitchen mixer"
    ]

    ground_truth = [
        ["samsung_galaxy", "iphone_14", "pixel_6"],
        ["fitbit_charge", "apple_watch", "garmin"],
        ["kitchenaid_mixer", "bosch_mixer", "stand_mixer"]
    ]

    try:
        metrics = retriever.evaluate_retrieval(
            test_queries,
            ground_truth,
            embedding_model,
            [1, 5, 10]
        )

        logger.info(f"Evaluation metrics: {metrics}")

    except Exception as e:
        logger.error(f"Error in evaluation: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Multimodal E-commerce Chatbot"
    )

    parser.add_argument(
        "--data",
        type=str,
        help="Path to dataset"
    )

    parser.add_argument(
        "--run-demo",
        action="store_true",
        help="Run demonstration"
    )

    parser.add_argument(
        "--log-file",
        type=str,
        help="Path to log file"
    )

    args = parser.parse_args()

    # Setup logging
    if args.log_file:
        setup_logging(args.log_file)

    logger.info("Starting Multimodal E-commerce Chatbot")

    try:
        # Initialize components
        logger.info("Initializing system components...")

        embedding_model = initialize_embedding_model()
        vector_store = initialize_vector_store(
            embedding_dim=embedding_model.get_embedding_dimension()
        )
        retriever = initialize_retriever(vector_store)
        llm = initialize_llm()

        logger.info("System initialized successfully")

        # Load data if provided
        if args.data:
            processed_data = load_data(args.data)
            logger.info(f"Loaded {len(processed_data)} products")

        # Run demo if requested
        if args.run_demo:
            logger.info("Running demonstration...")

            demo_text_query(embedding_model, retriever, llm)
            demo_evaluation(embedding_model, retriever)

            logger.info("Demonstration complete")

        logger.info("Chatbot ready for use")

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
