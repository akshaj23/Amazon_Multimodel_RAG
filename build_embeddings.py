#!/usr/bin/env python3
"""
Build CLIP embeddings for product dataset and populate vector store
"""

import logging
import json
import argparse
from pathlib import Path
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_products(filepath: str):
    """
    Load products from JSONL, CSV, or parquet file

    Args:
        filepath: Path to product file

    Returns:
        List of product dictionaries
    """
    path = Path(filepath)

    try:
        if path.suffix.lower() in {".csv", ".parquet"}:
            from data.preprocessing import DataPreprocessor

            preprocessor = DataPreprocessor()
            products = preprocessor.preprocess(filepath).to_dict("records")
            logger.info(f"Loaded and preprocessed {len(products)} products from {filepath}")
            return products

        products = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    products.append(json.loads(line))
        logger.info(f"Loaded {len(products)} products from {filepath}")
        return products
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise


def initialize_clip_model():
    """
    Initialize CLIP model for embeddings

    Returns:
        CLIP model instance
    """
    logger.info("Initializing CLIP model...")

    try:
        import clip
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        model, preprocess = clip.load("ViT-B/32", device=device)
        logger.info("CLIP model loaded successfully")

        return model, preprocess, device

    except ImportError as e:
        logger.error(f"CLIP not installed. Install with: pip install openai-clip")
        raise
    except Exception as e:
        logger.error(f"Error loading CLIP: {e}")
        raise


def generate_text_embeddings(products, model, device):
    """
    Generate CLIP embeddings for product descriptions

    Args:
        products: List of product dictionaries
        model: CLIP model
        device: Device to use (cuda/cpu)

    Returns:
        Dictionary mapping product_id to embeddings
    """
    import clip
    import torch

    logger.info(f"Generating embeddings for {len(products)} products...")

    embeddings = {}

    for i, product in enumerate(products):
        try:
            # Use combined description for better context
            text = product.get('combined_description', product.get('description', ''))

            if not text:
                logger.warning(f"No description for product {product.get('asin')}")
                continue

            # Generate embedding
            with torch.no_grad():
                text_tokens = clip.tokenize([text], truncate=True).to(device)
                text_embedding = model.encode_text(text_tokens)

                # Normalize
                text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
                embeddings[product['asin']] = text_embedding.cpu().numpy()[0]

            if (i + 1) % 5 == 0:
                logger.info(f"Processed {i + 1}/{len(products)} products")

        except Exception as e:
            logger.error(f"Error processing product {product.get('asin')}: {e}")
            continue

    logger.info(f"Generated {len(embeddings)} embeddings")
    return embeddings


def create_vector_store(products, embeddings, store_type="json", persist_directory="chroma_db"):
    """
    Create vector store with product embeddings

    Args:
        products: List of product dictionaries
        embeddings: Dictionary of product embeddings

    Returns:
        Populated VectorStore instance
    """
    from rag import ChromaVectorStore, VectorStore

    logger.info("Creating vector store...")

    if store_type == "chroma":
        vector_store = ChromaVectorStore(
            persist_directory=persist_directory,
            collection_name="amazon_products",
            dimension=512,
            reset=True,
        )
    else:
        vector_store = VectorStore(dimension=512)

    batch_embeddings = {}
    batch_metadata = {}

    for product in products:
        product_id = product['asin']

        if product_id not in embeddings:
            continue

        batch_embeddings[product_id] = embeddings[product_id]
        batch_metadata[product_id] = {
            'asin': product_id,
            'title': product.get('title', ''),
            'brand': product.get('brand', ''),
            'price': product.get('price', 0),
            'category': product.get('category', ''),
            'features': product.get('features', ''),
            'description': product.get('description', ''),
            'combined_description': product.get('combined_description', ''),
            'image': product.get('image', ''),
            'product_url': product.get('product_url', '')
        }

    vector_store.add_embeddings(batch_embeddings, batch_metadata)

    logger.info(f"Vector store created with {vector_store.get_size()} products")
    return vector_store


def save_vector_store(vector_store, output_path="vector_store.json"):
    """
    Save vector store to disk

    Args:
        vector_store: VectorStore instance
        output_path: Path to save
    """
    logger.info(f"Saving vector store to {output_path}...")

    parent = Path(output_path).parent
    if str(parent):
        parent.mkdir(parents=True, exist_ok=True)
    vector_store.save(output_path)

    logger.info(f"Vector store saved")


def build_embeddings(
    data_path: str,
    output_path: str = "vector_store.json",
    store_type: str = "json",
    persist_directory: str = "chroma_db",
):
    """
    Main function to build embeddings pipeline

    Args:
        data_path: Path to product data
        output_path: Path to save vector store
    """
    logger.info("=" * 50)
    logger.info("Building CLIP Embeddings Pipeline")
    logger.info("=" * 50)

    try:
        # 1. Load products
        products = load_products(data_path)

        # 2. Initialize CLIP
        model, preprocess, device = initialize_clip_model()

        # 3. Generate embeddings
        embeddings = generate_text_embeddings(products, model, device)

        # 4. Create vector store
        vector_store = create_vector_store(
            products,
            embeddings,
            store_type=store_type,
            persist_directory=persist_directory,
        )

        # 5. Save vector store
        if store_type == "chroma":
            vector_store.save()
        else:
            save_vector_store(vector_store, output_path)

        logger.info("=" * 50)
        logger.info("Pipeline completed successfully!")
        logger.info(f"Total products indexed: {vector_store.get_size()}")
        if store_type == "chroma":
            logger.info(f"Vector store saved to: {persist_directory}")
        else:
            logger.info(f"Vector store saved to: {output_path}")
        logger.info("=" * 50)

        return vector_store

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Build CLIP embeddings for product dataset"
    )

    parser.add_argument(
        "--data",
        type=str,
        default="data/raw/sample_products.json",
        help="Path to product data (JSONL format)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="vector_store.json",
        help="Path to save vector store"
    )

    parser.add_argument(
        "--store",
        type=str,
        choices=["json", "chroma"],
        default="chroma",
        help="Vector store backend to build"
    )

    parser.add_argument(
        "--persist-directory",
        type=str,
        default="chroma_db",
        help="Directory for persistent ChromaDB data"
    )

    parser.add_argument(
        "--generate-sample",
        action="store_true",
        help="Generate sample products first"
    )

    args = parser.parse_args()

    # Generate sample products if requested
    if args.generate_sample:
        logger.info("Generating sample products...")
        from data.sample_products import save_sample_products
        save_sample_products(args.data)

    # Build embeddings
    vector_store = build_embeddings(
        args.data,
        args.output,
        store_type=args.store,
        persist_directory=args.persist_directory,
    )

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
