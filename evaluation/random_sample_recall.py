#!/usr/bin/env python3
"""
Random-sample recall evaluation for product title and product image queries.

This script samples products from a dataset, queries the existing vector store
with each product's title and/or image, and checks whether the same ASIN is
retrieved. With one relevant item per query, Recall@K is equivalent to hit@K.
"""

import argparse
import json
import logging
import sys
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import numpy as np
import requests
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from build_embeddings import load_products
from evaluation.metrics import EvaluationMetrics
from models import CLIPEmbeddingModel
from rag import ChromaVectorStore, VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_k_values(value: str) -> List[int]:
    """Parse a comma-separated list of positive integer K cutoffs."""
    try:
        k_values = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--k-values must be comma-separated integers") from exc

    if not k_values or any(k <= 0 for k in k_values):
        raise argparse.ArgumentTypeError("--k-values must contain positive integers")

    return k_values


def normalize_product_id(product: Dict) -> str:
    """Return the canonical product id used by the embedding index."""
    return str(product.get("asin") or product.get("product_id") or "").strip()


def first_image_reference(image_field) -> Optional[str]:
    """Return the first usable image URL or local path from a dataset image field."""
    if image_field is None:
        return None

    if isinstance(image_field, (list, tuple)):
        candidates = image_field
    else:
        candidates = str(image_field).split("|")

    for candidate in candidates:
        image_ref = str(candidate).strip()
        if image_ref and "transparent-pixel" not in image_ref:
            return image_ref

    return None


def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def load_image(image_ref: str, timeout: float = 10.0) -> Image.Image:
    """Load an image from a URL or local path."""
    if is_url(image_ref):
        response = requests.get(image_ref, timeout=timeout)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")

    image_path = Path(image_ref)
    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path

    return Image.open(image_path).convert("RGB")


def load_vector_store(args):
    """Load the requested vector store backend."""
    if args.store == "chroma":
        return ChromaVectorStore(
            persist_directory=args.persist_directory,
            collection_name=args.collection_name,
            dimension=args.dimension,
            reset=False,
        )

    vector_store = VectorStore(dimension=args.dimension)
    vector_store.load(args.vector_store)
    return vector_store


def sample_products(
    products: Sequence[Dict],
    sample_size: int,
    seed: int,
    mode: str,
) -> List[Dict]:
    """Filter products to evaluable rows and return a deterministic random sample."""
    eligible = []
    for product in products:
        product_id = normalize_product_id(product)
        has_title = bool(str(product.get("title") or "").strip())
        has_image = bool(first_image_reference(product.get("image")))

        if not product_id:
            continue
        if mode in {"title", "both"} and not has_title:
            continue
        if mode in {"photo", "both"} and not has_image:
            continue

        eligible.append(product)

    if not eligible:
        raise ValueError(f"No products are eligible for mode={mode!r}")

    rng = np.random.default_rng(seed)
    sample_count = min(sample_size, len(eligible))
    indices = rng.choice(len(eligible), size=sample_count, replace=False)
    return [eligible[int(index)] for index in indices]


def retrieved_ids_from_results(results: Iterable[Tuple[str, float, Dict]]) -> List[str]:
    return [str(product_id) for product_id, _score, _metadata in results]


def evaluate_title_queries(
    sample: Sequence[Dict],
    vector_store,
    embedding_model: CLIPEmbeddingModel,
    k_values: Sequence[int],
) -> Tuple[Dict[str, float], List[Dict]]:
    """Evaluate retrieval when the query is the product title."""
    metrics = EvaluationMetrics()
    max_k = max(k_values)
    retrieved_lists = []
    ground_truth_lists = []
    details = []

    for product in sample:
        product_id = normalize_product_id(product)
        title = str(product.get("title") or "").strip()
        query_embedding = embedding_model.encode_text(title)
        if query_embedding.ndim > 1:
            query_embedding = query_embedding[0]

        results = vector_store.search(query_embedding, top_k=max_k)
        retrieved_ids = retrieved_ids_from_results(results)

        retrieved_lists.append(retrieved_ids)
        ground_truth_lists.append([product_id])
        details.append(
            {
                "query_type": "title",
                "asin": product_id,
                "title": title,
                "retrieved_ids": retrieved_ids,
                "first_rank": (retrieved_ids.index(product_id) + 1)
                if product_id in retrieved_ids
                else None,
            }
        )

    return metrics.evaluate_retrieval(retrieved_lists, ground_truth_lists, list(k_values)), details


def evaluate_photo_queries(
    sample: Sequence[Dict],
    vector_store,
    embedding_model: CLIPEmbeddingModel,
    k_values: Sequence[int],
    image_timeout: float,
) -> Tuple[Dict[str, float], List[Dict]]:
    """Evaluate retrieval when the query is the product image."""
    metrics = EvaluationMetrics()
    max_k = max(k_values)
    retrieved_lists = []
    ground_truth_lists = []
    details = []

    for product in sample:
        product_id = normalize_product_id(product)
        title = str(product.get("title") or "").strip()
        image_ref = first_image_reference(product.get("image"))

        try:
            image = load_image(image_ref, timeout=image_timeout)
            query_embedding = embedding_model.encode_image(image)
            if query_embedding.ndim > 1:
                query_embedding = query_embedding[0]
            results = vector_store.search(query_embedding, top_k=max_k)
            retrieved_ids = retrieved_ids_from_results(results)
            error = None
        except Exception as exc:
            logger.warning("Skipping image query for %s: %s", product_id, exc)
            retrieved_ids = []
            error = str(exc)

        retrieved_lists.append(retrieved_ids)
        ground_truth_lists.append([product_id])
        details.append(
            {
                "query_type": "photo",
                "asin": product_id,
                "title": title,
                "image": image_ref,
                "retrieved_ids": retrieved_ids,
                "first_rank": (retrieved_ids.index(product_id) + 1)
                if product_id in retrieved_ids
                else None,
                "error": error,
            }
        )

    return metrics.evaluate_retrieval(retrieved_lists, ground_truth_lists, list(k_values)), details


def summarize_metrics(result: Dict, k_values: Sequence[int]) -> Dict[str, float]:
    """Create cross-query averages for metrics available in the result."""
    query_metrics = result["query_metrics"]
    summary = {}

    for k in k_values:
        values = [
            metrics[f"recall@{k}"]
            for metrics in query_metrics.values()
            if f"recall@{k}" in metrics
        ]
        if values:
            summary[f"avg_recall@{k}"] = float(np.mean(values))

    for metric_name in ["mrr", "ndcg"]:
        values = [
            metrics[metric_name]
            for metrics in query_metrics.values()
            if metric_name in metrics
        ]
        if values:
            summary[f"avg_{metric_name}"] = float(np.mean(values))

    if "title" in query_metrics and "photo" in query_metrics:
        for k in k_values:
            summary[f"photo_minus_title_recall@{k}"] = (
                query_metrics["photo"][f"recall@{k}"]
                - query_metrics["title"][f"recall@{k}"]
            )

    return summary


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def write_details_csv(path: Path, details: Sequence[Dict]) -> None:
    """Write per-sample details without requiring pandas."""
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["query_type", "asin", "title", "image", "first_rank", "error", "retrieved_ids"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in details:
            csv_row = {field: row.get(field, "") for field in fields}
            csv_row["retrieved_ids"] = "|".join(row.get("retrieved_ids") or [])
            writer.writerow(csv_row)


def run(args) -> Dict:
    products = load_products(args.data)
    sample = sample_products(
        products,
        sample_size=args.sample_size,
        seed=args.seed,
        mode=args.mode,
    )

    logger.info("Loaded %s products; evaluating %s sampled products", len(products), len(sample))

    vector_store = load_vector_store(args)
    if vector_store.get_size() == 0:
        raise ValueError("Vector store is empty. Build embeddings before running metrics.")

    embedding_model = CLIPEmbeddingModel(model_name=args.clip_model, device=args.device)

    query_metrics = {}
    details = []

    if args.mode in {"title", "both"}:
        title_metrics, title_details = evaluate_title_queries(
            sample,
            vector_store,
            embedding_model,
            args.k_values,
        )
        query_metrics["title"] = title_metrics
        details.extend(title_details)

    if args.mode in {"photo", "both"}:
        photo_metrics, photo_details = evaluate_photo_queries(
            sample,
            vector_store,
            embedding_model,
            args.k_values,
            args.image_timeout,
        )
        query_metrics["photo"] = photo_metrics
        details.extend(photo_details)

    result = {
        "config": {
            "data": args.data,
            "store": args.store,
            "persist_directory": args.persist_directory if args.store == "chroma" else None,
            "vector_store": args.vector_store if args.store == "json" else None,
            "collection_name": args.collection_name if args.store == "chroma" else None,
            "sample_size_requested": args.sample_size,
            "sample_size_evaluated": len(sample),
            "seed": args.seed,
            "k_values": list(args.k_values),
            "mode": args.mode,
        },
        "query_metrics": query_metrics,
        "summary": {},
        "details": details,
    }
    result["summary"] = summarize_metrics(result, args.k_values)

    write_json(Path(args.output), result)
    if args.details_csv:
        write_details_csv(Path(args.details_csv), details)

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate average recall over random product title/photo samples."
    )
    parser.add_argument("--data", required=True, help="Product dataset path: JSONL, CSV, or parquet")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Number of random products to evaluate",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument(
        "--mode",
        choices=["title", "photo", "both"],
        default="both",
        help="Which query types to evaluate",
    )
    parser.add_argument(
        "--k-values",
        type=parse_k_values,
        default=[1, 5, 10],
        help="Comma-separated recall cutoffs, e.g. 1,5,10",
    )
    parser.add_argument(
        "--store",
        choices=["chroma", "json"],
        default="chroma",
        help="Vector store backend to query",
    )
    parser.add_argument(
        "--persist-directory",
        default="chroma_db",
        help="ChromaDB persistence directory",
    )
    parser.add_argument(
        "--collection-name",
        default="amazon_products",
        help="ChromaDB collection name",
    )
    parser.add_argument(
        "--vector-store",
        default="vector_store.json",
        help="JSON vector store path when --store=json",
    )
    parser.add_argument("--clip-model", default="ViT-B/32", help="CLIP model name")
    parser.add_argument("--device", default="cuda", help="Preferred CLIP device: cuda or cpu")
    parser.add_argument("--dimension", type=int, default=512, help="Embedding dimension")
    parser.add_argument(
        "--image-timeout",
        type=float,
        default=10.0,
        help="Seconds before an image URL request times out",
    )
    parser.add_argument(
        "--output",
        default="evaluation/results/random_sample_recall_metrics.json",
        help="Path for aggregate metrics JSON",
    )
    parser.add_argument(
        "--details-csv",
        default="evaluation/results/random_sample_recall_details.csv",
        help="Optional path for per-query CSV details. Use an empty string to skip.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.sample_size <= 0:
        parser.error("--sample-size must be positive")
    if args.details_csv == "":
        args.details_csv = None

    result = run(args)

    print(json.dumps({"query_metrics": result["query_metrics"], "summary": result["summary"]}, indent=2))
    print(f"\nWrote metrics to {args.output}")
    if args.details_csv:
        print(f"Wrote per-query details to {args.details_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
