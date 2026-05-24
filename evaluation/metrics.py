"""
Evaluation metrics for the multimodal chatbot system
"""

import logging
import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """
    Compute and track evaluation metrics for the chatbot system

    Includes:
    - Retrieval metrics (recall@k, accuracy)
    - Response quality metrics
    - System performance metrics
    """

    def __init__(self):
        """Initialize metrics tracker"""
        self.retrieval_metrics = {}
        self.response_metrics = {}
        self.system_metrics = {}

    def compute_recall_at_k(
        self,
        retrieved: List[str],
        ground_truth: List[str],
        k: int
    ) -> float:
        """
        Compute Recall@K

        Args:
            retrieved: List of retrieved item IDs
            ground_truth: List of relevant item IDs
            k: Cutoff position

        Returns:
            Recall@K score
        """
        if len(ground_truth) == 0:
            return 0.0

        # Get top-k retrieved
        top_k = set(retrieved[:k])
        relevant = set(ground_truth)

        # Compute recall
        if len(relevant) == 0:
            return 0.0

        recall = len(top_k & relevant) / len(relevant)
        return recall

    def compute_precision_at_k(
        self,
        retrieved: List[str],
        ground_truth: List[str],
        k: int
    ) -> float:
        """
        Compute Precision@K

        Args:
            retrieved: List of retrieved item IDs
            ground_truth: List of relevant item IDs
            k: Cutoff position

        Returns:
            Precision@K score
        """
        # Get top-k retrieved
        top_k = set(retrieved[:k])
        relevant = set(ground_truth)

        # Compute precision
        if len(top_k) == 0:
            return 0.0

        precision = len(top_k & relevant) / len(top_k)
        return precision

    def compute_mrr(
        self,
        retrieved: List[str],
        ground_truth: List[str]
    ) -> float:
        """
        Compute Mean Reciprocal Rank (MRR)

        Args:
            retrieved: List of retrieved item IDs
            ground_truth: List of relevant item IDs

        Returns:
            MRR score
        """
        relevant_set = set(ground_truth)

        for i, item in enumerate(retrieved):
            if item in relevant_set:
                return 1.0 / (i + 1)

        return 0.0

    def compute_ndcg(
        self,
        retrieved: List[Tuple[str, float]],
        ground_truth: List[str],
        k: int = None
    ) -> float:
        """
        Compute Normalized Discounted Cumulative Gain (NDCG)

        Args:
            retrieved: List of (item_id, score) tuples
            ground_truth: List of relevant item IDs
            k: Cutoff position

        Returns:
            NDCG score
        """
        if k is None:
            k = len(retrieved)

        # Compute DCG
        dcg = 0.0
        retrieved_ids = [item[0] for item in retrieved[:k]]
        relevant_set = set(ground_truth)

        for i, item_id in enumerate(retrieved_ids):
            if item_id in relevant_set:
                dcg += 1.0 / np.log2(i + 2)

        # Compute IDCG (ideal DCG)
        ideal_k = min(k, len(ground_truth))
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_k))

        # Compute NDCG
        if idcg == 0:
            return 0.0

        ndcg = dcg / idcg
        return ndcg

    def evaluate_retrieval(
        self,
        retrieved_list: List[List[str]],
        ground_truth_list: List[List[str]],
        k_values: List[int] = [1, 5, 10]
    ) -> Dict:
        """
        Evaluate retrieval performance across multiple queries

        Args:
            retrieved_list: List of retrieved items per query
            ground_truth_list: List of ground truth per query
            k_values: Cutoff values for recall/precision

        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {}

        # Recall and Precision at different K
        for k in k_values:
            recall_scores = []
            precision_scores = []

            for retrieved, ground_truth in zip(retrieved_list, ground_truth_list):
                recall = self.compute_recall_at_k(retrieved, ground_truth, k)
                precision = self.compute_precision_at_k(retrieved, ground_truth, k)

                recall_scores.append(recall)
                precision_scores.append(precision)

            metrics[f'recall@{k}'] = np.mean(recall_scores)
            metrics[f'precision@{k}'] = np.mean(precision_scores)

        # MRR
        mrr_scores = []
        for retrieved, ground_truth in zip(retrieved_list, ground_truth_list):
            mrr = self.compute_mrr(retrieved, ground_truth)
            mrr_scores.append(mrr)

        metrics['mrr'] = np.mean(mrr_scores)

        # NDCG
        ndcg_scores = []
        for retrieved, ground_truth in zip(retrieved_list, ground_truth_list):
            # Convert to proper format if needed
            if retrieved and isinstance(retrieved[0], tuple):
                ndcg = self.compute_ndcg(retrieved, ground_truth)
            else:
                # If no scores, treat all items equally
                retrieved_with_scores = [(item, 1.0) for item in retrieved]
                ndcg = self.compute_ndcg(retrieved_with_scores, ground_truth)

            ndcg_scores.append(ndcg)

        metrics['ndcg'] = np.mean(ndcg_scores)

        self.retrieval_metrics = metrics
        logger.info(f"Retrieval evaluation: {metrics}")

        return metrics

    def evaluate_response_quality(
        self,
        predictions: List[str],
        references: List[str],
        metric: str = "accuracy"
    ) -> float:
        """
        Evaluate response quality

        Args:
            predictions: Generated responses
            references: Reference responses
            metric: Evaluation metric type

        Returns:
            Score
        """
        if metric == "accuracy":
            # Exact match accuracy
            matches = sum(1 for p, r in zip(predictions, references) if p == r)
            score = matches / len(predictions) if predictions else 0
        else:
            raise ValueError(f"Unknown metric: {metric}")

        self.response_metrics[metric] = score
        logger.info(f"Response quality ({metric}): {score:.2%}")

        return score

    def compute_embedding_quality(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> Dict:
        """
        Compute embedding quality metrics

        Args:
            embeddings: Embedding vectors [batch_size, dim]
            labels: Class labels

        Returns:
            Dictionary with quality metrics
        """
        metrics = {}

        # Average distance within cluster
        unique_labels = np.unique(labels)

        intra_distances = []
        for label in unique_labels:
            mask = labels == label
            class_embeddings = embeddings[mask]

            if len(class_embeddings) > 1:
                # Compute pairwise distances
                distances = []
                for i in range(len(class_embeddings)):
                    for j in range(i+1, len(class_embeddings)):
                        dist = np.linalg.norm(
                            class_embeddings[i] - class_embeddings[j]
                        )
                        distances.append(dist)

                if distances:
                    intra_distances.append(np.mean(distances))

        metrics['avg_intra_distance'] = np.mean(intra_distances) if intra_distances else 0

        # Embedding dimension statistics
        metrics['embedding_dim'] = embeddings.shape[1]
        metrics['embedding_norm_mean'] = np.mean(
            np.linalg.norm(embeddings, axis=1)
        )

        return metrics

    def get_summary_report(self) -> Dict:
        """
        Get summary report of all metrics

        Returns:
            Dictionary with all computed metrics
        """
        report = {
            'retrieval': self.retrieval_metrics,
            'response': self.response_metrics,
            'system': self.system_metrics
        }

        logger.info(f"Metrics summary: {report}")
        return report

    def log_metrics(self) -> None:
        """Log all metrics to logger"""
        logger.info("=== Evaluation Metrics Summary ===")

        if self.retrieval_metrics:
            logger.info("Retrieval Metrics:")
            for metric, value in self.retrieval_metrics.items():
                logger.info(f"  {metric}: {value:.4f}")

        if self.response_metrics:
            logger.info("Response Metrics:")
            for metric, value in self.response_metrics.items():
                logger.info(f"  {metric}: {value:.4f}")

        if self.system_metrics:
            logger.info("System Metrics:")
            for metric, value in self.system_metrics.items():
                logger.info(f"  {metric}: {value}")

        logger.info("=================================")
