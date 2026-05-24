"""
Data loader for the Amazon Product Dataset 2020
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles loading and batching of product data for the chatbot system

    Responsibilities:
    - Load product data from various formats
    - Manage data batching
    - Provide efficient data access
    """

    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the data loader

        Args:
            data_path: Path to the dataset file
        """
        self.data = None
        self.data_path = data_path
        if data_path:
            self.load_data(data_path)

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load product data from file

        Args:
            filepath: Path to the dataset

        Returns:
            Loaded DataFrame
        """
        logger.info(f"Loading data from {filepath}")
        path = Path(filepath)

        try:
            if filepath.endswith('.json'):
                self.data = pd.read_json(filepath, lines=True)
            elif filepath.endswith('.csv'):
                self.data = pd.read_csv(filepath)
            elif filepath.endswith('.parquet'):
                self.data = pd.read_parquet(filepath)
            else:
                raise ValueError(f"Unsupported file format: {filepath}")

            logger.info(f"Loaded {len(self.data)} products")
            return self.data

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def get_product_by_asin(self, asin: str) -> Optional[Dict]:
        """
        Retrieve a product by its ASIN

        Args:
            asin: Amazon Standard Identification Number

        Returns:
            Product data as dictionary, or None if not found
        """
        if self.data is None:
            logger.warning("No data loaded")
            return None

        product = self.data[self.data['asin'] == asin]
        if len(product) > 0:
            return product.iloc[0].to_dict()
        return None

    def get_products_by_category(self, category: str) -> List[Dict]:
        """
        Get all products in a specific category

        Args:
            category: Product category

        Returns:
            List of products in the category
        """
        if self.data is None:
            return []

        products = self.data[self.data['category'] == category]
        return [row.to_dict() for _, row in products.iterrows()]

    def get_batch(self, batch_size: int = 32, shuffle: bool = False) -> List[List[Dict]]:
        """
        Create batches of product data

        Args:
            batch_size: Size of each batch
            shuffle: Whether to shuffle data before batching

        Returns:
            List of batches
        """
        if self.data is None:
            logger.warning("No data loaded")
            return []

        data = self.data.copy()

        if shuffle:
            data = data.sample(frac=1).reset_index(drop=True)

        batches = []
        for i in range(0, len(data), batch_size):
            batch = data.iloc[i:i+batch_size]
            batches.append([row.to_dict() for _, row in batch.iterrows()])

        return batches

    def get_sample(self, n: int = 100) -> pd.DataFrame:
        """
        Get a random sample of products

        Args:
            n: Number of products to sample

        Returns:
            Sample DataFrame
        """
        if self.data is None:
            logger.warning("No data loaded")
            return pd.DataFrame()

        return self.data.sample(n=min(n, len(self.data)))

    def search_products(self, query: str, fields: List[str] = None) -> List[Dict]:
        """
        Search products by text query

        Args:
            query: Search query
            fields: Fields to search in (default: title, description)

        Returns:
            List of matching products
        """
        if self.data is None:
            return []

        if fields is None:
            fields = ['title', 'description', 'features']

        # Filter columns that exist
        fields = [f for f in fields if f in self.data.columns]

        if not fields:
            logger.warning("No searchable fields found")
            return []

        # Case-insensitive search
        query_lower = query.lower()
        mask = pd.Series([False] * len(self.data))

        for field in fields:
            mask |= self.data[field].astype(str).str.lower().str.contains(query_lower, na=False)

        results = self.data[mask]
        return [row.to_dict() for _, row in results.iterrows()]

    def get_dataset_info(self) -> Dict:
        """
        Get information about the loaded dataset

        Returns:
            Dictionary with dataset statistics
        """
        if self.data is None:
            return {}

        info = {
            "total_products": len(self.data),
            "columns": list(self.data.columns),
            "memory_usage": self.data.memory_usage(deep=True).sum() / 1024**2,  # MB
            "missing_values": self.data.isnull().sum().to_dict(),
            "dtypes": self.data.dtypes.astype(str).to_dict(),
        }

        # Add numeric stats
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            info[f"{col}_stats"] = {
                "mean": self.data[col].mean(),
                "std": self.data[col].std(),
                "min": self.data[col].min(),
                "max": self.data[col].max(),
            }

        return info

    def filter_by_price(self, min_price: float = None, max_price: float = None) -> pd.DataFrame:
        """
        Filter products by price range

        Args:
            min_price: Minimum price
            max_price: Maximum price

        Returns:
            Filtered DataFrame
        """
        if self.data is None or 'price' not in self.data.columns:
            return pd.DataFrame()

        result = self.data.copy()

        if min_price is not None:
            result = result[result['price'] >= min_price]

        if max_price is not None:
            result = result[result['price'] <= max_price]

        return result

    def get_top_products(self, column: str = 'price', n: int = 10, ascending: bool = False) -> List[Dict]:
        """
        Get top N products sorted by a column

        Args:
            column: Column to sort by
            n: Number of top products
            ascending: Sort order

        Returns:
            List of top products
        """
        if self.data is None or column not in self.data.columns:
            return []

        top = self.data.nlargest(n, column) if not ascending else self.data.nsmallest(n, column)
        return [row.to_dict() for _, row in top.iterrows()]
