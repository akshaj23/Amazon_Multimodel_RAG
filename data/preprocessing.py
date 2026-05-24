"""
Data preprocessing module for the Amazon Product Dataset
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
import logging
import re

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Handles preprocessing of the Amazon Product Dataset 2020

    Key responsibilities:
    - Clean and validate product data
    - Combine product attributes into comprehensive descriptions
    - Ensure data consistency and quality
    - Handle missing values
    """

    def __init__(self):
        self.processed_data = None
        self.stats = {}

    def load_raw_data(self, filepath: str) -> pd.DataFrame:
        """
        Load raw product data from file

        Args:
            filepath: Path to the dataset file

        Returns:
            DataFrame with raw product data
        """
        logger.info(f"Loading data from {filepath}")
        try:
            if filepath.endswith(".csv"):
                df = pd.read_csv(filepath)
            elif filepath.endswith(".json") or filepath.endswith(".jsonl"):
                df = pd.read_json(filepath, lines=True)
            elif filepath.endswith(".parquet"):
                df = pd.read_parquet(filepath)
            else:
                raise ValueError(f"Unsupported file format: {filepath}")
            logger.info(f"Loaded {len(df)} products")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize known Amazon Product Dataset 2020 columns to the schema used by
        the embedding and UI layers.
        """
        df = df.copy()

        column_map = {
            "Uniq Id": "uniq_id",
            "Product Name": "title",
            "Brand Name": "brand",
            "Asin": "asin",
            "Category": "category",
            "Selling Price": "price",
            "About Product": "features",
            "Product Description": "description",
            "Product Url": "product_url",
            "Image": "image",
        }

        for source, target in column_map.items():
            if source in df.columns and target not in df.columns:
                df[target] = df[source]

        if "description" in df.columns and "Technical Details" in df.columns:
            df["description"] = df["description"].fillna(df["Technical Details"])
        elif "Technical Details" in df.columns:
            df["description"] = df["Technical Details"]

        if "asin" not in df.columns:
            df["asin"] = ""

        df["asin"] = df["asin"].apply(self.clean_text)
        if "product_url" in df.columns:
            missing_asin = df["asin"].eq("") | df["asin"].str.lower().eq("nan")
            df.loc[missing_asin, "asin"] = df.loc[missing_asin, "product_url"].apply(
                self.extract_asin_from_url
            )

        missing_asin = df["asin"].eq("") | df["asin"].str.lower().eq("nan")
        if "uniq_id" in df.columns:
            df.loc[missing_asin, "asin"] = df.loc[missing_asin, "uniq_id"].apply(self.clean_text)

        return df

    def extract_asin_from_url(self, url: str) -> str:
        """Extract ASIN-like product id from an Amazon product URL."""
        if pd.isna(url):
            return ""

        match = re.search(r"/dp/([A-Z0-9]{10})", str(url))
        return match.group(1) if match else ""

    def parse_price(self, price) -> float:
        """Parse Amazon price strings like '$237.68' or '$19.99 - $24.99'."""
        if pd.isna(price):
            return 0.0

        numbers = re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", str(price))
        if not numbers:
            return 0.0

        return float(numbers[0].replace(",", ""))

    def clean_text(self, text: str) -> str:
        """
        Clean text by removing special characters and extra whitespace

        Args:
            text: Input text

        Returns:
            Cleaned text
        """
        if pd.isna(text):
            return ""

        # Convert to string and strip whitespace
        text = str(text).strip()

        # Remove extra whitespace
        text = " ".join(text.split())

        return text

    def create_product_description(self, row: pd.Series) -> str:
        """
        Create comprehensive product description from multiple attributes

        Args:
            row: Product data row

        Returns:
            Combined product description
        """
        description_parts = []

        # Add product title
        if "title" in row and pd.notna(row["title"]):
            description_parts.append(f"Product: {self.clean_text(row['title'])}")

        # Add brand
        if "brand" in row and pd.notna(row["brand"]):
            description_parts.append(f"Brand: {self.clean_text(row['brand'])}")

        # Add price
        if "price" in row and pd.notna(row["price"]):
            description_parts.append(f"Price: {row['price']}")

        # Add features/description
        if "features" in row and pd.notna(row["features"]):
            features = self.clean_text(row["features"]).replace(" | ", ". ")
            description_parts.append(f"Features: {features}")

        if "description" in row and pd.notna(row["description"]):
            desc = self.clean_text(row["description"]).replace(" | ", ". ")
            description_parts.append(f"Description: {desc}")

        # Add category
        if "category" in row and pd.notna(row["category"]):
            category = self.clean_text(row["category"])
            description_parts.append(f"Category: {category}")

        return " | ".join(description_parts)

    def preprocess(self, filepath: str, output_path: str = None) -> pd.DataFrame:
        """
        Main preprocessing pipeline

        Args:
            filepath: Path to raw dataset
            output_path: Path to save processed data

        Returns:
            Processed DataFrame
        """
        # Load raw data
        df = self.load_raw_data(filepath)

        logger.info("Starting preprocessing pipeline...")

        df = self.normalize_columns(df)

        # Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates(subset=["asin"], keep="first")
        logger.info(f"Removed {initial_rows - len(df)} duplicate products")

        # Handle missing values
        logger.info("Handling missing values...")
        df = self._handle_missing_values(df)

        # Clean text fields
        logger.info("Cleaning text fields...")
        text_fields = ["title", "brand", "features", "description", "category"]
        for field in text_fields:
            if field in df.columns:
                df[field] = df[field].apply(self.clean_text)

        # Create combined descriptions
        logger.info("Creating product descriptions...")
        df["combined_description"] = df.apply(self.create_product_description, axis=1)

        # Validate data
        logger.info("Validating data...")
        self._validate_data(df)

        self.processed_data = df

        # Save if output path provided
        if output_path:
            logger.info(f"Saving processed data to {output_path}")
            from pathlib import Path
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_json(output_path, orient="records", lines=True)

        logger.info(f"Preprocessing complete. Final dataset: {len(df)} products")
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with missing values handled
        """
        # Fill missing text fields with empty strings
        text_fields = ["title", "brand", "features", "description", "category"]
        for field in text_fields:
            if field in df.columns:
                df[field] = df[field].fillna("")

        # Fill missing price with 0
        if "price" in df.columns:
            df["price"] = df["price"].apply(self.parse_price)

        # Remove rows with missing images if image column exists
        if "image" in df.columns:
            df = df[df["image"].notna()]

        return df

    def _validate_data(self, df: pd.DataFrame) -> None:
        """
        Validate preprocessed data quality

        Args:
            df: DataFrame to validate
        """
        # Check for empty descriptions
        empty_descriptions = df["combined_description"].str.len() == 0
        if empty_descriptions.any():
            logger.warning(f"Found {empty_descriptions.sum()} products with empty descriptions")

        # Check for missing required fields
        required_fields = ["asin", "combined_description"]
        for field in required_fields:
            if field not in df.columns:
                raise ValueError(f"Required field '{field}' not found in data")

        logger.info("Data validation complete")

    def get_statistics(self) -> Dict:
        """
        Get statistics about the processed dataset

        Returns:
            Dictionary with dataset statistics
        """
        if self.processed_data is None:
            logger.warning("No processed data available")
            return {}

        df = self.processed_data

        stats = {
            "total_products": len(df),
            "unique_brands": df["brand"].nunique() if "brand" in df.columns else 0,
            "avg_description_length": df["combined_description"].str.len().mean(),
            "price_range": {
                "min": df["price"].min() if "price" in df.columns else 0,
                "max": df["price"].max() if "price" in df.columns else 0,
            }
        }

        self.stats = stats
        return stats
