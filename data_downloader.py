"""
Dataset downloader module for fetching and preparing the Fashion Product Images dataset from Kaggle.
"""

import os
import zipfile
import shutil
import json
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import logging
from tqdm import tqdm

from config import Config

# Setup logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format=Config.LOG_FORMAT
)
logger = logging.getLogger(__name__)


class FashionDatasetDownloader:
    """Handles downloading and preparing the Fashion Product Images dataset."""

    def __init__(self, max_items: Optional[int] = None):
        """
        Initialize the dataset downloader.

        Args:
            max_items: Maximum number of items to process (for development)
        """
        self.config = Config
        self.max_items = max_items or self.config.MAX_PRODUCTS_TO_LOAD
        self.data_dir = self.config.DATA_DIR
        self.images_dir = self.config.IMAGES_DIR

        # Dataset paths
        self.styles_csv = self.data_dir / "styles.csv"
        self.images_folder = self.data_dir / "images"
        self.processed_data = self.data_dir / "processed_data.json"

    def check_kaggle_credentials(self) -> bool:
        """Check if Kaggle credentials are properly configured."""
        if not self.config.setup_kaggle_credentials():
            logger.error("Kaggle credentials not found. Please set KAGGLE_USERNAME and KAGGLE_KEY in .env file")
            logger.info("You can get your Kaggle API key from https://www.kaggle.com/account")
            return False
        return True

    def download_dataset(self) -> bool:
        """
        Download the Fashion Product Images dataset from Kaggle.

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.check_kaggle_credentials():
            return False

        try:
            import kaggle

            logger.info(f"Downloading dataset: {self.config.DATASET_NAME}")

            # Download dataset
            kaggle.api.dataset_download_files(
                self.config.DATASET_NAME,
                path=str(self.data_dir),
                unzip=True
            )

            logger.info("Dataset downloaded successfully")
            return True

        except ImportError:
            logger.error("Kaggle package not installed. Run: pip install kaggle")
            return False
        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            return False

    def prepare_sample_data(self) -> pd.DataFrame:
        """
        Prepare a sample of the dataset for faster development.

        Returns:
            pd.DataFrame: Sample of product data
        """
        try:
            # Check if styles.csv exists
            if not self.styles_csv.exists():
                logger.warning(f"styles.csv not found at {self.styles_csv}")
                return pd.DataFrame()

            # Load styles data
            logger.info(f"Loading product metadata from {self.styles_csv}")
            df = pd.read_csv(self.styles_csv, on_bad_lines='skip')

            # Clean the dataframe
            df = self.clean_dataframe(df)

            # Sample data if needed
            if self.max_items and len(df) > self.max_items:
                logger.info(f"Sampling {self.max_items} items from {len(df)} total products")
                df = df.sample(n=self.max_items, random_state=42)

            # Verify image files exist
            df = self.verify_images(df)

            logger.info(f"Prepared {len(df)} products for processing")
            return df

        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return pd.DataFrame()

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the product dataframe.

        Args:
            df: Raw dataframe from styles.csv

        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        # Remove rows with missing critical information
        required_cols = ['id', 'productDisplayName']
        df = df.dropna(subset=required_cols)

        # Fill missing values
        fill_values = {
            'gender': 'Unisex',
            'masterCategory': 'Unknown',
            'subCategory': 'Unknown',
            'articleType': 'Unknown',
            'baseColour': 'Unknown',
            'season': 'All Season',
            'year': 2020,
            'usage': 'Casual'
        }

        for col, default_val in fill_values.items():
            if col in df.columns:
                df[col] = df[col].fillna(default_val)

        # Convert id to string for consistency
        df['id'] = df['id'].astype(str)

        # Create image path column
        df['image_path'] = df['id'].apply(lambda x: f"{x}.jpg")

        return df

    def verify_images(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Verify that image files exist for products.

        Args:
            df: Product dataframe

        Returns:
            pd.DataFrame: Dataframe with only products that have images
        """
        valid_rows = []

        logger.info("Verifying image files...")
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Checking images"):
            image_path = self.images_folder / row['image_path']
            if image_path.exists():
                row['full_image_path'] = str(image_path)
                valid_rows.append(row)

        valid_df = pd.DataFrame(valid_rows).reset_index(drop=True)
        logger.info(f"Found {len(valid_df)} products with valid images")

        return valid_df

    def create_product_metadata(self, df: pd.DataFrame) -> List[Dict]:
        """
        Create structured metadata for each product.

        Args:
            df: Cleaned product dataframe

        Returns:
            List[Dict]: List of product metadata dictionaries
        """
        products = []

        for _, row in df.iterrows():
            product = {
                'id': row['id'],
                'name': row.get('productDisplayName', 'Unknown Product'),
                'gender': row.get('gender', 'Unisex'),
                'masterCategory': row.get('masterCategory', 'Unknown'),
                'subCategory': row.get('subCategory', 'Unknown'),
                'articleType': row.get('articleType', 'Unknown'),
                'baseColour': row.get('baseColour', 'Unknown'),
                'season': row.get('season', 'All Season'),
                'year': int(row.get('year', 2020)),
                'usage': row.get('usage', 'Casual'),
                'image_path': row['full_image_path'],
                'description': self.generate_product_description(row)
            }
            products.append(product)

        return products

    def generate_product_description(self, row: pd.Series) -> str:
        """
        Generate a natural language description for a product.

        Args:
            row: Product data row

        Returns:
            str: Product description
        """
        parts = []

        # Add basic description
        parts.append(row.get('productDisplayName', 'Fashion item'))

        # Add attributes
        if pd.notna(row.get('baseColour')) and row['baseColour'] != 'Unknown':
            parts.append(f"in {row['baseColour']}")

        if pd.notna(row.get('gender')) and row['gender'] != 'Unisex':
            parts.append(f"for {row['gender']}")

        if pd.notna(row.get('usage')) and row['usage'] != 'Casual':
            parts.append(f"perfect for {row['usage']} wear")

        if pd.notna(row.get('season')) and row['season'] != 'All Season':
            parts.append(f"ideal for {row['season']}")

        return " ".join(parts)

    def save_processed_data(self, products: List[Dict]) -> bool:
        """
        Save processed product data to JSON file.

        Args:
            products: List of product dictionaries

        Returns:
            bool: True if successful
        """
        try:
            with open(self.processed_data, 'w') as f:
                json.dump(products, f, indent=2)
            logger.info(f"Saved processed data to {self.processed_data}")
            return True
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            return False

    def load_processed_data(self) -> List[Dict]:
        """
        Load previously processed product data.

        Returns:
            List[Dict]: List of product dictionaries
        """
        if self.processed_data.exists():
            try:
                with open(self.processed_data, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading processed data: {e}")
        return []

    def setup_dataset(self, force_download: bool = False) -> Tuple[bool, List[Dict]]:
        """
        Main method to setup the dataset.

        Args:
            force_download: Force re-download even if data exists

        Returns:
            Tuple[bool, List[Dict]]: Success status and product data
        """
        # Check if processed data already exists
        if not force_download and self.processed_data.exists():
            logger.info("Loading existing processed data...")
            products = self.load_processed_data()
            if products:
                return True, products

        # Check if raw data exists
        if not self.styles_csv.exists() or force_download:
            logger.info("Dataset not found locally. Downloading from Kaggle...")
            if not self.download_dataset():
                logger.error("Failed to download dataset")
                return False, []

        # Prepare and process data
        df = self.prepare_sample_data()
        if df.empty:
            logger.error("No data to process")
            return False, []

        # Create product metadata
        products = self.create_product_metadata(df)

        # Save processed data
        self.save_processed_data(products)

        return True, products


def main():
    """Main function for testing the data downloader."""
    downloader = FashionDatasetDownloader(max_items=100)
    success, products = downloader.setup_dataset()

    if success:
        print(f"Successfully loaded {len(products)} products")
        if products:
            print(f"Sample product: {json.dumps(products[0], indent=2)}")
    else:
        print("Failed to setup dataset")


if __name__ == "__main__":
    main()
