#!/usr/bin/env python3
"""
Database initialization helper script for the GenAI Fashion Store.
This script helps reset and initialize the vector database with proper embeddings.
"""

import os
import sys
import logging
from pathlib import Path
import shutil
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def reset_vector_database():
    """Reset the vector database by clearing existing data."""
    from config import Config

    logger.info("Resetting vector database...")

    # Paths to clear
    paths_to_clear = [
        Config.CHROMA_PERSIST_DIR,
        Config.VECTOR_DB_DIR,
        Config.CACHE_DIR / "embeddings"
    ]

    for path in paths_to_clear:
        if Path(path).exists():
            try:
                if Path(path).is_dir():
                    shutil.rmtree(path)
                    logger.info(f"Cleared: {path}")
                else:
                    Path(path).unlink()
                    logger.info(f"Removed: {path}")
            except Exception as e:
                logger.warning(f"Could not clear {path}: {e}")

        # Recreate directories
        Path(path).mkdir(parents=True, exist_ok=True)

    logger.info("Vector database reset complete")


def initialize_components():
    """Initialize all components with proper configuration."""
    try:
        logger.info("Initializing components...")

        # Import modules
        from image_embedder import CLIPEmbedder
        from vector_database import FashionVectorDB
        from recommendation_engine import RecommendationEngine
        from genai_assistant import FashionShoppingAssistant

        # Initialize embedder first (this sets the embedding dimension)
        logger.info("Initializing CLIP embedder...")
        embedder = CLIPEmbedder()

        # Test embedding dimension
        test_text = "test"
        test_embedding = embedder.get_text_embedding(test_text)
        embedding_dim = len(test_embedding)
        logger.info(f"Embedding dimension: {embedding_dim}")

        # Initialize vector database with embedder
        logger.info("Initializing vector database with embedder...")
        vector_db = FashionVectorDB(embedder=embedder)

        # Initialize other components
        logger.info("Initializing recommendation engine...")
        rec_engine = RecommendationEngine(vector_db, embedder)

        logger.info("Initializing AI assistant...")
        assistant = FashionShoppingAssistant(vector_db, embedder, rec_engine)

        logger.info("✅ All components initialized successfully")

        return {
            'embedder': embedder,
            'vector_db': vector_db,
            'recommendation_engine': rec_engine,
            'assistant': assistant,
            'embedding_dim': embedding_dim
        }

    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        return None


def load_sample_data(components: dict, max_items: int = 100):
    """Load sample data into the database."""
    try:
        logger.info(f"Loading sample data (max {max_items} items)...")

        from data_downloader import FashionDatasetDownloader

        # Download/load dataset
        downloader = FashionDatasetDownloader(max_items=max_items)
        success, products = downloader.setup_dataset()

        if not success or not products:
            logger.error("Failed to load dataset")
            return False

        logger.info(f"Loaded {len(products)} products")

        # Generate embeddings
        logger.info("Generating embeddings...")
        embedder = components['embedder']
        vector_db = components['vector_db']

        image_paths = [p['image_path'] for p in products if 'image_path' in p]

        # Filter out non-existent images
        valid_paths = []
        valid_products = []
        for product, path in zip(products, image_paths):
            if Path(path).exists():
                valid_paths.append(path)
                valid_products.append(product)

        if not valid_paths:
            logger.warning("No valid image paths found")
            return False

        logger.info(f"Processing {len(valid_paths)} valid images...")

        # Generate embeddings in batches
        embeddings = embedder.get_batch_embeddings(
            valid_paths,
            batch_size=32,
            use_cache=True
        )

        # Add to vector database
        logger.info("Adding products to vector database...")
        success = vector_db.add_products(valid_products, embeddings)

        if success:
            # Get statistics
            stats = vector_db.get_statistics()
            logger.info(f"✅ Database initialized with {stats['total_products']} products")
            return True
        else:
            logger.error("Failed to add products to database")
            return False

    except Exception as e:
        logger.error(f"Error loading sample data: {e}")
        return False


def test_search(components: dict):
    """Test search functionality."""
    try:
        logger.info("\nTesting search functionality...")

        vector_db = components['vector_db']
        embedder = components['embedder']

        # Test text search
        test_queries = [
            "blue shirt",
            "red dress",
            "shoes for men",
            "summer clothing"
        ]

        for query in test_queries:
            logger.info(f"\nSearching for: '{query}'")
            results = vector_db.search_by_text(query, n_results=3)

            if results:
                logger.info(f"Found {len(results)} results:")
                for result in results[:3]:
                    logger.info(f"  - {result.get('name', 'Unknown')} ({result.get('baseColour', 'N/A')})")
            else:
                logger.warning("No results found")

        return True

    except Exception as e:
        logger.error(f"Search test failed: {e}")
        return False


def main():
    """Main initialization function."""
    print("="*60)
    print("GenAI Fashion Store - Database Initialization")
    print("="*60)

    # Check if user wants to reset
    if len(sys.argv) > 1 and sys.argv[1] == '--reset':
        response = input("\n⚠️  This will clear all existing data. Continue? (y/n): ")
        if response.lower() == 'y':
            reset_vector_database()
        else:
            print("Initialization cancelled")
            return

    # Initialize components
    components = initialize_components()
    if not components:
        logger.error("Failed to initialize components")
        sys.exit(1)

    # Ask about loading sample data
    response = input("\nDo you want to load sample data? (y/n): ")
    if response.lower() == 'y':
        max_items = input("How many items to load? (default: 100): ").strip()
        max_items = int(max_items) if max_items else 100

        if load_sample_data(components, max_items):
            # Test search
            test_search(components)
        else:
            logger.error("Failed to load sample data")

    print("\n" + "="*60)
    print("✅ Initialization complete!")
    print("="*60)
    print("\nYou can now run the application:")
    print("  streamlit run app.py")
    print("="*60)


if __name__ == "__main__":
    main()
