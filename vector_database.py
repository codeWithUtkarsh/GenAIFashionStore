"""
Vector database module using ChromaDB for efficient similarity search and retrieval.
"""

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import logging
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np

from config import Config

# Setup logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format=Config.LOG_FORMAT
)
logger = logging.getLogger(__name__)


class FashionVectorDB:
    """Vector database for fashion product embeddings and metadata."""

    def __init__(self, persist_directory: str = None, collection_name: str = None, embedder=None):
        """
        Initialize the vector database.

        Args:
            persist_directory: Directory to persist the database
            collection_name: Name of the collection
            embedder: Optional embedder for text search
        """
        self.persist_directory = persist_directory or Config.CHROMA_PERSIST_DIR
        self.collection_name = collection_name or Config.CHROMA_COLLECTION_NAME
        self.embedder = embedder

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Get or create collection
        self._initialize_collection()

        logger.info(f"Vector database initialized at {self.persist_directory}")

    def _initialize_collection(self):
        """Initialize or load the collection."""
        try:
            # Try to get existing collection
            self.collection = self.client.get_collection(
                name=self.collection_name
            )
            logger.info(f"Loaded existing collection: {self.collection_name}")
        except Exception:
            # Create new collection without default embedding function
            # We'll provide embeddings directly
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=None  # Don't use default embedding
            )
            logger.info(f"Created new collection: {self.collection_name}")

    def add_products(self,
                    products: List[Dict],
                    embeddings: Dict[str, np.ndarray],
                    batch_size: int = 100) -> bool:
        """
        Add products with their embeddings to the database.

        Args:
            products: List of product dictionaries
            embeddings: Dictionary mapping image paths to embeddings
            batch_size: Batch size for insertion

        Returns:
            bool: Success status
        """
        try:
            # Prepare data for insertion
            ids = []
            embeddings_list = []
            metadatas = []
            documents = []

            for product in products:
                if 'image_path' not in product:
                    continue

                image_path = product['image_path']
                if image_path not in embeddings:
                    logger.warning(f"No embedding found for {image_path}")
                    continue

                # Prepare product data
                product_id = f"product_{product['id']}"
                ids.append(product_id)

                # Add embedding
                embedding = embeddings[image_path]
                if isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()
                embeddings_list.append(embedding)

                # Prepare metadata (ChromaDB requires serializable types)
                metadata = {
                    'id': str(product['id']),
                    'name': product.get('name', 'Unknown'),
                    'gender': product.get('gender', 'Unisex'),
                    'masterCategory': product.get('masterCategory', 'Unknown'),
                    'subCategory': product.get('subCategory', 'Unknown'),
                    'articleType': product.get('articleType', 'Unknown'),
                    'baseColour': product.get('baseColour', 'Unknown'),
                    'season': product.get('season', 'All Season'),
                    'year': str(product.get('year', 2020)),
                    'usage': product.get('usage', 'Casual'),
                    'image_path': image_path
                }
                metadatas.append(metadata)

                # Add description as document
                documents.append(product.get('description', ''))

            # Insert in batches
            total_items = len(ids)
            logger.info(f"Adding {total_items} products to vector database...")

            for i in tqdm(range(0, total_items, batch_size), desc="Inserting products"):
                batch_end = min(i + batch_size, total_items)

                self.collection.add(
                    ids=ids[i:batch_end],
                    embeddings=embeddings_list[i:batch_end],
                    metadatas=metadatas[i:batch_end],
                    documents=documents[i:batch_end]
                )

            logger.info(f"Successfully added {total_items} products to database")
            return True

        except Exception as e:
            logger.error(f"Error adding products to database: {e}")
            return False

    def search_similar(self,
                      query_embedding: np.ndarray,
                      n_results: int = 10,
                      filters: Optional[Dict] = None) -> List[Dict]:
        """
        Search for similar products based on embedding.

        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            filters: Optional metadata filters

        Returns:
            List[Dict]: List of similar products with scores
        """
        try:
            # Convert numpy array to list if necessary
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()

            # Build where clause for filters
            where_clause = None
            if filters:
                where_clause = {}
                for key, value in filters.items():
                    if isinstance(value, list):
                        where_clause[key] = {"$in": value}
                    else:
                        where_clause[key] = value

            # Perform query
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_clause if where_clause else None,
                include=['metadatas', 'documents', 'distances']
            )

            # Process results
            products = []
            if results and results['ids'] and results['ids'][0]:
                for idx, product_id in enumerate(results['ids'][0]):
                    metadata = results['metadatas'][0][idx]
                    distance = results['distances'][0][idx]

                    # Convert distance to similarity score (1 - distance for cosine)
                    similarity_score = 1.0 - distance

                    # Extract the actual product ID (remove "product_" prefix if present)
                    actual_id = product_id.replace('product_', '') if product_id.startswith('product_') else product_id

                    product = {
                        'product_id': product_id,
                        'id': actual_id,  # Add clean ID for consistency
                        'similarity_score': float(similarity_score),
                        **metadata
                    }

                    if results['documents'][0][idx]:
                        product['description'] = results['documents'][0][idx]

                    products.append(product)

            return products

        except Exception as e:
            logger.error(f"Error searching similar products: {e}")
            return []

    def search_by_text(self,
                      text_query: str,
                      n_results: int = 10,
                      filters: Optional[Dict] = None) -> List[Dict]:
        """
        Search products by text description.

        Args:
            text_query: Text search query
            n_results: Number of results to return
            filters: Optional metadata filters

        Returns:
            List[Dict]: List of matching products
        """
        try:
            # Build where clause
            where_clause = None
            if filters:
                where_clause = {}
                for key, value in filters.items():
                    if isinstance(value, list):
                        where_clause[key] = {"$in": value}
                    else:
                        where_clause[key] = value

            # Generate text embedding using CLIP if embedder is available
            if self.embedder:
                # Use CLIP to generate text embedding
                text_embedding = self.embedder.get_text_embedding(text_query)
                if isinstance(text_embedding, np.ndarray):
                    text_embedding = text_embedding.tolist()

                # Use query with embedding instead of text
                results = self.collection.query(
                    query_embeddings=[text_embedding],
                    n_results=n_results,
                    where=where_clause if where_clause else None,
                    include=['metadatas', 'documents', 'distances']
                )
            else:
                # Fallback: search in documents if no embedder
                # This won't work well but prevents errors
                all_results = self.collection.get(
                    where=where_clause,
                    limit=n_results * 10,
                    include=['metadatas', 'documents']
                )

                # Simple text matching in documents
                results = {'ids': [[]], 'metadatas': [[]], 'documents': [[]], 'distances': [[]]}
                if all_results and all_results['ids']:
                    query_lower = text_query.lower()
                    for idx, doc in enumerate(all_results.get('documents', [])):
                        if doc and query_lower in doc.lower():
                            results['ids'][0].append(all_results['ids'][idx])
                            results['metadatas'][0].append(all_results['metadatas'][idx])
                            results['documents'][0].append(doc)
                            results['distances'][0].append(0.5)  # Default distance

                            if len(results['ids'][0]) >= n_results:
                                break

            # Process results
            products = []
            if results and results['ids'] and results['ids'][0]:
                for idx, product_id in enumerate(results['ids'][0]):
                    metadata = results['metadatas'][0][idx]
                    distance = results['distances'][0][idx]

                    # Extract the actual product ID (remove "product_" prefix if present)
                    actual_id = product_id.replace('product_', '') if product_id.startswith('product_') else product_id

                    product = {
                        'product_id': product_id,
                        'id': actual_id,  # Add clean ID for consistency
                        'relevance_score': float(1.0 - distance),
                        **metadata
                    }

                    if results['documents'][0][idx]:
                        product['description'] = results['documents'][0][idx]

                    products.append(product)

            return products

        except Exception as e:
            logger.error(f"Error searching by text: {e}")
            return []

    def get_product_by_id(self, product_id: str) -> Optional[Dict]:
        """
        Get a specific product by its ID.

        Args:
            product_id: Product ID

        Returns:
            Optional[Dict]: Product data if found
        """
        try:
            results = self.collection.get(
                ids=[product_id],
                include=['metadatas', 'documents', 'embeddings']
            )

            if results and results['ids']:
                product = results['metadatas'][0] if results['metadatas'] else {}
                if results['documents'] and results['documents'][0]:
                    product['description'] = results['documents'][0]
                return product

            return None

        except Exception as e:
            logger.error(f"Error getting product by ID: {e}")
            return None

    def get_products_by_category(self,
                                 category: str,
                                 category_type: str = 'masterCategory',
                                 limit: int = 50) -> List[Dict]:
        """
        Get products by category.

        Args:
            category: Category value
            category_type: Type of category (masterCategory, subCategory, etc.)
            limit: Maximum number of products to return

        Returns:
            List[Dict]: List of products in the category
        """
        try:
            where_clause = {category_type: category}

            results = self.collection.get(
                where=where_clause,
                limit=limit,
                include=['metadatas', 'documents']
            )

            products = []
            if results and results['ids']:
                for idx, product_id in enumerate(results['ids']):
                    # Extract the actual product ID (remove "product_" prefix if present)
                    actual_id = product_id.replace('product_', '') if product_id.startswith('product_') else product_id

                    product = {
                        'product_id': product_id,
                        'id': actual_id,  # Add clean ID for consistency
                        **results['metadatas'][idx]
                    }
                    if results['documents'] and results['documents'][idx]:
                        product['description'] = results['documents'][idx]
                    products.append(product)

            return products

        except Exception as e:
            logger.error(f"Error getting products by category: {e}")
            return []

    def get_unique_values(self, field: str) -> List[str]:
        """
        Get unique values for a metadata field.

        Args:
            field: Metadata field name

        Returns:
            List[str]: List of unique values
        """
        try:
            # Get a sample of products
            results = self.collection.get(
                limit=1000,
                include=['metadatas']
            )

            # Extract unique values
            unique_values = set()
            if results and results['metadatas']:
                for metadata in results['metadatas']:
                    if field in metadata:
                        value = metadata[field]
                        if isinstance(value, list):
                            unique_values.update(value)
                        else:
                            unique_values.add(value)

            return sorted(list(unique_values))

        except Exception as e:
            logger.error(f"Error getting unique values: {e}")
            return []

    def update_product(self, product_id: str, metadata: Dict) -> bool:
        """
        Update product metadata.

        Args:
            product_id: Product ID
            metadata: New metadata

        Returns:
            bool: Success status
        """
        try:
            self.collection.update(
                ids=[product_id],
                metadatas=[metadata]
            )
            return True
        except Exception as e:
            logger.error(f"Error updating product: {e}")
            return False

    def delete_product(self, product_id: str) -> bool:
        """
        Delete a product from the database.

        Args:
            product_id: Product ID

        Returns:
            bool: Success status
        """
        try:
            self.collection.delete(ids=[product_id])
            return True
        except Exception as e:
            logger.error(f"Error deleting product: {e}")
            return False

    def get_statistics(self) -> Dict:
        """
        Get database statistics.

        Returns:
            Dict: Database statistics
        """
        try:
            count = self.collection.count()

            # Get sample for category distribution
            results = self.collection.get(
                limit=1000,
                include=['metadatas']
            )

            stats = {
                'total_products': count,
                'categories': {}
            }

            if results and results['metadatas']:
                # Count category distributions
                for field in ['gender', 'masterCategory', 'subCategory', 'baseColour']:
                    field_values = {}
                    for metadata in results['metadatas']:
                        if field in metadata:
                            value = metadata[field]
                            field_values[value] = field_values.get(value, 0) + 1
                    stats['categories'][field] = field_values

            return stats

        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {'total_products': 0, 'categories': {}}

    def reset_database(self) -> bool:
        """
        Reset the entire database.

        Returns:
            bool: Success status
        """
        try:
            self.client.delete_collection(self.collection_name)
            self._initialize_collection()
            logger.info("Database reset successfully")
            return True
        except Exception as e:
            logger.error(f"Error resetting database: {e}")
            return False


def test_vector_db():
    """Test the vector database functionality."""
    from image_embedder import CLIPEmbedder

    # Initialize embedder and database
    embedder = CLIPEmbedder()
    db = FashionVectorDB(embedder=embedder)

    # Get statistics
    stats = db.get_statistics()
    print(f"Database statistics: {json.dumps(stats, indent=2)}")

    # Test product insertion
    test_products = [
        {
            'id': '1001',
            'name': 'Blue T-Shirt',
            'gender': 'Men',
            'masterCategory': 'Apparel',
            'subCategory': 'Topwear',
            'articleType': 'T-Shirts',
            'baseColour': 'Blue',
            'season': 'Summer',
            'year': 2023,
            'usage': 'Casual',
            'image_path': '/path/to/image1.jpg',
            'description': 'A comfortable blue t-shirt for casual wear'
        }
    ]

    # Create dummy embeddings
    test_embeddings = {
        '/path/to/image1.jpg': np.random.randn(512)
    }

    # Add products
    success = db.add_products(test_products, test_embeddings)
    print(f"Products added: {success}")

    # Search similar
    query_embedding = np.random.randn(512)
    similar_products = db.search_similar(query_embedding, n_results=5)
    print(f"Found {len(similar_products)} similar products")


if __name__ == "__main__":
    test_vector_db()
