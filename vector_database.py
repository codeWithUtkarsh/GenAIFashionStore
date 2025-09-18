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
import re
from collections import defaultdict

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
                      filters: Optional[Dict] = None,
                      query_attributes: Optional[Dict] = None) -> List[Dict]:
        """
        Search for similar products based on embedding with optional attribute filtering.

        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            filters: Optional metadata filters
            query_attributes: Optional attributes from query image (colors, category)

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

            # Perform query - get more results for better filtering
            initial_results = min(n_results * 5, 200) if query_attributes else min(n_results * 3, 100)
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=initial_results,  # Get more results for filtering
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

                    # Apply attribute-based adjustments if query attributes provided
                    if query_attributes:
                        # Check color match
                        if 'dominant_colors' in query_attributes and query_attributes['dominant_colors']:
                            query_colors = query_attributes['dominant_colors']
                            product_color = metadata.get('baseColour', '').lower()

                            # Boost if colors match
                            color_match = any(
                                qc.lower() in product_color or product_color in qc.lower()
                                for qc in query_colors if qc
                            )

                            if color_match:
                                similarity_score *= 1.3  # Boost by 30% for color match
                            elif product_color and query_colors:
                                # Penalty for clearly different colors
                                similarity_score *= 0.7  # Reduce by 30% for color mismatch

                        # Check category match
                        if 'predicted_category' in query_attributes and query_attributes['predicted_category']:
                            predicted_cats = query_attributes['predicted_category']
                            product_type = metadata.get('articleType', '').lower()
                            product_name = metadata.get('name', '').lower()

                            # Check if product matches predicted category
                            category_match = False
                            for cat, confidence in predicted_cats.items():
                                if confidence > 0.2:  # Only consider categories with reasonable confidence
                                    cat_lower = cat.lower()
                                    if cat_lower in product_type or cat_lower in product_name:
                                        category_match = True
                                        similarity_score *= (1.0 + confidence * 0.2)  # Boost based on confidence
                                        break

                            if not category_match and list(predicted_cats.values())[0] > 0.5:
                                # Penalty if high confidence category doesn't match
                                similarity_score *= 0.8

                    # Cap similarity score at 1.0
                    similarity_score = min(similarity_score, 1.0)

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

                    # Only include if similarity is above threshold after adjustments
                    if similarity_score > 0.3:  # Minimum threshold
                        products.append(product)

            # Sort by similarity score
            products.sort(key=lambda x: x['similarity_score'], reverse=True)

            # Additional filtering for image search with attributes
            if query_attributes and 'dominant_colors' in query_attributes:
                # Separate products into tiers based on color match
                perfect_matches = []
                good_matches = []
                other_matches = []

                query_colors = [c.lower() for c in query_attributes.get('dominant_colors', []) if c]

                for product in products:
                    product_color = product.get('baseColour', '').lower()

                    if any(qc in product_color or product_color in qc for qc in query_colors):
                        perfect_matches.append(product)
                    elif product.get('similarity_score', 0) > 0.7:
                        good_matches.append(product)
                    else:
                        other_matches.append(product)

                # Combine tiers
                products = perfect_matches + good_matches + other_matches

            return products[:n_results]

        except Exception as e:
            logger.error(f"Error searching similar products: {e}")
            return []

    def _extract_search_terms(self, query: str) -> Dict[str, List[str]]:
        """
        Extract meaningful search terms from query.

        Args:
            query: Search query text

        Returns:
            Dict with categorized search terms
        """
        query_lower = query.lower()

        # Define search patterns
        terms = {
            'colors': [],
            'categories': [],
            'gender': [],
            'article_types': []
        }

        # Color patterns with variations from config
        color_patterns = []
        if hasattr(Config, 'COLOR_VARIATIONS'):
            for base_color, variations in Config.COLOR_VARIATIONS.items():
                for variation in variations:
                    if variation in query_lower:
                        terms['colors'].append(base_color.capitalize())
                        break
        else:
            # Fallback color patterns
            color_patterns = ['black', 'white', 'blue', 'red', 'green', 'yellow', 'pink',
                             'purple', 'orange', 'brown', 'grey', 'gray', 'navy', 'beige',
                             'maroon', 'olive', 'teal', 'silver', 'gold']
            for color in color_patterns:
                if color in query_lower:
                    terms['colors'].append(color.capitalize())

        # Article type patterns using config mappings
        article_patterns = {}

        # Load mappings from config if available
        if hasattr(Config, 'ARTICLE_TYPE_MAPPING'):
            for key, mapping in Config.ARTICLE_TYPE_MAPPING.items():
                article_patterns[key] = mapping.get('keywords', [])
        else:
            # Fallback patterns
            article_patterns = {
                'shoes': ['shoe', 'shoes', 'footwear'],
                'shirts': ['shirt', 'shirts'],
                't-shirts': ['t-shirt', 'tshirt', 't shirt', 'tee'],
                'jeans': ['jean', 'jeans', 'denim'],
                'dress': ['dress', 'dresses'],
                'sandals': ['sandal', 'sandals'],
                'flip flops': ['flip flop', 'flipflop', 'flip-flop'],
                'heels': ['heel', 'heels'],
                'boots': ['boot', 'boots'],
                'sneakers': ['sneaker', 'sneakers'],
                'jacket': ['jacket', 'jackets'],
                'coat': ['coat', 'coats'],
                'sweater': ['sweater', 'sweaters'],
                'shorts': ['short', 'shorts'],
                'trousers': ['trouser', 'trousers', 'pants']
            }

        for article_type, patterns in article_patterns.items():
            for pattern in patterns:
                if pattern in query_lower:
                    terms['article_types'].append(article_type.title())
                    break

        # Gender patterns
        if 'men' in query_lower or "men's" in query_lower:
            terms['gender'].append('Men')
        if 'women' in query_lower or "women's" in query_lower:
            terms['gender'].append('Women')
        if 'boys' in query_lower or "boy's" in query_lower:
            terms['gender'].append('Boys')
        if 'girls' in query_lower or "girl's" in query_lower:
            terms['gender'].append('Girls')

        return terms

    def _calculate_text_match_score(self, product: Dict, search_terms: Dict) -> float:
        """
        Calculate text matching score for a product.

        Args:
            product: Product metadata
            search_terms: Extracted search terms

        Returns:
            float: Matching score between 0 and 1
        """
        score = 0.0
        max_score = 0.0

        # Color matching with variations (high weight)
        if search_terms['colors']:
            max_score += 0.3
            product_color = product.get('baseColour', '').lower()

            # Check for color match including variations
            matched = False
            for color in search_terms['colors']:
                if hasattr(Config, 'COLOR_VARIATIONS'):
                    # Check all variations of the color
                    variations = Config.COLOR_VARIATIONS.get(color.lower(), [color.lower()])
                    for variation in variations:
                        if variation in product_color:
                            score += 0.3
                            matched = True
                            break
                elif color.lower() in product_color:
                    score += 0.3
                    matched = True

                if matched:
                    break

        # Article type matching with exclusion rules (highest weight)
        if search_terms['article_types']:
            max_score += 0.4
            product_type = product.get('articleType', '')
            product_name = product.get('name', '').lower()

            matched = False
            for article in search_terms['article_types']:
                if hasattr(Config, 'ARTICLE_TYPE_MAPPING'):
                    mapping = Config.ARTICLE_TYPE_MAPPING.get(article.lower(), {})

                    # Check if product matches exact types
                    exact_matches = mapping.get('exact_matches', [])
                    exclude_types = mapping.get('exclude', [])

                    # Check for exclusions first
                    is_excluded = any(
                        excl.lower() in product_type.lower()
                        for excl in exclude_types
                    )

                    if not is_excluded:
                        # Check for exact matches
                        is_exact_match = any(
                            exact.lower() in product_type.lower()
                            for exact in exact_matches
                        )

                        if is_exact_match:
                            score += 0.4
                            matched = True
                            break

                        # Check in product name
                        keywords = mapping.get('keywords', [])
                        for keyword in keywords:
                            if keyword in product_name:
                                score += 0.35  # Slightly lower score for name matches
                                matched = True
                                break
                else:
                    # Fallback matching
                    if article.lower() in product_type.lower() or article.lower() in product_name:
                        score += 0.4
                        matched = True
                        break

                if matched:
                    break

        # Gender matching (medium weight)
        if search_terms['gender']:
            max_score += 0.2
            product_gender = product.get('gender', '')
            if product_gender in search_terms['gender']:
                score += 0.2

        # Partial name/description matching
        max_score += 0.1
        product_name = product.get('name', '').lower()
        product_desc = product.get('description', '').lower()
        query_words = set(search_terms.get('query_words', []))

        matching_words = sum(1 for word in query_words if word in product_name or word in product_desc)
        if query_words:
            score += 0.1 * (matching_words / len(query_words))

        # Normalize score
        if max_score > 0:
            return score / max_score
        return 0.0

    def search_by_text(self,
                      text_query: str,
                      n_results: int = 10,
                      filters: Optional[Dict] = None) -> List[Dict]:
        """
        Enhanced hybrid search combining text matching and semantic search.

        Args:
            text_query: Text search query
            n_results: Number of results to return
            filters: Optional metadata filters

        Returns:
            List[Dict]: List of matching products with combined scores
        """
        try:
            # Extract search terms
            search_terms = self._extract_search_terms(text_query)
            search_terms['query_words'] = text_query.lower().split()

            # Build enhanced filters based on extracted terms
            enhanced_filters = filters.copy() if filters else {}

            # Add extracted filters
            if search_terms['gender'] and 'gender' not in enhanced_filters:
                if len(search_terms['gender']) == 1:
                    enhanced_filters['gender'] = search_terms['gender'][0]

            # Build where clause
            where_clause = None
            if enhanced_filters:
                where_clause = {}
                for key, value in enhanced_filters.items():
                    if isinstance(value, list):
                        where_clause[key] = {"$in": value}
                    else:
                        where_clause[key] = value

            # Get more results for better filtering
            search_limit = min(n_results * 5, 200)

            # Generate text embedding using CLIP if embedder is available
            if self.embedder:
                # Use CLIP to generate text embedding
                text_embedding = self.embedder.get_text_embedding(text_query)
                if isinstance(text_embedding, np.ndarray):
                    text_embedding = text_embedding.tolist()

                # Query with larger limit for filtering
                results = self.collection.query(
                    query_embeddings=[text_embedding],
                    n_results=search_limit,
                    where=where_clause if where_clause else None,
                    include=['metadatas', 'documents', 'distances']
                )
            else:
                # Fallback without embedder
                results = {'ids': [[]], 'metadatas': [[]], 'documents': [[]], 'distances': [[]]}

            # Process and score results
            products = []
            if results and results['ids'] and results['ids'][0]:
                for idx, product_id in enumerate(results['ids'][0]):
                    metadata = results['metadatas'][0][idx]
                    distance = results['distances'][0][idx]

                    # Extract the actual product ID
                    actual_id = product_id.replace('product_', '') if product_id.startswith('product_') else product_id

                    # Calculate semantic similarity score
                    semantic_score = float(1.0 - distance)

                    # Calculate text matching score
                    text_match_score = self._calculate_text_match_score(metadata, search_terms)

                    # Combine scores (weighted average)
                    # Give more weight to text matching for specific queries
                    if search_terms['article_types'] or search_terms['colors']:
                        # Specific query - prioritize exact matches
                        combined_score = (text_match_score * 0.7) + (semantic_score * 0.3)
                    else:
                        # General query - balance both
                        combined_score = (text_match_score * 0.4) + (semantic_score * 0.6)

                    product = {
                        'product_id': product_id,
                        'id': actual_id,
                        'relevance_score': combined_score,
                        'semantic_score': semantic_score,
                        'text_match_score': text_match_score,
                        **metadata
                    }

                    if results['documents'][0][idx]:
                        product['description'] = results['documents'][0][idx]

                    # Only include products with reasonable scores
                    if combined_score > 0.2:  # Threshold for relevance
                        products.append(product)

            # Sort by combined relevance score
            products.sort(key=lambda x: x['relevance_score'], reverse=True)

            # Apply additional filtering for specific queries
            if search_terms['article_types'] or search_terms['colors']:
                # Filter out products that don't match key criteria
                filtered_products = []
                for product in products:
                    include = True

                    # Check article type with exclusion rules
                    if search_terms['article_types']:
                        product_type = product.get('articleType', '')
                        product_name = product.get('name', '').lower()

                        has_match = False
                        is_excluded = False

                        for article in search_terms['article_types']:
                            if hasattr(Config, 'ARTICLE_TYPE_MAPPING'):
                                mapping = Config.ARTICLE_TYPE_MAPPING.get(article.lower(), {})

                                # Check exclusion list
                                exclude_types = mapping.get('exclude', [])
                                if any(excl.lower() in product_type.lower() for excl in exclude_types):
                                    is_excluded = True
                                    break

                                # Check exact matches
                                exact_matches = mapping.get('exact_matches', [])
                                if any(exact.lower() in product_type.lower() for exact in exact_matches):
                                    has_match = True
                                    break

                                # Check keywords in name
                                keywords = mapping.get('keywords', [])
                                if any(keyword in product_name for keyword in keywords):
                                    has_match = True
                                    break
                            else:
                                # Fallback
                                if article.lower() in product_type.lower() or article.lower() in product_name:
                                    has_match = True
                                    break

                        if is_excluded or (not has_match and product.get('semantic_score', 0) < 0.8):
                            include = False

                    # Check color if specified and product matches article type
                    if include and search_terms['colors']:
                        product_color = product.get('baseColour', '').lower()
                        has_color_match = any(
                            color.lower() in product_color
                            for color in search_terms['colors']
                        )
                        if not has_color_match and product['text_match_score'] < 0.5:
                            include = False

                    if include:
                        filtered_products.append(product)

                products = filtered_products

            return products[:n_results]

        except Exception as e:
            logger.error(f"Error in hybrid text search: {e}")
            # Fallback to basic search
            return self._basic_text_search(text_query, n_results, filters)

    def _basic_text_search(self, text_query: str, n_results: int, filters: Optional[Dict]) -> List[Dict]:
        """Fallback basic text search."""
        try:
            if self.embedder:
                text_embedding = self.embedder.get_text_embedding(text_query)
                if isinstance(text_embedding, np.ndarray):
                    text_embedding = text_embedding.tolist()

                results = self.collection.query(
                    query_embeddings=[text_embedding],
                    n_results=n_results,
                    where=filters,
                    include=['metadatas', 'documents', 'distances']
                )

                products = []
                if results and results['ids'] and results['ids'][0]:
                    for idx, product_id in enumerate(results['ids'][0]):
                        metadata = results['metadatas'][0][idx]
                        distance = results['distances'][0][idx]

                        actual_id = product_id.replace('product_', '') if product_id.startswith('product_') else product_id

                        product = {
                            'product_id': product_id,
                            'id': actual_id,
                            'relevance_score': float(1.0 - distance),
                            **metadata
                        }
                        products.append(product)

                return products
            return []
        except Exception as e:
            logger.error(f"Error in basic text search: {e}")
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
