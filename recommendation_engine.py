"""
Recommendation engine module with multiple strategies for fashion product recommendations.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from collections import defaultdict
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from config import Config
from vector_database import FashionVectorDB
from image_embedder import CLIPEmbedder

# Setup logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format=Config.LOG_FORMAT
)
logger = logging.getLogger(__name__)


class RecommendationEngine:
    """Multi-strategy recommendation engine for fashion products."""

    def __init__(self,
                 vector_db: FashionVectorDB,
                 embedder: CLIPEmbedder = None):
        """
        Initialize the recommendation engine.

        Args:
            vector_db: Vector database instance
            embedder: Image embedder instance
        """
        self.vector_db = vector_db
        self.embedder = embedder or CLIPEmbedder()

        # Cache for user interactions (for collaborative filtering simulation)
        self.user_interactions = defaultdict(list)
        self.product_views = defaultdict(int)

        logger.info("Recommendation engine initialized")

    def get_recommendations(self,
                           product_id: str = None,
                           query_embedding: np.ndarray = None,
                           user_id: str = None,
                           method: str = "hybrid",
                           n_recommendations: int = 10,
                           filters: Dict = None) -> List[Dict]:
        """
        Get product recommendations using specified method.

        Args:
            product_id: Source product ID for recommendations
            query_embedding: Query embedding for similarity search
            user_id: User ID for personalized recommendations
            method: Recommendation method (visual_similarity, category_based, collaborative, hybrid)
            n_recommendations: Number of recommendations to return
            filters: Optional filters for recommendations

        Returns:
            List[Dict]: List of recommended products
        """
        method = method.lower()

        if method == "visual_similarity":
            return self._visual_similarity_recommendations(
                product_id=product_id,
                query_embedding=query_embedding,
                n_recommendations=n_recommendations,
                filters=filters
            )
        elif method == "category_based":
            return self._category_based_recommendations(
                product_id=product_id,
                n_recommendations=n_recommendations,
                filters=filters
            )
        elif method == "collaborative":
            return self._collaborative_recommendations(
                user_id=user_id,
                product_id=product_id,
                n_recommendations=n_recommendations,
                filters=filters
            )
        elif method == "hybrid":
            return self._hybrid_recommendations(
                product_id=product_id,
                query_embedding=query_embedding,
                user_id=user_id,
                n_recommendations=n_recommendations,
                filters=filters
            )
        else:
            logger.warning(f"Unknown recommendation method: {method}")
            return []

    def _visual_similarity_recommendations(self,
                                          product_id: str = None,
                                          query_embedding: np.ndarray = None,
                                          n_recommendations: int = 10,
                                          filters: Dict = None) -> List[Dict]:
        """
        Get recommendations based on visual similarity.

        Args:
            product_id: Product ID to find similar items
            query_embedding: Direct embedding for similarity search
            n_recommendations: Number of recommendations
            filters: Optional filters

        Returns:
            List[Dict]: Similar products
        """
        try:
            # Get query embedding
            if query_embedding is None and product_id:
                # Get product embedding from database
                product = self.vector_db.get_product_by_id(f"product_{product_id}")
                if not product or 'image_path' not in product:
                    logger.warning(f"Product {product_id} not found")
                    return []

                # Generate embedding for the product image
                query_embedding = self.embedder.get_image_embedding(product['image_path'])

            if query_embedding is None:
                logger.error("No query embedding provided")
                return []

            # Search similar products
            similar_products = self.vector_db.search_similar(
                query_embedding=query_embedding,
                n_results=n_recommendations + 1,  # +1 to exclude source product
                filters=filters
            )

            # Filter out the source product if present
            if product_id:
                similar_products = [
                    p for p in similar_products
                    if p.get('id') != product_id
                ]

            return similar_products[:n_recommendations]

        except Exception as e:
            logger.error(f"Error in visual similarity recommendations: {e}")
            return []

    def _category_based_recommendations(self,
                                       product_id: str,
                                       n_recommendations: int = 10,
                                       filters: Dict = None) -> List[Dict]:
        """
        Get recommendations based on product categories and attributes.

        Args:
            product_id: Source product ID
            n_recommendations: Number of recommendations
            filters: Optional filters

        Returns:
            List[Dict]: Recommended products from similar categories
        """
        try:
            # Get source product
            source_product = self.vector_db.get_product_by_id(f"product_{product_id}")
            if not source_product:
                logger.warning(f"Product {product_id} not found")
                return []

            # Build category-based filters
            category_filters = filters or {}

            # Priority 1: Same subcategory and gender
            priority_filters = [
                {
                    'subCategory': source_product.get('subCategory'),
                    'gender': source_product.get('gender'),
                    **category_filters
                },
                # Priority 2: Same master category and gender
                {
                    'masterCategory': source_product.get('masterCategory'),
                    'gender': source_product.get('gender'),
                    **category_filters
                },
                # Priority 3: Same article type
                {
                    'articleType': source_product.get('articleType'),
                    **category_filters
                },
                # Priority 4: Same usage
                {
                    'usage': source_product.get('usage'),
                    **category_filters
                }
            ]

            recommendations = []
            seen_ids = {product_id}

            # Try each filter priority
            for filter_set in priority_filters:
                if len(recommendations) >= n_recommendations:
                    break

                # Clean filter set (remove None values)
                clean_filters = {k: v for k, v in filter_set.items() if v is not None}

                # Get products with these filters
                products = self.vector_db.get_products_by_category(
                    category=clean_filters.get('subCategory', clean_filters.get('masterCategory', '')),
                    category_type='subCategory' if 'subCategory' in clean_filters else 'masterCategory',
                    limit=n_recommendations * 2
                )

                # Add unique products
                for product in products:
                    if product.get('id') not in seen_ids:
                        recommendations.append(product)
                        seen_ids.add(product.get('id'))

                        if len(recommendations) >= n_recommendations:
                            break

            # Add relevance scores based on attribute similarity
            for rec in recommendations:
                score = self._calculate_attribute_similarity(source_product, rec)
                rec['relevance_score'] = score

            # Sort by relevance score
            recommendations.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)

            return recommendations[:n_recommendations]

        except Exception as e:
            logger.error(f"Error in category-based recommendations: {e}")
            return []

    def _collaborative_recommendations(self,
                                      user_id: str = None,
                                      product_id: str = None,
                                      n_recommendations: int = 10,
                                      filters: Dict = None) -> List[Dict]:
        """
        Get recommendations based on collaborative filtering (simulated).

        Args:
            user_id: User ID for personalized recommendations
            product_id: Product currently being viewed
            n_recommendations: Number of recommendations
            filters: Optional filters

        Returns:
            List[Dict]: Recommended products based on user behavior
        """
        try:
            # Record interaction if provided
            if user_id and product_id:
                self.user_interactions[user_id].append(product_id)
                self.product_views[product_id] += 1

            recommendations = []

            # Get popular products (most viewed)
            popular_products = sorted(
                self.product_views.items(),
                key=lambda x: x[1],
                reverse=True
            )[:n_recommendations * 2]

            # Get products from database
            for prod_id, view_count in popular_products:
                if prod_id != product_id:
                    product = self.vector_db.get_product_by_id(f"product_{prod_id}")
                    if product:
                        product['popularity_score'] = view_count / max(self.product_views.values())
                        recommendations.append(product)

                        if len(recommendations) >= n_recommendations:
                            break

            # If not enough recommendations, add random products
            if len(recommendations) < n_recommendations:
                # Get some random products
                all_products = self.vector_db.get_products_by_category(
                    category='Apparel',
                    category_type='masterCategory',
                    limit=n_recommendations * 2
                )

                for product in all_products:
                    if product.get('id') != product_id and product not in recommendations:
                        product['popularity_score'] = random.uniform(0.3, 0.6)
                        recommendations.append(product)

                        if len(recommendations) >= n_recommendations:
                            break

            return recommendations[:n_recommendations]

        except Exception as e:
            logger.error(f"Error in collaborative recommendations: {e}")
            return []

    def _hybrid_recommendations(self,
                               product_id: str = None,
                               query_embedding: np.ndarray = None,
                               user_id: str = None,
                               n_recommendations: int = 10,
                               filters: Dict = None) -> List[Dict]:
        """
        Get recommendations using a hybrid approach combining multiple methods.

        Args:
            product_id: Source product ID
            query_embedding: Query embedding for similarity
            user_id: User ID for personalization
            n_recommendations: Number of recommendations
            filters: Optional filters

        Returns:
            List[Dict]: Hybrid recommendations
        """
        try:
            all_recommendations = {}
            weights = {
                'visual': 0.4,
                'category': 0.3,
                'collaborative': 0.3
            }

            # Get visual similarity recommendations
            if product_id or query_embedding:
                visual_recs = self._visual_similarity_recommendations(
                    product_id=product_id,
                    query_embedding=query_embedding,
                    n_recommendations=n_recommendations * 2,
                    filters=filters
                )

                for i, rec in enumerate(visual_recs):
                    rec_id = rec.get('id', rec.get('product_id'))
                    if rec_id not in all_recommendations:
                        all_recommendations[rec_id] = {
                            'product': rec,
                            'scores': {}
                        }
                    # Higher score for higher ranking
                    score = 1.0 - (i / len(visual_recs))
                    all_recommendations[rec_id]['scores']['visual'] = score * weights['visual']

            # Get category-based recommendations
            if product_id:
                category_recs = self._category_based_recommendations(
                    product_id=product_id,
                    n_recommendations=n_recommendations * 2,
                    filters=filters
                )

                for i, rec in enumerate(category_recs):
                    rec_id = rec.get('id', rec.get('product_id'))
                    if rec_id not in all_recommendations:
                        all_recommendations[rec_id] = {
                            'product': rec,
                            'scores': {}
                        }
                    score = 1.0 - (i / len(category_recs))
                    all_recommendations[rec_id]['scores']['category'] = score * weights['category']

            # Get collaborative recommendations
            collab_recs = self._collaborative_recommendations(
                user_id=user_id,
                product_id=product_id,
                n_recommendations=n_recommendations * 2,
                filters=filters
            )

            for i, rec in enumerate(collab_recs):
                rec_id = rec.get('id', rec.get('product_id'))
                if rec_id not in all_recommendations:
                    all_recommendations[rec_id] = {
                        'product': rec,
                        'scores': {}
                    }
                score = rec.get('popularity_score', 0.5)
                all_recommendations[rec_id]['scores']['collaborative'] = score * weights['collaborative']

            # Calculate final scores
            final_recommendations = []
            for rec_id, rec_data in all_recommendations.items():
                # Sum all scores
                total_score = sum(rec_data['scores'].values())
                product = rec_data['product']
                product['hybrid_score'] = total_score
                product['score_breakdown'] = rec_data['scores']
                final_recommendations.append(product)

            # Sort by hybrid score
            final_recommendations.sort(key=lambda x: x['hybrid_score'], reverse=True)

            # Diversify recommendations
            diversified = self._diversify_recommendations(final_recommendations, n_recommendations)

            return diversified

        except Exception as e:
            logger.error(f"Error in hybrid recommendations: {e}")
            return []

    def _calculate_attribute_similarity(self, product1: Dict, product2: Dict) -> float:
        """
        Calculate similarity between two products based on attributes.

        Args:
            product1: First product
            product2: Second product

        Returns:
            float: Similarity score between 0 and 1
        """
        attributes = ['gender', 'masterCategory', 'subCategory', 'articleType',
                     'baseColour', 'season', 'usage']

        matches = 0
        total = 0

        for attr in attributes:
            if attr in product1 and attr in product2:
                total += 1
                if product1[attr] == product2[attr]:
                    matches += 1

        return matches / total if total > 0 else 0

    def _diversify_recommendations(self,
                                  recommendations: List[Dict],
                                  n_recommendations: int) -> List[Dict]:
        """
        Diversify recommendations to avoid too similar items.

        Args:
            recommendations: List of recommendations
            n_recommendations: Number of final recommendations

        Returns:
            List[Dict]: Diversified recommendations
        """
        if len(recommendations) <= n_recommendations:
            return recommendations

        diversified = []
        categories_seen = defaultdict(int)
        max_per_category = max(2, n_recommendations // 3)

        for rec in recommendations:
            category = rec.get('subCategory', 'Unknown')

            # Check if we've already added too many from this category
            if categories_seen[category] < max_per_category:
                diversified.append(rec)
                categories_seen[category] += 1

                if len(diversified) >= n_recommendations:
                    break

        # If still need more, add remaining highest scored
        if len(diversified) < n_recommendations:
            for rec in recommendations:
                if rec not in diversified:
                    diversified.append(rec)
                    if len(diversified) >= n_recommendations:
                        break

        return diversified

    def get_complementary_items(self,
                               product_id: str,
                               n_recommendations: int = 6) -> List[Dict]:
        """
        Get complementary items (e.g., shoes with dress, belt with pants).

        Args:
            product_id: Source product ID
            n_recommendations: Number of recommendations

        Returns:
            List[Dict]: Complementary products
        """
        try:
            # Get source product
            source_product = self.vector_db.get_product_by_id(f"product_{product_id}")
            if not source_product:
                return []

            # Define complementary categories
            complementary_map = {
                'Shirts': ['Jeans', 'Trousers', 'Belts', 'Watches'],
                'T-Shirts': ['Jeans', 'Shorts', 'Caps', 'Sunglasses'],
                'Jeans': ['Shirts', 'T-Shirts', 'Belts', 'Shoes'],
                'Dresses': ['Heels', 'Sandals', 'Bags', 'Jewellery'],
                'Shoes': ['Socks', 'Bags'],
                'Watches': ['Bracelets', 'Wallets'],
                'Bags': ['Wallets', 'Sunglasses']
            }

            article_type = source_product.get('articleType', '')
            gender = source_product.get('gender', '')

            # Get complementary article types
            complementary_types = complementary_map.get(article_type, [])

            recommendations = []

            for comp_type in complementary_types:
                # Search for complementary items
                filters = {
                    'articleType': comp_type,
                    'gender': gender
                }

                products = self.vector_db.get_products_by_category(
                    category=comp_type,
                    category_type='articleType',
                    limit=n_recommendations // len(complementary_types) + 1
                )

                recommendations.extend(products)

            # Shuffle for variety
            random.shuffle(recommendations)

            return recommendations[:n_recommendations]

        except Exception as e:
            logger.error(f"Error getting complementary items: {e}")
            return []

    def record_interaction(self, user_id: str, product_id: str, interaction_type: str = "view"):
        """
        Record user interaction with a product.

        Args:
            user_id: User ID
            product_id: Product ID
            interaction_type: Type of interaction (view, click, purchase)
        """
        self.user_interactions[user_id].append({
            'product_id': product_id,
            'type': interaction_type,
            'timestamp': np.datetime64('now')
        })

        # Update product popularity
        if interaction_type == "view":
            self.product_views[product_id] += 1
        elif interaction_type == "click":
            self.product_views[product_id] += 2
        elif interaction_type == "purchase":
            self.product_views[product_id] += 5


def test_recommendation_engine():
    """Test the recommendation engine."""
    from vector_database import FashionVectorDB

    # Initialize components
    db = FashionVectorDB()
    engine = RecommendationEngine(db)

    # Test different recommendation methods
    print("Testing recommendation engine...")

    # Record some interactions
    engine.record_interaction("user1", "1001", "view")
    engine.record_interaction("user1", "1002", "click")
    engine.record_interaction("user2", "1001", "purchase")

    # Get recommendations
    recs = engine.get_recommendations(
        product_id="1001",
        method="hybrid",
        n_recommendations=5
    )

    print(f"Found {len(recs)} recommendations")
    for rec in recs[:3]:
        print(f"- {rec.get('name', 'Unknown')} (Score: {rec.get('hybrid_score', 0):.3f})")


if __name__ == "__main__":
    test_recommendation_engine()
