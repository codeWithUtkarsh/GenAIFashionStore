"""
GenAI Assistant module for natural language interactions and intelligent shopping assistance.
"""

import openai
from typing import List, Dict, Optional, Tuple, Any
import json
import logging
from datetime import datetime
import re

from config import Config
from vector_database import FashionVectorDB
from image_embedder import CLIPEmbedder
from recommendation_engine import RecommendationEngine

# Setup logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format=Config.LOG_FORMAT
)
logger = logging.getLogger(__name__)


class FashionShoppingAssistant:
    """AI-powered shopping assistant using OpenAI GPT models."""

    def __init__(self,
                 vector_db: FashionVectorDB,
                 embedder: CLIPEmbedder,
                 recommendation_engine: RecommendationEngine):
        """
        Initialize the shopping assistant.

        Args:
            vector_db: Vector database instance
            embedder: Image embedder instance
            recommendation_engine: Recommendation engine instance
        """
        self.vector_db = vector_db
        self.embedder = embedder
        self.recommendation_engine = recommendation_engine

        # Initialize OpenAI client
        if Config.OPENAI_API_KEY:
            openai.api_key = Config.OPENAI_API_KEY
            self.gpt_model = Config.GPT_MODEL if hasattr(Config, 'GPT_MODEL') else "gpt-3.5-turbo"
        else:
            logger.warning("OpenAI API key not configured")
            self.gpt_model = None

        # Conversation history
        self.conversation_history = []

        # System prompt for the assistant
        self.system_prompt = """You are a helpful and knowledgeable fashion shopping assistant.
        You help customers find the perfect fashion items based on their preferences, style, and needs.
        You have access to a large catalog of fashion products including clothing, accessories, and footwear.

        Your capabilities include:
        1. Understanding customer style preferences and requirements
        2. Recommending products based on visual similarity, categories, or descriptions
        3. Explaining fashion trends and styling tips
        4. Helping with outfit coordination and complementary items
        5. Answering questions about products, materials, and care instructions

        Always be friendly, helpful, and provide specific product recommendations when possible.
        If you're unsure about something, ask clarifying questions."""

        logger.info("Fashion Shopping Assistant initialized")

    def process_user_query(self,
                          query: str,
                          user_id: str = None,
                          context: Dict = None) -> Dict:
        """
        Process a user query and generate appropriate response.

        Args:
            query: User's natural language query
            user_id: Optional user ID for personalization
            context: Optional context (current product, filters, etc.)

        Returns:
            Dict: Response with text, recommendations, and actions
        """
        try:
            # Parse user intent
            intent = self._parse_intent(query)

            # Extract entities from query
            entities = self._extract_entities(query)

            # Generate response based on intent
            if intent == "product_search":
                return self._handle_product_search(query, entities, context)
            elif intent == "recommendation":
                return self._handle_recommendation_request(query, entities, context, user_id)
            elif intent == "outfit_coordination":
                return self._handle_outfit_coordination(query, entities, context)
            elif intent == "product_info":
                return self._handle_product_info(query, entities, context)
            elif intent == "style_advice":
                return self._handle_style_advice(query, entities)
            else:
                return self._handle_general_query(query, context)

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'response': "I apologize, but I encountered an error processing your request. Could you please rephrase your question?",
                'error': str(e)
            }

    def _parse_intent(self, query: str) -> str:
        """
        Parse user intent from the query.

        Args:
            query: User query

        Returns:
            str: Detected intent
        """
        query_lower = query.lower()

        # Define intent patterns
        search_keywords = ['find', 'search', 'looking for', 'show me', 'need', 'want']
        recommendation_keywords = ['recommend', 'suggest', 'similar', 'like this', 'alternatives']
        outfit_keywords = ['outfit', 'coordinate', 'match', 'goes with', 'wear with', 'complement']
        info_keywords = ['tell me about', 'information', 'details', 'what is', 'describe']
        style_keywords = ['style', 'fashion', 'trend', 'advice', 'tips', 'how to wear']

        if any(keyword in query_lower for keyword in search_keywords):
            return "product_search"
        elif any(keyword in query_lower for keyword in recommendation_keywords):
            return "recommendation"
        elif any(keyword in query_lower for keyword in outfit_keywords):
            return "outfit_coordination"
        elif any(keyword in query_lower for keyword in info_keywords):
            return "product_info"
        elif any(keyword in query_lower for keyword in style_keywords):
            return "style_advice"
        else:
            return "general"

    def _extract_entities(self, query: str) -> Dict:
        """
        Extract entities (colors, categories, attributes) from query.

        Args:
            query: User query

        Returns:
            Dict: Extracted entities
        """
        entities = {
            'colors': [],
            'categories': [],
            'gender': None,
            'attributes': [],
            'price_range': None,
            'brand': None,
            'season': None,
            'usage': None
        }

        query_lower = query.lower()

        # Extract colors
        colors = ['red', 'blue', 'green', 'black', 'white', 'yellow', 'pink',
                 'purple', 'orange', 'brown', 'grey', 'gray', 'navy', 'beige']
        entities['colors'] = [c for c in colors if c in query_lower]

        # Extract gender
        if 'men' in query_lower or "men's" in query_lower:
            entities['gender'] = 'Men'
        elif 'women' in query_lower or "women's" in query_lower:
            entities['gender'] = 'Women'
        elif 'unisex' in query_lower:
            entities['gender'] = 'Unisex'

        # Extract categories
        categories = ['shirt', 'jeans', 't-shirt', 'dress', 'shoes', 'watch',
                     'bag', 'jacket', 'sweater', 'shorts', 'sandals', 'boots']
        for cat in categories:
            if cat in query_lower:
                entities['categories'].append(cat)

        # Extract usage
        usage_types = ['casual', 'formal', 'sports', 'party', 'ethnic']
        for usage in usage_types:
            if usage in query_lower:
                entities['usage'] = usage.capitalize()

        # Extract season
        seasons = ['summer', 'winter', 'spring', 'fall', 'autumn']
        for season in seasons:
            if season in query_lower:
                entities['season'] = season.capitalize()

        return entities

    def _handle_product_search(self, query: str, entities: Dict, context: Dict) -> Dict:
        """
        Handle product search queries.

        Args:
            query: User query
            entities: Extracted entities
            context: Query context

        Returns:
            Dict: Response with search results
        """
        # Build filters from entities
        filters = {}
        if entities['gender']:
            filters['gender'] = entities['gender']
        if entities['usage']:
            filters['usage'] = entities['usage']
        if entities['season']:
            filters['season'] = entities['season']

        # Perform text-based search
        search_results = self.vector_db.search_by_text(
            text_query=query,
            n_results=10,
            filters=filters
        )

        # Generate natural language response
        if search_results:
            response = self._generate_search_response(query, search_results, entities)
        else:
            response = "I couldn't find any products matching your criteria. Would you like to try a different search?"

        return {
            'response': response,
            'products': search_results,
            'filters_applied': filters,
            'intent': 'product_search'
        }

    def _handle_recommendation_request(self,
                                      query: str,
                                      entities: Dict,
                                      context: Dict,
                                      user_id: str) -> Dict:
        """
        Handle recommendation requests.

        Args:
            query: User query
            entities: Extracted entities
            context: Query context
            user_id: User ID

        Returns:
            Dict: Response with recommendations
        """
        # Get current product from context if available
        current_product_id = context.get('current_product_id') if context else None

        # Build filters
        filters = {}
        if entities['gender']:
            filters['gender'] = entities['gender']

        # Get recommendations
        recommendations = self.recommendation_engine.get_recommendations(
            product_id=current_product_id,
            user_id=user_id,
            method='hybrid',
            n_recommendations=8,
            filters=filters
        )

        # Generate response
        if recommendations:
            response = self._generate_recommendation_response(query, recommendations, entities)
        else:
            response = "I'm having trouble finding recommendations right now. Could you provide more details about what you're looking for?"

        return {
            'response': response,
            'recommendations': recommendations,
            'intent': 'recommendation'
        }

    def _handle_outfit_coordination(self, query: str, entities: Dict, context: Dict) -> Dict:
        """
        Handle outfit coordination requests.

        Args:
            query: User query
            entities: Extracted entities
            context: Query context

        Returns:
            Dict: Response with outfit suggestions
        """
        current_product_id = context.get('current_product_id') if context else None

        if current_product_id:
            # Get complementary items
            complementary_items = self.recommendation_engine.get_complementary_items(
                product_id=current_product_id,
                n_recommendations=6
            )

            response = self._generate_outfit_response(query, complementary_items)

            return {
                'response': response,
                'complementary_items': complementary_items,
                'intent': 'outfit_coordination'
            }
        else:
            return {
                'response': "To help you coordinate an outfit, please select a product first or describe what item you'd like to build an outfit around.",
                'intent': 'outfit_coordination'
            }

    def _handle_product_info(self, query: str, entities: Dict, context: Dict) -> Dict:
        """
        Handle product information requests.

        Args:
            query: User query
            entities: Extracted entities
            context: Query context

        Returns:
            Dict: Response with product information
        """
        current_product_id = context.get('current_product_id') if context else None

        if current_product_id:
            product = self.vector_db.get_product_by_id(f"product_{current_product_id}")

            if product:
                response = self._generate_product_info_response(product)
            else:
                response = "I couldn't find information about this product."

            return {
                'response': response,
                'product': product,
                'intent': 'product_info'
            }
        else:
            return {
                'response': "Please select a product to get more information about it.",
                'intent': 'product_info'
            }

    def _handle_style_advice(self, query: str, entities: Dict) -> Dict:
        """
        Handle style advice requests.

        Args:
            query: User query
            entities: Extracted entities

        Returns:
            Dict: Response with style advice
        """
        # Generate style advice using GPT
        if self.gpt_model:
            response = self._generate_gpt_response(
                query,
                context="Provide helpful fashion and style advice."
            )
        else:
            response = self._generate_generic_style_advice(entities)

        return {
            'response': response,
            'intent': 'style_advice'
        }

    def _handle_general_query(self, query: str, context: Dict) -> Dict:
        """
        Handle general queries.

        Args:
            query: User query
            context: Query context

        Returns:
            Dict: General response
        """
        # Use GPT for general queries if available
        if self.gpt_model:
            response = self._generate_gpt_response(query, context)
        else:
            response = "I'm here to help you find the perfect fashion items. You can search for products, ask for recommendations, or get style advice. What would you like to know?"

        return {
            'response': response,
            'intent': 'general'
        }

    def _generate_search_response(self, query: str, results: List[Dict], entities: Dict) -> str:
        """Generate natural language response for search results."""
        if not results:
            return "I couldn't find any products matching your search."

        num_results = len(results)

        # Build response
        response_parts = []

        if entities['colors']:
            color_str = " and ".join(entities['colors'])
            response_parts.append(f"I found {num_results} {color_str} items")
        else:
            response_parts.append(f"I found {num_results} products")

        if entities['categories']:
            cat_str = ", ".join(entities['categories'])
            response_parts.append(f"in {cat_str}")

        response = " ".join(response_parts) + " for you. "

        # Highlight top results
        top_products = results[:3]
        product_descriptions = []
        for product in top_products:
            desc = f"{product.get('name', 'Item')} in {product.get('baseColour', 'various colors')}"
            product_descriptions.append(desc)

        response += f"The top matches include: {', '.join(product_descriptions)}."

        return response

    def _generate_recommendation_response(self, query: str, recommendations: List[Dict], entities: Dict) -> str:
        """Generate natural language response for recommendations."""
        if not recommendations:
            return "I couldn't find suitable recommendations at this time."

        response = f"Based on your preferences, I recommend these {len(recommendations)} items: "

        # Describe top recommendations
        top_recs = recommendations[:3]
        descriptions = []
        for rec in top_recs:
            name = rec.get('name', 'Product')
            category = rec.get('articleType', 'item')
            color = rec.get('baseColour', '')

            if color:
                desc = f"a {color} {name}"
            else:
                desc = f"a {name}"

            descriptions.append(desc)

        response += ", ".join(descriptions)
        response += ". These items match your style preferences and would be great additions to your wardrobe!"

        return response

    def _generate_outfit_response(self, query: str, complementary_items: List[Dict]) -> str:
        """Generate outfit coordination response."""
        if not complementary_items:
            return "I couldn't find complementary items for outfit coordination."

        response = "Here are some great items to complete your outfit: "

        items_by_type = {}
        for item in complementary_items:
            item_type = item.get('articleType', 'accessory')
            if item_type not in items_by_type:
                items_by_type[item_type] = []
            items_by_type[item_type].append(item)

        suggestions = []
        for item_type, items in items_by_type.items():
            if items:
                color = items[0].get('baseColour', '')
                if color:
                    suggestions.append(f"{color} {item_type.lower()}")
                else:
                    suggestions.append(item_type.lower())

        response += ", ".join(suggestions)
        response += ". These pieces would create a stylish and coordinated look!"

        return response

    def _generate_product_info_response(self, product: Dict) -> str:
        """Generate detailed product information response."""
        name = product.get('name', 'This product')
        category = product.get('articleType', 'item')
        color = product.get('baseColour', 'multi-color')
        gender = product.get('gender', 'unisex')
        usage = product.get('usage', 'versatile')
        season = product.get('season', 'all-season')

        response = f"{name} is a {gender.lower()} {category.lower()} in {color}. "
        response += f"It's perfect for {usage.lower()} wear "

        if season != 'all-season':
            response += f"and ideal for {season.lower()}. "
        else:
            response += "and suitable for any season. "

        if product.get('description'):
            response += product['description']

        return response

    def _generate_generic_style_advice(self, entities: Dict) -> str:
        """Generate generic style advice based on entities."""
        advice_parts = []

        if entities['colors']:
            color = entities['colors'][0]
            advice_parts.append(f"{color.capitalize()} is a versatile color that can be styled in many ways.")

        if entities['season']:
            season = entities['season']
            if season.lower() == 'summer':
                advice_parts.append("For summer, opt for lightweight, breathable fabrics and bright colors.")
            elif season.lower() == 'winter':
                advice_parts.append("For winter, layer your outfits and choose warmer fabrics like wool and cashmere.")

        if entities['usage']:
            usage = entities['usage']
            if usage.lower() == 'formal':
                advice_parts.append("For formal occasions, stick to classic cuts and neutral colors.")
            elif usage.lower() == 'casual':
                advice_parts.append("For casual wear, feel free to experiment with colors and comfortable fits.")

        if advice_parts:
            return " ".join(advice_parts)
        else:
            return "Fashion is about expressing your personality. Don't be afraid to experiment with different styles and find what makes you feel confident!"

    def _generate_gpt_response(self, query: str, context: Any = None) -> str:
        """
        Generate response using OpenAI GPT.

        Args:
            query: User query
            context: Additional context

        Returns:
            str: GPT-generated response
        """
        try:
            messages = [
                {"role": "system", "content": self.system_prompt}
            ]

            # Add conversation history
            for msg in self.conversation_history[-5:]:  # Last 5 messages
                messages.append(msg)

            # Add current query
            user_message = query
            if context:
                user_message += f"\n\nContext: {json.dumps(context) if isinstance(context, dict) else str(context)}"

            messages.append({"role": "user", "content": user_message})

            # Generate response
            response = openai.ChatCompletion.create(
                model=self.gpt_model,
                messages=messages,
                temperature=0.7,
                max_tokens=300
            )

            assistant_response = response.choices[0].message.content

            # Update conversation history
            self.conversation_history.append({"role": "user", "content": query})
            self.conversation_history.append({"role": "assistant", "content": assistant_response})

            return assistant_response

        except Exception as e:
            logger.error(f"Error generating GPT response: {e}")
            return "I'm having trouble generating a response right now. Please try again."

    def clear_conversation_history(self):
        """Clear the conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")


def test_assistant():
    """Test the GenAI assistant."""
    from vector_database import FashionVectorDB
    from image_embedder import CLIPEmbedder
    from recommendation_engine import RecommendationEngine

    # Initialize components
    db = FashionVectorDB()
    embedder = CLIPEmbedder()
    rec_engine = RecommendationEngine(db, embedder)

    # Initialize assistant
    assistant = FashionShoppingAssistant(db, embedder, rec_engine)

    # Test queries
    test_queries = [
        "Show me blue shirts for men",
        "I need something for a formal event",
        "What would go well with jeans?",
        "Recommend some summer clothing"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        response = assistant.process_user_query(query)
        print(f"Response: {response['response']}")
        if 'products' in response and response['products']:
            print(f"Found {len(response['products'])} products")


if __name__ == "__main__":
    test_assistant()
