"""
Main Streamlit application for the GenAI Fashion Store.
A comprehensive shopping assistant with image-based search, recommendations, and natural language interaction.
"""

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import json
import time
from pathlib import Path
import logging
from typing import Dict, List, Optional
import base64
from io import BytesIO
import hashlib

# Import custom modules
from config import Config
from data_downloader import FashionDatasetDownloader
from image_embedder import CLIPEmbedder
from vector_database import FashionVectorDB
from recommendation_engine import RecommendationEngine
from genai_assistant import FashionShoppingAssistant

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format=Config.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="GenAI Fashion Store",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton > button {
        background-color: #FF6B6B;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #FF5252;
        transform: translateY(-2px);
    }
    .product-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem;
        transition: all 0.3s;
        background: white;
    }
    .product-card:hover {
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        transform: translateY(-3px);
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #f0f2f6;
        text-align: right;
    }
    .assistant-message {
        background-color: #e8f4fd;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)


class FashionStoreApp:
    """Main application class for the Fashion Store."""

    def __init__(self):
        """Initialize the Fashion Store application."""
        self.config = Config
        self.initialize_session_state()
        self.setup_components()

    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'initialized' not in st.session_state:
            st.session_state.initialized = False
            st.session_state.products = []
            st.session_state.current_product = None
            st.session_state.search_results = []
            st.session_state.recommendations = []
            st.session_state.chat_history = []
            st.session_state.user_id = f"user_{int(time.time())}"
            st.session_state.cart = []
            st.session_state.favorites = []
            st.session_state.view_history = []
            st.session_state.uploaded_image = None

    def setup_components(self):
        """Setup and cache application components."""
        # Use session state to cache components
        if 'components_initialized' not in st.session_state:
            with st.spinner("🚀 Initializing Fashion Store components..."):
                try:
                    # Initialize embedder first
                    st.session_state.embedder = CLIPEmbedder()

                    # Initialize vector database with embedder
                    st.session_state.vector_db = FashionVectorDB(embedder=st.session_state.embedder)

                    # Initialize recommendation engine
                    st.session_state.recommendation_engine = RecommendationEngine(
                        st.session_state.vector_db,
                        st.session_state.embedder
                    )

                    # Initialize AI assistant
                    st.session_state.assistant = FashionShoppingAssistant(
                        st.session_state.vector_db,
                        st.session_state.embedder,
                        st.session_state.recommendation_engine
                    )

                    st.session_state.components_initialized = True
                    logger.info("All components initialized successfully")

                except Exception as e:
                    logger.error(f"Error initializing components: {e}")
                    st.error(f"Failed to initialize components: {e}")
                    return False

        # Set instance attributes from session state
        self.vector_db = st.session_state.vector_db
        self.embedder = st.session_state.embedder
        self.recommendation_engine = st.session_state.recommendation_engine
        self.assistant = st.session_state.assistant
        return True

    def load_data(self, force_reload: bool = False):
        """Load and index product data."""
        if not st.session_state.initialized or force_reload:
            with st.spinner("📊 Loading fashion dataset..."):
                downloader = FashionDatasetDownloader(max_items=Config.MAX_PRODUCTS_TO_LOAD)
                success, products = downloader.setup_dataset()

                if success and products:
                    st.session_state.products = products

                    # Index products in vector database
                    with st.spinner("🔍 Indexing products for search..."):
                        # Generate embeddings
                        image_paths = [p['image_path'] for p in products if 'image_path' in p]

                        if hasattr(self, 'embedder') and self.embedder:
                            embeddings = self.embedder.get_batch_embeddings(
                                image_paths[:Config.MAX_PRODUCTS_TO_LOAD],
                                batch_size=32,
                                use_cache=True
                            )

                            # Add to vector database
                            if hasattr(self, 'vector_db') and self.vector_db:
                                self.vector_db.add_products(products, embeddings)

                    st.session_state.initialized = True
                    st.success(f"✅ Loaded {len(products)} products successfully!")
                else:
                    st.error("Failed to load dataset. Please check your configuration.")

    def display_header(self):
        """Display application header."""
        col1, col2, col3 = st.columns([1, 3, 1])

        with col2:
            st.markdown("""
                <h1 style='text-align: center; color: #FF6B6B; font-size: 3rem;'>
                    👗 GenAI Fashion Store
                </h1>
                <p style='text-align: center; color: #666; font-size: 1.2rem;'>
                    Your AI-Powered Personal Fashion Assistant
                </p>
            """, unsafe_allow_html=True)

    def display_search_section(self):
        """Display search functionality."""
        st.markdown("### 🔍 Search Products")

        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            search_query = st.text_input(
                "Search for fashion items",
                placeholder="e.g., 'blue shirt for men', 'summer dress', 'formal shoes'",
                key="search_input"
            )

        with col2:
            search_type = st.selectbox(
                "Search by",
                ["Text", "Image", "Both"],
                key="search_type"
            )

        with col3:
            search_button = st.button("🔍 Search", key="search_button", use_container_width=True)

        # Image upload for visual search
        if search_type in ["Image", "Both"]:
            uploaded_file = st.file_uploader(
                "Upload an image for visual search",
                type=['png', 'jpg', 'jpeg'],
                key="image_upload"
            )

            if uploaded_file:
                st.session_state.uploaded_image = Image.open(uploaded_file)
                st.image(st.session_state.uploaded_image, caption="Uploaded Image", width=200)

        # Perform search
        if search_button or search_query:
            self.perform_search(search_query, search_type)

    def perform_search(self, query: str, search_type: str):
        """Perform product search."""
        with st.spinner("Searching products..."):
            results = []

            if search_type == "Text" and query:
                # Text-based search
                if hasattr(self, 'vector_db') and self.vector_db:
                    results = self.vector_db.search_by_text(query, n_results=20)

            elif search_type == "Image" and st.session_state.uploaded_image:
                # Image-based search
                if hasattr(self, 'embedder') and self.embedder and hasattr(self, 'vector_db') and self.vector_db:
                    image_embedding = self.embedder.get_image_embedding(st.session_state.uploaded_image)
                    results = self.vector_db.search_similar(image_embedding, n_results=20)

            elif search_type == "Both":
                # Hybrid search
                text_results = []
                if query and hasattr(self, 'vector_db') and self.vector_db:
                    text_results = self.vector_db.search_by_text(query, n_results=10)

                image_results = []
                if st.session_state.uploaded_image and hasattr(self, 'embedder') and self.embedder and hasattr(self, 'vector_db') and self.vector_db:
                    image_embedding = self.embedder.get_image_embedding(st.session_state.uploaded_image)
                    image_results = self.vector_db.search_similar(image_embedding, n_results=10)

                # Merge results
                results = text_results + image_results

            st.session_state.search_results = results

            if results:
                st.success(f"Found {len(results)} products!")
            else:
                st.warning("No products found. Try different keywords or filters.")

    def display_filters(self):
        """Display filter options in sidebar."""
        st.sidebar.markdown("### 🎯 Filters")

        # Gender filter
        gender_options = ["All", "Men", "Women", "Boys", "Girls", "Unisex"]
        selected_gender = st.sidebar.selectbox("Gender", gender_options)

        # Category filter
        category_options = ["All", "Apparel", "Footwear", "Accessories", "Personal Care"]
        selected_category = st.sidebar.selectbox("Category", category_options)

        # Color filter
        color_options = ["All", "Black", "Blue", "White", "Red", "Green", "Yellow", "Pink", "Grey", "Brown"]
        selected_color = st.sidebar.selectbox("Color", color_options)

        # Price range (simulated)
        price_range = st.sidebar.slider(
            "Price Range ($)",
            min_value=0,
            max_value=500,
            value=(0, 500),
            step=10
        )

        # Season filter
        season_options = ["All", "Summer", "Winter", "Spring", "Fall"]
        selected_season = st.sidebar.selectbox("Season", season_options)

        # Usage filter
        usage_options = ["All", "Casual", "Formal", "Sports", "Ethnic", "Party"]
        selected_usage = st.sidebar.selectbox("Usage", usage_options)

        # Apply filters button
        if st.sidebar.button("Apply Filters", use_container_width=True):
            # Build filter dictionary
            filters = {}
            if selected_gender != "All":
                filters['gender'] = selected_gender
            if selected_category != "All":
                filters['masterCategory'] = selected_category
            if selected_color != "All":
                filters['baseColour'] = selected_color
            if selected_season != "All":
                filters['season'] = selected_season
            if selected_usage != "All":
                filters['usage'] = selected_usage

            # Apply filters to search
            st.session_state.filters = filters
            st.rerun()

    def display_products(self, products: List[Dict], title: str = "Products"):
        """Display product grid."""
        if not products:
            st.info("No products to display.")
            return

        st.markdown(f"### {title}")

        # Create product grid
        cols = st.columns(4)

        for idx, product in enumerate(products[:20]):  # Limit to 20 products
            # Generate unique key for this product instance
            product_str = f"{title}_{idx}_{product.get('id', '')}_{product.get('name', '')}"
            unique_key = hashlib.md5(product_str.encode()).hexdigest()[:8]
            col_idx = idx % 4

            with cols[col_idx]:
                # Product card
                with st.container():
                    # Try to display image
                    if 'image_path' in product and Path(product['image_path']).exists():
                        try:
                            img = Image.open(product['image_path'])
                            st.image(img, width='content')
                        except:
                            st.image("https://via.placeholder.com/200x200?text=No+Image", use_column_width=True)
                    else:
                        st.image("https://via.placeholder.com/200x200?text=No+Image", use_column_width=True)

                    # Product info
                    st.markdown(f"**{product.get('name', 'Unknown Product')}**")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption(f"🎨 {product.get('baseColour', 'N/A')}")
                    with col2:
                        st.caption(f"👤 {product.get('gender', 'N/A')}")

                    st.caption(f"📦 {product.get('articleType', 'N/A')}")

                    # Similarity score if available
                    if 'similarity_score' in product:
                        st.progress(product['similarity_score'])
                        st.caption(f"Match: {product['similarity_score']*100:.1f}%")

                    # Action buttons
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        if st.button("👁️", key=f"view_{unique_key}", help="View Details"):
                            st.session_state.current_product = product
                            self.view_product_details(product)

                    with col2:
                        if st.button("❤️", key=f"fav_{unique_key}", help="Add to Favorites"):
                            self.add_to_favorites(product)

                    with col3:
                        if st.button("🛒", key=f"cart_{unique_key}", help="Add to Cart"):
                            self.add_to_cart(product)

    def display_chat_interface(self):
        """Display AI chat interface."""
        st.markdown("### 💬 AI Shopping Assistant")

        # Chat container
        chat_container = st.container()

        # Input area
        col1, col2 = st.columns([5, 1])

        with col1:
            user_input = st.text_input(
                "Ask me anything about fashion!",
                placeholder="e.g., 'What would go well with blue jeans?', 'Show me formal wear for a wedding'",
                key="chat_input"
            )

        with col2:
            send_button = st.button("Send", key="send_button", use_container_width=True)

        # Process chat input
        if send_button and user_input:
            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            # Get assistant response
            context = {
                'current_product_id': st.session_state.current_product.get('id') if st.session_state.current_product else None,
                'filters': getattr(st.session_state, 'filters', {})
            }

            if hasattr(self, 'assistant') and self.assistant:
                response = self.assistant.process_user_query(
                    user_input,
                    user_id=st.session_state.user_id,
                    context=context
                )
            else:
                response = {'response': 'Assistant is not initialized yet. Please wait a moment.'}

            # Add assistant response to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response['response']
            })

            # Handle recommendations or products
            if 'products' in response:
                st.session_state.search_results = response['products']
            elif 'recommendations' in response:
                st.session_state.recommendations = response['recommendations']

        # Display chat history
        with chat_container:
            for message in st.session_state.chat_history[-5:]:  # Show last 5 messages
                if message['role'] == 'user':
                    st.markdown(f"""
                        <div class='chat-message user-message'>
                            <strong>You:</strong> {message['content']}
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class='chat-message assistant-message'>
                            <strong>Assistant:</strong> {message['content']}
                        </div>
                    """, unsafe_allow_html=True)

    def display_recommendations(self):
        """Display personalized recommendations."""
        if st.session_state.current_product:
            st.markdown("### 🎯 Recommended for You")

            # Get recommendations
            if hasattr(self, 'recommendation_engine') and self.recommendation_engine:
                recommendations = self.recommendation_engine.get_recommendations(
                    product_id=st.session_state.current_product.get('id'),
                    user_id=st.session_state.user_id,
                    method='hybrid',
                    n_recommendations=8
                )
            else:
                recommendations = []

            if recommendations:
                self.display_products(recommendations, "Similar Products You Might Like")

            # Get complementary items
            st.markdown("### 👔 Complete the Look")
            if hasattr(self, 'recommendation_engine') and self.recommendation_engine:
                complementary = self.recommendation_engine.get_complementary_items(
                    product_id=st.session_state.current_product.get('id'),
                    n_recommendations=4
                )
            else:
                complementary = []

            if complementary:
                self.display_products(complementary, "Goes Well With")

    def display_sidebar_info(self):
        """Display information in sidebar."""
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Your Activity")

        # Display metrics
        col1, col2 = st.sidebar.columns(2)

        with col1:
            st.metric("Cart Items", len(st.session_state.cart))

        with col2:
            st.metric("Favorites", len(st.session_state.favorites))

        # Recent views
        if st.session_state.view_history:
            st.sidebar.markdown("### 👁️ Recently Viewed")
            for item in st.session_state.view_history[-3:]:
                st.sidebar.caption(f"• {item.get('name', 'Product')}")

        st.sidebar.markdown("---")

        # Database stats
        stats = self.vector_db.get_statistics() if hasattr(self, 'vector_db') and self.vector_db else {'total_products': 0}
        st.sidebar.markdown("### 📈 Store Statistics")
        st.sidebar.info(f"Total Products: {stats.get('total_products', 0)}")

    def view_product_details(self, product: Dict):
        """View detailed product information."""
        st.session_state.current_product = product

        # Add to view history (check for duplicates by ID)
        product_id = product.get('id', product.get('product_id', ''))
        history_ids = [p.get('id', p.get('product_id', '')) for p in st.session_state.view_history]

        if product_id not in history_ids:
            st.session_state.view_history.append(product)
            # Keep only last 10 items in history
            if len(st.session_state.view_history) > 10:
                st.session_state.view_history = st.session_state.view_history[-10:]

        # Record interaction
        if hasattr(self, 'recommendation_engine') and self.recommendation_engine:
                self.recommendation_engine.record_interaction(
                    st.session_state.user_id,
                    product.get('id'),
                    'view'
                )

    def add_to_cart(self, product: Dict):
        """Add product to cart."""
        # Check for duplicate by ID
        product_id = product.get('id', product.get('product_id', ''))
        cart_ids = [p.get('id', p.get('product_id', '')) for p in st.session_state.cart]

        if product_id not in cart_ids:
            st.session_state.cart.append(product)
            st.success(f"Added {product.get('name', 'Product')} to cart!")
        else:
            st.info("Product already in cart!")

    def add_to_favorites(self, product: Dict):
        """Add product to favorites."""
        # Check for duplicate by ID
        product_id = product.get('id', product.get('product_id', ''))
        favorite_ids = [p.get('id', p.get('product_id', '')) for p in st.session_state.favorites]

        if product_id not in favorite_ids:
            st.session_state.favorites.append(product)
            st.success(f"Added {product.get('name', 'Product')} to favorites!")
        else:
            st.info("Product already in favorites!")

    def run(self):
        """Main application entry point."""
        # Display header
        self.display_header()

        # Setup components first
        if not self.setup_components():
            st.error("Failed to initialize application components. Please refresh the page.")
            st.stop()

        # Initialize data
        if not st.session_state.initialized:
            self.load_data()

        # Sidebar filters
        self.display_filters()
        self.display_sidebar_info()

        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Search", "💬 AI Assistant", "🎯 For You", "❤️ Favorites", "🛒 Cart"])

        with tab1:
            self.display_search_section()
            if st.session_state.search_results:
                self.display_products(st.session_state.search_results, "Search Results")

        with tab2:
            self.display_chat_interface()
            if st.session_state.recommendations:
                self.display_products(st.session_state.recommendations, "AI Recommendations")

        with tab3:
            self.display_recommendations()

        with tab4:
            if st.session_state.favorites:
                self.display_products(st.session_state.favorites, "Your Favorites")
            else:
                st.info("No favorites yet! Browse products and click ❤️ to add to favorites.")

        with tab5:
            if st.session_state.cart:
                self.display_products(st.session_state.cart, "Shopping Cart")

                # Cart summary
                st.markdown("### 📝 Cart Summary")
                total_items = len(st.session_state.cart)
                estimated_total = total_items * 50  # Dummy pricing

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Items", total_items)
                with col2:
                    st.metric("Estimated Total", f"${estimated_total}")
                with col3:
                    if st.button("Checkout", use_container_width=True):
                        st.success("Thank you for shopping with us! 🎉")
                        st.session_state.cart = []
            else:
                st.info("Your cart is empty! Start shopping to add items.")

        # Footer
        st.markdown("---")
        st.markdown("""
            <p style='text-align: center; color: #999;'>
                Made with ❤️ using Streamlit and GenAI | Powered by CLIP & OpenAI
            </p>
        """, unsafe_allow_html=True)


def main():
    """Main function to run the application."""
    app = FashionStoreApp()
    app.run()


if __name__ == "__main__":
    main()
