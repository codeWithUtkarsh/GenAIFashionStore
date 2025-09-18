"""
Configuration module for the GenAI Fashion Store application.
Handles environment variables, paths, and application settings.
"""

import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration class."""

    # Base paths
    BASE_DIR = Path(__file__).parent.absolute()
    DATA_DIR = BASE_DIR / "data"
    IMAGES_DIR = DATA_DIR / "images"
    MODELS_DIR = BASE_DIR / "models"
    VECTOR_DB_DIR = BASE_DIR / "vector_db"
    CACHE_DIR = BASE_DIR / ".cache"

    # Create directories if they don't exist
    for dir_path in [DATA_DIR, IMAGES_DIR, MODELS_DIR, VECTOR_DB_DIR, CACHE_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME", "")
    KAGGLE_KEY = os.getenv("KAGGLE_KEY", "")
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")

    # Model Configuration
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "clip-ViT-B-32")
    CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
    GPT_MODEL = "gpt-4-turbo-preview"

    # Search Configuration
    MAX_RESULTS = int(os.getenv("MAX_RESULTS", "20"))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    ENABLE_SEMANTIC_SEARCH = True
    ENABLE_VISUAL_SEARCH = True

    # ChromaDB Configuration
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "chroma_db"))
    CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "fashion_products")

    # Dataset Configuration
    DATASET_NAME = "paramaggarwal/fashion-product-images-small"
    MAX_PRODUCTS_TO_LOAD = 2000  # For faster iteration during development
    BATCH_SIZE = 32

    # UI Configuration
    STREAMLIT_THEME = {
        "primaryColor": "#FF6B6B",
        "backgroundColor": "#FFFFFF",
        "secondaryBackgroundColor": "#F5F5F5",
        "textColor": "#262730",
        "font": "sans serif"
    }

    # Image Processing
    IMAGE_SIZE = (224, 224)  # CLIP input size
    THUMBNAIL_SIZE = (150, 150)
    SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.webp']

    # Cache Configuration
    ENABLE_CACHE = True
    CACHE_TTL = 3600  # 1 hour

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Product Categories (based on Fashion Product dataset)
    CATEGORIES = {
        "gender": ["Men", "Women", "Boys", "Girls", "Unisex"],
        "masterCategory": ["Apparel", "Accessories", "Footwear", "Personal Care", "Free Items"],
        "subCategory": [
            "Topwear", "Bottomwear", "Watches", "Bags", "Shoes",
            "Sandals", "Belts", "Flip Flops", "Wallets", "Sunglasses"
        ],
        "articleType": [
            "Shirts", "Jeans", "T-Shirts", "Shoes", "Watches", "Bags",
            "Shorts", "Dresses", "Sarees", "Kurtas", "Jackets", "Sweaters"
        ],
        "season": ["Summer", "Winter", "Fall", "Spring"],
        "usage": ["Casual", "Formal", "Sports", "Ethnic", "Party"]
    }

    # Article Type Mapping for better search accuracy
    ARTICLE_TYPE_MAPPING = {
        'shoes': {
            'exact_matches': ['Shoes', 'Casual Shoes', 'Formal Shoes', 'Sports Shoes'],
            'exclude': ['Flip Flops', 'Sandals', 'Heels', 'Flats'],
            'keywords': ['shoe', 'shoes', 'footwear', 'sneaker', 'loafer', 'oxford']
        },
        'sandals': {
            'exact_matches': ['Sandals', 'Flats', 'Heels'],
            'exclude': ['Shoes', 'Flip Flops'],
            'keywords': ['sandal', 'sandals', 'heels', 'flats']
        },
        'flip flops': {
            'exact_matches': ['Flip Flops'],
            'exclude': ['Shoes', 'Sandals', 'Heels'],
            'keywords': ['flip flop', 'flipflop', 'flip-flop', 'slippers']
        },
        'shirts': {
            'exact_matches': ['Shirts', 'Casual Shirts', 'Formal Shirts'],
            'exclude': ['T-Shirts', 'Tops'],
            'keywords': ['shirt', 'shirts']
        },
        't-shirts': {
            'exact_matches': ['Tshirts', 'T-Shirts', 'Tops'],
            'exclude': ['Shirts'],
            'keywords': ['t-shirt', 'tshirt', 't shirt', 'tee', 'tops']
        },
        'jeans': {
            'exact_matches': ['Jeans', 'Denim'],
            'exclude': ['Shorts', 'Trousers'],
            'keywords': ['jean', 'jeans', 'denim']
        },
        'dresses': {
            'exact_matches': ['Dresses', 'Gowns', 'Tunics'],
            'exclude': ['Tops', 'Shirts'],
            'keywords': ['dress', 'dresses', 'gown', 'tunic']
        }
    }

    # Color variations for better matching
    COLOR_VARIATIONS = {
        'black': ['black', 'charcoal', 'ebony'],
        'white': ['white', 'cream', 'ivory', 'off-white'],
        'blue': ['blue', 'navy', 'navy blue', 'royal blue', 'teal'],
        'red': ['red', 'maroon', 'burgundy', 'crimson'],
        'green': ['green', 'olive', 'mint', 'emerald'],
        'yellow': ['yellow', 'mustard', 'golden'],
        'pink': ['pink', 'rose', 'fuchsia'],
        'grey': ['grey', 'gray', 'silver', 'ash'],
        'brown': ['brown', 'tan', 'beige', 'khaki', 'coffee']
    }

    # Recommendation Engine Settings
    RECOMMENDATION_METHODS = ["visual_similarity", "category_based", "collaborative", "hybrid"]
    DEFAULT_RECOMMENDATION_METHOD = "hybrid"
    MIN_RECOMMENDATIONS = 4
    MAX_RECOMMENDATIONS = 12

    # Performance Settings
    USE_GPU = torch.cuda.is_available() if 'torch' in globals() else False
    NUM_WORKERS = 4
    PREFETCH_FACTOR = 2

    @classmethod
    def get_config_dict(cls) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return {
            key: value for key, value in cls.__dict__.items()
            if not key.startswith('_') and not callable(value)
        }

    @classmethod
    def validate_config(cls) -> bool:
        """Validate required configuration values."""
        required_keys = ['OPENAI_API_KEY']
        missing_keys = [key for key in required_keys if not getattr(cls, key)]

        if missing_keys:
            print(f"Warning: Missing required configuration keys: {missing_keys}")
            print("Please set them in your .env file")
            return False
        return True

    @classmethod
    def setup_kaggle_credentials(cls):
        """Setup Kaggle API credentials from environment variables."""
        if cls.KAGGLE_USERNAME and cls.KAGGLE_KEY:
            os.environ['KAGGLE_USERNAME'] = cls.KAGGLE_USERNAME
            os.environ['KAGGLE_KEY'] = cls.KAGGLE_KEY
            return True
        return False


# Create a singleton instance
config = Config()
