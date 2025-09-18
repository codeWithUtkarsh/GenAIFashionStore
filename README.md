# 🛍️ GenAI Fashion Store

An AI-powered fashion shopping assistant that uses computer vision, natural language processing, and recommendation algorithms to provide a personalized shopping experience. Built with Streamlit, OpenAI's CLIP, and GPT models.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-brightgreen.svg)

## ✨ Features

### 🔍 **Smart Search Capabilities**
- **Text-based Search**: Find products using natural language queries
- **Visual Search**: Upload an image to find similar fashion items
- **Hybrid Search**: Combine text and image search for precise results
- **Advanced Filtering**: Filter by gender, category, color, season, and usage

### 🤖 **AI Shopping Assistant**
- Natural language interaction powered by GPT
- Style advice and fashion tips
- Outfit coordination suggestions
- Product information and recommendations

### 🎯 **Intelligent Recommendations**
- **Visual Similarity**: Find products with similar visual features using CLIP embeddings
- **Category-based**: Discover items from related categories
- **Collaborative Filtering**: Get recommendations based on user behavior
- **Hybrid Approach**: Combines multiple strategies for best results
- **Complementary Items**: Find accessories and items that complete an outfit

### 📊 **Advanced Backend**
- Vector database (ChromaDB) for efficient similarity search
- CLIP embeddings for visual feature extraction
- Caching system for improved performance
- Batch processing for large datasets

## 🚀 Quick Start

### Prerequisites
- Python 3.8 - 3.12 (Note: Python 3.13 may have compatibility issues)
- 4GB+ free disk space
- 8GB+ RAM recommended
- CUDA-capable GPU (optional, for better performance)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/codeWithUtkarsh/GenAIFashionStore
cd GenAIFashionStore
```

2. **Create and activate virtual environment**
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# OR
.venv\Scripts\activate  # On Windows
```

3. **Install dependencies**

For stable installation (recommended):
```bash
# Use stable requirements to avoid compatibility issues
pip install -r requirements_stable.txt
```

For latest versions:
```bash
pip install -r requirements.txt
```

4. **Run the setup script**
```bash
python setup.py
```

This will:
- Verify your Python version
- Set up configuration files
- Create necessary directories
- Guide you through API key configuration

5. **Configure API Keys**

Edit the `.env` file with your API keys:
```env
OPENAI_API_KEY=your_openai_api_key_here
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

Get your keys from:
- OpenAI: https://platform.openai.com/api-keys
- Kaggle: https://www.kaggle.com/settings/account

### Running the Application

#### Safe Start (Recommended for First Run)
```bash
# Use the safe runner script to handle common issues
python run_safe.py --clear-cache
```

#### Normal Start
```bash
streamlit run app.py
```

#### Custom Options
```bash
# Run on different port
python run_safe.py --port 8502

# Enable debug mode
python run_safe.py --debug

# Clear all caches before starting
python run_safe.py --clear-cache
```

5. **Open in browser**
Navigate to: http://localhost:8501

## 📁 Project Structure

```
GenAIFashionStore/
│
├── app.py                  # Main Streamlit application
├── config.py               # Configuration settings
├── setup.py                # Setup script
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
│
├── data_downloader.py      # Kaggle dataset downloader
├── image_embedder.py       # CLIP-based image embedding
├── vector_database.py      # ChromaDB vector storage
├── recommendation_engine.py # Multi-strategy recommendations
├── genai_assistant.py      # GPT-powered chat assistant
│
├── data/                   # Dataset directory
│   ├── images/            # Product images
│   └── styles.csv         # Product metadata
│
├── models/                 # Saved models
├── vector_db/             # Vector database storage
├── chroma_db/             # ChromaDB persistence
└── .cache/                # Embedding cache
```

## 🎯 Usage Guide

### Searching for Products

1. **Text Search**: Type queries like:
   - "blue shirt for men"
   - "summer dresses"
   - "formal shoes"

2. **Visual Search**: Upload an image of a fashion item to find similar products

3. **Filters**: Use sidebar filters to narrow down results by:
   - Gender
   - Category
   - Color
   - Season
   - Usage/Occasion

### Using the AI Assistant

Ask questions like:
- "What goes well with blue jeans?"
- "Show me outfit ideas for a summer wedding"
- "I need business casual recommendations"
- "What's trending this season?"

### Getting Recommendations

1. Click on any product to view details
2. System automatically shows:
   - Similar items
   - Complementary pieces
   - Complete outfit suggestions

## 🔧 Configuration

### Adjusting Dataset Size

Edit `config.py` to change the number of products loaded:
```python
MAX_PRODUCTS_TO_LOAD = 2000  # Adjust this value
```

### Model Selection

Configure embedding models in `config.py`:
```python
EMBEDDING_MODEL = "clip-ViT-B-32"  # Options: clip-ViT-B-32, clip-ViT-L-14
GPT_MODEL = "gpt-4-turbo-preview"  # Options: gpt-3.5-turbo, gpt-4
```

### Performance Tuning

```python
# Enable GPU acceleration (if available)
USE_GPU = True

# Adjust batch size for processing
BATCH_SIZE = 32

# Configure cache settings
ENABLE_CACHE = True
CACHE_TTL = 3600  # Cache duration in seconds
```

## 🗃️ Dataset

This project uses the **Fashion Product Images (Small)** dataset from Kaggle:
- **Source**: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small
- **Size**: ~44,000 products
- **Contents**: Product images, categories, attributes, and metadata

### Dataset Features
- Product images in multiple categories
- Detailed metadata (gender, category, color, season, usage)
- Structured CSV format for easy processing

## 🛠️ Technical Stack

### Core Technologies
- **Frontend**: Streamlit
- **Backend**: Python 3.8-3.12
- **Database**: ChromaDB (Vector Database)
- **AI/ML**: OpenAI CLIP, GPT, Sentence Transformers

### Key Libraries
- `streamlit`: Web application framework
- `torch` & `torchvision`: Deep learning
- `transformers`: Hugging Face models
- `chromadb`: Vector database
- `openai`: GPT integration
- `Pillow`: Image processing
- `pandas` & `numpy`: Data manipulation

## 📊 Performance Considerations

### System Requirements
- **Minimum**: 4GB RAM, 2 CPU cores
- **Recommended**: 8GB RAM, 4+ CPU cores, CUDA GPU
- **Storage**: 5-10GB for dataset and models

## 🚀 Advanced Features

### Adding Custom Models

1. Implement embedding interface in `image_embedder.py`
2. Update `config.py` with model configuration
3. Modify vector database dimensions if needed

### Extending Recommendations

Add new recommendation strategies in `recommendation_engine.py`:
```python
def _custom_recommendations(self, ...):
    # Your custom logic here
    pass
```


## 📈 Future Enhancements

- [ ] User authentication and profiles
- [ ] Purchase history tracking
- [ ] Size and fit recommendations
- [ ] Virtual try-on using AR
- [ ] Multi-language support
- [ ] Price tracking and alerts
- [ ] Social features (sharing, reviews)
- [ ] Mobile app version
- [ ] Integration with e-commerce platforms


**Made with ❤️ for AffinityLabs**