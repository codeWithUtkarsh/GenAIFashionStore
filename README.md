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
- Python 3.8 or higher
- 4GB+ free disk space
- 8GB+ RAM recommended
- CUDA-capable GPU (optional, for better performance)

### Installation

1. **Clone the repository**
```bash
cd GenAIFashionStore
```

2. **Run the setup script**
```bash
python setup.py
```

This will:
- Create a virtual environment
- Install all dependencies
- Set up configuration files
- Create necessary directories
- Optionally download sample data

3. **Configure API Keys**

Edit the `.env` file with your API keys:
```env
OPENAI_API_KEY=your_openai_api_key_here
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

Get your keys from:
- OpenAI: https://platform.openai.com/api-keys
- Kaggle: https://www.kaggle.com/settings/account

4. **Run the application**
```bash
# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Run Streamlit app
streamlit run app.py
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
- **Backend**: Python 3.8+
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

### Optimization Tips
1. Use GPU acceleration when available
2. Enable caching for repeated operations
3. Start with smaller dataset for development
4. Use batch processing for embeddings
5. Implement pagination for large result sets

## 🔍 Troubleshooting

### Common Issues

**1. API Key Errors**
```
Solution: Ensure API keys are correctly set in .env file
```

**2. Memory Issues**
```
Solution: Reduce MAX_PRODUCTS_TO_LOAD or BATCH_SIZE in config.py
```

**3. Port Already in Use**
```
Solution: Change port with: streamlit run app.py --server.port 8502
```

**4. Dataset Download Fails**
```
Solution: Check Kaggle credentials and internet connection
```

**5. CUDA/GPU Errors**
```
Solution: Set USE_GPU = False in config.py to use CPU
```

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

### Custom UI Themes

Modify Streamlit theme in `config.py`:
```python
STREAMLIT_THEME = {
    "primaryColor": "#FF6B6B",
    "backgroundColor": "#FFFFFF",
    # ... more settings
}
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

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Kaggle** for providing the Fashion Product Images dataset
- **OpenAI** for CLIP and GPT models
- **Streamlit** for the amazing web framework
- **Hugging Face** for transformer models
- **ChromaDB** for vector database functionality

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Check the troubleshooting section
- Review the documentation

## 📝 Citation

If you use this project in your research or work, please cite:
```bibtex
@software{genai_fashion_store,
  title = {GenAI Fashion Store},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/GenAIFashionStore}
}
```

---

**Made with ❤️ using GenAI technologies**