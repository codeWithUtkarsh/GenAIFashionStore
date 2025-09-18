"""
Image embedding module using CLIP and other models for visual similarity search.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional, Union
import logging
from pathlib import Path
from tqdm import tqdm
import pickle
import hashlib

from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
import torchvision.transforms as transforms

from config import Config

# Setup logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format=Config.LOG_FORMAT
)
logger = logging.getLogger(__name__)


class FashionImageDataset(Dataset):
    """Custom dataset for fashion product images."""

    def __init__(self, image_paths: List[str], transform=None):
        """
        Initialize the dataset.

        Args:
            image_paths: List of image file paths
            transform: Image transformations to apply
        """
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, image_path


class CLIPEmbedder:
    """CLIP-based image and text embedding generator."""

    def __init__(self, model_name: str = None, device: str = None):
        """
        Initialize CLIP embedder.

        Args:
            model_name: CLIP model name
            device: Device to use (cuda/cpu)
        """
        self.model_name = model_name or Config.CLIP_MODEL_NAME
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        logger.info(f"Initializing CLIP model: {self.model_name} on {self.device}")

        # Load CLIP model and processor
        self.model = CLIPModel.from_pretrained(self.model_name)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        # Cache directory for embeddings
        self.cache_dir = Config.CACHE_DIR / "embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info("CLIP model initialized successfully")

    def get_image_embedding(self, image: Union[str, Image.Image, torch.Tensor]) -> np.ndarray:
        """
        Generate embedding for a single image.

        Args:
            image: Image path, PIL Image, or tensor

        Returns:
            np.ndarray: Image embedding
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')

        # Process image
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate embedding
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        return image_features.cpu().numpy().squeeze()

    def get_text_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for text.

        Args:
            text: Input text

        Returns:
            np.ndarray: Text embedding
        """
        # Process text
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate embedding
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

        return text_features.cpu().numpy().squeeze()

    def get_batch_embeddings(self,
                            image_paths: List[str],
                            batch_size: int = 32,
                            use_cache: bool = True) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for multiple images in batches.

        Args:
            image_paths: List of image file paths
            batch_size: Batch size for processing
            use_cache: Whether to use cached embeddings

        Returns:
            Dict[str, np.ndarray]: Mapping of image paths to embeddings
        """
        embeddings = {}

        # Check cache first
        if use_cache:
            cached_embeddings = self._load_cached_embeddings(image_paths)
            embeddings.update(cached_embeddings)

            # Filter out already processed images
            remaining_paths = [p for p in image_paths if p not in embeddings]

            if not remaining_paths:
                logger.info(f"All {len(image_paths)} embeddings loaded from cache")
                return embeddings

            logger.info(f"Processing {len(remaining_paths)} new images ({len(embeddings)} from cache)")
            image_paths = remaining_paths

        # Process remaining images
        dataset = FashionImageDataset(
            image_paths,
            transform=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]
                )
            ])
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True if self.device == 'cuda' else False
        )

        # Generate embeddings
        for batch_images, batch_paths in tqdm(dataloader, desc="Generating embeddings"):
            batch_images = batch_images.to(self.device)

            with torch.no_grad():
                # Get image features directly from preprocessed tensors
                image_features = self.model.vision_model(batch_images)[1]
                image_features = self.model.visual_projection(image_features)
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

                # Store embeddings
                for path, embedding in zip(batch_paths, image_features.cpu().numpy()):
                    embeddings[path] = embedding

        # Cache new embeddings
        if use_cache:
            self._save_cached_embeddings(embeddings)

        return embeddings

    def _get_cache_key(self, image_path: str) -> str:
        """Generate cache key for an image path."""
        return hashlib.md5(f"{image_path}_{self.model_name}".encode()).hexdigest()

    def _load_cached_embeddings(self, image_paths: List[str]) -> Dict[str, np.ndarray]:
        """Load cached embeddings for given image paths."""
        cached = {}

        for path in image_paths:
            cache_key = self._get_cache_key(path)
            cache_file = self.cache_dir / f"{cache_key}.pkl"

            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        cached[path] = pickle.load(f)
                except Exception:
                    continue

        return cached

    def _save_cached_embeddings(self, embeddings: Dict[str, np.ndarray]):
        """Save embeddings to cache."""
        for path, embedding in embeddings.items():
            cache_key = self._get_cache_key(path)
            cache_file = self.cache_dir / f"{cache_key}.pkl"

            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(embedding, f)
            except Exception:
                continue

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            float: Cosine similarity score
        """
        # Normalize embeddings
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)

        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2)

        return float(similarity)

    def find_similar_images(self,
                           query_embedding: np.ndarray,
                           database_embeddings: Dict[str, np.ndarray],
                           top_k: int = 10,
                           threshold: float = 0.0) -> List[Tuple[str, float]]:
        """
        Find similar images based on embedding similarity.

        Args:
            query_embedding: Query image embedding
            database_embeddings: Database of image embeddings
            top_k: Number of top results to return
            threshold: Minimum similarity threshold

        Returns:
            List[Tuple[str, float]]: List of (image_path, similarity_score) tuples
        """
        similarities = []

        # Normalize query embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Compute similarities
        for path, db_embedding in database_embeddings.items():
            similarity = self.compute_similarity(query_embedding, db_embedding)

            if similarity >= threshold:
                similarities.append((path, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]


class HybridEmbedder:
    """Combines multiple embedding models for enhanced similarity search."""

    def __init__(self):
        """Initialize hybrid embedder with multiple models."""
        self.clip_embedder = CLIPEmbedder()

        # Optional: Add sentence transformer for text
        try:
            self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence transformer initialized")
        except Exception as e:
            logger.warning(f"Could not load sentence transformer: {e}")
            self.text_model = None

    def get_hybrid_embedding(self,
                            image_path: str,
                            text_description: str = None) -> np.ndarray:
        """
        Generate hybrid embedding combining visual and textual features.

        Args:
            image_path: Path to image
            text_description: Optional text description

        Returns:
            np.ndarray: Hybrid embedding
        """
        # Get image embedding
        image_embedding = self.clip_embedder.get_image_embedding(image_path)

        if text_description and self.text_model:
            # Get text embedding
            text_embedding = self.text_model.encode(text_description)

            # Combine embeddings (weighted average)
            hybrid_embedding = 0.7 * image_embedding + 0.3 * text_embedding

            # Normalize
            hybrid_embedding = hybrid_embedding / np.linalg.norm(hybrid_embedding)

            return hybrid_embedding

        return image_embedding


def test_embedder():
    """Test the embedding module."""
    # Initialize embedder
    embedder = CLIPEmbedder()

    # Test text embedding
    text = "A blue shirt for men"
    text_embedding = embedder.get_text_embedding(text)
    print(f"Text embedding shape: {text_embedding.shape}")

    # Test similarity computation
    text2 = "A red dress for women"
    text_embedding2 = embedder.get_text_embedding(text2)

    similarity = embedder.compute_similarity(text_embedding, text_embedding2)
    print(f"Similarity between '{text}' and '{text2}': {similarity:.3f}")


if __name__ == "__main__":
    test_embedder()
