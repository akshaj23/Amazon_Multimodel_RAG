"""
CLIP model for generating multimodal embeddings
"""

import torch
import logging
from typing import List, Tuple, Union
from PIL import Image
import numpy as np

try:
    import clip
except ImportError:
    clip = None

logger = logging.getLogger(__name__)


class CLIPEmbeddingModel:
    """
    Generates embeddings for both text and images using CLIP model

    The CLIP model (Contrastive Language-Image Pre-training) aligns
    visual and textual data in a shared embedding space, making it ideal
    for multimodal retrieval and similarity matching.
    """

    def __init__(self, model_name: str = "ViT-B/32", device: str = "cuda"):
        """
        Initialize CLIP model

        Args:
            model_name: CLIP model variant (default: ViT-B/32)
            device: Device to load model on ("cuda" or "cpu")
        """
        if clip is None:
            raise ImportError("CLIP is not installed. Install with: pip install openai-clip")

        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_name = model_name

        logger.info(f"Loading CLIP model: {model_name} on {self.device}")

        try:
            self.model, self.preprocess = clip.load(model_name, device=self.device)
            self.model.eval()
            logger.info("CLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading CLIP model: {e}")
            raise

    def encode_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text

        Args:
            texts: Single text string or list of text strings

        Returns:
            Embeddings as numpy array [batch_size, embedding_dim]
        """
        if isinstance(texts, str):
            texts = [texts]

        logger.debug(f"Encoding {len(texts)} text samples")

        try:
            with torch.no_grad():
                text_tokens = clip.tokenize(texts, truncate=True).to(self.device)
                embeddings = self.model.encode_text(text_tokens)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                return embeddings.cpu().numpy()
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            raise

    def encode_image(self, images: Union[str, Image.Image, List[Union[str, Image.Image]]]) -> np.ndarray:
        """
        Generate embeddings for images

        Args:
            images: Image path, PIL Image, or list of images

        Returns:
            Embeddings as numpy array [batch_size, embedding_dim]
        """
        if isinstance(images, (str, Image.Image)):
            images = [images]

        logger.debug(f"Encoding {len(images)} image samples")

        processed_images = []

        try:
            for img in images:
                if isinstance(img, str):
                    # Load image from path
                    image = Image.open(img).convert("RGB")
                elif isinstance(img, Image.Image):
                    image = img.convert("RGB")
                else:
                    raise ValueError(f"Unsupported image type: {type(img)}")

                # Preprocess image
                processed_img = self.preprocess(image)
                processed_images.append(processed_img)

            # Stack images and move to device
            image_tensor = torch.stack(processed_images).to(self.device)

            with torch.no_grad():
                embeddings = self.model.encode_image(image_tensor)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                return embeddings.cpu().numpy()

        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            raise

    def encode_multimodal(self, texts: List[str], images: List[Union[str, Image.Image]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate embeddings for both text and images

        Args:
            texts: List of text descriptions
            images: List of image paths or PIL Images

        Returns:
            Tuple of (text_embeddings, image_embeddings)
        """
        if len(texts) != len(images):
            raise ValueError("Number of texts must match number of images")

        text_embeddings = self.encode_text(texts)
        image_embeddings = self.encode_image(images)

        return text_embeddings, image_embeddings

    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Similarity score (0 to 1)
        """
        embedding1 = embedding1.flatten()
        embedding2 = embedding2.flatten()

        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )

        return float(similarity)

    def get_text_image_similarity(self, text_embedding: np.ndarray, image_embedding: np.ndarray) -> float:
        """
        Calculate similarity between text and image embeddings

        Args:
            text_embedding: Text embedding
            image_embedding: Image embedding

        Returns:
            Cross-modal similarity score
        """
        return self.similarity(text_embedding, image_embedding)

    def batch_similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """
        Calculate pairwise similarities between two sets of embeddings

        Args:
            embeddings1: Array of embeddings [batch_size1, embedding_dim]
            embeddings2: Array of embeddings [batch_size2, embedding_dim]

        Returns:
            Similarity matrix [batch_size1, batch_size2]
        """
        # Normalize embeddings
        embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
        embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)

        # Compute cosine similarity
        similarities = np.dot(embeddings1, embeddings2.T)

        return similarities

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of generated embeddings

        Returns:
            Embedding dimension
        """
        # Use a dummy text to get embedding dimension
        dummy_embedding = self.encode_text("dummy")
        return dummy_embedding.shape[1]

    def save_model(self, path: str) -> None:
        """
        Save model state (for CLIP, this mainly saves configuration)

        Args:
            path: Path to save model
        """
        logger.info(f"Saving model configuration to {path}")
        torch.save({
            'model_name': self.model_name,
            'device': self.device,
        }, path)

    @staticmethod
    def load_model(path: str, device: str = "cuda") -> "CLIPEmbeddingModel":
        """
        Load CLIP model from saved configuration

        Args:
            path: Path to saved model config
            device: Device to load on

        Returns:
            CLIPEmbeddingModel instance
        """
        checkpoint = torch.load(path)
        return CLIPEmbeddingModel(
            model_name=checkpoint['model_name'],
            device=device
        )
