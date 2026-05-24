"""
Configuration settings for the Multimodal E-commerce Chatbot
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ===== Project Settings =====
PROJECT_NAME = "Multimodal Conversational AI for E-commerce"
PROJECT_VERSION = "1.0.0"
DEBUG = False

# ===== Data Settings =====
DATASET_NAME = "Amazon Product Dataset 2020"
DATA_DIRECTORY = "./data/raw"
PROCESSED_DATA_DIRECTORY = "./data/processed"
CHUNK_SIZE = 256  # For text chunking

# ===== Model Settings =====
# CLIP Model Configuration
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_DEVICE = "cuda"  # or "cpu"

# LLM Configuration
LLM_MODEL_NAME = "meta-llama/Llama-2-7b-hf"  # or "mistralai/Mixtral-8x7B"
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 512
LLM_TOP_P = 0.9

# ===== RAG Settings =====
# Vector Database (Google Vertex AI)
GOOGLE_PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")
GOOGLE_REGION = "us-central1"
VECTOR_STORE_INDEX = "product-embeddings-index"
EMBEDDING_DIMENSION = 512
TOP_K_RETRIEVAL = 5

# ===== Prompt Templates =====
SYSTEM_PROMPT = """You are a helpful e-commerce product assistant. You have access to product information
from a comprehensive database. Answer customer questions about products accurately and helpfully.
If you don't know something, say so clearly."""

PROMPT_TEMPLATE_TEXT = """Context from product database:
{context}

Customer Question: {question}

Please provide a detailed and helpful answer based on the product information above."""

PROMPT_TEMPLATE_IMAGE = """I have found the following product information based on the image:
{context}

Customer Question: {question}

Please provide a detailed answer about this product."""

# ===== Evaluation Settings =====
RECALL_CUTOFFS = [1, 5, 10]
EVALUATION_METRICS = ["recall", "accuracy", "relevance"]

# ===== Streamlit UI Settings =====
STREAMLIT_PAGE_TITLE = "E-commerce Product Assistant"
STREAMLIT_PAGE_ICON = "🛍️"
STREAMLIT_LAYOUT = "wide"
MAX_UPLOAD_SIZE_MB = 10

# ===== Logging Settings =====
LOG_LEVEL = "INFO"
LOG_FILE = "./logs/chatbot.log"

# ===== Default Parameters =====
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 4
RANDOM_SEED = 42
