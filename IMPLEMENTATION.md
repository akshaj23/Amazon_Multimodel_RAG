# Implementation Guide

## Multimodal Conversational AI for E-commerce

This document provides a detailed implementation guide for the project.

## Table of Contents

1. [Setup](#setup)
2. [Data Preparation](#data-preparation)
3. [Component Implementation](#component-implementation)
4. [Integration](#integration)
5. [Evaluation](#evaluation)
6. [Deployment](#deployment)

## Setup

### Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your credentials
```

### API Credentials

Configure the following in `.env`:

```
GOOGLE_PROJECT_ID=your-project-id
HUGGINGFACE_TOKEN=your-token
OPENAI_API_KEY=your-api-key
```

## Data Preparation

### 1. Download Dataset

Download the Amazon Product Dataset 2020 from Kaggle:

```python
# Using Kaggle CLI
kaggle datasets download -d promptcloud/amazon-product-data-2020

# Or download manually and place in data/raw/
```

### 2. Preprocess Data

```python
from data import DataPreprocessor

preprocessor = DataPreprocessor()
processed_data = preprocessor.preprocess(
    filepath="data/raw/amazon_products.json",
    output_path="data/processed/amazon_products.json"
)

# View statistics
stats = preprocessor.get_statistics()
print(stats)
```

Key preprocessing steps:
- Remove duplicates
- Clean text fields
- Handle missing values
- Combine attributes into descriptions
- Validate data quality

## Component Implementation

### 1. Embedding Generation (CLIP)

```python
from models import CLIPEmbeddingModel

# Initialize CLIP model
clip_model = CLIPEmbeddingModel(
    model_name="ViT-B/32",
    device="cuda"
)

# Generate text embeddings
text_embeddings = clip_model.encode_text([
    "Samsung Galaxy S21 smartphone",
    "Apple AirPods Pro",
    "KitchenAid mixer"
])

# Generate image embeddings
image_embeddings = clip_model.encode_image([
    "path/to/product1.jpg",
    "path/to/product2.jpg"
])

# Get embedding dimension
dim = clip_model.get_embedding_dimension()  # Returns 512
```

**Key Features:**
- Aligns text and images in shared embedding space
- 512-dimensional embeddings
- Supports batch processing
- GPU acceleration

### 2. Vector Storage

```python
from rag import VectorStore

# Create vector store
vector_store = VectorStore(dimension=512)

# Add embeddings
vector_store.add_embedding(
    embedding_id="asin_123",
    embedding=text_embedding,
    metadata={
        "title": "Samsung Galaxy S21",
        "price": 799,
        "brand": "Samsung"
    }
)

# Search
results = vector_store.search(query_embedding, top_k=5)
# Returns: [(product_id, similarity_score, metadata), ...]

# Save for later use
vector_store.save("vector_store.json")
```

**Key Features:**
- In-memory storage (can scale to Google Vertex AI)
- Cosine similarity search
- Metadata support
- Persistence to disk

### 3. Retrieval System

```python
from rag import RetrieverSystem

retriever = RetrieverSystem(vector_store)

# Index products
retriever.index_product(
    product_id="asin_123",
    text_embedding=text_emb,
    image_embedding=image_emb,
    product_info={"title": "...", "price": ...}
)

# Retrieve by text
results = retriever.retrieve_by_text(
    query="what is the best smartphone?",
    embedding_model=clip_model,
    top_k=5
)

# Retrieve by image
results = retriever.retrieve_by_image(
    image_path="product.jpg",
    embedding_model=clip_model,
    top_k=5
)

# Multimodal retrieval
results = retriever.retrieve_by_multimodal(
    query_text="smartphone under 500",
    image_path="reference_phone.jpg",
    embedding_model=clip_model,
    top_k=5
)

# Format results for LLM
context = retriever.format_retrieval_results(results)
```

**Key Features:**
- Text, image, and multimodal retrieval
- Flexible weighting for combined queries
- Result formatting for LLM context
- Evaluation capabilities

### 4. Large Language Model Integration

```python
from models import LLMInterface, PromptStrategy

llm = LLMInterface(
    model_name="meta-llama/Llama-2-7b-hf",
    temperature=0.7,
    max_tokens=512
)

# Load model (requires sufficient GPU memory)
llm.load_model()

# Zero-shot prompting
response = llm.generate_with_strategy(
    question="What are the features of this product?",
    strategy=PromptStrategy.ZERO_SHOT
)

# Few-shot prompting
examples = [
    {
        "question": "What is this product?",
        "answer": "This is a smartphone with..."
    }
]
response = llm.generate_with_strategy(
    question="Describe this product",
    strategy=PromptStrategy.FEW_SHOT,
    examples=examples
)

# RAG-augmented generation
context = retriever.format_retrieval_results(retrieved_products)
response = llm.generate_with_strategy(
    question="What are the specs?",
    strategy=PromptStrategy.RAG_AUGMENTED,
    context=context
)
```

**Key Features:**
- Multiple prompt engineering strategies
- Context-aware generation
- RAG integration
- Temperature and token control

### 5. Streamlit User Interface

```bash
# Run the Streamlit app
streamlit run ui/streamlit_app.py

# Access at http://localhost:8501
```

**Features:**
- Text input for queries
- Image upload support
- Multimodal input combination
- Chat history
- Real-time responses
- Performance metrics display

## Integration

### Full Pipeline Example

```python
import numpy as np
from data import DataLoader
from models import CLIPEmbeddingModel, LLMInterface, PromptStrategy
from rag import VectorStore, RetrieverSystem
from evaluation import EvaluationMetrics

# 1. Load data
loader = DataLoader("data/processed/amazon_products.json")
products = loader.get_sample(n=1000)

# 2. Initialize embedding model
clip_model = CLIPEmbeddingModel()

# 3. Create vector store
vector_store = VectorStore(dimension=512)
retriever = RetrieverSystem(vector_store)

# 4. Index products
for product in products:
    # Generate embeddings
    text_emb = clip_model.encode_text(product['combined_description'])
    if text_emb.ndim > 1:
        text_emb = text_emb[0]
    
    # Index
    retriever.index_product(
        product_id=product['asin'],
        text_embedding=text_emb,
        product_info=product
    )

# 5. Initialize LLM
llm = LLMInterface()
llm.load_model()

# 6. Query system
query = "What smartphones have good cameras?"

# Retrieve
retrieved = retriever.retrieve_by_text(query, clip_model, top_k=5)
context = retriever.format_retrieval_results(retrieved)

# Generate response
response = llm.generate_with_strategy(
    question=query,
    strategy=PromptStrategy.RAG_AUGMENTED,
    context=context
)

print(f"Query: {query}")
print(f"Response: {response}")
```

## Evaluation

### Retrieval Evaluation

```python
from evaluation import EvaluationMetrics

evaluator = EvaluationMetrics()

# Test queries and ground truth
test_queries = ["smartphone", "fitness tracker", "kitchen mixer"]
ground_truth = [
    ["samsung_s21", "iphone_14", "pixel_6"],
    ["fitbit_charge", "apple_watch"],
    ["kitchenaid", "bosch_mixer"]
]

# Evaluate
metrics = evaluator.evaluate_retrieval(
    retrieved_list=[...],  # Retrieved items for each query
    ground_truth_list=ground_truth,
    k_values=[1, 5, 10]
)

print(f"Recall@5: {metrics['recall@5']:.2%}")
print(f"Precision@5: {metrics['precision@5']:.2%}")
print(f"MRR: {metrics['mrr']:.4f}")
print(f"NDCG: {metrics['ndcg']:.4f}")
```

**Key Metrics:**
- Recall@K: Proportion of relevant items in top K
- Precision@K: Proportion of top K results that are relevant
- MRR: Mean Reciprocal Rank
- NDCG: Normalized Discounted Cumulative Gain

### Response Evaluation

```python
# Evaluate response relevance
response_quality = evaluator.evaluate_response_quality(
    predictions=[generated_response],
    references=[expected_response],
    metric="accuracy"
)

# Get comprehensive report
report = evaluator.get_summary_report()
print(report)
```

## Deployment

### Local Deployment

```bash
# Run Streamlit app
streamlit run ui/streamlit_app.py

# Run main script
python main.py --data data/processed/amazon_products.json --run-demo
```

### Production Deployment

For production, consider:

1. **Vector Database**: Use Google Vertex AI Vector Search instead of in-memory storage
2. **Model Serving**: Deploy LLM using vLLM or TensorRT
3. **Scaling**: Use containerization (Docker) and orchestration (Kubernetes)
4. **Monitoring**: Implement logging and metrics tracking
5. **Caching**: Cache embeddings and frequent queries

Example Dockerfile:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["streamlit", "run", "ui/streamlit_app.py"]
```

## Troubleshooting

### GPU Memory Issues

If you encounter CUDA out of memory errors:

```python
# Reduce batch size
clip_model = CLIPEmbeddingModel(device="cuda")

# Or use CPU
clip_model = CLIPEmbeddingModel(device="cpu")
```

### Model Loading Issues

Ensure you have the correct HuggingFace token:

```bash
huggingface-cli login
```

### Vector Store Performance

For large-scale deployments (>1M embeddings), use Google Vertex AI Vector Search:

```python
# Replace VectorStore with Google Vertex AI
from google.cloud import aiplatform

# Configure with your project ID and index name
# Implementation details in production deployment
```

## References

- [CLIP Paper](https://arxiv.org/abs/2103.14030)
- [RAG Paper](https://arxiv.org/abs/2005.11401)
- [Llama 2 Paper](https://arxiv.org/abs/2307.09288)
- [Streamlit Documentation](https://docs.streamlit.io/)
