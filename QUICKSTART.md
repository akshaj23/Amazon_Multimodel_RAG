# Quick Start Guide

## Multimodal E-commerce Chatbot with CLIP + RAG

Get the chatbot running in 5 minutes!

## Prerequisites

- Python 3.9+
- NVIDIA GPU recommended (but CPU works too, just slower)

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Generate Sample Products

```bash
python data/sample_products.py
```

This creates a sample dataset with 10 products:
- Samsung Galaxy S21
- iPhone 14 Pro
- Fitbit Charge 5
- KitchenAid Stand Mixer
- Sony WF-1000XM4 Earbuds
- iPad Pro
- Canon EOS R5
- Sonos Move Speaker
- MacBook Pro
- DJI Air 2S Drone

## Step 3: Build CLIP Embeddings

```bash
python build_embeddings.py --generate-sample --output vector_store.json
```

This will:
1. Load the sample products
2. Initialize CLIP model (downloads on first run)
3. Generate embeddings for each product description
4. Store embeddings in `vector_store.json`

**Time**: ~2-5 minutes depending on GPU

## Step 4: Run the Streamlit App

```bash
streamlit run ui/streamlit_app.py
```

The app will open at `http://localhost:8501`

## Usage Examples

### Text Search

Try these queries:
- "What smartphones have 5G and great cameras?"
- "I need a fitness tracker"
- "Show me professional camera options"
- "What's the best mixer for baking?"

### Image Search

Upload an image of any product and the chatbot will:
1. Extract visual features using CLIP
2. Find similar products in the database
3. Display matches with similarity scores

### Combined Search

Combine text and image:
- Upload a smartphone image
- Add text: "I want something with better battery life"
- Results will be ranked by both visual and text similarity

## How It Works

```
User Input
    ↓
[Text] → CLIP Text Encoder → Embedding (512D)
[Image] → CLIP Image Encoder → Embedding (512D)
    ↓
    Combine Embeddings (with configurable weights)
    ↓
    Vector Search (find nearest products)
    ↓
    Return Top K Results with Similarity Scores
    ↓
    Display in Streamlit UI
```

## Key Features

### CLIP Model
- OpenAI's CLIP (ViT-B/32)
- 512-dimensional embeddings
- Aligns text and images in shared space
- No training needed - uses pre-trained weights

### Vector Store
- In-memory database with 10 products
- Cosine similarity search
- O(n) time complexity
- Can be scaled with production databases

### Streamlit UI
- 3 input modes: Text, Image, Combined
- Adjustable parameters (temperature, top_k)
- Chat history
- Real-time results
- Similarity score visualization

## Troubleshooting

### "CLIP not installed"
```bash
pip install openai-clip
```

### "Vector store not found"
```bash
python build_embeddings.py --generate-sample
```

### "CUDA out of memory"
The app automatically uses CPU if GPU memory is insufficient. No action needed.

### "Port 8501 already in use"
```bash
streamlit run ui/streamlit_app.py --server.port 8502
```

## Configuration

Edit `config.py` to customize:

```python
# Model settings
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# Retrieval settings
TOP_K_RETRIEVAL = 5  # Default number of results

# UI settings
STREAMLIT_PAGE_TITLE = "E-commerce Product Assistant"
```

## Scale to Production

For production with real Amazon dataset:

1. **Download dataset**
   ```bash
   kaggle datasets download -d promptcloud/amazon-product-data-2020
   ```

2. **Preprocess**
   ```python
   from data import DataPreprocessor
   
   preprocessor = DataPreprocessor()
   products = preprocessor.preprocess(
       "data/raw/amazon_products.json",
       "data/processed/amazon_products.json"
   )
   ```

3. **Build embeddings** (takes longer with millions of products)
   ```bash
   python build_embeddings.py --data data/processed/amazon_products.json
   ```

4. **Use production vector DB**
   - Replace VectorStore with Google Vertex AI Vector Search
   - Or use Pinecone, Weaviate, Milvus

## Next Steps

1. ✅ Try text searches
2. ✅ Try image uploads
3. ✅ Try combined queries
4. 🔄 Adjust temperature and top_k settings
5. 🚀 Integrate with real Amazon dataset

## API Reference

### Search by Text
```python
from models import CLIPEmbeddingModel
from rag import RetrieverSystem

clip_model = CLIPEmbeddingModel()
retriever = RetrieverSystem(vector_store)

results = retriever.retrieve_by_text(
    "smartphone with good camera",
    clip_model,
    top_k=5
)
# Returns: [(product_id, similarity_score, metadata), ...]
```

### Search by Image
```python
results = retriever.retrieve_by_image(
    "path/to/product.jpg",
    clip_model,
    top_k=5
)
```

### Multimodal Search
```python
results = retriever.retrieve_by_multimodal(
    query_text="smartphone",
    image_path="phone.jpg",
    embedding_model=clip_model,
    top_k=5,
    text_weight=0.5  # 50% text, 50% image
)
```

## Performance Metrics

After building embeddings, you'll see:
- **Recall@K**: What proportion of relevant items are in top K
- **Similarity Score**: How well each result matches (0-1)
- **Embedding Dimension**: 512 (CLIP's output)

## Community & Support

For issues or questions:
1. Check the `IMPLEMENTATION.md` guide
2. Review `PROJECT_STRUCTURE.md` for architecture
3. See `config.py` for all settings

## License

Academic use only - UChicago 2024
