# Getting Started with CLIP + RAG System

## What You Have

A fully functional **multimodal e-commerce chatbot** that uses:

- **CLIP** (OpenAI's Contrastive Language-Image Pre-training)
  - Generates 512-dimensional embeddings
  - Aligns text and images in shared space
  - No training needed - uses pre-trained model

- **RAG** (Retrieval-Augmented Generation)
  - Vector database for fast similarity search
  - Retrieves relevant products based on queries
  - Supports text, image, and multimodal searches

- **Streamlit UI**
  - Interactive web interface
  - Real-time search results
  - Similarity score visualization

## 🚀 Quick Start (5 minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Sample Data
```bash
python data/sample_products.py
```
Creates 10 sample products (smartphones, cameras, fitness trackers, etc.)

### 3. Build Embeddings
```bash
python build_embeddings.py --generate-sample --output vector_store.json
```
Generates CLIP embeddings for all products (~2-3 minutes)

### 4. Run the App
```bash
streamlit run ui/streamlit_app.py
```

Visit `http://localhost:8501` 🎉

## How It Works

### Text Search
```
User Query: "What smartphones have good cameras?"
    ↓
CLIP Text Encoder
    ↓
512D Embedding
    ↓
Vector Search (find nearest products)
    ↓
Return Top 5 Results with Similarity Scores
    ↓
Display in UI
```

### Image Search
```
User Uploads Image
    ↓
CLIP Image Encoder
    ↓
512D Embedding
    ↓
Vector Search
    ↓
Display Similar Products
```

### Multimodal Search
```
Text + Image Inputs
    ↓
Generate Both Embeddings
    ↓
Weighted Combination (configurable)
    ↓
Vector Search
    ↓
Display Results
```

## Usage Examples

### Try These Searches

**Text queries:**
- "smartphone with 5G and great camera"
- "fitness tracker for health monitoring"
- "professional camera for photography"
- "portable wireless speaker"
- "laptop for work and gaming"

**Image uploads:**
- Any smartphone image → finds similar phones
- Any camera image → finds similar cameras
- Any product image → finds similar products

**Combined:**
- Upload image + "I want something cheaper"
- Upload image + "Show me newer models"

## Project Files

### Core Implementation
- `data/sample_products.py` - Sample product dataset
- `data/preprocessing.py` - Data cleaning and preparation
- `data/data_loader.py` - Load and manage products
- `models/clip_embeddings.py` - CLIP embedding generation
- `models/llm_integration.py` - LLM interface (for future use)
- `rag/vector_store.py` - Vector database (512D)
- `rag/retrieval.py` - Product retrieval system
- `ui/streamlit_app.py` - Web interface
- `evaluation/metrics.py` - Performance metrics

### Build & Test
- `build_embeddings.py` - Generate embeddings from products
- `test_clip_rag.py` - Automated test suite
- `example_usage.py` - Usage examples

### Configuration
- `config.py` - All settings
- `.env.example` - Environment variables
- `requirements.txt` - Dependencies

### Documentation
- `README.md` - Full project overview
- `QUICKSTART.md` - 5-minute quick start
- `IMPLEMENTATION.md` - Detailed guide
- `PROJECT_STRUCTURE.md` - Architecture
- `GETTING_STARTED.md` - This file

## Key Capabilities

✅ **Text Search**
- Query products by description
- Semantic understanding (not keyword matching)
- Fast retrieval from database

✅ **Image Search**
- Upload any product image
- CLIP extracts visual features
- Finds visually similar products

✅ **Multimodal Search**
- Combine text and image
- Adjustable text/image weights
- Best of both worlds

✅ **Product Details**
- Title, brand, price
- Features and descriptions
- Similarity scores

✅ **Real-time Performance**
- Instant search results
- Similarity scores (0-100%)
- No latency issues

## Architecture

```
┌─────────────────┐
│  User Input     │
│ (Text/Image)    │
└────────┬────────┘
         │
    ┌────▼─────┐
    │   CLIP   │ ← OpenAI's Vision-Language Model
    │  Model   │   (Pre-trained, 512D embeddings)
    └────┬─────┘
         │
    ┌────▼──────────┐
    │ Embedding     │
    │ (512D vector) │
    └────┬──────────┘
         │
    ┌────▼────────────┐
    │ Vector Search   │ ← Cosine Similarity
    │ (VectorStore)   │
    └────┬────────────┘
         │
    ┌────▼──────────┐
    │  Top K        │
    │  Results      │
    └────┬──────────┘
         │
    ┌────▼─────────┐
    │  Streamlit   │
    │  Display     │
    └──────────────┘
```

## Performance

- **Search latency**: <100ms (after CLIP encoding)
- **CLIP encoding**: 1-2 seconds per query
- **Vector search**: O(n) - linear scan
- **Similarity range**: 0.0-1.0 (0-100%)

### Scaling Notes

Current system:
- 10 sample products
- ~1 MB memory for embeddings
- Single-machine, CPU or GPU

For production (millions of products):
- Use Google Vertex AI Vector Search
- Or Pinecone, Weaviate, Milvus
- Add approximate nearest neighbor (HNSW)

## Configuration

Edit `config.py` to customize:

```python
# CLIP Model
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_DEVICE = "cuda"  # or "cpu"

# Retrieval
TOP_K_RETRIEVAL = 5  # Default results

# UI
STREAMLIT_PAGE_TITLE = "E-commerce Product Assistant"
MAX_UPLOAD_SIZE_MB = 10
```

## Troubleshooting

### CLIP not loading
```bash
pip install openai-clip
```

### Vector store not found
```bash
python build_embeddings.py --generate-sample
```

### Port already in use
```bash
streamlit run ui/streamlit_app.py --server.port 8502
```

### GPU memory issues
The app automatically uses CPU if GPU is unavailable.

## Testing

Run the automated test suite:
```bash
python test_clip_rag.py
```

Run usage examples:
```bash
python example_usage.py
```

## Next Steps

1. **Explore the system**
   - Try different queries
   - Upload your own images
   - Adjust parameters

2. **Understand the code**
   - Read `IMPLEMENTATION.md`
   - Check `example_usage.py`
   - Review `PROJECT_STRUCTURE.md`

3. **Scale to production**
   - Download real Amazon dataset
   - Preprocess products
   - Build embeddings (takes longer)
   - Deploy to cloud

4. **Add features**
   - Add filters (price, brand, category)
   - Implement feedback loop
   - Add chat history persistence
   - Integrate with LLM for detailed responses

## API Reference

### Simple Text Search
```python
from models import CLIPEmbeddingModel
from rag import RetrieverSystem

clip = CLIPEmbeddingModel()
retriever = RetrieverSystem(vector_store)

# Search
results = retriever.retrieve_by_text(
    "smartphone",
    clip,
    top_k=5
)

# Results: [(product_id, similarity, metadata), ...]
for product_id, sim, meta in results:
    print(f"{meta['title']}: {sim*100:.1f}%")
```

### Image Search
```python
results = retriever.retrieve_by_image(
    "phone.jpg",
    clip,
    top_k=5
)
```

### Combined Search
```python
results = retriever.retrieve_by_multimodal(
    query_text="smartphone",
    image_path="phone.jpg",
    embedding_model=clip,
    top_k=5,
    text_weight=0.7  # 70% text, 30% image
)
```

## Support & Learning

- **Issues**: Check QUICKSTART.md first
- **Code examples**: See example_usage.py
- **Architecture**: Read PROJECT_STRUCTURE.md
- **Implementation details**: See IMPLEMENTATION.md

## Key Concepts

### CLIP
- Contrastive learning between text and images
- Maps both modalities to same embedding space
- Enables cross-modal search

### Embeddings
- 512-dimensional vectors
- Normalized to unit length
- Represent semantic meaning

### Vector Search
- Cosine similarity between embeddings
- Fast retrieval of nearest neighbors
- O(n) complexity for exact search

### RAG Pattern
- Retrieve relevant context
- Pass to LLM for generation
- Improves answer quality

## Resources

- [CLIP Paper](https://arxiv.org/abs/2103.14030)
- [CLIP GitHub](https://github.com/openai/CLIP)
- [Streamlit Docs](https://docs.streamlit.io/)
- [Vector Databases](https://www.pinecone.io/)

---

**Ready to go!** Start with `streamlit run ui/streamlit_app.py` 🚀
