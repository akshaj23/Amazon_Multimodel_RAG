# Project Structure Overview

## Directory Tree

```
GEN_AI_FINAL_Project/
├── data/                              # Data handling and preprocessing
│   ├── __init__.py
│   ├── preprocessing.py              # Data cleaning and preparation
│   ├── data_loader.py                # Dataset loading utilities
│   ├── raw/                          # Raw dataset files
│   └── processed/                    # Processed dataset files
│
├── models/                            # Machine learning models
│   ├── __init__.py
│   ├── clip_embeddings.py            # CLIP embedding generation
│   ├── llm_integration.py            # LLM interface and prompt engineering
│   └── checkpoints/                  # Model weights
│
├── rag/                               # RAG (Retrieval-Augmented Generation)
│   ├── __init__.py
│   ├── vector_store.py               # Vector database management
│   ├── retrieval.py                  # Retrieval system
│   └── indices/                      # Vector indices
│
├── ui/                                # User interface
│   ├── __init__.py
│   └── streamlit_app.py              # Streamlit web application
│
├── evaluation/                        # Performance evaluation
│   ├── __init__.py
│   └── metrics.py                    # Evaluation metrics
│
├── notebooks/                         # Jupyter notebooks
│   └── exploration.ipynb             # Data exploration template
│
├── config.py                          # Configuration settings
├── main.py                            # Main entry point
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment variables template
├── .gitignore                         # Git ignore rules
├── README.md                          # Project documentation
├── IMPLEMENTATION.md                  # Implementation guide
└── PROJECT_STRUCTURE.md              # This file
```

## File Descriptions

### Core Configuration

- **config.py**: Central configuration for all components
  - Model names and parameters
  - API credentials paths
  - Prompt templates
  - Evaluation settings

- **requirements.txt**: All Python dependencies
  - PyTorch, Transformers
  - Streamlit for UI
  - Google Cloud libraries
  - Evaluation tools

### Data Module (`data/`)

- **preprocessing.py**: DataPreprocessor class
  - Load raw data
  - Clean text fields
  - Create product descriptions
  - Handle missing values
  - Validate data quality

- **data_loader.py**: DataLoader class
  - Load processed data
  - Search products
  - Filter by price/category
  - Batch products
  - Get dataset statistics

### Models Module (`models/`)

- **clip_embeddings.py**: CLIPEmbeddingModel class
  - Generate text embeddings
  - Generate image embeddings
  - Compute similarities
  - Normalize embeddings
  - Support batch processing

- **llm_integration.py**: LLMInterface class
  - Support multiple LLMs
  - Implement prompt strategies (zero-shot, few-shot, RAG)
  - Generate responses
  - Configure temperature and tokens

### RAG Module (`rag/`)

- **vector_store.py**: VectorStore class
  - Store embeddings in-memory
  - Search by similarity
  - Batch operations
  - Persistence (save/load)
  - Statistics tracking

- **retrieval.py**: RetrieverSystem class
  - Index products
  - Retrieve by text
  - Retrieve by image
  - Retrieve with multimodal queries
  - Format results
  - Evaluate performance

### UI Module (`ui/`)

- **streamlit_app.py**: Streamlit application
  - Text query interface
  - Image upload support
  - Chat history
  - Settings panel
  - Metrics display

### Evaluation Module (`evaluation/`)

- **metrics.py**: EvaluationMetrics class
  - Recall@K computation
  - Precision@K computation
  - MRR (Mean Reciprocal Rank)
  - NDCG (Normalized Discounted Cumulative Gain)
  - Summary reports

### Entry Points

- **main.py**: Main script
  - Data loading
  - Component initialization
  - Demo execution
  - Evaluation running

### Documentation

- **README.md**: Project overview and setup
- **IMPLEMENTATION.md**: Detailed implementation guide
- **PROJECT_STRUCTURE.md**: This file

## Module Dependencies

```
Streamlit UI
    ↓
    Main Components
    ├─ CLIP Embeddings (models/)
    ├─ Vector Store (rag/)
    ├─ Retrieval System (rag/)
    └─ LLM Interface (models/)
    
    Supporting Modules
    ├─ Data Loading (data/)
    ├─ Preprocessing (data/)
    └─ Evaluation (evaluation/)
```

## Configuration Files

### .env (Environment Variables)
```
GOOGLE_PROJECT_ID=...
HUGGINGFACE_TOKEN=...
OPENAI_API_KEY=...
LOG_LEVEL=INFO
DEBUG=False
```

### config.py (Project Configuration)
- CLIP model: `openai/clip-vit-base-patch32`
- LLM model: `meta-llama/Llama-2-7b-hf`
- Vector dimension: 512
- Top K retrieval: 5
- Temperature: 0.7

## Data Flow

```
1. Raw Data (Amazon Products)
   ↓
2. Preprocessing (Clean, validate)
   ↓
3. Embedding Generation (CLIP)
   ├─ Text embeddings
   └─ Image embeddings
   ↓
4. Vector Storage (VectorStore)
   ↓
5. User Query
   ├─ Text query → Generate embedding
   ├─ Image query → Generate embedding
   └─ Combined query → Weighted combination
   ↓
6. Retrieval (Find similar products)
   ↓
7. LLM Generation (Generate response)
   ↓
8. User Response (Display in UI)
```

## Key Classes and Methods

### DataPreprocessor
- `load_raw_data(filepath)`: Load JSON/CSV data
- `clean_text(text)`: Clean text fields
- `create_product_description(row)`: Combine attributes
- `preprocess(filepath, output_path)`: Full pipeline
- `get_statistics()`: Dataset statistics

### DataLoader
- `load_data(filepath)`: Load processed data
- `get_product_by_asin(asin)`: Get single product
- `get_batch(batch_size)`: Create batches
- `search_products(query)`: Text search
- `filter_by_price(min, max)`: Price filter

### CLIPEmbeddingModel
- `encode_text(texts)`: Generate text embeddings
- `encode_image(images)`: Generate image embeddings
- `encode_multimodal(texts, images)`: Both embeddings
- `similarity(emb1, emb2)`: Compute similarity
- `batch_similarity(embs1, embs2)`: Batch similarities

### VectorStore
- `add_embedding(id, embedding, metadata)`: Add single
- `add_embeddings(embeddings, metadata)`: Batch add
- `search(query_embedding, top_k)`: Find similar
- `delete_embedding(id)`: Remove
- `save(filepath)`: Persist

### RetrieverSystem
- `index_product(id, embedding, image, info)`: Index
- `retrieve_by_text(query, model, top_k)`: Text search
- `retrieve_by_image(image, model, top_k)`: Image search
- `retrieve_by_multimodal(text, image, model)`: Combined
- `format_retrieval_results(results)`: Format for LLM
- `evaluate_retrieval(queries, ground_truth)`: Evaluate

### LLMInterface
- `load_model()`: Load LLM
- `generate(prompt, strategy)`: Generate response
- `generate_with_strategy(question, strategy, context)`: With strategy
- `zero_shot_prompt(question)`: Zero-shot
- `few_shot_prompt(question, examples)`: Few-shot
- `rag_augmented_prompt(question, context)`: RAG

### EvaluationMetrics
- `compute_recall_at_k(retrieved, ground_truth, k)`: Recall
- `compute_precision_at_k(...)`: Precision
- `compute_mrr(...)`: Mean Reciprocal Rank
- `compute_ndcg(...)`: NDCG
- `evaluate_retrieval(...)`: Full evaluation

## Running the Project

### Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run ui/streamlit_app.py
```

### Full Pipeline
```bash
# Load data
python main.py --data data/raw/amazon_products.json

# With demo
python main.py --data data/raw/amazon_products.json --run-demo

# With logging
python main.py --log-file logs/chatbot.log --run-demo
```

### Development
```bash
# In notebooks/exploration.ipynb
# Experiment with components individually
jupyter notebook notebooks/exploration.ipynb
```

## Extension Points

1. **Vector Database**: Replace VectorStore with Google Vertex AI
2. **Models**: Swap CLIP with other encoders or LLMs
3. **UI**: Enhance Streamlit app with additional features
4. **Evaluation**: Add more metrics or custom evaluations
5. **Data**: Support additional data sources

## Performance Considerations

- Embedding dimension: 512 (configurable)
- Vector search: O(n) linear (can use approximate search)
- LLM inference: GPU recommended (8GB+ VRAM)
- Data size: Tested with 100K+ products

## Next Steps

1. Implement actual model loading
2. Connect to real vector database
3. Add authentication and multi-user support
4. Optimize for scale
5. Deploy to production
