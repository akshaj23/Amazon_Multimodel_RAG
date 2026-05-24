# Project Structure

```text
GEN_AI_FINAL_Project/
├── build_embeddings.py              # Build CLIP embeddings and ChromaDB index
├── config.py                        # Project configuration
├── example_usage.py                 # Example code usage
├── main.py                          # Simple entry/demo script
├── requirements.txt                 # Python dependencies
├── test_clip_rag.py                 # Local tests/demo checks
├── README.md                        # Main GitHub documentation
├── QUICKSTART.md                    # Short setup instructions
├── GETTING_STARTED.md               # More detailed setup guide
├── IMPLEMENTATION.md                # Implementation notes
├── PROJECT_STRUCTURE.md             # This file
├── CONTRIBUTORS.md                  # Project contributor
│
├── data/
│   ├── __init__.py
│   ├── data_loader.py               # Dataset loading helpers
│   ├── preprocessing.py             # Amazon product preprocessing
│   ├── sample_products.py           # Small sample-data generator
│   ├── raw/                         # Local raw dataset files, ignored by Git
│   └── processed/                   # Local processed dataset files, ignored by Git
│
├── evaluation/
│   ├── __init__.py
│   └── metrics.py                   # Recall, precision, MRR, NDCG helpers
│
├── models/
│   ├── __init__.py
│   ├── clip_embeddings.py           # CLIP text/image embedding utilities
│   └── llm_integration.py           # Ollama-backed Llama 3.1 interface
│
├── rag/
│   ├── __init__.py
│   ├── chroma_store.py              # Persistent ChromaDB store
│   ├── retrieval.py                 # Retrieval system abstraction
│   └── vector_store.py              # JSON/in-memory fallback store
│
└── ui/
    ├── __init__.py
    └── streamlit_app.py             # Streamlit app
```

## Runtime Artifacts

These are generated locally and ignored by Git:

```text
data/raw/*
data/processed/*
chroma_db/
vector_store.json
.env
logs/
```

## Core Components

### Data Layer

`data/preprocessing.py` cleans Amazon product data and creates combined product descriptions from fields such as title, brand, category, features, description, price, image, and product URL.

### Embedding Layer

`build_embeddings.py` loads products, initializes CLIP ViT-B/32, generates 512-dimensional text embeddings, and stores them in ChromaDB.

### Retrieval Layer

`rag/chroma_store.py` wraps persistent ChromaDB search. The Streamlit app adds product-term filtering and keyword boosts for precise product-name queries.

### LLM Layer

`models/llm_integration.py` connects to Ollama at `http://localhost:11434` and uses `llama3.1` for RAG responses. Prompts instruct the model to use only retrieved product context.

### UI Layer

`ui/streamlit_app.py` provides three workflows:

- Text Query
- Image Query
- Combined Text + Image Query

Combined mode treats question text differently from product-description text so uploaded images remain the retrieval anchor when the user asks about the image.

## Data Flow

```text
Amazon product file
    -> preprocessing
    -> CLIP product embeddings
    -> ChromaDB
    -> user query/image
    -> CLIP query embedding
    -> retrieved products
    -> Llama 3.1 answer
    -> Streamlit display
```
