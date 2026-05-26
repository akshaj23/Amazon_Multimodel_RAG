# Amazon Multimodal RAG

Multimodal product search and question answering for the Amazon Product Dataset 2020. The app supports text queries, image queries, and combined text-image queries using CLIP embeddings, ChromaDB retrieval, and local Llama 3.1 responses through Ollama.

## Features

- Text search over Amazon product titles, categories, descriptions, features, and metadata
- Image-based product retrieval with CLIP image embeddings
- Combined text + image search with question-aware behavior
- ChromaDB persistent vector store
- RAG answers generated with Llama 3.1 through Ollama
- Product cards with title, ASIN, brand, price, category, features, image, and product link
- Keyword-aware retrieval boosts for product-name queries such as `monopoly`
- Local execution with no API key required

## Tech Stack

- **UI**: Streamlit
- **Vision-language model**: CLIP ViT-B/32
- **Vector database**: ChromaDB
- **LLM**: Llama 3.1 served locally with Ollama
- **Core libraries**: PyTorch, NumPy, Pandas, Pillow, Requests
- **Dataset**: Amazon Product Dataset 2020 from Kaggle

## Project Structure

```text
.
├── build_embeddings.py          # Builds CLIP embeddings and ChromaDB index
├── config.py                    # App, model, and retrieval settings
├── data/
│   ├── data_loader.py           # Product loading utilities
│   ├── preprocessing.py         # Amazon dataset cleaning/preprocessing
│   ├── raw/                     # Local raw data, ignored by Git
│   └── processed/               # Local processed data, ignored by Git
├── evaluation/
│   └── metrics.py               # Retrieval evaluation metrics
├── models/
│   ├── clip_embeddings.py       # CLIP embedding helpers
│   └── llm_integration.py       # Ollama/Llama 3.1 interface and prompts
├── rag/
│   ├── chroma_store.py          # Persistent ChromaDB vector store
│   ├── retrieval.py             # Retrieval system helpers
│   └── vector_store.py          # JSON/in-memory fallback vector store
└── ui/
    └── streamlit_app.py         # Streamlit application
```

See [ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md) for the full system architecture diagram.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Install and start Ollama

Install Ollama from `https://ollama.com`, then pull Llama 3.1:

```bash
ollama pull llama3.1
ollama serve
```

The app uses `http://localhost:11434` by default.

### 3. Add the Amazon dataset

Download the Kaggle dataset:

```bash
kaggle datasets download -d promptcloud/amazon-product-dataset-2020
```

Place the downloaded product file under `data/raw/`. The repository ignores raw and processed data files so large dataset files are not committed.

### 4. Build embeddings

Use ChromaDB for the final app:

```bash
python build_embeddings.py --data data/processed/amazon_products_2020.json --store chroma
```

If starting from a CSV or parquet dataset file, `build_embeddings.py` can preprocess it through `data/preprocessing.py`:

```bash
python build_embeddings.py --data data/raw/your_amazon_file.csv --store chroma
```

For a small demo only:

```bash
python build_embeddings.py --generate-sample --store chroma
```

### 5. Run the app

```bash
streamlit run ui/streamlit_app.py
```

Open:

```text
http://localhost:8501
```

## How It Works

1. Product metadata is cleaned and combined into searchable descriptions.
2. CLIP generates 512-dimensional embeddings for product text.
3. ChromaDB stores embeddings and product metadata persistently in `chroma_db/`.
4. User text or uploaded images are encoded with CLIP.
5. ChromaDB retrieves the nearest products.
6. Retrieval logic applies keyword/product-name boosts where needed.
7. Llama 3.1 receives the retrieved product context and generates the final answer.

## Query Modes

### Text Query

Use this for product-name or natural-language searches:

```text
show me monopoly and how it is played
which headphones are best
what is the price of this product
```

Text retrieval combines CLIP similarity with product keyword filtering when a query contains specific product terms.

### Image Query

Upload a product image. The app uses CLIP image embeddings to retrieve visually similar Amazon products.

### Combined Query

Upload an image and add text. If the text is a question, the image is used for retrieval and the text is used as the Llama question. If the text describes a desired product, the app blends text and image embeddings.

## Retrieval Metrics

This project now includes a random-sample retrieval evaluation file at `evaluation/random_sample_recall.py`.
It tests whether the retrieval index can find the correct product when the query comes from known product data.

What the evaluator does:

- Randomly samples products from the dataset, with `50` samples by default.
- Uses each sampled product title as a text query.
- Optionally creates harder title-derived queries, such as shortened titles, brand/category queries, natural-language queries, and feature-only queries.
- Uses each sampled product image as a photo query when an image URL or path is available.
- Cross-checks retrieved product IDs against the sampled product's ASIN.
- Computes average Recall@1, Recall@5, Recall@10, MRR, and NDCG.
- Saves both aggregate metrics and per-query retrieval details for inspection.

Run a random 50-product recall check for product title and image queries:

```bash
python evaluation/random_sample_recall.py \
  --data data/processed/amazon_products_2020.json \
  --sample-size 50 \
  --mode all
```

The script reports averaged Recall@1, Recall@5, Recall@10, MRR, and NDCG, then writes aggregate metrics to `evaluation/results/random_sample_recall_metrics.json` and per-query details to `evaluation/results/random_sample_recall_details.csv`.
If image URLs cannot be downloaded, photo queries are counted as skipped instead of being averaged as failed retrievals.

Useful variants:

```bash
# Title-only recall test
python evaluation/random_sample_recall.py \
  --data data/processed/amazon_products_2020.json \
  --sample-size 50 \
  --mode title

# Harder title-derived recall test
python evaluation/random_sample_recall.py \
  --data data/processed/amazon_products_2020.json \
  --sample-size 50 \
  --mode hard-title

# Photo-only recall test
python evaluation/random_sample_recall.py \
  --data data/processed/amazon_products_2020.json \
  --sample-size 50 \
  --mode photo

# Use custom recall cutoffs
python evaluation/random_sample_recall.py \
  --data data/processed/amazon_products_2020.json \
  --sample-size 50 \
  --mode all \
  --k-values 1,3,5,10
```

Before running the evaluator, build the vector store with the same dataset:

```bash
python build_embeddings.py \
  --data data/processed/amazon_products_2020.json \
  --store chroma
```

## Configuration

Important settings are in `config.py`:

```python
LLM_MODEL_NAME = "llama3.1"
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 220
TOP_K_RETRIEVAL = 5
```

Local overrides can be placed in `.env`, which is ignored by Git.

## Notes

- `data/raw/`, `data/processed/`, and `chroma_db/` are ignored by Git.
- Ollama must be running for Llama 3.1 answers.
- If Ollama is unavailable, the UI falls back to metadata-based answers so retrieval can still be tested.
- Search quality depends on the products indexed in ChromaDB.

## License

Academic project use.
