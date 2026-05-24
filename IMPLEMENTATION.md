# Implementation Notes

## Final Architecture

The final app uses a local multimodal RAG pipeline:

```text
Amazon Product Dataset 2020
        |
        v
Product preprocessing
        |
        v
CLIP ViT-B/32 text embeddings
        |
        v
ChromaDB vector index
        |
        v
Text / image / combined query
        |
        v
Retrieved product metadata
        |
        v
Llama 3.1 through Ollama
        |
        v
Streamlit answer and product cards
```

## Data Processing

The data pipeline accepts JSONL, CSV, or parquet input. CSV and parquet files are processed with `data/preprocessing.py`; JSONL files are loaded directly.

Important product fields:

- `asin`
- `title`
- `brand`
- `price`
- `category`
- `features`
- `description`
- `combined_description`
- `image`
- `product_url`

Raw data, processed data, and ChromaDB files are local runtime artifacts and are not committed.

## Embeddings

`build_embeddings.py` initializes CLIP ViT-B/32 and creates normalized 512-dimensional embeddings from product descriptions.

Final command:

```bash
python build_embeddings.py --data data/processed/amazon_products_2020.json --store chroma
```

The `--store chroma` option builds the persistent ChromaDB index used by the Streamlit app.

## Retrieval

The app loads `chroma_db/` first. If ChromaDB is unavailable, it can fall back to `vector_store.json`.

Retrieval uses:

- CLIP text embeddings for text queries
- CLIP image embeddings for image queries
- Weighted text-image embeddings for descriptive combined queries
- Image-only retrieval for combined image + question workflows
- Keyword/product-term filtering for queries with specific product names

This improves cases where CLIP alone is too broad, such as:

```text
show me monopoly and how it is played
```

## LLM Integration

`models/llm_integration.py` uses Ollama with Llama 3.1:

```python
LLM_MODEL_NAME = "llama3.1"
OLLAMA_BASE_URL = "http://localhost:11434"
```

The prompt tells Llama to answer only from retrieved product data and to treat Product 1 as the primary product unless the user asks for comparison or recommendations.

Ollama setup:

```bash
ollama pull llama3.1
ollama serve
```

## Streamlit UI

`ui/streamlit_app.py` contains the final user workflow:

- Text Query: text embedding plus product keyword boost
- Image Query: image embedding retrieval
- Combined Query: question-aware handling of text + image

The UI displays:

- Llama answer
- Answer source
- Retrieved product cards
- Match/similarity score
- Image, ASIN, brand, price, category, features, and product URL

## Configuration

Important final settings in `config.py`:

```python
LLM_MODEL_NAME = "llama3.1"
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 220
TOP_K_RETRIEVAL = 5
EMBEDDING_DIMENSION = 512
```

The lower temperature keeps answers grounded and reduces unrelated product mixing.

## Verification

Basic syntax check:

```bash
python -m py_compile ui/streamlit_app.py models/llm_integration.py config.py
```

Run the app:

```bash
streamlit run ui/streamlit_app.py
```
