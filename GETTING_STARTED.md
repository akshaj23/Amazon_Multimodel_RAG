# Getting Started

## What This Project Does

This project is a multimodal Amazon product assistant. It retrieves products with CLIP embeddings and ChromaDB, then uses Llama 3.1 through Ollama to answer questions from the retrieved product context.

The app supports:

- Text product search
- Image-based product search
- Combined image + question workflows
- RAG answers grounded in retrieved product metadata

## Requirements

- Python 3.9+
- Ollama
- Llama 3.1 model pulled locally
- Amazon Product Dataset 2020 data file

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Start Ollama:

```bash
ollama pull llama3.1
ollama serve
```

Build embeddings:

```bash
python build_embeddings.py --data data/processed/amazon_products_2020.json --store chroma
```

Run the app:

```bash
streamlit run ui/streamlit_app.py
```

Open `http://localhost:8501`.

## Data

The intended dataset is:

```text
https://www.kaggle.com/datasets/promptcloud/amazon-product-dataset-2020
```

Raw and processed dataset files are intentionally ignored by Git:

```text
data/raw/
data/processed/
chroma_db/
```

This keeps the repository lightweight and avoids committing large generated data.

## System Flow

```text
Amazon product metadata
        |
        v
Preprocessing and combined product descriptions
        |
        v
CLIP text embeddings
        |
        v
ChromaDB persistent vector store
        |
        v
Text, image, or combined user query
        |
        v
CLIP query embedding and retrieval
        |
        v
Retrieved product context
        |
        v
Llama 3.1 RAG answer
```

## Query Modes

### Text Query

Text queries use CLIP text embeddings. Product-name words are also extracted and used to improve retrieval for specific searches such as:

```text
show me monopoly and how it is played
```

### Image Query

Image queries use CLIP image embeddings to retrieve visually similar products.

### Combined Query

Combined mode handles two cases:

- If the text is a question, the image drives retrieval and the text is sent to Llama as the question.
- If the text is a product description, the app blends text and image embeddings.

## Main Files

- `ui/streamlit_app.py`: Streamlit UI and query workflow
- `build_embeddings.py`: CLIP embedding build pipeline
- `rag/chroma_store.py`: ChromaDB vector store
- `models/llm_integration.py`: Ollama/Llama 3.1 integration
- `data/preprocessing.py`: Dataset preprocessing
- `config.py`: Model, retrieval, and UI settings

## Common Commands

Run app:

```bash
streamlit run ui/streamlit_app.py
```

Build ChromaDB embeddings:

```bash
python build_embeddings.py --data data/processed/amazon_products_2020.json --store chroma
```

Generate sample data for testing only:

```bash
python build_embeddings.py --generate-sample --store chroma
```

Check Ollama:

```bash
curl http://localhost:11434/api/tags
```
