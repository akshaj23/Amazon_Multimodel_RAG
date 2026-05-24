# Quickstart

This is the shortest path to run the final Amazon multimodal RAG app.

## 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

## 2. Start Llama 3.1 with Ollama

```bash
ollama pull llama3.1
ollama serve
```

Keep Ollama running while using the Streamlit app.

## 3. Prepare product data

Download the Amazon Product Dataset 2020 from Kaggle:

```bash
kaggle datasets download -d promptcloud/amazon-product-dataset-2020
```

Place the dataset file in `data/raw/`.

If you already have a processed JSONL file, place it at:

```text
data/processed/amazon_products_2020.json
```

## 4. Build the ChromaDB index

```bash
python build_embeddings.py --data data/processed/amazon_products_2020.json --store chroma
```

For CSV or parquet input:

```bash
python build_embeddings.py --data data/raw/your_amazon_file.csv --store chroma
```

For a tiny demo dataset only:

```bash
python build_embeddings.py --generate-sample --store chroma
```

## 5. Run Streamlit

```bash
streamlit run ui/streamlit_app.py
```

Open:

```text
http://localhost:8501
```

## What To Try

Text query:

```text
show me monopoly and how it is played
```

Image query:

Upload a product image to retrieve visually similar Amazon products.

Combined query:

Upload a product image and ask:

```text
what is the price
is this suitable for kids
what are the features
```

In combined mode, question text is used for the Llama answer while the image drives retrieval.

## Troubleshooting

If Llama answers are unavailable:

```bash
ollama serve
ollama pull llama3.1
```

If Streamlit says embeddings are missing:

```bash
python build_embeddings.py --data data/processed/amazon_products_2020.json --store chroma
```

If port `8501` is busy:

```bash
streamlit run ui/streamlit_app.py --server.port 8502
```
