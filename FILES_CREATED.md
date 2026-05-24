# Repository Files

This repository contains the final code and documentation for the Amazon multimodal RAG project.

## Core Files

```text
build_embeddings.py
config.py
main.py
example_usage.py
test_clip_rag.py
requirements.txt
```

## Data Module

```text
data/
├── __init__.py
├── data_loader.py
├── preprocessing.py
├── sample_products.py
├── raw/.gitkeep
└── processed/.gitkeep
```

## Model Module

```text
models/
├── __init__.py
├── clip_embeddings.py
└── llm_integration.py
```

## RAG Module

```text
rag/
├── __init__.py
├── chroma_store.py
├── retrieval.py
└── vector_store.py
```

## UI Module

```text
ui/
├── __init__.py
└── streamlit_app.py
```

## Evaluation Module

```text
evaluation/
├── __init__.py
└── metrics.py
```

## Documentation

```text
README.md
QUICKSTART.md
GETTING_STARTED.md
IMPLEMENTATION.md
PROJECT_STRUCTURE.md
CONTRIBUTORS.md
FILES_CREATED.md
```

## Ignored Local Artifacts

The following are intentionally not committed:

```text
data/raw/*
data/processed/*
chroma_db/
vector_store.json
.env
logs/
__pycache__/
```

These files can be regenerated locally from the Kaggle dataset and embedding build command.
