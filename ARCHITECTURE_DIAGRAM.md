# Architecture Diagram

```mermaid
flowchart TD
    A["Amazon Product Dataset 2020<br/>Kaggle"] --> B["Data Preprocessing<br/>data/preprocessing.py"]
    B --> C["Processed Product Records<br/>title, brand, price, category,<br/>features, description, image URL"]
    C --> D["CLIP Text Embedding Builder<br/>build_embeddings.py"]
    D --> E["ChromaDB Vector Store<br/>rag/chroma_store.py"]

    U["User<br/>Streamlit UI"] --> M{"Input Mode"}

    M --> T["Text Query"]
    M --> I["Image Query"]
    M --> X["Combined Text + Image Query"]

    T --> T1["Normalize Query + Extract Product Terms<br/>ui/streamlit_app.py"]
    T1 --> T2["CLIP Text Encoder<br/>ViT-B/32"]
    T2 --> R["ChromaDB Retrieval<br/>Semantic Similarity + Keyword Boost"]

    I --> I1["Upload Product Image"]
    I1 --> I2["CLIP Image Encoder<br/>ViT-B/32"]
    I2 --> R

    X --> X1{"Is text a question?"}
    X1 -->|Yes| X2["Use image for retrieval<br/>Use text as LLM question"]
    X1 -->|No| X3["Blend text + image embeddings<br/>using text/image weight"]
    X2 --> I2
    X3 --> R

    E --> R
    R --> P["Top K Retrieved Products<br/>ASIN, title, price, category,<br/>features, image, product URL"]

    P --> L["RAG Prompt Builder<br/>models/llm_integration.py"]
    L --> O["Ollama Local Server<br/>localhost:11434"]
    O --> G["Llama 3.1"]
    G --> A2["Grounded Product Answer"]

    P --> C2["Retrieved Product Cards"]
    A2 --> S["Streamlit Response Page"]
    C2 --> S

    S --> U
```

## Component Summary

| Layer | Component | Purpose |
| --- | --- | --- |
| Data | Amazon Product Dataset 2020 | Product metadata and image URLs |
| Preprocessing | `data/preprocessing.py` | Cleans fields and builds combined product descriptions |
| Embeddings | CLIP ViT-B/32 | Creates shared text/image embeddings |
| Vector DB | ChromaDB | Stores product embeddings and metadata |
| Retrieval | `rag/chroma_store.py`, `ui/streamlit_app.py` | Finds similar products and applies keyword boosts |
| LLM | Llama 3.1 through Ollama | Generates grounded RAG answers |
| UI | Streamlit | Provides text, image, and combined query workflows |

## Runtime Flow

1. Product data is preprocessed into clean records.
2. CLIP creates product text embeddings.
3. ChromaDB stores embeddings and metadata.
4. The user submits text, image, or combined input in Streamlit.
5. CLIP encodes the query.
6. ChromaDB retrieves the most relevant products.
7. Llama 3.1 receives retrieved metadata as context.
8. Streamlit displays the answer and product cards.
