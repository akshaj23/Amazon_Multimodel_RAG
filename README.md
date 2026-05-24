# Multimodal Conversational AI for E-commerce

A Vision-Language Approach to Customer Support

## Project Overview

This project develops a multimodal conversational chatbot capable of answering product-related questions using both text and image inputs. The system integrates Vision-Language models, Retrieval-Augmented Generation (RAG), and Large Language Models to provide comprehensive customer support for e-commerce platforms.

## Key Features

- **Multimodal Input Processing**: Handle both text queries and image uploads
- **Vision-Language Embeddings**: Use CLIP for text and image alignment
- **Retrieval System**: Google Vertex AI Vector Search for efficient data retrieval
- **Conversational Interface**: LLM integration (Meta-Llama-3.1 or Mixtral)
- **User-Friendly UI**: Streamlit-based interface
- **Performance Evaluation**: Retrieval accuracy and response quality metrics

## Project Components

### 1. Understanding Multimodal Data
- Analyze and preprocess the Amazon Product Dataset 2020
- Define optimal product attribute combinations
- Ensure data consistency and quality

### 2. Vision-Language RAG Implementation
- Generate embeddings using CLIP model
- Store embeddings in vector database
- Implement efficient retrieval system
- Evaluate with Recall@1, Recall@5, Recall@10 metrics

### 3. LLM Integration
- Integrate open-source LLM (Meta-Llama-3.1 or Mixtral)
- Implement prompt engineering (zero-shot, few-shot, multi-shot)
- Enable context-aware response generation

### 4. User Interface
- Build Streamlit application
- Support text and image input
- Display comprehensive responses

## Project Structure

```
├── data/                          # Data processing modules
│   ├── __init__.py
│   ├── preprocessing.py           # Data cleaning and preparation
│   ├── data_loader.py             # Dataset loading utilities
│   └── amazon_dataset.py          # Amazon Product Dataset 2020 handler
├── models/                        # Model implementations
│   ├── __init__.py
│   ├── clip_embeddings.py         # CLIP-based embedding generation
│   └── llm_integration.py         # LLM setup and integration
├── rag/                           # RAG system components
│   ├── __init__.py
│   ├── retrieval.py               # Retrieval mechanism
│   └── vector_store.py            # Vector database management
├── ui/                            # User interface
│   ├── __init__.py
│   └── streamlit_app.py           # Main Streamlit application
├── evaluation/                    # Evaluation metrics
│   ├── __init__.py
│   └── metrics.py                 # Retrieval and response metrics
├── notebooks/                     # Jupyter notebooks
│   └── exploration.ipynb          # Data exploration and testing
├── config.py                      # Configuration settings
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up API keys and credentials (if needed):
   - Google Cloud credentials for Vertex AI Vector Search
   - HuggingFace token for model access

## Usage

### Run the Chatbot UI

```bash
streamlit run ui/streamlit_app.py
```

### Example Interactions

**Text-based query:**
```
Query: "What are the features of the Samsung Galaxy S21?"
Response: "The Samsung Galaxy S21 comes with a 6.2-inch Dynamic AMOLED display, 
a triple-camera setup (12MP wide, 64MP telephoto, 12MP ultrawide), 
and a 4000mAh battery..."
```

**Image-based query:**
```
Upload product image → System identifies product and describes features/usage
```

## Dataset

- **Source**: Amazon Product Dataset 2020 (Kaggle)
- **Components**: Product images, descriptions, attributes
- **Focus**: Title, brand, price, features, images

## Technologies & Models

- **Embedding Model**: CLIP (Contrastive Language-Image Pre-training)
- **Vector Database**: Google Vertex AI Vector Search
- **Language Models**: Meta-Llama-3.1 or Mixtral
- **Framework**: Streamlit
- **Libraries**: PyTorch, Transformers, LangChain

## Evaluation Metrics

- **Retrieval Accuracy**: Correctness of retrieved items
- **Recall@K**: Recall@1, Recall@5, Recall@10
- **Response Quality**: Relevance and accuracy of generated responses
- **User Experience**: Interface usability and response time

## Expected Deliverables

- ✓ Fully functional multimodal chatbot
- ✓ Detailed documentation
- ✓ Streamlit UI interface
- ✓ Research report with findings
- ✓ Evaluation metrics and results

## References

1. Radford et al. - Learning Transferable Visual Models From Natural Language Supervision (CLIP)
2. Liu et al. - Vision-Language Alignment and Variance Adjustment (VLAVA)
3. Lewis et al. - Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
4. Li et al. - Pre-trained Vision and Language Transformer for Multimodal Understanding

## License

Academic use only

## Contact

For questions or issues, contact: akshajchandwani@uchicago.edu
