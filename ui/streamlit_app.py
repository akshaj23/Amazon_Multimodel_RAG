"""
Streamlit UI for the Multimodal E-commerce Chatbot
"""

import streamlit as st
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import logging
import re

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    LLM_MODEL_NAME,
    STREAMLIT_PAGE_TITLE,
    STREAMLIT_PAGE_ICON,
    STREAMLIT_LAYOUT,
    PROMPT_TEMPLATE_TEXT,
    PROMPT_TEMPLATE_IMAGE
)

logger = logging.getLogger(__name__)

QUERY_STOPWORDS = {
    "about",
    "and",
    "are",
    "best",
    "can",
    "could",
    "describe",
    "does",
    "for",
    "give",
    "have",
    "how",
    "include",
    "includes",
    "is",
    "it",
    "me",
    "played",
    "price",
    "show",
    "tell",
    "the",
    "this",
    "what",
    "which",
    "with",
    "would",
    "you",
}


def get_first_image_url(metadata):
    """Return the first real product image URL from a pipe-delimited image field."""
    image_field = metadata.get("image", "")
    if not image_field:
        return None

    for image_url in str(image_field).split("|"):
        image_url = image_url.strip()
        if image_url and "transparent-pixel" not in image_url:
            return image_url

    return None


def format_price(price):
    """Format numeric prices without crashing on empty or malformed values."""
    try:
        return f"${float(price):.2f}"
    except (TypeError, ValueError):
        return "N/A"


def summarize_features(metadata, max_chars=450):
    """Return a compact feature/description summary for answer text."""
    text = metadata.get("features") or metadata.get("description") or ""
    text = str(text).replace(" | ", ". ").strip()
    if not text:
        return "No detailed features are available in the dataset for this product."
    if len(text) > max_chars:
        return f"{text[:max_chars].rsplit(' ', 1)[0]}..."
    return text


def answer_product_question(question, results):
    """Generate a concise RAG answer from the top retrieved product metadata."""
    if not results:
        return "I could not find a matching product in the indexed dataset."

    product_id, similarity, metadata = results[0]
    title = metadata.get("title") or "this product"
    brand = metadata.get("brand") or "N/A"
    price = format_price(metadata.get("price"))
    category = metadata.get("category") or "N/A"
    features = summarize_features(metadata)
    question_lower = (question or "").lower().strip()

    if not question_lower:
        return (
            f"This looks most similar to **{title}**. "
            f"The listed price is **{price}**, and it is categorized as **{category}**."
        )

    if any(term in question_lower for term in ["price", "cost", "how much", "sell", "selling"]):
        return f"The listed price for **{title}** is **{price}**."

    if "brand" in question_lower or "who makes" in question_lower:
        return f"The listed brand for **{title}** is **{brand}**."

    if "category" in question_lower or "type" in question_lower:
        return f"**{title}** is categorized as **{category}**."

    if any(term in question_lower for term in ["feature", "spec", "include", "come with"]):
        return f"Key details for **{title}**: {features}"

    if any(term in question_lower for term in ["use", "usage", "how do i", "what does it do"]):
        return f"**{title}** is used as a product in the **{category}** category. Details from the dataset: {features}"

    if any(term in question_lower for term in ["name", "identify", "what is this", "which product"]):
        return f"This appears to be **{title}**."

    if any(term in question_lower for term in ["best", "recommend", "which"]):
        return (
            f"The best match in the indexed data is **{title}** "
            f"with a similarity score of **{similarity * 100:.1f}%**. "
            f"It is priced at **{price}**."
        )

    return (
        f"Based on the top retrieved product, this is **{title}**. "
        f"Price: **{price}**. Category: **{category}**. Details: {features}"
    )


def results_to_llm_products(results):
    """Convert retrieval tuples into the format expected by LLMInterface."""
    products = []
    for product_id, similarity, metadata in results:
        product_metadata = dict(metadata)
        product_metadata.setdefault("asin", product_id)
        product_metadata["similarity"] = f"{similarity * 100:.1f}%"
        products.append({"metadata": product_metadata})
    return products


@st.cache_resource
def load_llm():
    """Load the local Llama 3.1 interface if Ollama is available."""
    try:
        from models.llm_integration import LLMInterface

        llm = LLMInterface()
        if not llm.is_available():
            return None
        llm.load_model()
        return llm
    except Exception as exc:
        logger.warning("LLM unavailable, falling back to metadata answers: %s", exc)
        return None


def generate_rag_answer(question, results):
    """Generate an LLM-grounded answer, falling back to deterministic metadata QA."""
    fallback_answer = answer_product_question(question, results)
    llm = st.session_state.get("llm")

    if llm is None:
        return fallback_answer, "metadata fallback"

    try:
        question_lower = (question or "").lower()
        multi_product_terms = ["compare", "similar", "options", "recommend", "best", "which"]
        context_results = results[:3] if any(term in question_lower for term in multi_product_terms) else results[:1]
        answer = llm.generate_product_response(
            question or "Describe the product.",
            results_to_llm_products(context_results),
        )
        if answer:
            return answer, "llama3.1"
    except Exception as exc:
        logger.warning("LLM answer failed, falling back to metadata answer: %s", exc)

    return fallback_answer, "metadata"


def normalize_query(query):
    """Clean common product-query typos before embedding and filtering."""
    replacements = {
        "heaphone": "headphone",
        "heaphones": "headphones",
        "earbud": "earbud",
        "airpod": "airpod",
    }

    normalized = query
    for typo, correction in replacements.items():
        normalized = re.sub(rf"\b{typo}\b", correction, normalized, flags=re.IGNORECASE)

    return normalized


def is_product_question(text):
    """Return True when text is a question about an uploaded/retrieved product."""
    text_lower = (text or "").lower().strip()
    if not text_lower:
        return False

    question_terms = [
        "what",
        "which",
        "how",
        "why",
        "is",
        "are",
        "can",
        "do",
        "does",
        "should",
        "would",
        "could",
        "tell me",
        "describe",
        "price",
        "cost",
        "brand",
        "feature",
        "spec",
        "include",
        "category",
    ]
    starts_like_question = re.match(
        r"^(what|which|how|why|is|are|can|do|does|should|would|could|tell me|describe)\b",
        text_lower,
    )
    return text_lower.endswith("?") or bool(starts_like_question) or any(
        f" {term} " in f" {text_lower} " for term in question_terms
    )


def get_product_type_terms(query):
    """Return domain terms used for deterministic filtering on narrow product asks."""
    query_lower = query.lower()

    if re.search(r"\b(headphones?|headsets?|earbuds?|earphones?|airpods?)\b", query_lower):
        return [
            "headphone",
            "headphones",
            "headset",
            "headsets",
            "earbud",
            "earbuds",
            "earphone",
            "earphones",
            "airpod",
            "airpods",
        ]

    return []


def get_query_product_terms(query):
    """Extract likely product-name terms from a natural language query."""
    terms = get_product_type_terms(query)
    if terms:
        return terms

    words = re.findall(r"[a-z0-9]+", query.lower())
    product_terms = []
    for word in words:
        if len(word) < 3 or word in QUERY_STOPWORDS:
            continue
        if word not in product_terms:
            product_terms.append(word)

    return product_terms[:4]


def metadata_text(metadata):
    """Create searchable text from product metadata."""
    fields = [
        metadata.get("title", ""),
        metadata.get("brand", ""),
        metadata.get("category", ""),
        metadata.get("features", ""),
        metadata.get("description", ""),
    ]
    return " ".join(str(field) for field in fields if field).lower()


def search_products(query_embedding, query_text, top_k):
    """Search vector store with optional product-type filtering for precise asks."""
    vector_store = st.session_state.vector_store
    terms = get_query_product_terms(query_text)

    if not terms:
        return vector_store.search(query_embedding, top_k=top_k)

    if hasattr(vector_store, "search_product_terms"):
        return vector_store.search_product_terms(query_embedding, terms, top_k=top_k)

    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
    primary_candidates = []
    fallback_candidates = []

    for product_id, embedding in vector_store.embeddings.items():
        metadata = vector_store.metadata.get(product_id, {})
        text = metadata_text(metadata)
        if not any(term in text for term in terms):
            continue

        similarity = float(np.dot(query_norm, embedding))
        title_category = f"{metadata.get('title', '')} {metadata.get('category', '')}".lower()
        result = (product_id, similarity, metadata)
        if any(term in title_category for term in terms):
            primary_candidates.append((product_id, similarity + 0.12, metadata))
        else:
            fallback_candidates.append((product_id, similarity + 0.02, metadata))

    candidates = primary_candidates or fallback_candidates

    if not candidates:
        return vector_store.search(query_embedding, top_k=top_k)

    candidates.sort(key=lambda result: result[1], reverse=True)
    return candidates[:top_k]


def render_product_result(index, product_id, similarity, metadata, score_label="Match"):
    """Render a retrieved product with image, metadata, and score."""
    col1, col2, col3 = st.columns([1.2, 3, 1])

    with col1:
        image_url = get_first_image_url(metadata)
        if image_url:
            st.image(image_url, use_column_width=True)
        else:
            st.caption("No image available")

    with col2:
        st.write(f"### {index}. {metadata.get('title', 'Unknown')}")
        st.write(f"**ASIN:** {product_id}")
        st.write(f"**Brand:** {metadata.get('brand') or 'N/A'}")
        st.write(f"**Price:** {format_price(metadata.get('price'))}")
        st.write(f"**Category:** {metadata.get('category') or 'N/A'}")

        features = metadata.get("features") or "N/A"
        if len(str(features)) > 700:
            features = f"{str(features)[:700]}..."
        st.write(f"**Features:** {features}")

        product_url = metadata.get("product_url")
        if product_url:
            st.link_button("Open product page", product_url)

    with col3:
        st.metric(score_label, f"{similarity * 100:.1f}%")

    st.divider()


@st.cache_resource
def load_clip_model():
    """Load CLIP model once"""
    try:
        import clip
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        return model, preprocess, device
    except ImportError:
        st.error("CLIP not installed. Install with: pip install openai-clip")
        return None, None, None


@st.cache_resource
def load_vector_store():
    """Load vector store once"""
    chroma_path = "chroma_db"
    vector_store_path = "vector_store.json"

    try:
        if Path(chroma_path).exists():
            from rag import ChromaVectorStore
            vector_store = ChromaVectorStore(persist_directory=chroma_path)
            if vector_store.get_size() > 0:
                return vector_store

        if Path(vector_store_path).exists():
            from rag import VectorStore
            vector_store = VectorStore()
            vector_store.load(vector_store_path)
            return vector_store

        st.warning(
            "Vector store not found. Please run: "
            "python build_embeddings.py --data data/processed/amazon_products_2020.json --store chroma"
        )
        return None
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        return None


def init_session_state():
    """Initialize Streamlit session state"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'retrieved_products' not in st.session_state:
        st.session_state.retrieved_products = []
    if 'clip_model' not in st.session_state:
        st.session_state.clip_model, st.session_state.preprocess, st.session_state.device = load_clip_model()
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = load_vector_store()
    if 'llm' not in st.session_state:
        st.session_state.llm = load_llm()


def configure_page():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title=STREAMLIT_PAGE_TITLE,
        page_icon=STREAMLIT_PAGE_ICON,
        layout=STREAMLIT_LAYOUT,
        initial_sidebar_state="expanded"
    )


def render_sidebar():
    """Render sidebar with controls and information"""
    with st.sidebar:
        st.title("⚙️ Settings")

        st.markdown("---")

        # Model settings
        st.subheader("Model Configuration")
        llm_status = "Connected" if st.session_state.get("llm") else "Unavailable - metadata fallback"
        st.caption(f"LLM: {LLM_MODEL_NAME} ({llm_status})")

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Controls response creativity (0=deterministic, 1=random)"
        )

        top_k = st.slider(
            "Top K Results",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of products to retrieve"
        )

        st.markdown("---")

        # Input mode
        st.subheader("Input Mode")
        input_mode = st.radio(
            "Choose input type",
            ["Text Query", "Image Query", "Combined (Text + Image)"]
        )

        st.markdown("---")

        # About
        st.subheader("ℹ️ About")
        st.markdown(
            """
            **Multimodal E-commerce Chatbot**

            This chatbot uses Vision-Language models to understand
            and answer questions about products using both text
            and images.

            **Technologies:**
            - CLIP for embeddings
            - Vector search for retrieval
            - LLM for responses
            - Streamlit for UI
            """
        )

        return temperature, top_k, input_mode


def render_chat_interface():
    """Render main chat interface"""
    st.title(f"{STREAMLIT_PAGE_ICON} E-commerce Product Assistant")

    st.markdown(
        """
        Welcome to our multimodal chatbot! Ask questions about products using:
        - **Text**: Describe what you're looking for
        - **Images**: Upload product images
        - **Both**: Combine text and image queries for better results
        """
    )

    st.markdown("---")

    # Display chat history
    if st.session_state.chat_history:
        st.subheader("Chat History")
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.write(f"👤 **You:** {message['content']}")
            else:
                st.write(f"🤖 **Assistant:** {message['content']}")
            st.markdown("---")


def render_text_query_interface(temperature, top_k):
    """Render text query input interface"""
    st.subheader("📝 Text Query")

    user_query = st.text_input(
        "Ask a question about products",
        placeholder="E.g., What are the features of Samsung Galaxy S21?"
    )

    if st.button("🔍 Search", key="text_search"):
        if user_query:
            if st.session_state.clip_model is None or st.session_state.vector_store is None:
                st.error("❌ Models not loaded. Please build embeddings first: python build_embeddings.py")
                return

            with st.spinner("🔄 Searching products..."):
                try:
                    import clip
                    import torch

                    normalized_query = normalize_query(user_query)

                    # Generate query embedding
                    with torch.no_grad():
                        query_tokens = clip.tokenize([normalized_query], truncate=True).to(st.session_state.device)
                        query_embedding = st.session_state.clip_model.encode_text(query_tokens)
                        query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
                        query_embedding = query_embedding.cpu().numpy()[0]

                    # Search vector store
                    results = search_products(query_embedding, normalized_query, top_k=top_k)

                    # Add to chat history
                    st.session_state.chat_history.append({
                        'role': 'user',
                        'content': user_query
                    })

                    if results:
                        st.session_state.retrieved_products = results

                        if normalized_query != user_query:
                            st.info(f"Interpreting query as: {normalized_query}")

                        # Create response
                        direct_answer, answer_source = generate_rag_answer(normalized_query, results)
                        response_text = f"{direct_answer}\n\nFound {len(results)} relevant products for your query.\n\n"
                        for i, (product_id, similarity, metadata) in enumerate(results, 1):
                            response_text += f"**{i}. {metadata.get('title', 'Unknown')}** (Match: {similarity*100:.1f}%)\n"
                            response_text += f"- Brand: {metadata.get('brand') or 'N/A'}\n"
                            response_text += f"- Price: {format_price(metadata.get('price'))}\n"
                            response_text += f"- Features: {metadata.get('features', 'N/A')}\n\n"

                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': response_text
                        })

                        st.success("✅ Retrieved relevant products!")
                        st.subheader("Answer")
                        st.markdown(direct_answer)
                        st.caption(f"Answer source: {answer_source}")

                        # Display results
                        st.subheader("Retrieved Products")

                        for i, (product_id, similarity, metadata) in enumerate(results, 1):
                            render_product_result(i, product_id, similarity, metadata, "Match Score")

                    else:
                        st.warning("⚠️ No products found matching your query")

                except Exception as e:
                    st.error(f"❌ Error during search: {str(e)}")
                    logger.error(f"Search error: {e}")
        else:
            st.warning("Please enter a query")


def render_image_query_interface(temperature, top_k):
    """Render image query input interface"""
    st.subheader("🖼️ Image Query")

    uploaded_image = st.file_uploader(
        "Upload a product image",
        type=['jpg', 'jpeg', 'png']
    )

    if uploaded_image is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.image(uploaded_image, caption="Uploaded image", use_column_width=True)

        with col2:
            st.info("✅ Image uploaded successfully!")

            # Additional question
            additional_question = st.text_input(
                "Ask a specific question about this product (optional)"
            )

            if st.button("🔍 Analyze Image", key="image_search"):
                if st.session_state.clip_model is None or st.session_state.vector_store is None:
                    st.error("❌ Models not loaded. Please build embeddings first: python build_embeddings.py")
                    return

                with st.spinner("🔄 Analyzing image and finding similar products..."):
                    try:
                        import torch

                        # Process image
                        image = Image.open(uploaded_image).convert('RGB')

                        # Generate image embedding
                        with torch.no_grad():
                            image_tensor = st.session_state.preprocess(image).unsqueeze(0).to(st.session_state.device)
                            image_embedding = st.session_state.clip_model.encode_image(image_tensor)
                            image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
                            image_embedding = image_embedding.cpu().numpy()[0]

                        # Search vector store
                        results = st.session_state.vector_store.search(image_embedding, top_k=top_k)

                        st.session_state.chat_history.append({
                            'role': 'user',
                            'content': f"Image analysis {' + ' + additional_question if additional_question else ''}"
                        })

                        if results:
                            st.session_state.retrieved_products = results

                            direct_answer, answer_source = generate_rag_answer(additional_question, results)
                            response_text = f"{direct_answer}\n\n✅ Found {len(results)} similar products based on the image.\n\n"
                            for i, (product_id, similarity, metadata) in enumerate(results, 1):
                                response_text += f"**{i}. {metadata.get('title', 'Unknown')}** (Similarity: {similarity*100:.1f}%)\n"
                                response_text += f"- Brand: {metadata.get('brand') or 'N/A'}\n"
                                response_text += f"- Price: {format_price(metadata.get('price'))}\n\n"

                            st.session_state.chat_history.append({
                                'role': 'assistant',
                                'content': response_text
                            })

                            st.success("✅ Image analysis complete!")
                            st.subheader("Answer")
                            st.markdown(direct_answer)
                            st.caption(f"Answer source: {answer_source}")

                            st.subheader("Similar Products Found")

                            for i, (product_id, similarity, metadata) in enumerate(results, 1):
                                render_product_result(i, product_id, similarity, metadata, "Similarity")

                        else:
                            st.warning("⚠️ No similar products found")

                    except Exception as e:
                        st.error(f"❌ Error during image analysis: {str(e)}")
                        logger.error(f"Image analysis error: {e}")


def render_combined_query_interface(temperature, top_k):
    """Render combined text + image query interface"""
    st.subheader("🎨 Combined Query (Text + Image)")

    col1, col2 = st.columns(2)

    with col1:
        user_text = st.text_area(
            "Describe what you're looking for",
            placeholder="E.g., I want a smartphone with good camera and 5G..."
        )

    with col2:
        uploaded_image = st.file_uploader(
            "Upload product image",
            type=['jpg', 'jpeg', 'png'],
            key="combined_upload"
        )

        if uploaded_image is not None:
            st.image(uploaded_image, caption="Reference image", use_column_width=True)

    text_weight = st.slider(
        "Weight text vs image (0=image only, 1=text only)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1
    )

    if st.button("🔍 Search", key="combined_search"):
        if user_text or uploaded_image:
            if st.session_state.clip_model is None or st.session_state.vector_store is None:
                st.error("❌ Models not loaded. Please build embeddings first: python build_embeddings.py")
                return

            with st.spinner("🔄 Performing multimodal search..."):
                try:
                    import torch
                    import clip

                    combined_embedding = None
                    use_text_for_retrieval = bool(user_text) and not (
                        uploaded_image is not None and is_product_question(user_text)
                    )

                    # Generate text embedding
                    if use_text_for_retrieval:
                        with torch.no_grad():
                            text_tokens = clip.tokenize([user_text], truncate=True).to(st.session_state.device)
                            text_embedding = st.session_state.clip_model.encode_text(text_tokens)
                            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
                            text_embedding = text_embedding.cpu().numpy()[0]

                        combined_embedding = text_embedding * text_weight

                    # Generate image embedding
                    if uploaded_image is not None:
                        image = Image.open(uploaded_image).convert('RGB')
                        with torch.no_grad():
                            image_tensor = st.session_state.preprocess(image).unsqueeze(0).to(st.session_state.device)
                            image_embedding = st.session_state.clip_model.encode_image(image_tensor)
                            image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
                            image_embedding = image_embedding.cpu().numpy()[0]

                        if combined_embedding is None:
                            image_factor = 1 if not use_text_for_retrieval else (1 - text_weight)
                            combined_embedding = image_embedding * image_factor
                        else:
                            combined_embedding += image_embedding * (1 - text_weight)

                    # Normalize combined embedding
                    combined_embedding = combined_embedding / (np.linalg.norm(combined_embedding) + 1e-8)

                    # Search
                    results = st.session_state.vector_store.search(combined_embedding, top_k=top_k)

                    st.session_state.chat_history.append({
                        'role': 'user',
                        'content': f"Multimodal search: {user_text if user_text else '[Image query]'}"
                    })

                    if results:
                        st.session_state.retrieved_products = results

                        direct_answer, answer_source = generate_rag_answer(user_text, results)
                        response_text = f"{direct_answer}\n\n✅ Found {len(results)} products matching your multimodal query.\n\n"
                        for i, (product_id, similarity, metadata) in enumerate(results, 1):
                            response_text += f"**{i}. {metadata.get('title', 'Unknown')}** (Match: {similarity*100:.1f}%)\n"
                            response_text += f"- Brand: {metadata.get('brand') or 'N/A'}\n"

                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': response_text
                        })

                        st.success("✅ Multimodal search complete!")
                        st.subheader("Answer")
                        st.markdown(direct_answer)
                        st.caption(f"Answer source: {answer_source}")

                        st.subheader("Results")

                        for i, (product_id, similarity, metadata) in enumerate(results, 1):
                            render_product_result(i, product_id, similarity, metadata, "Match")

                    else:
                        st.warning("⚠️ No products found")

                except Exception as e:
                    st.error(f"❌ Error during search: {str(e)}")
                    logger.error(f"Multimodal search error: {e}")

        else:
            st.warning("Please provide text or upload an image")


def render_evaluation_metrics():
    """Render evaluation metrics section"""
    return


def main():
    """Main Streamlit application"""
    configure_page()
    init_session_state()

    temperature, top_k, input_mode = render_sidebar()

    render_chat_interface()

    # Main content area
    st.markdown("---")

    if input_mode == "Text Query":
        render_text_query_interface(temperature, top_k)
    elif input_mode == "Image Query":
        render_image_query_interface(temperature, top_k)
    else:
        render_combined_query_interface(temperature, top_k)

    st.markdown("---")

    # Evaluation metrics are reported separately after running a labeled eval set.


if __name__ == "__main__":
    main()
