"""Core data loading and processing functionality."""
import streamlit as st
import pickle
import os
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

import config


@st.cache_resource
def load_model() -> SentenceTransformer:
    """Load SentenceTransformer model."""
    return SentenceTransformer(config.MODEL_NAME)


def _download_vocab() -> List[str]:
    """Download or load vocabulary from cache."""
    if os.path.exists(config.VOCAB_FILE):
        with open(config.VOCAB_FILE, 'r') as f:
            return [line.strip() for line in f]
    
    response = requests.get(config.VOCAB_URL)
    response.raise_for_status()
    words = [line.strip() for line in response.text.split('\n') if line.strip()]
    words = words[:config.TOP_K]
    
    # Cache to file
    with open(config.VOCAB_FILE, 'w') as f:
        f.write('\n'.join(words))
    
    return words


@st.cache_resource
def load_resources() -> Tuple[List[str], np.ndarray, faiss.IndexFlatIP]:
    """Load vocabulary, embeddings, and FAISS index."""
    vocab = _download_vocab()
    
    # Load or create embeddings
    if os.path.exists(config.EMBEDDINGS_FILE):
        with open(config.EMBEDDINGS_FILE, 'rb') as f:
            embeddings = pickle.load(f)
    else:
        model = load_model()
        with st.spinner("Creating embeddings for vocabulary..."):
            embeddings = model.encode(vocab, show_progress_bar=False, convert_to_numpy=True)
        
        # Cache embeddings
        with open(config.EMBEDDINGS_FILE, 'wb') as f:
            pickle.dump(embeddings, f)
    
    # Normalize embeddings for cosine similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Create FAISS index
    index = faiss.IndexFlatIP(config.EMBEDDING_DIM)
    index.add(embeddings.astype('float32'))
    
    return vocab, embeddings, index
