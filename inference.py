"""AI inference logic for word analogies."""
import re
from typing import List, Optional, Tuple
import streamlit as st
import numpy as np
import faiss

from core import load_model


def parse_equation(equation: str) -> Optional[List[str]]:
    """Parse equation into tokens with flexible regex."""
    equation = equation.strip().lower()
    
    # Flexible regex pattern for any length equation
    pattern = r'^[a-z]+(\s+[+-]\s+[a-z]+)*$'
    
    if not re.match(pattern, equation):
        return None
    
    # Split into tokens
    tokens = re.findall(r'[a-z]+|[+-]', equation)
    return tokens


def compute_analogy(tokens: List[str], vocab: List[str]) -> Optional[np.ndarray]:
    """Compute result vector from tokens using vector arithmetic."""
    model = load_model()
    result_vector = None
    current_op = '+'
    vocab_set = set(vocab)
    
    for token in tokens:
        if token in ['+', '-']:
            current_op = token
        else:
            # Check if word is in vocabulary
            if token not in vocab_set:
                st.error(f"Word '{token}' not found in vocabulary")
                return None
            
            # Get embedding
            word_emb = model.encode([token], convert_to_numpy=True)[0]
            word_emb = word_emb / np.linalg.norm(word_emb)
            
            # Apply operation
            if result_vector is None:
                result_vector = word_emb.copy()
            else:
                if current_op == '+':
                    result_vector += word_emb
                else:
                    result_vector -= word_emb
    
    # Normalize result
    if result_vector is not None:
        result_vector = result_vector / np.linalg.norm(result_vector)
    
    return result_vector


def search_similar(vector: np.ndarray, index: faiss.IndexFlatIP, vocab: List[str], tokens: List[str]) -> List[Tuple[str, float]]:
    """Search for top k similar words using FAISS index."""
    scores, indices = index.search(vector.reshape(1, -1).astype('float32'), 10)
    results = [(vocab[idx], float(scores[0][i])) for i, idx in enumerate(indices[0])]
    return [x for x in results if x[0] not in tokens]
