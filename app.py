import streamlit as st
import os
import time

import config
from core import load_resources
from inference import parse_equation, compute_analogy, search_similar

# Page config
st.set_page_config(
    page_title="Word Vector Analogy Solver",
    page_icon="🔤",
    layout="centered"
)

# Title
st.title("🔤 Word Vector Analogy Solver")
st.markdown("Solve word analogies using semantic embeddings and vector arithmetic")

# Load resources
with st.spinner("Loading vocabulary and embeddings..."):
    try:
        vocab, embeddings, index = load_resources()
        vocab_size = len(vocab)
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        st.stop()

# Input section
col1, col2 = st.columns([4, 1])
with col1:
    equation = st.text_input(
        "Enter word equation:",
        value="king - man + woman",
        placeholder="e.g., king - man + woman"
    )
with col2:
    st.write("")
    st.write("")
    solve_button = st.button("Solve ➡️", type="primary", use_container_width=True)

# Solve equation
if solve_button and equation:
    tokens = parse_equation(equation)
    
    if tokens is None:
        st.error("Invalid equation format. Use: word1 ± word2 ± word3 (e.g., 'king - man + woman')")
    else:
        start_time = time.time()
        
        result_vector = compute_analogy(tokens, vocab)
        
        if result_vector is not None:
            results = search_similar(result_vector, index, vocab, tokens)
            query_time = (time.time() - start_time) * 1000
            
            # Display result
            st.markdown("---")
            st.markdown(f"### **{equation}** → **{results[0][0]}**")
            st.metric("Similarity Score", f"{results[0][1]:.4f}")
            
            # Top 5 matches
            st.markdown("#### Top 5 Matches")
            for i, (word, score) in enumerate(results):
                st.markdown(f"{i + 1}. **{word}** - {score:.4f}")
            
            st.markdown("---")
            st.caption(f"⚡ Query time: {query_time:.2f}ms")

# Reset cache button
if st.button("🔄 Reset Cache"):
    if os.path.exists(config.VOCAB_FILE):
        os.remove(config.VOCAB_FILE)
    if os.path.exists(config.EMBEDDINGS_FILE):
        os.remove(config.EMBEDDINGS_FILE)
    st.cache_resource.clear()
    st.success("Cache cleared! Please refresh the page.")
    st.rerun()

# Examples
with st.expander("📚 Examples"):
    examples_data = {
        "Equation": [
            "king - man + woman",
            "doctor - man + woman",
            "dog - puppy + kitten",
        ],
        "Expected": [
            "queen",
            "nurse",
            "cat",
        ]
    }
    st.table(examples_data)

# Footer
st.markdown("---")
st.caption(f"📊 Vocabulary size: {vocab_size:,} words | Model: {config.MODEL_NAME}")