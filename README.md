# 🔤 Word Vector Analogy Solver

A semantic word analogy solver powered by neural embeddings and vector arithmetic. Solve equations like **king - man + woman = queen** by computing arithmetic operations on word embeddings.

## Overview

This app demonstrates how semantic relationships between words can be captured and manipulated using dense vector representations. Instead of traditional symbolic approaches, it uses pre-trained neural embeddings to understand meaning and perform analogical reasoning.

**Example analogies:**
- `king - man + woman` → `queen`
- `doctor - man + woman` → `nurse`
- `dog - puppy + kitten` → `cat`

## How It Works

### 1. **Semantic Embeddings**

The app uses **SentenceTransformers** (`all-MiniLM-L6-v2`), a lightweight transformer model that converts words into dense 384-dimensional vectors. These vectors capture semantic meaning:
- Words with similar meanings are close together in vector space
- Analogies correspond to parallelograms in this space

### 2. **Vector Arithmetic (The Core Magic)**

When you input an equation like `king - man + woman`, the app:

1. **Retrieves embeddings** for each word from the pre-computed cache
2. **Normalizes** each embedding (converts to unit length)
3. **Performs vector arithmetic**:
   ```
   result = embedding(king) - embedding(man) + embedding(woman)
   ```
4. **Re-normalizes** the result vector for cosine similarity

The resulting vector now captures the semantic relationship: "king with male-ness removed and female-ness added."

#### Vector Normalization
Embeddings are normalized using L2 norm normalization:
```
normalized_vector = vector / ||vector||
```

This ensures all vectors have unit length (magnitude = 1), making cosine similarity equivalent to a simple dot product. This is crucial for:
- Consistent similarity comparisons
- Efficient FAISS retrieval (which uses Inner Product)
- Fair distance metrics across the vocabulary

### 3. **Similarity Retrieval**

After computing the result vector, the app finds the most similar words using **FAISS (Facebook AI Similarity Search)**:

- **Index Type**: `IndexFlatIP` (Inner Product Index)
- **Similarity Metric**: Cosine similarity (via dot product on normalized vectors)
- **Search**: O(n) exact search through all vocabulary embeddings
- **Output**: Top 10 candidates, excluding input words

The FAISS index stores all 10,000 vocabulary embeddings in a searchable format and quickly computes dot products between the result vector and all vocabulary vectors.

#### Why FAISS?
- **Speed**: Optimized C++ implementation for vector operations
- **Simplicity**: IndexFlatIP provides exact nearest neighbors
- **Scalability**: Designed for searching millions of vectors (overkill here, but future-proof)

### 4. **Data Pipeline**

#### Preprocessing & Caching

```
Initial Run:
  1. Download vocabulary (10k words) from online source
  2. Cache to vocab.txt
  3. Load SentenceTransformer model
  4. Encode all 10k words → 10k × 384 matrix
  5. Normalize embeddings (row-wise L2 norm)
  6. Pickle & cache embeddings to vocab_embeddings.pkl
  7. Build FAISS index
  8. Serve via Streamlit

Subsequent Runs:
  1. Load vocab.txt (instant)
  2. Load pickled embeddings (instant)
  3. Rebuild FAISS index (< 1ms)
  4. Serve via Streamlit
```

#### Pickle File Format

The `vocab_embeddings.pkl` is a binary serialized NumPy array:
- **Shape**: (10000, 384) — 10,000 words × 384-dimensional embeddings
- **Data Type**: `float32`
- **Content**: Pre-computed, normalized embeddings
- **Loading**: Python's `pickle.load()` deserializes into memory

Advantages of pickling:
- Binary compression (smaller than JSON/CSV)
- Preserves NumPy array structure and dtype
- Fast deserialization
- Language-agnostic (can be read by other systems)

### 5. **Query Process (Step-by-Step)**

```
User Input: "king - man + woman"
    ↓
[Parsing] Extract tokens: ['king', '-', 'man', '+', 'woman']
    ↓
[Encoding] Get embeddings:
  - e_king = model.encode("king")          # 384-dim vector
  - e_man = model.encode("man")            # 384-dim vector
  - e_woman = model.encode("woman")        # 384-dim vector
    ↓
[Vector Math] Compute result:
  - result = e_king - e_man + e_woman      # Element-wise arithmetic
  - result = result / ||result||           # Normalize to unit length
    ↓
[Search] Find closest words:
  - similarities = FAISS.search(result, top_10)
  - Exclude input words from results
    ↓
[Output] Display top matches with similarity scores
  - queen: 0.8234
  - princess: 0.7891
  - ...
```

### 6. **Similarity Scoring**

The similarity score between the result vector and a vocabulary word is **cosine similarity**:

```
similarity(v1, v2) = (v1 · v2) / (||v1|| × ||v2||)
```

Since both vectors are normalized:
```
similarity(v1, v2) = v1 · v2  (simple dot product)
```

**Range**: [-1, 1]
- **1.0** = identical direction (highly similar)
- **0.0** = orthogonal (no relationship)
- **-1.0** = opposite directions (opposite meaning)

Scores typically range **0.5–0.9** for related words.

## Architecture

```
config.py          → Constants (model name, paths, dimensions)
core.py            → Data loading (vocab download, embeddings, FAISS)
inference.py       → AI logic (parsing, vector math, similarity search)
app.py             → Streamlit UI
```

### Key Components

| Module | Responsibility |
|--------|-----------------|
| **config.py** | All hyperparameters and file paths |
| **core.py** | Model loading, embedding generation, FAISS index creation |
| **inference.py** | Equation parsing, vector arithmetic, similarity retrieval |
| **app.py** | Streamlit interface and user interactions |

## Technical Stack

- **Neural Embeddings**: [SentenceTransformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`)
- **Vector Search**: [FAISS](https://github.com/facebookresearch/faiss)
- **Web Framework**: [Streamlit](https://streamlit.io/)
- **Linear Algebra**: NumPy
- **Serialization**: Pickle
- **Vocabulary**: [Desi Quintans' Noun List](https://www.desiquintans.com/downloads/nounlist/)

## Performance

- **First load**: ~10-15 seconds (downloads model + computes embeddings)
- **Subsequent loads**: < 1 second (cached)
- **Query time**: 50-200 ms (encoding + FAISS search)
- **Memory**: ~150 MB (model + embeddings + index)

## Limitations & Future Work

**Current Limitations:**
- Limited to 10,000 words (configurable in `config.py`)
- Only supports single words (no multi-word phrases)
- Works best for concrete noun-like concepts
- No support for unknown words

**Potential Improvements:**
- Increase vocabulary size (91k words available)
- Add support for subword tokenization
- Use quantization for faster searches
- Add word similarity explanations
- Fine-tune on specific domains
- Support multi-word expressions

## References

- Mikolov et al. (2013): "Efficient Estimation of Word Representations in Vector Space" — foundational work on word embeddings
- Levy & Goldberg (2014): "Neural Word Embeddings as Implicit Matrix Factorization" — understanding vector arithmetic
- Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
