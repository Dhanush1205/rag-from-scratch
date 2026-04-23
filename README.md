# RAG from Scratch

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Ollama](https://img.shields.io/badge/Ollama-0.9.5-blue.svg)](https://ollama.com)

> Retrieval-Augmented Generation built from the ground up — no LangChain, no vector DB library, just Python and math.

Most RAG tutorials hand you an abstraction. This one doesn't. Every component — embedding, vector storage, cosine similarity, retrieval, generation — is implemented explicitly so you understand what frameworks are actually doing under the hood.

---

## Why "From Scratch"?

Libraries like LangChain and LlamaIndex are useful, but they hide the mechanics. This project exposes them:

- **Embeddings** — text converted to 768-dim vectors using `bge-base-en-v1.5-gguf`
- **Vector store** — a plain Python list of `(chunk, embedding)` tuples
- **Retrieval** — cosine similarity computed manually with dot products and L2 norms
- **Generation** — Llama-3.2-1B via Ollama, fully local, no API keys

---

## Architecture

```
User Query
    │
    ▼
Embed Query (bge-base-en-v1.5)
    │
    ▼
Cosine Similarity → In-Memory Vector DB
    │
    ▼
Top-K Retrieved Chunks
    │
    ▼
Prompt Construction
    │
    ▼
Llama-3.2-1B (Ollama) → Response
```

---

## Quick Start

**Prerequisites:** Install [Ollama](https://ollama.com)

```bash
git clone https://github.com/Dhanush1205/rag-from-scratch.git
cd rag-from-scratch
bash start_rag.sh
```

The script pulls both models (~875MB total), indexes the knowledge base, and launches an interactive Q&A session.

---

## How It Works

### 1. Indexing

```python
for chunk in dataset:
    embedding = embed_model(chunk)        # text → 768-dim vector
    VECTOR_DB.append((chunk, embedding))  # store as tuple
```

### 2. Retrieval

```python
def retrieve(query, top_n=3):
    query_vec = embed_model(query)
    similarities = [
        (chunk, cosine_similarity(query_vec, vec))
        for chunk, vec in VECTOR_DB
    ]
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]
```

### 3. Cosine Similarity (no libraries)

```python
def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x ** 2 for x in a) ** 0.5
    norm_b = sum(x ** 2 for x in b) ** 0.5
    return dot / (norm_a * norm_b)
```

### 4. Generation

```python
context = "\n".join(retrieved_chunks)
prompt = f"Use only this context:\n{context}\n\nQuestion: {query}"
response = language_model(prompt)
```

---

## Example Output

```
Ask me a question: How fast can cats run?

Retrieved knowledge:
  [0.82] Cats can run at speeds of up to 31 mph over short distances.
  [0.58] Cats can jump up to six times their length.
  [0.51] A cat's heart beats nearly twice as fast as a human heart.

Response:
Cats can reach approximately 31 mph (49 km/h) over short distances.
```

---

## Known Limitations & What I'd Add Next

| Limitation | Fix |
|---|---|
| Line-by-line chunking dilutes embedding signal | Sentence-level chunking with overlap |
| In-memory storage doesn't scale | ChromaDB or Qdrant |
| No reranking step | Cross-encoder reranker between retrieval and generation |
| Single-query only | Multi-query retrieval |
| No eval metrics | Retrieval precision + answer faithfulness harness |

The biggest lesson from building this: **most RAG failures are retrieval bugs, not generation bugs.** Fixing chunking strategy had more impact than swapping models.

---

## Project Structure

```
rag-from-scratch/
├── rag_demo.py       # Core implementation (~150 lines)
├── cat-facts.txt     # Knowledge base
├── start_rag.sh      # Setup + launch script
└── README.md
```

---

## Stack

- **Embedding model:** `bge-base-en-v1.5-gguf` (68MB, 768-dim)
- **Language model:** `Llama-3.2-1B-Instruct` (807MB)
- **Inference runtime:** Ollama
- **Dependencies:** Pure Python — no LangChain, no FAISS, no NumPy

---

## License

MIT — see [LICENSE](LICENSE)

---

*Built by [Dhanush1205](https://github.com/Dhanush1205)*
