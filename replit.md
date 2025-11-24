# RAG System from Scratch

## Project Overview
An educational implementation of a Retrieval-Augmented Generation (RAG) system built from scratch using Python and Ollama. This project demonstrates how RAG works without using heavy ML frameworks, implementing core concepts like vector embeddings, cosine similarity, and semantic search.

**Current Status**: Fully functional RAG system with interactive Q&A interface

**Last Updated**: November 24, 2025

## Architecture

### Core Components
1. **Vector Database** - In-memory list storing (chunk, embedding) tuples
2. **Embedding Model** - `hf.co/CompendiumLabs/bge-base-en-v1.5-gguf` (68MB)
3. **Language Model** - `hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF` (807MB)

### Key Files
- `rag_demo.py` - Main RAG implementation with retrieval and generation logic
- `cat-facts.txt` - Knowledge base (20 cat facts used as demo dataset)
- `start_rag.sh` - Startup script that initializes Ollama server and pulls models
- `README.md` - Comprehensive documentation with usage examples

### Technical Implementation
- **Cosine Similarity**: Custom implementation for vector comparison
- **Chunking Strategy**: Line-by-line chunking (simple approach for demo)
- **Retrieval**: Top-N similarity search (default N=3)
- **Generation**: Ollama chat API with system prompt containing retrieved context

## How to Use

### Starting the System
The workflow "RAG Demo" automatically:
1. Starts the Ollama server
2. Downloads required models (if not already cached)
3. Indexes the cat-facts.txt dataset
4. Launches the interactive Q&A interface

### Asking Questions
Users can interact with the RAG system through the console:
- Ask questions about cats
- System retrieves relevant facts
- Generates answers based only on retrieved context
- Shows similarity scores for transparency

### Example Questions
- "How fast can cats run?"
- "Tell me about cat sleep patterns"
- "What's unique about a cat's nose?"

## Dependencies

### System Packages
- `ollama` - Local LLM server (version 0.9.5)

### Python Packages
- `ollama` (0.6.1) - Python client for Ollama API
- Standard library only (no numpy, no transformers)

## Project Structure
```
.
├── rag_demo.py          # Main RAG implementation
├── cat-facts.txt        # Knowledge base dataset
├── start_rag.sh         # Startup script
├── README.md            # User documentation
├── replit.md            # Project memory/documentation
└── .gitignore           # Python-specific ignores
```

## Technical Details

### Indexing Phase
1. Load dataset from text file
2. For each line/chunk:
   - Generate embedding vector using Ollama
   - Store (chunk, embedding) in VECTOR_DB list

### Retrieval Phase
1. Convert user query to embedding vector
2. Calculate cosine similarity with all stored chunks
3. Sort by similarity (descending)
4. Return top N most relevant chunks

### Generation Phase
1. Construct system prompt with retrieved chunks
2. Send to Ollama chat API
3. Stream response back to user in real-time

### Cosine Similarity Formula
```python
similarity = dot_product(a, b) / (norm(a) * norm(b))
```

## Future Improvements

### Potential Enhancements
- Support for PDF and multi-format documents
- Semantic text chunking with overlap
- Vector database integration (ChromaDB, FAISS, Qdrant)
- Reranking models for better relevance
- Multi-query retrieval for complex questions
- Conversation memory for multi-turn dialogues
- Larger language models for improved responses

### Known Limitations
- Simple line-by-line chunking (not suitable for complex documents)
- In-memory storage (not scalable for large datasets)
- No reranking (relies solely on cosine similarity)
- Single-query retrieval (may miss context for complex questions)
- Small 1B parameter model (limited response quality)

## Learning Resources
- Based on: [Code a simple RAG from scratch](https://huggingface.co/blog/ngxson/rag-from-scratch) by Xuan-Son Nguyen
- RAG Paper: [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
- Ollama Docs: [github.com/ollama/ollama](https://github.com/ollama/ollama)

## User Preferences
None specified yet.

## Recent Changes
- **2025-11-24**: Initial project setup
  - Installed Ollama and Python dependencies
  - Implemented vector database from scratch
  - Created interactive Q&A system
  - Added comprehensive documentation
  - Configured workflow for automatic startup
