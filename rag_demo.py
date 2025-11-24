"""
RAG (Retrieval-Augmented Generation) System - Educational Implementation

This script demonstrates how RAG systems work from scratch without using
frameworks like LangChain or LlamaIndex. Perfect for learning!

Components:
    1. Vector Database (in-memory list of tuples)
    2. Embedding Model (converts text to vectors)
    3. Language Model (generates responses)

Author: Educational Project
License: MIT
"""

import ollama
import sys

# Model configurations - these are small models that run on CPU
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'  # 68MB, creates 768-dim vectors
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'   # 807MB, 1B parameters

# In-memory vector database: list of (text_chunk, embedding_vector) tuples
VECTOR_DB = []


def cosine_similarity(a, b):
    """
    Calculate cosine similarity between two vectors.
    
    Cosine similarity measures the angle between two vectors in high-dimensional
    space. It returns a value between -1 and 1:
        1.0  = vectors point in same direction (very similar)
        0.0  = vectors are perpendicular (unrelated)
        -1.0 = vectors point in opposite directions (opposite meaning)
    
    Formula: similarity = (A·B) / (||A|| × ||B||)
    
    Args:
        a (list): First vector (e.g., query embedding)
        b (list): Second vector (e.g., chunk embedding)
    
    Returns:
        float: Similarity score between 0 and 1 (we use absolute values)
    
    Example:
        >>> vec1 = [1, 2, 3]
        >>> vec2 = [4, 5, 6]
        >>> cosine_similarity(vec1, vec2)
        0.974... (very similar)
    """
    # Dot product: multiply corresponding elements and sum
    dot_product = sum([x * y for x, y in zip(a, b)])
    
    # Magnitude (L2 norm): square root of sum of squares
    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([x ** 2 for x in b]) ** 0.5
    
    # Prevent division by zero for empty vectors
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    # Final similarity calculation
    return dot_product / (norm_a * norm_b)


def add_chunk_to_database(chunk):
    """
    Create an embedding for a text chunk and store it in the vector database.
    
    This is the "indexing" phase of RAG. We convert each piece of knowledge
    into a vector representation that can be efficiently searched later.
    
    Args:
        chunk (str): Text to be indexed (e.g., a fact, paragraph, or sentence)
    
    Raises:
        ConnectionError: If Ollama server is not running
        Exception: If embedding generation fails
    
    Side Effects:
        Appends (chunk, embedding) tuple to VECTOR_DB
    
    Example:
        >>> add_chunk_to_database("Cats can jump up to 6 times their length")
        # Stores: ("Cats can jump...", [0.1, -0.2, 0.3, ...])
    """
    try:
        # Call Ollama's embedding API to convert text -> vector
        embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
        
        # Store the chunk with its vector representation
        VECTOR_DB.append((chunk, embedding))
        
    except ConnectionError:
        print("\n❌ Error: Cannot connect to Ollama server.")
        print("   Please make sure Ollama is running with: ollama serve")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error creating embedding: {e}")
        print("   Make sure the embedding model is available.")
        sys.exit(1)


def retrieve(query, top_n=3):
    """
    Retrieve the most relevant chunks from the vector database for a given query.
    
    This is the "retrieval" phase of RAG. We:
        1. Convert the query to a vector
        2. Compare it with all stored chunk vectors
        3. Return the top N most similar chunks
    
    Args:
        query (str): User's question or search query
        top_n (int): Number of top results to return (default: 3)
    
    Returns:
        list: Top N chunks as [(chunk_text, similarity_score), ...] 
              sorted by similarity (highest first)
    
    Example:
        >>> retrieve("How fast can cats run?", top_n=2)
        [("Cats can run 31 mph...", 0.82),
         ("Cats can jump 6 times...", 0.51)]
    """
    try:
        # Step 1: Convert query to embedding vector (same model as indexing!)
        query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
        
        # Step 2: Calculate similarity with every chunk in the database
        similarities = []
        for chunk, embedding in VECTOR_DB:
            similarity = cosine_similarity(query_embedding, embedding)
            similarities.append((chunk, similarity))
        
        # Step 3: Sort by similarity score (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Step 4: Return top N results
        return similarities[:top_n]
        
    except ConnectionError:
        print("\n❌ Error: Lost connection to Ollama server.")
        print("   The Ollama server may have stopped.")
        return []
    except Exception as e:
        print(f"\n❌ Error during retrieval: {e}")
        return []

def main():
    print("=" * 80)
    print("RAG System - Retrieval-Augmented Generation Demo")
    print("=" * 80)
    print()
    
    print("Loading dataset...")
    dataset = []
    try:
        with open('cat-facts.txt', 'r') as file:
            dataset = file.readlines()
        print(f'✓ Loaded {len(dataset)} entries from cat-facts.txt')
    except FileNotFoundError:
        print("Error: cat-facts.txt not found!")
        return
    
    print()
    print("Indexing dataset (creating embeddings)...")
    print("This may take a moment as we download and run the embedding model...")
    for i, chunk in enumerate(dataset):
        chunk = chunk.strip()
        if chunk:
            add_chunk_to_database(chunk)
            print(f'  Added chunk {i+1}/{len(dataset)} to the database')
    
    print()
    print(f"✓ Indexing complete! Vector database contains {len(VECTOR_DB)} chunks.")
    print()
    print("=" * 80)
    print("You can now ask questions about cats!")
    print("The system will retrieve relevant facts and generate answers.")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 80)
    print()
    
    while True:
        input_query = input('Ask me a question: ').strip()
        
        if input_query.lower() in ['quit', 'exit', 'q']:
            print("\nThank you for using the RAG system!")
            print("\nNote: The Ollama server is still running in the background.")
            print("To stop it, you can restart the workflow or use: pkill ollama")
            break
        
        if not input_query:
            continue
        
        print()
        print("Retrieving relevant knowledge...")
        retrieved_knowledge = retrieve(input_query)
        
        if not retrieved_knowledge:
            print("⚠️  Could not retrieve knowledge. Please try again.")
            continue
        
        print('\nRetrieved knowledge:')
        for chunk, similarity in retrieved_knowledge:
            print(f'  - (similarity: {similarity:.2f}) {chunk}')
        
        contexts = '\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])
        instruction_prompt = f'''You are a helpful chatbot.
Use only the following pieces of context to answer the question. Don't make up any new information:
{contexts}'''
        
        print('\nGenerating response...')
        print('Chatbot response:')
        print('-' * 80)
        
        try:
            stream = ollama.chat(
                model=LANGUAGE_MODEL,
                messages=[
                    {'role': 'system', 'content': instruction_prompt},
                    {'role': 'user', 'content': input_query},
                ],
                stream=True,
            )
            
            for chunk in stream:
                print(chunk['message']['content'], end='', flush=True)
            
            print()
            print('-' * 80)
            print()
        except ConnectionError:
            print("\n❌ Error: Lost connection to Ollama server.")
            print("   The language model server may have stopped.")
            print('-' * 80)
            print()
        except Exception as e:
            print(f"\n❌ Error generating response: {e}")
            print("   Make sure the language model is available.")
            print('-' * 80)
            print()

if __name__ == '__main__':
    main()
