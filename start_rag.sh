#!/bin/bash

echo "Starting Ollama server..."
ollama serve > ollama.log 2>&1 &
OLLAMA_PID=$!

echo "Waiting for Ollama server to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:11434 > /dev/null 2>&1; then
        echo "✓ Ollama server is ready!"
        break
    fi
    echo "  Attempt $i/30: Waiting..."
    sleep 2
done

if ! curl -s http://localhost:11434 > /dev/null 2>&1; then
    echo "ERROR: Ollama server failed to start"
    exit 1
fi

echo ""
echo "Checking for required models..."
echo "This will download models if they're not already available."
echo "The models are large (embedding: ~68MB, language: ~807MB)"
echo ""

echo "Pulling embedding model..."
ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf

echo "Pulling language model..."
ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF

echo ""
echo "✓ All models ready!"
echo ""
echo "Starting RAG Demo..."
echo ""

python rag_demo.py
