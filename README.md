# RAG-Enabled Python Application with Open-Source Embeddings

## Overview

This Python application implements a Retrieval-Augmented Generation (RAG) system using open-source embeddings and ChromaDB for vector storage. Key features include:

- Document processing with text splitting
- Sentence Transformers for local embeddings (no API keys needed)
- ChromaDB vector store with cosine similarity search
- Conversation memory with automatic compression
- OpenAI integration for response generation (optional)

## Components

### 1. SentenceTransformerEmbeddings
- Uses Sentence Transformers models (`all-MiniLM-L6-v2` by default)
- Supports both document and query embeddings
- Embedding dimension detection

### 2. DocumentProcessor
- Loads text documents from directory (supports `.txt`, `.md`, `.markdown`)
- Splits documents into chunks (1000 chars with 200 char overlap)
- Preserves metadata including source file information

### 3. ChromaVectorStore
- Persistent ChromaDB storage with cosine similarity
- Batch document ingestion with progress tracking
- Similarity search with relevance scoring
- Collection management functions

### 4. ChatMemoryManager
- Maintains conversation history
- Automatic compression when exceeding 8 messages
- OpenAI-based summarization (with fallback)
- Context management for LLM prompts

### 5. RAGApplication
- Main application class combining all components
- Document loading and indexing
- Context-aware response generation
- Interactive chat interface

## Usage

### Initialization
```python
app = RAGApplication(openai_api_key=None, embedding_model="all-MiniLM-L6-v2")
