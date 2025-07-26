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

## 📋 Requirements

### System Requirements
- Python 3.8+
- 4GB+ RAM (recommended for embedding models)
- 2GB+ disk space for models and vector database

### Dependencies
```
chromadb>=0.4.0
sentence-transformers>=2.2.0
langchain>=0.1.0
openai>=1.0.0
numpy>=1.21.0
```

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/avinash00134/rag-chat-ai
cd rag-chat-ai
```

### 2. Create Virtual Environment
```bash
python -m venv rag_env
source rag_env/bin/activate  # On Windows: rag_env\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables (Optional)
For full response generation capabilities, set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or create a `.env` file:
```
OPENAI_API_KEY=your-api-key-here
```


## 🚀 Quick Start

### 1. Prepare Your Documents
Create a `documents` folder and add your text files:
```bash
mkdir documents
# Add your .txt, .md, or .markdown files to this folder
```

### 2. Run the Application
```bash
python rag_chat.py
```

### 3. Basic Usage
The application will automatically:
- Load and process documents from the `documents` folder
- Create embeddings using Sentence Transformers
- Start an interactive chat session

## 💡 Usage Examples

### Basic Chat
```
You: What is machine learning?
Assistant: Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed...
```

### Document Search
```
You: search artificial intelligence
# Returns relevant document chunks with similarity scores
```

### Available Commands
- `quit` - Exit the application
- `clear` - Clear chat history
- `search <query>` - Search documents without generating response
- `info` - Show collection information
- `history` - Display chat history
- `responses` - Show stored LLM responses with metadata
- `tokens` - Display total token usage

  
