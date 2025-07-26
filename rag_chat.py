import os
import logging
import argparse
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import numpy as np
import datetime

# Core dependencies
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Sentence Transformers for embeddings
from sentence_transformers import SentenceTransformer

# LangChain components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.memory import ConversationSummaryBufferMemory

# OpenAI for LLM (only for response generation, not embeddings)
import openai

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('rag_application.log')
    ]
)
logger = logging.getLogger(__name__)
# Set logging level for chromadb to WARNING to reduce noise
logging.getLogger('chromadb').setLevel(logging.WARNING)


class SentenceTransformerEmbeddings:
    """Custom embedding class using Sentence Transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize with a sentence transformer model
        Popular models:
        - all-MiniLM-L6-v2: Fast and good quality (384 dimensions)
        - all-mpnet-base-v2: Higher quality (768 dimensions)
        - multi-qa-MiniLM-L6-cos-v1: Optimized for Q&A
        """
        self.model_name = model_name
        try:
            logger.info(f"Loading Sentence Transformer model: {model_name}")
            self.model = SentenceTransformer(model_name)
            logger.info(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        embeddings = self.model.encode(texts, convert_to_tensor=False, show_progress_bar=True)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        embedding = self.model.encode([text], convert_to_tensor=False)
        return embedding[0].tolist()
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.model.get_sentence_embedding_dimension()


class DocumentProcessor:
    """Handle document loading and processing"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    
    def load_documents(self, directory_path: str) -> List[Document]:
        """Load documents from directory"""
        documents = []
        directory = Path(directory_path)
        
        if not directory.exists():
            logger.error(f"Directory {directory_path} does not exist")
            return documents
        
        supported_extensions = {'.txt', '.md', '.markdown'}
        
        for file_path in directory.rglob('*'):
            if file_path.suffix.lower() in supported_extensions:
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                        if content.strip():
                            doc = Document(
                                page_content=content,
                                metadata={
                                    'source': str(file_path),
                                    'filename': file_path.name,
                                    'file_size': len(content)
                                }
                            )
                            documents.append(doc)
                            logger.info(f"Loaded document: {file_path.name} ({len(content)} chars)")
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
        
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        logger.info(f"Splitting {len(documents)} documents into chunks...")
        split_docs = self.text_splitter.split_documents(documents)
        logger.info(f"Created {len(split_docs)} document chunks")
        return split_docs


class ChromaVectorStore:
    """ChromaDB vector store with Sentence Transformers"""
    
    def __init__(self, collection_name: str = "rag_documents", embedding_model: str = "all-MiniLM-L6-v2"):
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        
        # Initialize Sentence Transformer embeddings
        self.embeddings = SentenceTransformerEmbeddings(embedding_model)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path="./chroma_db")
        
        # Initialize collection
        self.collection = None
        self._initialize_collection()
    
    def _initialize_collection(self):
        """Initialize or get existing collection"""
        try:
            self.collection = self.client.get_collection(
                name=self.collection_name
            )
            logger.info(f"Loaded existing collection: {self.collection_name}")
            logger.info(f"Collection contains {self.collection.count()} documents")
        except Exception:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "embedding_model": self.embedding_model_name
                }
            )
            logger.info(f"Created new collection: {self.collection_name}")
    
    def add_documents(self, documents: List[Document], batch_size: int = 100):
        """Add documents to vector store in batches"""
        if not documents:
            logger.warning("No documents to add")
            return
        
        logger.info(f"Adding {len(documents)} documents to vector store in batches of {batch_size}")
        
        try:
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_texts = [doc.page_content for doc in batch]
                batch_metadatas = [doc.metadata for doc in batch]
                
                logger.debug(f"Processing batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
                
                try:
                    batch_embeddings = self.embeddings.embed_documents(batch_texts)
                    batch_ids = [f"doc_{i+j}" for j in range(len(batch))]
                    
                    self.collection.add(
                        embeddings=batch_embeddings,
                        documents=batch_texts,
                        metadatas=batch_metadatas,
                        ids=batch_ids
                    )
                    logger.debug(f"Added batch {i//batch_size + 1} ({len(batch)} documents)")
                    
                except Exception as batch_error:
                    logger.error(f"Error processing batch {i//batch_size + 1}: {str(batch_error)}")
                    continue
            
            logger.info(f"Successfully added {len(documents)} document chunks to vector store")
            
        except Exception as e:
            logger.error(f"Fatal error adding documents: {str(e)}")
            raise

    
    def similarity_search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        logger.info(f"Searching for similar documents to query: '{query[:50]}...'")
        
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self.collection.count()),
            include=['documents', 'metadatas', 'distances']
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                formatted_results.append({
                    'content': doc,
                    'metadata': results['metadatas'][0][i] if results['metadatas'][0] else {},
                    'distance': results['distances'][0][i] if results['distances'][0] else 0.0,
                    'similarity_score': 1 - results['distances'][0][i] if results['distances'][0] else 1.0
                })
        
        logger.info(f"Found {len(formatted_results)} similar documents")
        return formatted_results
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        count = self.collection.count()
        return {
            'name': self.collection_name,
            'document_count': count,
            'embedding_model': self.embedding_model_name,
            'embedding_dimension': self.embeddings.get_dimension()
        }
    
    def delete_collection(self):
        """Delete the collection"""
        self.client.delete_collection(self.collection_name)
        logger.info(f"Deleted collection: {self.collection_name}")


class ChatMemoryManager:
    """Manage chat history with automatic compression and full response storage"""
    
    def __init__(self, max_messages: int = 8, llm=None):
        self.max_messages = max_messages
        self.llm = llm
        self.memory = ConversationSummaryBufferMemory(
            llm=llm,
            max_token_limit=2000,
            memory_key="chat_history",
            return_messages=True
        )
        self.chat_history = []  # For recent messages
        self.compressed_history = ""
        self.full_message_history = []  # Stores all messages exactly as sent/received
        self.full_llm_responses = []  # Stores complete LLM responses with metadata
        self.total_tokens_used = 0  # Track total token usage
    
    def add_message(self, role: str, content: str):
        """Add a message to chat history and compress if needed"""
        try:
            if role == "user":
                self.memory.chat_memory.add_user_message(content)
            else:
                self.memory.chat_memory.add_ai_message(content)
            
            message = {"role": role, "content": content}
            self.chat_history.append(message)
            self.full_message_history.append(message)
            logger.debug(f"Added {role} message: {content[:50]}...")
            
            if len(self.chat_history) > self.max_messages:
                logger.debug("Chat history exceeds max messages, compressing...")
                self._compress_history()
                
        except Exception as e:
            logger.error(f"Error adding message: {str(e)}")
            raise
    
    def add_llm_response(self, query: str, response: str, context: str, metadata: Dict[str, Any] = None):
        """Store complete LLM response with metadata and track token usage"""
        llm_response = {
            "timestamp": datetime.datetime.now().isoformat(),
            "query": query,
            "response": response,
            "context": context,
            "metadata": metadata or {}
        }
        self.full_llm_responses.append(llm_response)
        
        # Track token usage if available in metadata
        if metadata and 'usage' in metadata:
            self.total_tokens_used += metadata['usage'].get('total_tokens', 0)
        
        logger.info(f"Stored complete LLM response in memory. Total tokens used: {self.total_tokens_used}")
    
    def _compress_history(self):
        """Compress chat history when it exceeds max_messages"""
        logger.info("Compressing chat history...")
        
        messages_to_compress = self.chat_history[:-4]
        recent_messages = self.chat_history[-4:]
        
        conversation_text = ""
        for msg in messages_to_compress:
            conversation_text += f"{msg['role']}: {msg['content']}\n"
        
        try:
            if openai.api_key:
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": "Summarize the following conversation concisely, preserving key information and context. Keep it under 200 words."
                        },
                        {
                            "role": "user",
                            "content": conversation_text
                        }
                    ],
                    max_tokens=250,
                    temperature=0
                )
                summary = response.choices[0].message.content
                
                # Track token usage for compression
                if hasattr(response, 'usage'):
                    self.total_tokens_used += response.usage.total_tokens
            else:
                summary = self._simple_compress(conversation_text)
            
            if self.compressed_history:
                self.compressed_history += f"\n\n[Previous conversation]: {summary}"
            else:
                self.compressed_history = f"[Previous conversation]: {summary}"
            
            self.chat_history = recent_messages
            
            logger.info("Chat history compressed successfully")
            
        except Exception as e:
            logger.error(f"Error compressing history: {e}")
            self.chat_history = recent_messages
    
    def clear_full_history(self):
        """Clear all chat history including full messages and LLM responses"""
        self.memory.clear()
        self.chat_history = []
        self.compressed_history = ""
        self.full_message_history = []
        self.full_llm_responses = []
        self.total_tokens_used = 0
        logger.info("Cleared all chat history including full messages and LLM responses")

    def _simple_compress(self, text: str) -> str:
        """Simple rule-based compression fallback"""
        lines = text.strip().split('\n')
        if len(lines) <= 4:
            return text
        
        compressed = '\n'.join(lines[:2]) + '\n[... conversation continued ...]\n' + '\n'.join(lines[-2:])
        return compressed
    
    def get_context_for_llm(self) -> str:
        """Get formatted context from memory"""
        return self.memory.load_memory_variables({})["chat_history"]
    
    def get_chat_history(self) -> List[Dict[str, str]]:
        """Get current chat history as list of dicts"""
        messages = self.memory.chat_memory.messages
        history = []
        for msg in messages:
            history.append({
                "role": "user" if msg.type == "human" else "assistant",
                "content": msg.content
            })
        return history
    
    def get_full_history(self) -> List[Dict[str, str]]:
        """Get complete chat history including all messages"""
        return self.full_message_history
    
    def get_full_llm_responses(self) -> List[Dict[str, Any]]:
        """Get all stored LLM responses with metadata"""
        return self.full_llm_responses
    
    def get_token_usage(self) -> int:
        """Get total tokens used"""
        return self.total_tokens_used
    
    def clear_history(self):
        """Clear all chat history except full LLM responses"""
        self.memory.clear()
        self.chat_history = []
        self.compressed_history = ""
        logger.info("Cleared chat history (preserved full LLM responses)")


class RAGApplication:
    """Main RAG Application class with open-source embeddings"""
    
    def __init__(self, openai_api_key: Optional[str] = None, embedding_model: str = "all-MiniLM-L6-v2"):
        # Set OpenAI API key (optional, only needed for response generation)
        if openai_api_key:
            openai.api_key = openai_api_key
            os.environ["OPENAI_API_KEY"] = openai_api_key
            logger.info("OpenAI API key configured for response generation")
            
            # Create proper LLM instance for memory with fallback
            try:
                from langchain_openai import ChatOpenAI
                self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.6)
                logger.info("Using langchain_openai.ChatOpenAI")
            except ImportError:
                try:
                    # Fallback for older versions
                    from langchain.chat_models import ChatOpenAI
                    self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.6)
                    logger.info("Using langchain.chat_models.ChatOpenAI (fallback)")
                except ImportError as e:
                    logger.error(f"Failed to import ChatOpenAI from either langchain_openai or langchain.chat_models: {e}")
                    self.llm = None
        else:
            logger.warning("No OpenAI API key provided. Response generation will be limited.")
            self.llm = None
        
        # Initialize components with proper memory integration
        self.doc_processor = DocumentProcessor()
        self.vector_store = ChromaVectorStore(embedding_model=embedding_model)
        self.memory_manager = ChatMemoryManager(llm=self.llm)
        
        logger.info(f"RAG Application initialized with embedding model: {embedding_model}")

    def load_documents(self, directory_path: str):
        """Load and index documents"""
        logger.info(f"Loading documents from: {directory_path}")
        
        documents = self.doc_processor.load_documents(directory_path)
        if not documents:
            logger.warning("No documents loaded")
            return
        
        split_docs = self.doc_processor.split_documents(documents)
        
        self.vector_store.add_documents(split_docs)
        
        info = self.vector_store.get_collection_info()
        logger.info(f"Documents indexed successfully: {info}")
    
    def _retrieve_context(self, query: str, k: int = 3) -> str:
        """Retrieve relevant context from vector store"""
        results = self.vector_store.similarity_search(query, k=k)
        
        if not results:
            return "No relevant context found in the knowledge base."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            content = result['content']
            similarity = result['similarity_score']
            context_parts.append(
                f"[Context {i} (relevance: {similarity:.2f})]:\n{content}"
            )
        
        return "\n\n".join(context_parts)
    
    def search_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search documents without generating a response"""
        return self.vector_store.similarity_search(query, k=k)
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the document collection"""
        return self.vector_store.get_collection_info()
    
    def get_chat_history(self) -> List[Dict[str, str]]:
        """Delegate to memory manager to get chat history"""
        return self.memory_manager.get_chat_history()

    def get_full_chat_history(self) -> List[Dict[str, str]]:
        """Get complete chat history including all messages"""
        return self.memory_manager.get_full_history()
    
    def get_full_llm_responses(self) -> List[Dict[str, Any]]:
        """Get all stored LLM responses with metadata"""
        return self.memory_manager.get_full_llm_responses()
    
    def get_token_usage(self) -> int:
        """Get total tokens used"""
        return self.memory_manager.get_token_usage()
    
    def clear_chat_history(self):
        """Clear all chat history including full messages"""
        self.memory_manager.clear_full_history()
        logger.info("Chat history cleared (including full messages)")

    def generate_response(self, user_query: str, use_context: bool = True) -> str:
        """Generate concise, summary-style response using RAG
        
        Args:
            user_query: The user's input query
            use_context: Whether to use context from the vector store (default: True)
            
        Returns:
            str: The generated response or error message
            
        Raises:
            Logs errors but doesn't raise them to maintain user experience
        """
        logger.info(f"Processing query: '{user_query[:50]}'... (length: {len(user_query)} chars)")
        
        # Initialize variables with default values
        context = ""
        assistant_response = ""
        metadata = {
            "timestamp": datetime.datetime.now().isoformat(),
            "query_length": len(user_query),
            "use_context": use_context
        }
        
        try:
            # Start timing for performance monitoring
            start_time = datetime.datetime.now()
            
            # 1. Add user message to memory
            logger.debug("Adding user message to memory...")
            self.memory_manager.add_message("user", user_query)
            
            # 2. Retrieve context if needed
            if use_context:
                logger.debug("Retrieving context from vector store...")
                context_start = datetime.datetime.now()
                context = self._retrieve_context(user_query)
                context_time = (datetime.datetime.now() - context_start).total_seconds()
                logger.info(f"Context retrieval took {context_time:.2f} seconds")
                metadata.update({
                    "context_retrieval_time": context_time,
                    "context_length": len(context)
                })
            
            # 3. Get chat history context
            logger.debug("Loading conversation context...")
            chat_context = self.memory_manager.get_context_for_llm()
            metadata["chat_history_length"] = len(str(chat_context))
            
            # 4. Construct optimized prompt
            prompt = f"""You are an AI assistant that provides concise, accurate responses using Retrieval-Augmented Generation (RAG). Follow these rules:
            1. Response Structure:
            - First provide a 1-2 sentence direct answer
            - Then include 3-5 bullet points of key supporting facts
            - Keep total response under 150 words unless more detail is explicitly requested

            2. Context Usage:
            - Prioritize information from the provided context
            - Never mention "according to the context" or similar phrases
            - Synthesize information naturally into your response

            3. Style Guidelines:
            - Use professional but conversational tone
            - Avoid redundant phrases like "based on the information provided"
            - Format bullet points clearly with line breaks

            4. Safety & Compliance:
            - If context contradicts general knowledge, note this discreetly
            - Flag any potentially harmful requests
            - Never reveal file names or metadata from the context

            5. Memory Management:
            [When chat history exceeds 8 messages, automatically summarize earlier messages into:]
            "Previous conversation summary: [concise 3-sentence summary of key points]"

            Current Context:
            {chat_context}

            {'Context from knowledge base:' + context if context else 'No additional context available.'}

            User Query: {user_query}

            Keep the response under 150 words unless more detail is explicitly requested.
            DO NOT mention specific source file names."""
            
            metadata.update({
                "prompt_length": len(prompt),
                "has_openai_key": bool(openai.api_key)
            })
            
            # 5. Generate response using OpenAI or fallback
            if openai.api_key:
                logger.info("Generating response using OpenAI...")
                try:
                    llm_start = datetime.datetime.now()
                    response = openai.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a concise AI assistant that summarizes information effectively."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=500,
                        temperature=0.5
                    )
                    llm_time = (datetime.datetime.now() - llm_start).total_seconds()
                    
                    assistant_response = response.choices[0].message.content
                    logger.debug(f"Received response from OpenAI: {assistant_response[:100]}...")
                    
                    # Update metadata with LLM details
                    metadata.update({
                        "model": "gpt-3.5-turbo",
                        "temperature": 0.5,
                        "max_tokens": 500,
                        "llm_response_time": llm_time,
                        "response_length": len(assistant_response)
                    })
                    
                    if hasattr(response, 'usage'):
                        metadata['usage'] = {
                            'prompt_tokens': response.usage.prompt_tokens,
                            'completion_tokens': response.usage.completion_tokens,
                            'total_tokens': response.usage.total_tokens
                        }
                        logger.info(f"Token usage: {response.usage.total_tokens} (prompt: {response.usage.prompt_tokens}, completion: {response.usage.completion_tokens})")
                    
                except Exception as llm_error:
                    logger.error(f"OpenAI API error: {str(llm_error)}", exc_info=True)
                    assistant_response = f"""I encountered an error generating a response. Here's the relevant context:

    {context if context else "No relevant context found."}"""
                    metadata.update({
                        "error": str(llm_error),
                        "model": "openai_error"
                    })
            else:
                logger.warning("No OpenAI API key available - using fallback response")
                assistant_response = f"""I can provide context but cannot generate full responses without an API key.

    Relevant context for "{user_query}":
    {context if context else "No relevant context found."}"""
                metadata.update({
                    "model": "local_context_only"
                })
            
            # 6. Store the complete response
            self.memory_manager.add_llm_response(
                query=user_query,
                response=assistant_response,
                context=context,
                metadata=metadata
            )
            
            # 7. Add assistant response to memory
            self.memory_manager.add_message("assistant", assistant_response)
            
            # Calculate total processing time
            total_time = (datetime.datetime.now() - start_time).total_seconds()
            logger.info(f"Response generation completed in {total_time:.2f} seconds")
            metadata["total_processing_time"] = total_time
            
            return assistant_response
            
        except Exception as e:
            logger.error(f"Unexpected error in generate_response: {str(e)}", exc_info=True)
            error_response = f"An unexpected error occurred while generating a response: {str(e)}"
            if context:
                error_response += f"\n\nRelevant context found:\n{context}"
            
            # Store error response with metadata
            metadata.update({
                "error": str(e),
                "model": "error"
            })
            self.memory_manager.add_llm_response(
                query=user_query,
                response=error_response,
                context=context,
                metadata=metadata
            )
            
            return error_response

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="RAG Application with Open-Source Embeddings")
    
    parser.add_argument(
        "--model",
        type=str,
        choices=["all-MiniLM-L6-v2", "all-mpnet-base-v2", "multi-qa-MiniLM-L6-cos-v1"],
        default="all-MiniLM-L6-v2",
        help="Embedding model to use"
    )
    
    parser.add_argument(
        "--documents",
        type=str,
        default="./documents",
        help="Path to documents directory"
    )
    
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Run in non-interactive mode (just load documents)"
    )
    
    parser.add_argument(
        "--clear-db",
        action="store_true",
        help="Clear the vector database before starting"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the RAG application"""
    args = parse_arguments()
    
    print("=== RAG Application with Open-Source Embeddings ===")
    print(f"Using embedding model: {args.model}")
    print("=" * 55)
    
    # Get API key (optional) for the llm
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("No OpenAI API key found. You can still use document search functionality.")
        print("Set OPENAI_API_KEY environment variable for full response generation.")
    
    # Initialize the application
    app = RAGApplication(api_key, embedding_model=args.model)
    
    # Check for documents directory
    if not os.path.exists(args.documents):
        print(f"\nError: Documents directory '{args.documents}' not found.")
        print("Please create a 'documents' folder with your text files or specify path with --documents")
        return
    
    # Load documents
    app.load_documents(args.documents)
    info = app.get_collection_info()
    print(f"\nLoaded {info['document_count']} document chunks")
    print(f"Embedding dimension: {info['embedding_dimension']}")
    
    if args.no_interactive:
        print("\nRunning in non-interactive mode (documents loaded, exiting)")
        return
    
    # Interactive chat loop
    print("\n=== Interactive Chat ===")
    print("Commands: 'quit' to exit, 'clear' to clear history, 'search <query>' for document search")
    print("         'info' for collection info, 'history' to see chat history")
    print("         'responses' to see stored LLM responses, 'tokens' to see token usage")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'clear':
                app.clear_chat_history()
                print("Chat history cleared.")
                continue
            elif user_input.lower() == 'info':
                info = app.get_collection_info()
                print(f"\nCollection Info:")
                print(f"- Name: {info['name']}")
                print(f"- Documents: {info['document_count']}")
                print(f"- Embedding Model: {info['embedding_model']}")
                print(f"- Embedding Dimension: {info['embedding_dimension']}")
                continue
            
            elif user_input.lower() == 'history':
                history = app.get_chat_history()
                print("\n-------------- Chat History --------------")
                for msg in history:
                    print(f"{msg['role'].title()}: {msg['content']}\n")
                continue

            elif user_input.lower() == 'responses':
                responses = app.get_full_llm_responses()
                print(f"\n-------------- Stored LLM Responses ({len(responses)}) --------------")
                for i, resp in enumerate(responses, 1):
                    print(f"\nResponse {i} ({resp['timestamp']}):")
                    print(f"Query: {resp['query'][:80]}...")
                    print(f"Response: {resp['response'][:200]}...")
                    print(f"Model: {resp['metadata'].get('model', 'unknown')}")
                    if 'usage' in resp['metadata']:
                        usage = resp['metadata']['usage']
                        print(f"Tokens: {usage['total_tokens']} (prompt: {usage['prompt_tokens']}, completion: {usage['completion_tokens']})")
                continue
            elif user_input.lower() == 'tokens':
                tokens = app.get_token_usage()
                print(f"\nTotal tokens used: {tokens}")
                continue
            elif user_input.lower().startswith('search '):
                query = user_input[7:].strip()
                if query:
                    results = app.search_documents(query, k=3)
                    print(f"\n-------------- Search Results for: '{query}' --------------")
                    for i, result in enumerate(results, 1):
                        source = result['metadata'].get('filename', 'Unknown')
                        similarity = result['similarity_score']
                        content = result['content'][:200] + '...' if len(result['content']) > 200 else result['content']
                        print(f"\n{i}. {source} (similarity: {similarity:.3f})")
                        print(f"   {content}")
                continue
            elif not user_input:
                continue
            
            # Generate response
            response = app.generate_response(user_input)
            print(f"\nAssistant: {response}")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
