"""
FAISS-based RAG implementation.

This module provides a RAG system using FAISS vector store for efficient
similarity search with persistent indexing.
"""

import os
from typing import List
from llama_index.core import Document, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.settings import Settings
from llama_index.core.schema import NodeWithScore
from utils.llamaindex_llm import CustomLLMWrapper
from code.RAG.rag_interface import BaseRAG
import faiss


class FaissRAG(BaseRAG):
    """
    FAISS-based RAG implementation.
    
    This class provides efficient similarity search using FAISS indexing
    with persistence capabilities.
    """

    def __init__(self, data_paths: List[str], **kwargs):
        """
        Initialize FAISS RAG system.
        
        Args:
            data_paths: List of paths to data files
            **kwargs: Additional configuration parameters
        """
        super().__init__(data_paths, **kwargs)
        self.index = None
        self.db_path = os.path.join(kwargs.get('db_path', 'output/rag_db/'), 'faiss')
        self.faiss_index_path = os.path.join(self.db_path, "faiss_index.index")

        # Configuration
        self.embedding_dimension = kwargs.get('embedding_dimension', 1024)
        self.embedding_model_name = kwargs.get('embedding_model_name', 'bge-m3')
        self.ollama_base_url = kwargs.get('ollama_base_url', 'http://localhost:11434')

    def initialize(self) -> None:
        """
        Initialize the FAISS RAG system components.
        """
        print("Initializing FAISS RAG system...")

        # Load documents
        documents = self.load_documents(self.data_paths)
        if not documents:
            raise ValueError("No documents could be loaded from the provided paths.")

        # Initialize LLM and embedding model
        llm = CustomLLMWrapper()
        embed_model = OllamaEmbedding(
            model_name=self.embedding_model_name,
            base_url=self.ollama_base_url
        )

        # Set global settings
        Settings.llm = llm
        Settings.embed_model = embed_model

        # Create the output directory if it doesn't exist
        os.makedirs(self.db_path, exist_ok=True)

        # Check if we have a saved FAISS index
        if len(os.listdir(self.db_path)) > 0:
            print(f"Loading existing FAISS index from {self.db_path}...")
            try:
                # Load the persisted index
                storage_context = StorageContext.from_defaults(
                    persist_dir=self.db_path,
                    vector_store=FaissVectorStore.from_persist_dir(self.db_path)
                )

                self.index = load_index_from_storage(storage_context)
                print("Successfully loaded existing FAISS index!")
            except Exception as e:
                print(f"Error loading existing FAISS index: {e}")
                print("Creating new FAISS index...")
                self._create_new_index(documents)
        else:
            print("Creating new FAISS index...")
            self._create_new_index(documents)

        self.is_initialized = True
        print("FAISS RAG system initialized successfully!")

    def _create_new_index(self, documents: List[Document]) -> None:
        """
        Create a new FAISS index from documents.
        
        Args:
            documents: List of documents to index
        """
        # Create new FAISS index with specified dimension
        faiss_index = faiss.IndexFlatIP(self.embedding_dimension)  # Inner product for cosine similarity
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        print("Building FAISS index... This may take a while as it processes the documents.")
        self.index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

        # Persist the index to disk
        self.index.storage_context.persist(persist_dir=self.db_path)
        print(f"FAISS index and data saved to {self.db_path}")

    def query(self, query: str, similarity_top_k: int = 3, **kwargs) -> List[NodeWithScore]:
        """
        Query the FAISS RAG system with a given question.
        
        Args:
            query: The question to ask the RAG system
            similarity_top_k: Number of similar documents to retrieve
            **kwargs: Additional query parameters
            
        Returns:
            List of retrieved nodes with relevance scores
        """
        self.ensure_initialized()

        if self.index is None:
            print("Failed to initialize FAISS RAG index")
            return []

        print(f"Running FAISS query: '{query}'")

        try:
            # Create a retriever
            retriever = self.index.as_retriever(similarity_top_k=similarity_top_k)

            # Retrieve relevant nodes
            nodes = retriever.retrieve(query)

            print(f"Found {len(nodes)} relevant nodes")
            for i, node in enumerate(nodes):
                score = getattr(node, 'score', 'N/A')
                content = getattr(node, 'text', str(node))
                print(f"Node {i + 1} (Score: {score}): {content[:100]}...")

            return nodes

        except Exception as e:
            print(f"Error during FAISS retrieval: {e}")
            return []

    def get_info(self) -> dict:
        """
        Get information about the FAISS RAG system.
        
        Returns:
            Dictionary containing system information
        """
        info = super().get_info()
        info.update({
            "embedding_dimension": self.embedding_dimension,
            "embedding_model": self.embedding_model_name,
            "index_path": self.faiss_index_path,
            "index_exists": os.path.exists(self.faiss_index_path)
        })
        return info


if __name__ == "__main__":
    rag = FaissRAG([
        "data/高考简单题.json"
    ])
    rag.query("解方程")
