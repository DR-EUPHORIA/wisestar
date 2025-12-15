"""
RAPTOR-based RAG implementation.

This module provides a RAG system using RAPTOR (Recursive Abstractive Processing 
for Tree-Organized Retrieval) with hierarchical clustering and summarization.
"""
import nest_asyncio
import os
import logging
from typing import List, Optional

import chromadb
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import QueryBundle
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.packs.raptor import RaptorPack, RaptorRetriever
from llama_index.vector_stores.chroma import ChromaVectorStore

from code.RAG.rag_interface import BaseRAG

nest_asyncio.apply()


class RaptorRAG(BaseRAG):
    """
    RAPTOR-based RAG implementation.
    
    This class provides hierarchical retrieval using RAPTOR's tree-structured
    approach with clustering and summarization.
    """

    def __init__(
        self,
        llm,
        embed_model,
        data_paths: List[str],
        **kwargs):
        """
        Initialize RAPTOR RAG system.
        
        Args:
            data_paths: List of paths to data files
            **kwargs: Additional configuration parameters
        """
        super().__init__(
            llm,
            embed_model,
            data_paths
            )
        self.raptor_pack = None
        self.retriever = None

        # Configuration
        self.db_path = os.path.join(kwargs.get('db_path', 'code/RAG/db/'), 'raptor')
        self.collection_name = kwargs.get('collection_name', 'raptor')
        self.similarity_top_k = kwargs.get('topk', 2)
        self.mode = kwargs.get('mode', 'collapsed')  # 'collapsed' or 'tree_traversal'
        self.ollama_base_url = kwargs.get('ollama_base_url', 'http://localhost:11434')
        self.reindex = kwargs.get('reindex', False)

        # Initialize ChromaDB client_index and vector store
        print("DB path:", self.db_path)
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(self.collection_name)
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)

        # # Initialize LLM and embedding model
        # self.llm = Ollama(
        #     model=self.llm_model_name,
        #     base_url=self.ollama_base_url,
        #     thinking=False
        # )
        # self.embed_model = OllamaEmbedding(
        #     model_name=self.embedding_model_name,
        #     base_url=self.ollama_base_url
        # )

        # Setup logger
        self.logger = logging.getLogger("RaptorRAG")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        if not self.logger.handlers:
            log_path = kwargs.get('log_path', 'code/RAG/log/rag_query.log')
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            file_handler = logging.FileHandler(log_path)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def initialize(self) -> None:
        """
        Initialize the RAPTOR RAG system components.
        """
        print("Initializing RAPTOR RAG system...")
        # If reindexing, clear the existing collection first.
        if self.reindex and self.collection.count() > 0:
            print(f"Re-indexing requested. Clearing existing collection '{self.collection_name}'...")
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(self.collection_name)
            self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
            print("Collection cleared.")

        # Check if the collection_index is empty to decide whether to build a new index.
        if self.collection.count() == 0:
            print("Creating new RAPTOR index...")
            # Load documents and create nodes
            documents = self.load_documents(self.data_paths)
            if not documents:
                raise ValueError("No documents could be loaded from the provided paths.")

            unique_texts = set()
            unique_documents = []
            for doc in documents:
                if doc.text not in unique_texts:
                    unique_texts.add(doc.text)
                    unique_documents.append(doc)
            print(f"Loaded {len(documents)} documents, found {len(unique_documents)} unique documents.")

            parser = SimpleNodeParser()
            nodes = parser.get_nodes_from_documents(unique_documents)

            # Create RAPTOR pack
            self.raptor_pack = RaptorPack(
                nodes,
                embed_model=self.embed_model,
                llm=self.llm,
                vector_store=self.vector_store,
                similarity_top_k=self.similarity_top_k,
                mode=self.mode,
            )

        else:
            print("Loading existing RAPTOR index...")
        self.retriever = RaptorRetriever(
            [],  # Empty nodes list since we're loading from persistence
            embed_model=self.embed_model,
            llm=self.llm,
            vector_store=self.vector_store,
            similarity_top_k=self.similarity_top_k,
            mode=self.mode,
        )

        self.is_initialized = True
        print("RAPTOR RAG system initialized successfully!")

    def query(self, query: str, topk: Optional[int] = None, mode: Optional[str] = None, **kwargs) -> List[
        dict]:
        """
        Query the RAPTOR RAG system with a given question.
        
        Args:
            query: The question to ask the RAG system
            similarity_top_k: Number of similar documents to retrieve (overrides default)
            mode: Retrieval mode ('collapsed' or 'tree_traversal', overrides default)
            **kwargs: Additional query parameters
            
        Returns:
            List of retrieved nodes with relevance scores
        """
        self.ensure_initialized()

        if self.retriever is None:
            print("Failed to initialize RAPTOR retriever")
            return []

        # Use provided parameters or fall back to defaults
        top_k = topk if topk is not None else self.similarity_top_k
        retrieval_mode = mode if mode is not None else self.mode

        print(f"Running RAPTOR query: '{query}' (mode: {retrieval_mode}, top_k: {top_k})")
        self.logger.info(f"Running RAPTOR query: '{query}' (mode: {retrieval_mode}, top_k: {top_k})")

        try:
            # Update retriever settings if needed
            if hasattr(self.retriever, 'similarity_top_k'):
                self.retriever.similarity_top_k = top_k
            # Note: mode is typically set during initialization and may not be changeable

            # Create query engine and retrieve
            query_engine = RetrieverQueryEngine.from_args(self.retriever, llm=self.llm)
            nodes = query_engine.retrieve(QueryBundle(query))

            print(f"Found {len(nodes)} relevant nodes")
            self.logger.info(f"Found {len(nodes)} relevant nodes for query: '{query}'")

            records = []
            for node in nodes:
                question_hash = node.text.split("问题标识符：")[-1]
                self.logger.info(f"Retrieved node with hash: {question_hash} and score: {node.score}")
                full_text = self.get_solution_by_hash(question_hash)
                if full_text is None:
                    full_text = node.text
                
                record = dict(text=full_text, score=node.score)
                records.append(record)
            return records
            # return [
            #     dict(text=self.get_solution_by_hash(node.text.split("问题标识符：")[-1])[1], score=node.score)
            #     for node in nodes]
            # return nodes

        except Exception as e:
            print(f"Error during RAPTOR retrieval: {e}")
            self.logger.error(f"Error during RAPTOR retrieval for query '{query}': {e}", exc_info=True)
            return []

    def get_info(self) -> dict:
        """
        Get information about the RAPTOR RAG system.
        
        Returns:
            Dictionary containing system information
        """
        info = super().get_info()
        info.update({
            "db_path": self.db_path,
            "collection_name": self.collection_name,
            "similarity_top_k": self.similarity_top_k,
            "mode": self.mode,
            "embedding_model": self.embed_model.class_name,
            "llm_model": self.llm.class_name,
            "db_exists": os.path.exists(self.db_path)
        })
        return info


if __name__ == "__main__":
    rag = RaptorRAG(
        ["/Users/dengxiangyu/dev/MathAgent/data/test.json"]
    )
    rag.initialize()
    nodes = rag.query("虚部")
    print(nodes)
