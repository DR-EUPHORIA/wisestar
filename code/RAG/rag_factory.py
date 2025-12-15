"""
RAG factory and main interface.

This module provides a unified interface for creating and using different RAG implementations.
"""

from typing import List, Type, Dict
from code.RAG.rag_interface import BaseRAG
from code.RAG.faiss_rag import FaissRAG
from code.RAG.raptor_rag import RaptorRAG


class RAGFactory:
    """
    Factory class for creating RAG instances.
    """

    _implementations: Dict[str, Type[BaseRAG]] = {
        "faiss": FaissRAG,
        "raptor": RaptorRAG,
    }

    @classmethod
    def create_rag(
        cls, llm, embed_model, rag_type: str, data_paths: List[str], **kwargs
    ) -> BaseRAG:
        """
        Create a RAG instance of the specified type.

        Args:
            rag_type: Type of RAG to create ('faiss' or 'raptor')
            data_paths: List of paths to data files
            **kwargs: Additional configuration parameters

        Returns:
            Initialized RAG instance

        Raises:
            ValueError: If rag_type is not supported
        """
        if rag_type not in cls._implementations:
            available_types = list(cls._implementations.keys())
            raise ValueError(
                f"Unsupported RAG type '{rag_type}'. Available types: {available_types}"
            )

        rag_class = cls._implementations[rag_type]
        return rag_class(llm, embed_model, data_paths, **kwargs)

    @classmethod
    def register_implementation(cls, name: str, implementation: Type[BaseRAG]) -> None:
        """
        Register a new RAG implementation.

        Args:
            name: Name to register the implementation under
            implementation: RAG implementation class
        """
        cls._implementations[name] = implementation

    @classmethod
    def list_implementations(cls) -> List[str]:
        """
        List all available RAG implementations.

        Returns:
            List of implementation names
        """
        return list(cls._implementations.keys())


def create_rag_system(
    llm,
    embed_model,
    rag_type: str = "faiss",
    data_paths: list[str] | None = None,
    **kwargs,
) -> BaseRAG:
    """
    Convenience function to create a RAG system.

    Args:
        rag_type: Type of RAG to create ('faiss' or 'raptor')
        data_paths: List of paths to data files (defaults to common path)
        **kwargs: Additional configuration parameters

    Returns:
        Initialized RAG system
    """
    if data_paths is None:
        data_paths = ["/Users/dengxiangyu/dev/MathAgent/data/高考简单题.json"]

    rag_system = RAGFactory.create_rag(llm, embed_model, rag_type, data_paths, **kwargs)
    rag_system.initialize()
    return rag_system


def main():
    """
    Example usage of the RAG systems.
    """
    # Example query
    query = r"""已知 $a>0, b>0$ ，则
（A）$a^{2}+b^{2}>2 a b$
（B）$\frac{1}{a}+\frac{1}{b} \geq \frac{1}{a b}$
（C）$a+b>\sqr..."""

    print("=" * 60)
    print("RAG System Comparison")
    print("=" * 60)

    # Test FAISS RAG
    print("\n1. Testing FAISS RAG")
    print("-" * 30)
    try:
        faiss_rag = create_rag_system("faiss")
        print(f"FAISS RAG Info: {faiss_rag.get_info()}")

        results = faiss_rag.query(query, similarity_top_k=3)
        print(f"FAISS Results: {len(results)} nodes retrieved")
        if results:
            print(f"First result: {results[0].text[:100]}...")
    except Exception as e:
        print(f"Error with FAISS RAG: {e}")

    # Test RAPTOR RAG
    print("\n2. Testing RAPTOR RAG")
    print("-" * 30)
    try:
        raptor_rag = create_rag_system("raptor")
        print(f"RAPTOR RAG Info: {raptor_rag.get_info()}")

        results = raptor_rag.query(query, similarity_top_k=2)
        print(f"RAPTOR Results: {len(results)} nodes retrieved")
        if results:
            print(f"First result: {results[0].text[:100]}...")
    except Exception as e:
        print(f"Error with RAPTOR RAG: {e}")

    # Show available implementations
    print(f"\nAvailable RAG implementations: {RAGFactory.list_implementations()}")


if __name__ == "__main__":
    main()
