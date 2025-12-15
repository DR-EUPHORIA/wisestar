"""
Abstract base class for RAG (Retrieval-Augmented Generation) systems.

This module defines the interface for extensible RAG implementations,
allowing different retrieval strategies while maintaining a consistent API.
"""

from abc import ABC, abstractmethod
from typing import List
from llama_index.core import Document
import chromadb
import hashlib
import sqlite3
from llama_index.core.llms import LLM
from llama_index.core.embeddings import BaseEmbedding


class BaseRAG(ABC):
    """
    Abstract base class for RAG systems.

    This class defines the interface that all RAG implementations must follow,
    enabling easy swapping between different retrieval strategies (FAISS, RAPTOR, etc.).
    """

    def __init__(self, 
                 llm: LLM,
                 embed_model: BaseEmbedding,
                 data_paths: List[str]):
        """
        Initialize the RAG system.

        Args:
            data_paths: List of paths to data files
            **kwargs: Additional configuration parameters
        """
        self.data_paths = data_paths
        self.is_initialized = False
        self.client_index = chromadb.PersistentClient(
            path="./output/rag_db/solutionIndex"
        )
        self.sqlite_conn = sqlite3.connect("code/RAG/db/question2solution")
        self.sqlite_cursor = self.sqlite_conn.cursor()
        self.llm = llm
        self.embed_model = embed_model
        self._create_sqlite_table()

    def _create_sqlite_table(self):
        """创建SQLite表"""
        self.sqlite_cursor.execute('''
                                   CREATE TABLE IF NOT EXISTS text_mappings
                                   (
                                       id
                                       INTEGER
                                       PRIMARY
                                       KEY
                                       AUTOINCREMENT,
                                       question
                                       TEXT
                                       NOT
                                       NULL
                                       UNIQUE,
                                       solution
                                       TEXT
                                       NOT
                                       NULL,
                                       text_hash
                                       TEXT
                                       NOT
                                       NULL
                                       UNIQUE,
                                       created_at
                                       TIMESTAMP
                                       DEFAULT
                                       CURRENT_TIMESTAMP,
                                       updated_at
                                       TIMESTAMP
                                       DEFAULT
                                       CURRENT_TIMESTAMP
                                   )
                                   ''')

    def get_solution_by_hash(self, hash):
        self.sqlite_cursor.execute("""
                                   SELECT solution
                                   from text_mappings
                                   where text_hash = ?
                                   """, (hash,))
        fetched = self.sqlite_cursor.fetchone()
        if fetched:
            return fetched[0]
        else:
            return None

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the RAG system components (embeddings, index, etc.).

        This method should set up all necessary components and set
        self.is_initialized = True upon successful completion.
        """
        pass

    @abstractmethod
    def query(
            self, query: str, similarity_top_k: int = 3, **kwargs
    ) -> List[dict]:
        """
        Query the RAG system with a given question.

        Args:
            query: The question to ask the RAG system
            similarity_top_k: Number of similar documents to retrieve
            **kwargs: Additional query parameters specific to implementation

        Returns:
            List of retrieved nodes with relevance scores
        """
        pass

    def _create_metadata(self, q):
        score = eval(q["score"].split("=")[-1])
        pairs = zip(
            [key for key in q.keys() if "得分" in key],
            [key for key in q.keys() if "结果" in key],
            [key for key in q.keys() if "错误原因" in key],
        )
        _, key_result, key_error = min(pairs, key=lambda pair: q[pair[0]] - score)

        solution = rf"""题目：
{q["question（纯文本）"]}
题解：
{q[key_result]} 
错误原因（无误则为None）:
{q[key_error]}
"""
        return solution

    def _generate_text_hash(self, text: str) -> str:
        """生成文本哈希值用作唯一标识"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def load_documents(self, paths: List[str]) -> List[Document]:
        """
        Load documents from the specified JSON files.

        Args:
            paths: List of file paths to load documents from

        Returns:
            List of Document objects
        """
        import json
        questions = []
        sqlite_data = []
        docs = []
        for data_path in paths:
            try:

                with open(data_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                    for item in data:
                        question = item.get("question（纯文本）")
                        question_hash = self._generate_text_hash(question)
                        question = question + f"\n问题标识符：{question_hash}"
                        questions.append(question)
                        solution = self._create_metadata(item)

                        sqlite_data.append((question, solution, question_hash))


            except FileNotFoundError:
                print(f"Error: The file {data_path} was not found.")
                continue
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from the file {data_path}.")
                continue

            # Extract the 'question（纯文本）' field from each item in the JSON data

            # if not question:
            #     print(f"No question found in the data file: {data_path}")
            #     continue
            docs.extend([Document(text=q) for q in questions])
            try:
                self.sqlite_cursor.executemany('''
                    INSERT OR REPLACE INTO text_mappings (question, solution, text_hash)
                    VALUES (?, ?, ?)
                ''', sqlite_data)
                self.sqlite_conn.commit()
            except Exception as e:
                self.sqlite_conn.rollback()
                print(f"Database transaction failed: {e}")
                raise e
        # id_sol = dict(zip([result.metadata() for result in results], solutions))
        # self.collection_index.add(
        #     ids=list(id_sol.keys()), documents=list(id_sol.values())
        # )
        return docs

    def ensure_initialized(self) -> None:
        """
        Ensure the RAG system is initialized before use.
        """
        if not self.is_initialized:
            self.initialize()

    def get_info(self) -> dict:
        """
        Get information about the RAG system.

        Returns:
            Dictionary containing system information
        """
        return {
            "type": self.__class__.__name__,
            "data_paths": self.data_paths,
            "initialized": self.is_initialized,
        }
