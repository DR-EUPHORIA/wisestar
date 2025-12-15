# RAG 系统重构说明

本目录已重构为一个简洁、可扩展的 RAG（检索增强生成）架构。

## 架构说明

### 文件结构

1. **`base_rag.py`** - 定义 RAG 接口的抽象基类
2. **`faiss_rag.py`** - 基于 FAISS 的 RAG 实现
3. **`raptor_rag.py`** - 基于 RAPTOR 的 RAG 实现
4. **`rag_interface.py`** - RAG 系统的工厂与主接口
5. **`raptor.py`** - 兼容旧版的适配层（为向后兼容保留）

### 抽象基类（`BaseRAG`）

所有 RAG 实现均继承自 `BaseRAG`，其定义了：

- `initialize()` —— 初始化 RAG 系统
- `query(query, similarity_top_k, **kwargs)` —— 查询接口
- `load_documents(paths)` —— 从 JSON 文件加载文档
- `get_info()` —— 获取系统信息

### 具体实现

#### FAISS RAG（`FaissRAG`）

- 使用 FAISS 向量库实现高效相似度检索
- 支持持久化索引
- 可配置嵌入维度和模型

#### RAPTOR RAG (`RaptorRAG`)

- Uses hierarchical clustering and summarization
- ChromaDB for persistence
- Supports different retrieval modes ('collapsed', 'tree_traversal')

## Usage

### Basic Usage

```python
from code.RAG.rag_interface import create_rag_system

# Create a FAISS RAG system
faiss_rag = create_rag_system("faiss")
results = faiss_rag.query("解方程", similarity_top_k=3)

# Create a RAPTOR RAG system  
raptor_rag = create_rag_system("raptor")
results = raptor_rag.query("解方程", similarity_top_k=2)
```

### Advanced Configuration

```python
from code.RAG.rag_interface import RAGFactory

# Custom configuration for FAISS
faiss_rag = RAGFactory.create_rag(
    "faiss",
    data_paths=["/path/to/data.json"],
    embedding_dimension=1024,
    embedding_model_name="bge-m3"
)

# Custom configuration for RAPTOR
raptor_rag = RAGFactory.create_rag(
    "raptor",
    data_paths=["/path/to/data.json"],
    mode="tree_traversal",
    similarity_top_k=3
)
```

### Extending with New Implementations

```python
from code.RAG.base_rag import BaseRAG
from code.RAG.rag_interface import RAGFactory


class MyCustomRAG(BaseRAG):
    def initialize(self):
        # Custom initialization logic
        pass

    def query(self, query, similarity_top_k=3, **kwargs):
        # Custom query logic
        pass


# Register the new implementation
RAGFactory.register_implementation("custom", MyCustomRAG)

# Use it
custom_rag = RAGFactory.create_rag("custom", ["/path/to/data.json"])
```

## Benefits

1. **Extensibility** - Easy to add new RAG implementations
2. **Consistency** - All implementations follow the same interface
3. **Flexibility** - Configurable parameters for each implementation
4. **Maintainability** - Clear separation of concerns
5. **Backward Compatibility** - Legacy code continues to work

## Migration from Legacy Code

The old `raptor.py` file has been converted to a compatibility layer. Existing code should continue to work without
changes, but new code should use the new interface:

```python
# Old way (still works)
from code.RAG.raptor import initialize_faiss_rag, query_faiss_rag

index = initialize_faiss_rag()
results = query_faiss_rag("query")

# New way (recommended)
from code.RAG.rag_interface import create_rag_system

rag = create_rag_system("faiss")
results = rag.query("query")
```
