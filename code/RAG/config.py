import os

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
ENCODING = "utf-8"
DB_MODE = "dense"
EMBEDDING_DIM = 1024
TOP_K = 5
TOP_N = 3
CHUNK_SIZE_K = 5

# 嵌入模型配置
EMBEDDING_PROVIDER = "openai"  # 支持 "ollama"、"openai" 等 OpenAI 兼容接口，默认使用 DeepSeek 兼容端点
