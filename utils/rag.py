import os
from typing import List

import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

load_dotenv()


def get_embedding(texts, dimensions=1024, model_provider="openai"):
    """
    输入:
        texts: List[str] 待计算embedding的文本列表
        dimensions: int   向量维度
        model_provider: str 模型提供商，支持 "ollama", "openai"
    输出:
        List[List[float]]，每个文本的embedding向量
    """
    if model_provider == "ollama":
        # 使用本地Ollama服务 (BGE-M3)
        client = OpenAI(
            api_key="ollama",  # ollama不需要真实API key
            base_url="http://localhost:11434/v1"
        )
        response = client.embeddings.create(
            model="bge-m3",
            input=texts
        )
    else:  # openai (默认)
        # 使用OpenAI官方API或其他兼容接口（默认DeepSeek）
        client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL") or os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.deepseek.com/v1",
        )
        response = client.embeddings.create(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", os.getenv("DEEPSEEK_EMBEDDING_MODEL", "deepseek-embedding")),
            input=texts,
            dimensions=dimensions
        )

    # 提取每个embedding
    try:
        # 兼容OpenAI风格，通常为response.data[i].embedding
        embeddings = [item.embedding for item in response.data]
    except Exception as e:
        raise RuntimeError(f"Get embedding failed with {model_provider}: {e}\nResponse: {response}")
    return embeddings


def _bm25_rerank(question: str, docs: List[str], top_n: int) -> List[str]:
    tokenized_docs = [doc.split() for doc in docs]
    bm25 = BM25Okapi(tokenized_docs)
    scores = bm25.get_scores(question.split())
    ranked_indices = np.argsort(scores)[::-1][:top_n]
    return [docs[i] for i in ranked_indices]


def _embedding_rerank(question: str, docs: List[str], top_n: int) -> List[str]:
    texts = [question] + docs
    embeddings = get_embedding(texts, model_provider="openai")
    query_vec, doc_vecs = embeddings[0], embeddings[1:]
    query_norm = np.linalg.norm(query_vec)
    doc_norms = np.linalg.norm(doc_vecs, axis=1)
    similarities = np.dot(doc_vecs, query_vec) / (doc_norms * query_norm + 1e-8)
    ranked_indices = np.argsort(similarities)[::-1][:top_n]
    return [docs[i] for i in ranked_indices]


def rerank(question, docs, top_n):
    """
    计算top k docs相对于question的相关性分数，然后按分数排序，返回top n个文档文本。
    优先使用本地BM25排序，不依赖专有服务，必要时再退回到 embedding 相似度。

    :param question: str，问题
    :param docs: List[str]，待排序的候选文档
    :param top_n: int，返回前top_n个
    :return: reranked_docs: List[str]，按分数降序排序的文本列表
    """
    try:
        return _bm25_rerank(question, docs, top_n)
    except Exception:
        # 当BM25不可用时，退化到embedding相似度排序
        try:
            return _embedding_rerank(question, docs, top_n)
        except Exception:
            return docs[:top_n]


# 示例用法
if __name__ == "__main__":
    print("embed test===================")
    texts = [
        '风急天高猿啸哀',
        '渚清沙白鸟飞回',
        '无边落木萧萧下',
        '不尽长江滚滚来'
    ]

    # 测试不同的embedding提供商
    providers_to_test = [
        ("ollama", "本地BGE-M3"),
        ("openai", "OpenAI官方/兼容接口"),
    ]

    for provider, description in providers_to_test:
        try:
            print(f"\n--- 测试{description} ({provider}) ---")
            vecs = get_embedding(texts, 1024, model_provider=provider)
            print(f"返回向量数: {len(vecs)}")
            print(f"每个向量长度: {len(vecs[0]) if vecs else 'N/A'}")
            if vecs and len(vecs[0]) >= 10:
                print(f"第一个向量前10维: {vecs[0][:10]}")
        except Exception as e:
            print(f"❌ {description}测试失败: {e}")
    print("\nrerank test====================")

    question = "什么是文本排序模型"
    docs = [
        "文本排序模型广泛用于搜索引擎和推荐系统中，它们根据文本相关性对候选文本进行排序",
        "量子计算是计算科学的一个前沿领域",
        "预训练语言模型的发展给文本排序模型带来了新的进展"
    ]
    try:
        result = rerank(question, docs, 2)
        print(f"Rerank结果: {result}")
    except Exception as e:
        print(f"❌ Rerank测试失败: {e}")
