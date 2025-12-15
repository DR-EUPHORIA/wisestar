from pocketflow import Flow
from sympy import is_zero_dimensional

from code.RAG.node import *


def create_prep_flow():
    parse = PdfParseNode()
    rewrite = RewriteNode()
    chunk = RecursiveChunkNode()

    parse >> rewrite >> chunk

    return Flow(start=parse)


# 中间这些流程实际上不需要flow,各自用一个节点就能处理了

# def create_dense_index_flow():
#     index = EmbedNode()
#
#     return Flow(start=index)
#
#
# def create_bm25_index_flow():
#     parse = PdfParseNode()
#     rewrite = RewriteNode()
#     chunk = RecursiveChunkNode()
#     index = BM25Node()
#
#     parse >> rewrite >> chunk >> index
#
#     return Flow(start=parse)


# def create_hybrid_index_flow():
#     pass
#
#
# def create_cluster_index_flow():
#     pass
#
#
# def create_graph_index_flow():
#     pass
#
#
# def create_dense_retrieve_flow():
#     pass
#
#
# def create_bm25_retrieve_flow():
#     pass
#
#
# def create_hybrid_retrieve_flow():
#     pass
#
#
# def create_graph_retrieve_flow():
#     pass


def create_rerank_generate_flow():
    rerank = RerankNode()
    generate = GenerateNode()

    rerank >> generate
    return Flow(start=rerank)


def create_rerank_summarize_generate_flow():
    rerank = RerankNode()
    summarize = SummarizeNode()
    generate = GenerateNode()

    rerank >> summarize >> generate
    return Flow(start=rerank)


def create_rag_flow(llm, embed_model, docs, reindex, rag_type='faiss'):
    rag = RAGNode(
        llm,
        embed_model,
        rag_type=rag_type,
        docs=docs,
        reindex=reindex,
    )
    generate = GenerateNode()
    rag >> generate

    return Flow(start=rag)


if __name__ == "__main__":
    # docs = glob.glob("data/高考简单题.json")
    import os
    from dotenv import load_dotenv

    load_dotenv()
    
    docs = [
        "data/高考/高考简单题.json",
        "data/高考/高考难题.json",
    ]

    
    # from llama_index.embeddings.ollama import OllamaEmbedding
    # from llama_index.llms.ollama import Ollama
    # # Initialize LLM and embedding model, Personally recommended for testing rag system cause they are FREE and fast.
    # llm = Ollama(
    #     model="qwen3:1.7b",
    #     base_url="http://localhost:11434",
    #     thinking=False
    # )
    # embed_model = OllamaEmbedding(
    #     model_name="bge-m3:latest",
    #     base_url="http://localhost:11434"
    # )
    from llama_index.llms.openai_like import OpenAILike
    from llama_index.embeddings.openai_like import OpenAILikeEmbedding
    llm = OpenAILike(
        model=os.getenv('DEEPSEEK_MODEL') or os.getenv('MODEL_NAME'),
        api_base=os.getenv('DEEPSEEK_BASE_URL') or os.getenv('LLM_BASE_URL') or "https://api.deepseek.com/v1",
        api_key=os.getenv('DEEPSEEK_API_KEY') or os.getenv('LLM_API_KEY'),
        context_window=3900,
        is_chat_model=True,
        is_function_calling_model=False,
    )
    
    # Use at Your Own Risk: raptor calls embedding model for tons of times during indexing, 
    # which significantly exceeds your Maximum TPM.
    embed_model = OpenAILikeEmbedding(
        model_name=os.getenv('DEEPSEEK_EMBEDDING_MODEL') or os.getenv('EMBED_NAME'),
        api_base=os.getenv('DEEPSEEK_BASE_URL') or os.getenv('LLM_BASE_URL') or "https://api.deepseek.com/v1",
        api_key=os.getenv('DEEPSEEK_API_KEY') or os.getenv('LLM_API_KEY'),
        additional_kwargs={
            "encoding_format": "float", # qwen embedding required such param.
        }
    )
    
    flow = create_rag_flow(
        llm,
        embed_model,
        rag_type="raptor",
        docs=docs,
        reindex=True # set True to reindex the documents, takes a while. If the database is set, switch to False.
    )
    shared = dict(
        question=r"""'$(1+5 \\mathrm{i}) \\mathrm{i}$ 的虚部为
A．-1
B． 0
C． 1
D． 6'""")
    shared["rag_topk"] = 7  # set topk for rag
    
    flow.run(shared)
    print(shared["answer"])
