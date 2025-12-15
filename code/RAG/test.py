from code.RAG.node import *


# from multiprocessing import freeze_support


def main():
    shared = {
        "input_pdf_folder_path": "../../data/rag_data/reference2",
        "json_path": "../../output/十年真题 常用定理/test_output06.json",
        "txt_path": "../../output/十年真题 常用定理/test_output07.txt",
        "md_path": "../../output/十年真题 常用定理//test_output07.md",
        "chunks_path": "../../output/knowledgebase/chunks3.json"
    }

    """将所有执行逻辑封装在一个函数中"""
    print("Starting pipeline...")

    # # 步骤 1: 解析 PDF
    # parse_node = PdfParseNode()
    # print("Step 1: Parsing PDFs...")
    # parse_node.run(shared)
    # print("PDF parsing complete.")

    # 步骤 2: 重写内容
    rewrite_node = RewriteNode()
    print("Step 2: Rewriting content...")
    rewrite_node.run(shared)
    print("Content rewriting complete.")

    print("Pipeline finished successfully.")

    # 步骤3：chunking
    # chunk_node = RecursiveChunkNode()
    #
    # chunk_node.run(shared)

    # 步骤4:embedding
    # 4.1 dense embedding
    shared["dense_db_path"] = "../../output/knowledgebase/embedding1.index"
    #
    # embed_node = EmbedNode()
    # embed_node.run(shared)

    # 4.2 bm25 embedding
    shared["bm25_db_path"] = "../../output/knowledgebase/bm25db1.json"
    # bm25_node = BM25Node()
    # bm25_node.run(shared)

    # 步骤5:retrieval
    shared["question"] = """
        **已知函数 \( f(x) = x^4 - 4x^2 + 2 \)。**

        1. 求 \( f(x) \) 的最小值及取得最小值的 \( x \)；
        2. 设 \( a, b, c, d \) 是该函数的所有实数极值点和零点，求 \( S = a^2 + b^2 + c^2 + d^2 \) 的值。"""

    # # 5.1:dense retrieval
    # dense_retrieve_node = DenseRetrieveNode()
    # dense_retrieve_node.run(shared)
    # print("\ndense retrieval results:")
    # top_k_docs=shared["top_k_docs"]
    # for i in range(len(top_k_docs)):
    #     print(f"\ndoc{i+1}:\n", top_k_docs[i])

    # # 5.2:bm25 retrieval
    # bm25_retrieve_node = BM25RetrieveNode()
    # bm25_retrieve_node.run(shared)
    # print("\nbm25 retrieval results:")
    # top_k_docs = shared["top_k_docs"]
    # for i in range(len(top_k_docs)):
    #     print(f"\ndoc{i+1}:\n", top_k_docs[i])

    # # 5.3: hybrid retrieval
    # hybrid_retrieve_node = HybridRetrieveNode()
    # hybrid_retrieve_node.run(shared)
    # print("\nhybrid retrieval results:")
    # top_k_docs = shared["top_k_docs"]
    # for i in range(len(top_k_docs)):
    #     print(f"\ndoc{i + 1}:\n", top_k_docs[i])
    #
    # # 6. rerank
    # rerank_node = RerankNode()
    # rerank_node.run(shared)
    # print("\nrerank results:")
    # top_n_docs = shared["top_n_docs"]
    # for i in range(len(top_n_docs)):
    #     print(f"\ndoc{i + 1}:\n", top_n_docs[i])
    #
    # # 7. summarize: LLM对片段整合去重清洗（可选）
    # summarize_node = SummarizeNode()
    # summarize_node.run(shared)
    # print("\nSummarized context:")
    # print(shared["context"])
    #
    # # 8. generate: 生成最终答案
    # generate_node = GenerateNode()
    # generate_node.run(shared)
    # print("\nGenerated answer:")
    # print(shared["answer"])


# ==============================================================================
#  程序入口点 (The Guard)
# ==============================================================================
if __name__ == '__main__':
    # 在Windows上，如果计划将脚本打包成exe，freeze_support()是必需的。
    # 即使不打包，加上它也是一个好习惯。
    # 调用主函数
    rag()
