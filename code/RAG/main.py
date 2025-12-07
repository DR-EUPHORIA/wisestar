from code.RAG.node import *
from pocketflow import Flow
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy

from enum import Enum
from tqdm import tqdm
import time


# 只需要提前声明预处理和重排生成/重排总结生成的Flow即可,indexing和retrieve直接使用节点(直接用节点run)

# 预处理流程
def get_prep_flow():
    parse = PdfParseNode()
    rewrite = RewriteNode()
    chunk = RecursiveChunkNode()

    parse >> rewrite >> chunk

    return Flow(start=parse)


# 重排+生成
def get_rerank_generate_flow():
    rerank = RerankNode()
    generate = GenerateNode()

    rerank >> generate
    return Flow(start=rerank)


# 重排+总结+生成
def get_rerank_summarize_generate_flow():
    rerank = RerankNode()
    summarize = SummarizeNode()
    generate = GenerateNode()

    rerank >> summarize >> generate
    return Flow(start=rerank)


# 索引模式
class Mode(Enum):
    embedding = "embedding"
    bm25 = "bm25"
    hybrid = "hybrid"
    cluster = "cluster"
    # TODO 其他索引形式


def make_shared(
        input_pdf_folder_path: str,
        json_path: str,
        txt_path: str,
        md_path: str,
        chunks_path: str,
        dense_db_path: str,
        bm25_db_path: str,
        cluster_db_path: str,
        question: str
):
    """创建shared字典（必须字段全部显式传递）"""
    return {
        "input_pdf_folder_path": input_pdf_folder_path,
        "json_path": json_path,
        "txt_path": txt_path,
        "md_path": md_path,
        "chunks_path": chunks_path,
        "dense_db_path": dense_db_path,
        "bm25_db_path": bm25_db_path,
        "cluster_db_path": cluster_db_path,
        "question": question,
        # 下方为流程中产生的中间量，初始化为空
        "top_k_docs": [],
        "top_n_docs": [],
        "context": "",
        "answer": "",
    }


def preprocess(shared):
    """1. 预处理（如 PDF to json/txt/md/chunks）"""
    prep_flow = get_prep_flow()
    prep_flow.run(shared)


def index(shared):
    """2. 离线索引（embedding、BM25等）"""
    embed_node = EmbedNode()
    bm25_node = BM25Node()
    embed_node.run(shared)
    bm25_node.run(shared)
    # 可补充其它构建索引类型，比如聚类等


def retrieve(shared, mode: Mode):
    """3. 检索"""
    if mode == Mode.embedding:
        retrieve_node = DenseRetrieveNode()
    elif mode == Mode.bm25:
        retrieve_node = BM25RetrieveNode()
    elif mode == Mode.hybrid:
        retrieve_node = HybridRetrieveNode()
    # TODO 其他检索节点
    else:
        raise NotImplementedError(f"{mode}检索未实现")
    retrieve_node.run(shared)


def generate(shared, question, summarize=False):
    """4. 精排 + 答案生成"""
    shared["question"] = question
    if summarize:
        generate_flow = get_rerank_summarize_generate_flow()
    else:
        generate_flow = get_rerank_generate_flow()
    generate_flow.run(shared)


def print_result(shared):
    """结果输出"""
    print(f"最终答案：{shared['answer']}")
    print("关联参考：")
    if shared.get("context"):
        print(shared["context"])
    else:
        for doc in shared["top_n_docs"]:
            print(doc)


"""
用于并发的答题函数:R&Q
"""


def handle_question(shared_template, question, mode, id=None):
    shared = deepcopy(shared_template)
    shared["question"] = question

    # 直接写节点和流程的 run 方法
    # 检索节点
    if mode == Mode.embedding:
        DenseRetrieveNode().run(shared)
    elif mode == Mode.bm25:
        BM25RetrieveNode().run(shared)
    elif mode == Mode.hybrid:
        HybridRetrieveNode().run(shared)

    # 生成节点/流程
    flow = get_rerank_generate_flow()
    flow.run(shared)

    # 返回问题和答案
    return {
        # "question": question,
        "id": id,
        "context": shared['context'],
        "solution": shared["solution"],
        "answer": shared["answer"],
        # "docs": shared["top_n_docs"],
    }


def AIME_evaluate(standard_answer, our_answer):
    # 只处理字符串类型，否则直接转字符串
    if not isinstance(our_answer, str):
        our_answer = str(our_answer) if our_answer is not None else ''
    match = re.search(r'\d+', our_answer)
    if match:
        extracted = match.group()
        extracted = extracted.lstrip('0')
        if extracted == '':
            extracted = '0'
        our_answer = extracted
    if our_answer == standard_answer:
        return True
    return False


def batch_run(shared_template, questions, mode, max_workers=8):
    results = []
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future2qid = {
            pool.submit(
                handle_question,
                deepcopy(shared_template),
                q["question"],
                mode,
                q["id"]
            ): q["id"]
            for q in questions
        }
        for f in tqdm(as_completed(future2qid), total=len(future2qid), desc="答题进度"):
            qid = future2qid[f]
            try:
                res = f.result()
            except Exception as e:
                print(f"处理过程中报错: {e}")
                res = {"id": qid, "answer": None, "context": None, "solution": None}
            results.append(res)
    end_time = time.time()
    print(f"\n所有问题处理完毕，用时 {end_time - start_time:.2f} 秒。")
    return results


if __name__ == "__main__":
    # 用法示例
    shared = make_shared(
        input_pdf_folder_path="../../data/rag_data/reference2",
        json_path="../../output/十年真题 常用定理/test_output06.json",
        txt_path="../../output/十年真题 常用定理/test_output08.txt",
        md_path="../../output/十年真题 常用定理//test_output08.md",
        chunks_path="../../output/knowledgebase/chunks3.json",
        dense_db_path="../../output/knowledgebase/embedding_db2.index",
        bm25_db_path="../../output/knowledgebase/bm25_db2.json",
        cluster_db_path="",
        question=""
    )

    # get_prep_flow().run(shared)
    # RecursiveChunkNode().run(shared)
    # index(shared)
    # retrieve(shared, Mode.embedding)
    # generate(shared, summarize=False)

    # shared_template = make_shared(...)  # 不带具体问题的template
    # questions = ["问题1", "问题2", ]
    mode = Mode.embedding

    # 读取数据集
    json_input_path = "../../data/AIME/AIME_1983_2025_10.json"
    # 读取原文件到 list
    with open(json_input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    question_10 = []
    for item in data:
        question_10.append(item['question'])

    # 测试单个运行
    res1 = handle_question(shared, question_10[0], mode)
    print(res1)

    # 测试并发
    # res2 = batch_answer_questions(shared, question_10, mode, 5)
    # for i, res in enumerate(res2):
    #     print(f"answer{i + 1}:{res}")
    
    
    # --------------------------------Raptor Rag Demo------------------------------------------#
    
    from llama_index.llms.openai_like import OpenAILike
    from llama_index.embeddings.openai_like import OpenAILikeEmbedding
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    docs = [
        "data/高考/高考简单题.json",
        "data/高考/高考难题.json",
    ]
    
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
    
    # from llama_index.embeddings.ollama import OllamaEmbedding
    # from llama_index.llms.ollama import Ollama
    
    # llm = Ollama(
    #     model="qwen3:1.7b",
    #     base_url="http://localhost:11434",
    #     thinking=False
    # )
    # embed_model = OllamaEmbedding(
    #     model_name="bge-m3:latest",
    #     base_url="http://localhost:11434"
    # )
    
    from .flow import create_rag_flow
    
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
