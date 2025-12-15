from pocketflow import Node
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from pdf2image import convert_from_path
import pytesseract
from rag_factory import create_rag_system
from llama_index.core.llms import LLM
from llama_index.core.embeddings import BaseEmbedding

from utils.llm import call_llm_stream
from utils.rag import get_embedding, rerank
from utils.prompt_templates import (
    RAG_RWRITENODE_PROMPT,
    RAG_GENERATE_NODE_PROMPT,
    RAG_SUMMARIZE_PROMPT,
)
from code.RAG.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    ENCODING,
    EMBEDDING_DIM,
    TOP_K,
    TOP_N,
    CHUNK_SIZE_K,
    EMBEDDING_PROVIDER,
)

import os
import json
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
import jieba
from tqdm import tqdm
import re
import sys

load_dotenv()

"""

本离线索引流程的节点之间的转交借用文件系统实现,每个节点都在prep中读文件,在exec中执行所需功能,在post中写文件;
shared中只存储in&out的文件路径;

RAG schema

shared{
    input_pdf_folder_path:str,参考资料pdf所在的文件夹
    json_path:str,参考资料经过阅读和解析得到的json文件
    txt_path:str,参考资料转为json格式拼接得到的txt文件
    md_path:str,参考资料转为markdown+latex后存入的md文件
    
    chunks_path:str,对txt进行循环字符分割得到的切块json文件,.json
    dense_db_path:str,基于embedding+faiss建立的稠密向量库路径,.index
    bm25_db_path:str,基于BM25建立的知识库路径,.json
    cluster_db_path:str,类似RAPTOR的层次聚类后得到的知识库路径

    question:str,每条flow只处理一个问题,在flow外部并发
    top_k_docs:List(Dict(id,content)),粗排检索结果
    top_n_docs:List(Dict(id,content)),精排结果
    context:str,最终的上下文
    
    solution:str,题解/解题过程
    answer:str,最终答案
}
 
"""

# offline nodes=========================================================================================================

"""
ReadNode
- 处理pdf格式的参考资料,并进行分段
- 输入为pdf路径
- 使用pdf阅读工具,将文件读成str,对于编码的pdf,使用PyPDF,对于影印图像,使用OCR相关的库
- 对解析内容进行重写
- 功能过于耦合，后续拆分为PdfParser和Rewrite两部分

"""


class ReadNode(Node):
    def prep(self, shared):
        if "input_pdf_folder_path" not in shared or not shared["input_pdf_folder_path"]:
            raise ValueError(
                "缺少PDF输入文件夹路径，请确保 shared['input_pdf_folder_path'] 已设置且非空"
            )
        if "md_path" not in shared or not shared["md_path"]:
            raise ValueError("缺少输出md路径，请确保 shared['md_path'] 已设置且非空")

        in_folder_path = shared["input_pdf_folder_path"]
        pdf_files = [
            f for f in os.listdir(in_folder_path) if f.lower().endswith(".pdf")
        ]
        result = {}
        for fname in tqdm(pdf_files, desc="Processing PDFs"):
            fpath = os.path.join(in_folder_path, fname)
            reader = PdfReader(fpath)
            images = convert_from_path(fpath)
            text_list = []
            for idx, page in enumerate(
                tqdm(reader.pages, desc=f"{fname}", leave=False)
            ):
                page_text = page.extract_text() or ""
                if not page_text.strip():
                    page_img = images[idx]
                    page_text = pytesseract.image_to_string(
                        page_img, lang="chi_sim+eng"
                    )
                text_list.append(page_text)
            title = os.path.splitext(fname)[0]
            result[title] = text_list
        return result

    def exec(self, prep_res):
        def llm_convert(text):
            prompt = RAG_RWRITENODE_PROMPT.format(text=text)
            try:
                latex_text = call_llm_stream(prompt)
            except Exception:
                latex_text = text
            return latex_text

        def extract_result(response):
            match = re.search(r"<result>(.*?)</result>", response, re.DOTALL)
            if match:
                return match.group(1).strip()
            else:
                return response.strip()

        result = {}
        for title, page_text_list in tqdm(
            prep_res.items(), desc="Converting per title"
        ):
            reformatted_list = []
            for page_text in tqdm(page_text_list, desc=f"{title}", leave=False):
                if page_text.strip():
                    response = llm_convert(page_text)
                    reformatted_result = extract_result(response)
                    reformatted_list.append(reformatted_result)
                    reformatted_list.append(page_text)
                else:
                    reformatted_list.append("")
            result[title] = reformatted_list
        return result

    def post(self, shared, prep_res, exec_res):
        if "txt_path" not in shared or not shared["txt_path"]:
            raise ValueError("缺少输出txt路径，请确保 shared['txt_path'] 已设置且非空")
        if "json_path" not in shared or not shared["json_path"]:
            raise ValueError(
                "缺少输出json路径，请确保 shared['json_path'] 已设置且非空"
            )
        if "md_path" not in shared or not shared["md_path"]:
            raise ValueError("缺少输出md路径，请确保 shared['md_path'] 已设置且非空")
        txt_path = shared["txt_path"]
        json_path = shared["json_path"]
        md_path = shared["md_path"]

        # 拼接大文本
        all_text = ""
        all_md = ""
        for title, latex_texts in exec_res.items():
            all_text += f"<SOF>{title}>\n"
            all_md += f"<SOF>{title}>\n"
            for page_text in latex_texts:
                all_text += page_text + "\n"
                # md格式：每页用 --- 分隔，markdown常用；可酌情调整
                all_md += page_text + "\n\n---\n"
            all_text += f"<EOF>{title}>\n"
            all_md += f"<EOF>{title}>\n"

        # 写txt
        os.makedirs(os.path.dirname(txt_path), exist_ok=True)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(all_text)
        # 写md
        os.makedirs(os.path.dirname(md_path), exist_ok=True)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(all_md)
        # 写json
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(prep_res, f, ensure_ascii=False, indent=2)
        return "default"


"""
PdfParseNode
- 仅负责PDF读取和OCR，输出json
"""


class PdfParseNode(Node):
    """仅负责PDF读取+OCR，输出json（分步写）"""

    def prep(self, shared):
        input_pdf_folder_path = shared.get("input_pdf_folder_path")
        output_json_path = shared.get("json_path")
        if not output_json_path.lower().endswith(".json"):
            output_json_path += ".json"
            shared["json_path"] = output_json_path
        if not input_pdf_folder_path:
            raise ValueError("缺少PDF输入文件夹路径")
        if not output_json_path:
            raise ValueError("缺少输出json路径")
        return {
            "input_pdf_folder_path": input_pdf_folder_path,
            "output_json_path": output_json_path,
        }

    def exec(self, prep_res):
        in_folder = prep_res["input_pdf_folder_path"]
        output_json_path = prep_res["output_json_path"]
        pdf_files = [f for f in os.listdir(in_folder) if f.lower().endswith(".pdf")]

        # 清空文件，写入 {
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            f.write("{\n")  # 开头

        for idx, fname in enumerate(tqdm(pdf_files, desc="OCR Reading PDFs")):
            fpath = os.path.join(in_folder, fname)
            images = convert_from_path(fpath)
            text_list = []
            for page_img in tqdm(images, desc=f"{fname} OCR", leave=False):
                page_text = pytesseract.image_to_string(page_img, lang="chi_sim+eng")
                text_list.append(page_text)
            title = os.path.splitext(fname)[0]
            # 追加写入一项
            with open(output_json_path, "a", encoding="utf-8") as f:
                key_value = f"  {json.dumps(title, ensure_ascii=False)}: {json.dumps(text_list, ensure_ascii=False)}"
                if idx < len(pdf_files) - 1:
                    f.write(key_value + ",\n")
                else:
                    f.write(key_value + "\n")  # 最后一项不加逗号

        # 最后写入 }
        with open(output_json_path, "a", encoding="utf-8") as f:
            f.write("}\n")

    def post(self, shared, prep_res, exec_res):
        return "default"


"""
RewriteNode
- 对解析内容进行重写，保存至txt和md文件
"""


class RewriteNode(Node):
    def prep(self, shared):
        input_json_path = shared.get("json_path")
        output_md_path = shared.get("md_path")
        if not output_md_path.lower().endswith(".md"):
            output_md_path += ".md"
            shared["md_path"] = output_md_path
        output_txt_path = shared.get("txt_path")
        if not output_txt_path.lower().endswith(".txt"):
            output_txt_path += ".txt"
            shared["txt_path"] = output_txt_path
        if not input_json_path:
            raise ValueError("缺少输入json路径")
        if not output_md_path:
            raise ValueError("缺少输出md路径")
        if not output_txt_path:
            raise ValueError("缺少输出txt路径")
        return {
            "input_json_path": input_json_path,
            "output_md_path": output_md_path,
            "output_txt_path": output_txt_path,
        }

    def exec(self, prep_res):
        input_json_path = prep_res["input_json_path"]
        output_md_path = prep_res["output_md_path"]
        output_txt_path = prep_res["output_txt_path"]
        os.makedirs(os.path.dirname(output_md_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)

        with open(input_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        def llm_convert(text):
            prompt = RAG_RWRITENODE_PROMPT.format(text=text)
            try:
                # 使用turbo模型进行重写
                latex_text = call_llm_stream(
                    prompt, model_name=os.getenv("REWRITE_MODEL_NAME")
                )
                match = re.search(r"<result>(.*?)</result>", latex_text, re.DOTALL)
                if match:
                    latex_text = match.group(1)

            except Exception:
                latex_text = text
            return latex_text

        with (
            open(output_md_path, "a", encoding="utf-8") as fmd,
            open(output_txt_path, "a", encoding="utf-8") as ftxt,
        ):
            for title, page_list in tqdm(
                data.items(),
                desc="LLM Rewrite per file",
                file=sys.stdout,
                dynamic_ncols=True,
            ):
                new_chunk_list = []
                for i in tqdm(
                    range(0, len(page_list), CHUNK_SIZE_K),
                    desc=f"Processing '{title}' in chunks",
                    leave=False,
                    file=sys.stdout,
                    dynamic_ncols=True,
                ):
                    chunk_of_pages = page_list[i : i + CHUNK_SIZE_K]
                    separator = "\n\n---\n\n"
                    combined_text = separator.join(
                        [p.strip() for p in chunk_of_pages if p.strip()]
                    )
                    if combined_text:
                        rewritten_chunk = llm_convert(combined_text)
                        new_chunk_list.append(rewritten_chunk)

                # 写入磁盘
                fmd.write(f"# {title}\n\n")
                ftxt.write(f"<SOF>{title}>\n")
                for idx, page in enumerate(new_chunk_list):
                    fmd.write(f"## Page {idx + 1}\n\n{page}\n\n")
                    ftxt.write(page + "\n")
                ftxt.write(f"<EOF>{title}>\n")

    def post(self, shared, prep_res, exec_res):
        return "default"


"""
RecursiveChunkNode
    - 类似langchain的循环字符切块方式,按照chunk_size和overlap_size进行切分
    - 切完的chunks形式上为List(id:int,content:str),其中id为从0开始的顺序标记
    - 将chunks存储为json格式的文件,写入chunks_path中
"""


class RecursiveChunkNode(Node):
    def __init__(self):
        super().__init__()
        self.chunk_size = CHUNK_SIZE
        self.overlap = CHUNK_OVERLAP
        self.encoding = ENCODING
        self.separators = [
            "\n\n",
            "\n",
            "。",
            "！",
            "？",
            ".",
            "!",
            "?",
            "，",
            ",",
            " ",
            "",
        ]  # 中英文标点支持

    def prep(self, shared):
        txt_path = shared["txt_path"]
        with open(txt_path, "r", encoding=self.encoding) as f:
            reformat_txt = f.read()
        return reformat_txt

    def exec(self, prep_res):
        text = prep_res

        def recursive_split(text, separators, chunk_size):
            if not separators:
                # 最细不能再分直接硬切
                return [
                    text[i : i + chunk_size] for i in range(0, len(text), chunk_size)
                ]
            sep = separators[0]
            # 如果找不到分隔符，继续递归下一个
            if sep and (sep in text):
                parts = text.split(sep)
                chunks = []
                tmp = ""
                for idx, part in enumerate(parts):
                    if tmp:
                        # 预估加上分隔符长度
                        next_len = len(tmp) + len(sep) + len(part)
                    else:
                        next_len = len(part)
                    if tmp and next_len > chunk_size:
                        # 当前tmp已满,推入
                        chunks.append(tmp)
                        tmp = part
                    else:
                        if tmp:
                            tmp += sep + part
                        else:
                            tmp = part
                if tmp:
                    chunks.append(tmp)
                # 针对每个超过size的递归细分
                result = []
                for c in chunks:
                    if len(c) > chunk_size and len(separators) > 1:
                        result += recursive_split(c, separators[1:], chunk_size)
                    else:
                        result.append(c)
                return result
            else:
                # sep无效，递归下一级
                return recursive_split(text, separators[1:], chunk_size)

        all_chunks = recursive_split(text, self.separators, self.chunk_size)
        chunks_result = []

        for i, chunk in enumerate(all_chunks):
            if not chunk.strip():
                continue
            # overlap处理
            prev_content = ""
            overlap_len = 0
            j = i - 1
            while j >= 0 and overlap_len < self.overlap:
                prev_chunk = all_chunks[j]
                need = min(self.overlap - overlap_len, len(prev_chunk))
                prev_content = prev_chunk[-need:] + prev_content
                overlap_len += need
                j -= 1
            # 拼成带overlap的文本
            merged = prev_content + chunk
            # 保证不超出chunk_size
            merged = merged[-self.chunk_size :]
            chunks_result.append(
                {
                    "id": i,
                    "content": merged,
                }
            )

        return chunks_result

    def post(self, shared, prep_res, exec_res):
        chunks_path = shared["chunks_path"]
        # 自动添加 .json 后缀
        if not chunks_path.lower().endswith(".json"):
            chunks_path += ".json"
            # 可选：同步更新 shared（下游用到时更一致）
            shared["chunks_path"] = chunks_path
        # 确保目录存在
        os.makedirs(os.path.dirname(chunks_path), exist_ok=True)
        with open(chunks_path, "w", encoding=self.encoding) as f:
            json.dump(exec_res, f, ensure_ascii=False, indent=2)
        return "default"


"""
EmbedNode
    - 使用embedding model生成稠密向量,并写入faiss向量数据库
"""


class EmbedNode(Node):
    def prep(self, shared):
        # 从shared["chunks_path"]中读取chunks
        with open(shared["chunks_path"], "r", encoding=ENCODING) as f:
            chunks = json.load(f)
        return chunks

    def exec(self, prep_res):
        texts = [chunk["content"] for chunk in prep_res]
        ids = [chunk["id"] for chunk in prep_res]
        # 分批每次不超过 10 条
        BATCH = 10
        all_embeddings = []
        for i in tqdm(range(0, len(texts), BATCH), desc="Embedding Batches"):
            batch_texts = texts[i : i + BATCH]
            batch_embeddings = get_embedding(batch_texts, dimensions=EMBEDDING_DIM)
            all_embeddings.extend(batch_embeddings)
        assert len(all_embeddings) == len(texts), "Mismatch in embedding count"
        return {
            "ids": ids,
            "embeddings": all_embeddings,
        }

    def post(self, shared, prep_res, exec_res):
        # 将embedding写入FAISS向量数据库
        # FAISS写入
        ids = np.array(exec_res["ids"]).astype("int64")
        embs = np.array(exec_res["embeddings"]).astype("float32")
        db_path = shared["dense_db_path"]
        dim = embs.shape[1]

        # 自动添加 .index 后缀
        if not db_path.lower().endswith(".index"):
            db_path += ".index"
            # 可选：更新 shared["dense_db_path"]，便于后续节点统一引用
            shared["dense_db_path"] = db_path

        # 建立FAISS索引
        index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
        index.add_with_ids(embs, ids)

        faiss.write_index(index, db_path)
        print(f"FAISS index saved to {db_path}")

        return "default"


"""
BM25Node
    - 使用BM25生成稀疏向量
"""


class BM25Node(Node):
    def prep(self, shared):
        with open(shared["chunks_path"], "r", encoding="utf-8") as f:
            chunks = json.load(f)
        return chunks

    def exec(self, prep_res):
        # 1. 分词并构建 tokens list
        docs = prep_res
        # tokens 支持自定义分词，这里用 jieba，适合中文
        for doc in docs:
            doc["tokens"] = list(jieba.cut(doc["content"]))
        # 2. 按需求返回 tokens 数据
        return docs

    def post(self, shared, prep_res, exec_res):
        # 保存为vector库 (本地json存["id","content","tokens"])
        bm25_db_path = shared["bm25_db_path"]  # 如 bm25_vec.json
        if not bm25_db_path.lower().endswith(".json"):
            bm25_db_path += ".json"
            # 更新 shared，若后续节点还会用到 bm25_db_path
            shared["bm25_db_path"] = bm25_db_path
        with open(bm25_db_path, "w", encoding="utf-8") as f:
            json.dump(exec_res, f, ensure_ascii=False, indent=2)

        print(f"BM25 vector saved to {bm25_db_path}")
        return "default"



class RAGNode(Node):
    def __init__(
        self,
        llm: LLM,
        embed_model: BaseEmbedding,
        rag_type: str = "raptor",
        docs: list[str] | None = [],
        reindex: bool = False,
        **kwargs
    ):
        super().__init__()
        self.rag = create_rag_system(llm, embed_model, rag_type, docs, reindex=reindex, **kwargs)

    def prep(self, shared):
        question = shared["question"]
        topk = shared.get("rag_topk", 10)
        return {"question": question, "topk": topk}

    def exec(self, prep_res):
        print("executing RAG...")
        query = prep_res["question"]
        topk = prep_res["topk"]
        nodes_with_score = self.rag.query(query, topk)
        result = [
            {"text": node["text"], "score": node["score"]} for node in nodes_with_score
        ]
        print("result:", result)
        return {"result": result}

    def post(self, shared, prep_res, exec_res):
        result = exec_res["result"]
        print(f"Fetched nodes: {len(result)}")
        shared["rag_result"] = result
        shared["context"] = "\n".join([info["text"] for info in result])
        return "default"


# online nodes=========================================================================================================


class DenseRetrieveNode(Node):
    def prep(self, shared):
        # 读取当前question
        question = shared["question"]
        db_path = shared["dense_db_path"]
        if not os.path.exists(db_path):
            print(f"数据库文件不存在: {db_path}")
            return None
        # 计算嵌入
        question_embedding = get_embedding([question], dimensions=EMBEDDING_DIM)[
            0
        ]  # 取第一个元素
        return {
            "question": question,
            "question_embedding": question_embedding,
            "db_path": db_path,
        }

    def exec(self, prep_res):
        # 在FAISS库中查top k

        index = faiss.read_index(prep_res["db_path"])

        emb = np.array([prep_res["question_embedding"]]).astype("float32")
        similarity, related_ids = index.search(emb, TOP_K)
        ids = related_ids[0].tolist()
        return {"related_docs_id": ids}

    def post(self, shared, prep_res, exec_res):
        # 查找chunks内容，输出top k docs
        with open(shared["chunks_path"], "r", encoding="utf-8") as f:
            all_chunks = json.load(f)
        id2chunk = {chunk["id"]: chunk["content"] for chunk in all_chunks}
        # 按顺序组装 [{id, content}]
        docs = [
            {"id": _id, "content": id2chunk.get(_id, "")}
            for _id in exec_res["related_docs_id"]
        ]
        shared["top_k_docs"] = docs
        return "default"


class BM25RetrieveNode(Node):
    def prep(self, shared):
        # 读取分词后的bm25库（bm25_vec.json），内容为[{id, content, tokens}]
        question = shared["question"]
        db_path = shared["bm25_db_path"]
        with open(db_path, "r", encoding="utf-8") as f:
            docs = json.load(f)
        return {"docs": docs, "question": question}

    def exec(self, prep_res):
        """
        对 shared["question"] 的问题进行 BM25 检索
        """
        docs = prep_res["docs"]
        question = prep_res["question"]
        corpus = [doc["tokens"] for doc in docs]
        bm25 = BM25Okapi(corpus)
        q_tokens = list(jieba.cut(question))
        scores = bm25.get_scores(q_tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :TOP_K
        ]
        docs_result = [
            {
                "id": docs[idx]["id"],
                "content": docs[idx]["content"],
                # "score": float(scores[idx]),
            }
            # docs[idx]["content"]
            for idx in ranked
        ]
        return docs_result

    def post(self, shared, prep_res, exec_res):
        # 检索结果写入 shared
        shared["top_k_docs"] = exec_res  # 单条list
        return "default"


"""
HybridRetrieveNode
混合dense和bm25检索,检索后去重
"""


class HybridRetrieveNode(Node):
    def prep(self, shared):
        question = shared["question"]
        dense_db_path = shared["dense_db_path"]
        bm25_db_path = shared["bm25_db_path"]
        chunks_path = shared["chunks_path"]
        return {
            "question": question,
            "dense_db_path": dense_db_path,
            "bm25_db_path": bm25_db_path,
            "chunks_path": chunks_path,
        }

    def exec(self, prep_res):
        question = prep_res["question"]
        # ------- dense检索 -------
        dense_embedding = get_embedding([question], dimensions=EMBEDDING_DIM)[0]
        dense_index = faiss.read_index(prep_res["dense_db_path"])
        emb = np.array([dense_embedding]).astype("float32")
        _, dense_ids = dense_index.search(emb, TOP_K)
        dense_ids = dense_ids[0].tolist()

        # ------- bm25检索 -------
        with open(prep_res["bm25_db_path"], "r", encoding="utf-8") as f:
            bm25_docs = json.load(f)
        corpus = [doc["tokens"] for doc in bm25_docs]
        bm25 = BM25Okapi(corpus)
        q_tokens = list(jieba.cut(question))
        scores = bm25.get_scores(q_tokens)
        bm25_ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :TOP_K
        ]
        bm25_ids = [bm25_docs[idx]["id"] for idx in bm25_ranked]

        # ------- 合并去重 -------
        all_ids = []
        seen = set()
        # 保证dense优先，顺序依次取dense和bm25，已出现的不再放入
        for _id in dense_ids + bm25_ids:
            if _id not in seen:
                all_ids.append(_id)
                seen.add(_id)

        return {"related_docs_id": all_ids}

    def post(self, shared, prep_res, exec_res):
        # 查回内容，List[Dict(id, content)]
        with open(prep_res["chunks_path"], "r", encoding="utf-8") as f:
            all_chunks = json.load(f)
        id2chunk = {chunk["id"]: chunk["content"] for chunk in all_chunks}
        docs = [
            {"id": _id, "content": id2chunk.get(_id, "")}
            for _id in exec_res["related_docs_id"]
        ]
        shared["top_k_docs"] = docs
        return "default"


class RerankNode(Node):
    def prep(self, shared):
        # print("💬[RerankNode]处理中...")

        question = shared.get("question")
        top_k_docs = shared.get("top_k_docs")
        return {"question": question, "top_k_docs": top_k_docs}

    def exec(self, prep_res):
        question = prep_res["question"]
        top_k_docs = prep_res["top_k_docs"]
        content2doc = {doc["content"]: doc for doc in top_k_docs}
        contents = [doc["content"] for doc in top_k_docs]
        reranked_contents = rerank(question, contents, TOP_N)
        top_n_docs = [content2doc[c] for c in reranked_contents if c in content2doc]
        return top_n_docs

    def post(self, shared, prep_res, exec_res):
        shared["top_n_docs"] = exec_res

        # print("💬[RerankNode]处理结束.")
        return "default"


class SummarizeNode(Node):
    """
    对上下文进行重写
    """

    def prep(self, shared):
        # print("💬[SummarizeNode]处理中...")

        question = shared.get("question")
        docs = []
        # 优先选取精排结果,否则选择粗排结果
        if shared.get("top_n_docs"):
            docs = shared["top_n_docs"]
        elif shared.get("top_k_docs"):
            docs = shared.get("top_k_docs")
        else:
            raise Exception("没有找到相关文档")

        return {"question": question, "docs": docs}

    def exec(self, prep_res):
        question = prep_res["question"]
        docs = prep_res["docs"]
        context = ""
        # 使用LLM进行总结去重
        prompt = RAG_SUMMARIZE_PROMPT.format(
            question=question, docs="\n".join([doc["content"] for doc in docs])
        )
        response = call_llm_stream(prompt)
        match = re.search(r"<context>(.*?)</context>", response, re.DOTALL)
        if match:
            context = match.group(1)
        return context

    def post(self, shared, prep_res, exec_res):
        shared["context"] = exec_res

        # print(f"💬[SummarizeNode]处理结束.")
        return "default"


class GenerateNode(Node):
    def prep(self, shared):
        """
        生成答案
        """
        # print(f"💬[GenerateNode]生成答案中...")
        # 优先选取精排结果,否则选取粗排结果
        # related_docs = shared.get("top_n_docs") if shared.get("top_n_docs") else shared.get("top_k_docs")
        question = shared.get("question")

        context = ""
        if shared.get("context"):
            context = shared.get("context")
        elif shared.get("top_n_docs"):
            context = "\n".join([doc["content"] for doc in shared.get("top_n_docs")])
        elif shared.get("top_k_docs"):
            context = "\n".join([doc["content"] for doc in shared.get("top_k_docs")])

        # 写入context
        shared["context"] = context

        return {"context": context, "question": question}

    def exec(self, prep_res):
        # 拼接上下文
        # context = "\n".join(prep_res["related_docs"])
        context = prep_res["context"]
        prompt = RAG_GENERATE_NODE_PROMPT.format(
            question=prep_res["question"], context=context
        )
        solution = ""
        answer = ""

        response = call_llm_stream(prompt)
        solution_match = re.search(r"<solution>(.*?)</solution>", response, re.DOTALL)
        answer_match = re.search(r"<answer>(.*?)</answer", response, re.DOTALL)
        if solution_match:
            solution = solution_match.group(1)
        if answer_match:
            answer = answer_match.group(1)

        return {"solution": solution, "answer": answer}

    def post(self, shared, prep_res, exec_res):
        shared["solution"] = exec_res["solution"]
        shared["answer"] = exec_res["answer"]

        # print(f"💬[GenerateNode]生成答案结束.")
        return "default"
