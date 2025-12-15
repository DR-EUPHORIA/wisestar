## RAG

"闭卷变开卷"

### shared Schema

```
shared{
    input_pdf_folder_path:str,参考资料pdf所在的文件夹
    json_path:str,参考资料经过阅读和解析得到的json文件
    txt_path:str,参考资料转为json格式拼接得到的txt文件
    md_path:str,参考资料转为markdown+latex后存入的md文件
    chunks_path:str,对txt进行循环字符分割得到的切块json文件
    dense_db_path:str,基于embedding+faiss建立的稠密向量库路径
    bm25_db_path:str,基于BM25建立的知识库路径
    cluster_db_path:str,类似RAPTOR的层次聚类后得到的知识库路径

    question:str,每条flow只处理一个问题,在flow外部并发
    top_k_docs:List(str),粗排检索结果
    top_n_docs:List(str),精排结果
    context:str,最终的上下文
}
```

### 离线节点：

- PdfparserNode pdf进行OCR解析
- RewriteNode 对文本进行重写,转为markdown+latex格式,保存为md/txt文件
- RecursiveChunkNode 递归字符分块
- EmbedNode 使用嵌入模型生成稠密向量库
- BM25Node 使用BM25算法生成词元库
- ClusterNode 使用RAPTOR方法生成聚类摘要节点&摘要总结节点

### 在线节点

- DenseRetrieveNode 使用稠密向量库进行向量检索
- BM25RetrieveNode 使用BM25算法进行向量检索
- HybridRetrieveNode 混合Dense和BM25检索
- RerankNode 重排/精排
- GenerateNode 生成答案