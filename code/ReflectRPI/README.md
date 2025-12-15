## Re-RAG-PI-Reflect Agent

motivation:**"开卷考试"**  
分为两大部分:离线索引,和在线Agent流程

### online_node.py中的在线流程节点

1. Re节点
    1. 解题
    2. 选择动作(工具), retrieve(RAG)或calculation(PI)
2. RAG节点
    1. 从知识库中进行检索,寻找能帮助解决当前问题的参考资料段落
3. PI节点
    1. python代码解释器,和前两个实现相同
4. Reflect节点
    1. 和ReflectPI基本相同
5. Answer节点

### offline_node.py中的离线索引节点和重排节点

1. ReadNode
    - 处理pdf格式的参考资料
2. ChunkNode
    - 文档切块
3. EmbeddNode
    - 使用embedding model生成稠密向量
4. BM25Node
    - 使用BM25生成稀疏向量
5. ClusterNode
    - 基于文本进行聚类,逐层向上生成摘要节点
6. KGNode
    - 基于LLM从文本中提取实体-关系对,生成知识图谱
7. RerankNode
    - 对粗排检索到的top k文档/知识载体,做重排选出top n

### 整体流程

![img.png](../../images/ReflectRPI.png)