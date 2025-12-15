from pocketflow import Node
from dotenv import load_dotenv
from utils.llm import call_llm_stream
from utils.prompt_templates import RAG_RWRITENODE_PROMPT, RAG_GENERATE_NODE_PROMPT, RAG_SUMMARIZE_PROMPT


class ReadNode(Node):
    """
    ReadNode用于读取原始题目数据，可能涉及到预处理的内容，主要是对shared字典做一个初始化。
    需要读取的内容：
    - question
    """

    def prep(self, shared):
        pass

    def exec(self, prep_res):
        pass

    def post(self, shared, prep_res, exec_res):
        pass


class RoutingNode(Node):
    """
    RoutingNode用于对题目进行一个分类，这个地方直接用LLM+Prompt进行分类
    可分为的类别：
    - 计算题
    - 证明题
    ··· 当然，也可能还会有些其他的加进来
    然后去将Routing后的结果储存在shared字典中，返回下一步动作<action></action>
    从而去选择使用RAG或者RePI进行解答
    此处可优化成小模型来进行分类以节省资源
    """

    def prep(self, shared):
        pass

    def exec(self, prep_res):
        pass

    def post(self, shared, prep_res, exec_res):
        pass


class RAGNode(Node):
    """
    使用RAG来进行解答，直接调用实现
    """

    def prep(self, shared):
        pass

    def exec(self, prep_res):
        pass

    def post(self, shared, prep_res, exec_res):
        pass


class RePINode(Node):
    """
    使用RAG来进行解答，直接调用实现
    """

    def prep(self, shared):
        pass

    def exec(self, prep_res):
        pass

    def post(self, shared, prep_res, exec_res):
        pass


class AnswerNode(Node):
    """
    对生成的答案进行输出
    """

    def prep(self, shared):
        pass

    def exec(self, prep_res):
        pass

    def post(self, shared, prep_res, exec_res):
        pass
