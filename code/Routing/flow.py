from pocketflow import Flow
from code.Routing.node import ReadNode, RoutingNode, AnswerNode, RAGNode, RePINode
from utils.tool_functions import print_shared


def create_Routing_Agent():
    """实现一个Routing的工作流"""

    read = ReadNode()
    Routing = RoutingNode()
    RAG = RAGNode()
    RePI = RePINode()
    answer = AnswerNode()

    read - "process" >> Routing
    Routing - "calculate" >> RePI
    Routing - "prove" >> RAG

    RePI - "answer" >> answer
    RAG - "answer" >> answer

    return Flow(start=read)


if __name__ == '__main__':
    Routing = create_Routing_Agent()
    question = ""
    shared = {"question": question}
    Routing.run(shared)

    print("shared完整信息如下=======================================")
    print_shared(shared)
