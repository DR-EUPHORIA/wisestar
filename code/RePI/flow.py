from pocketflow import Flow
from code.RePI.node import ReNode, PINode, AnswerNode
from utils.tool_functions import print_shared


def create_RePI_Agent(enable_memory=True):
    """
    创建RePI Agent
    
    Args:
        enable_memory: 是否启用Memory记录，默认True
        
    Returns:
        Flow: RePI工作流
    """
    re = ReNode()
    pi = PINode()
    answer = AnswerNode(enable_memory=enable_memory)

    re - "answer" >> answer
    re - "calculate" >> pi
    pi - "feedback" >> re

    return Flow(start=re)


if __name__ == '__main__':
    test_question = "若一个等比数列的前 4 项和为 4 ，前 8 项和为 68 ，则该等比数列的公比为 $\qquad$。"
    RePI = create_RePI_Agent()
    shared = {"question": test_question}
    RePI.run(shared)

    print("shared完整信息如下=======================================")
    print_shared(shared)
