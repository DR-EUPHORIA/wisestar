from pocketflow import Flow
from code.ReflectPI.node import ReNode, ReflectNode, PINode, AnswerNode
from utils.tool_functions import print_shared


def create_ReflectPI_Agent():
    re = ReNode()
    pi = PINode()
    answer = AnswerNode()
    reflect = ReflectNode()

    re - "calculate" >> pi
    re - "reflect" >> reflect
    re - "answer" >> answer
    pi - "feedback" >> re
    reflect - "feedback" >> re
    reflect - "answer" >> answer

    return Flow(start=re)


if __name__ == '__main__':
    test_question = "若一个等比数列的前 4 项和为 4 ，前 8 项和为 68 ，则该等比数列的公比为 $\qquad$。"
    ReflectPI = create_ReflectPI_Agent()
    shared = {"question": test_question}
    ReflectPI.run(shared)

    print("shared完整信息如下=======================================")
    print_shared(shared)
