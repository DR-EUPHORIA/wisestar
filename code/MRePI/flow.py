from pocketflow import Flow
from code.MRePI.node import ReNode, PINode, AnswerNode, ReadNode
from utils.tool_functions import print_shared


def create_MRePI_Agent():
    re = ReNode()
    pi = PINode()
    answer = AnswerNode()
    read = ReadNode()

    read - "process" >> re
    re - "answer" >> answer
    re - "calculate" >> pi
    pi - "feedback" >> re

    return Flow(start=read)


if __name__ == '__main__':
    image_url = "https://gitee.com/hwangpengyu/math-picture/raw/master/image/10.png"
    question = "在平面直角坐标系 xOy 中，曲线 C1 : y2 = 4x，曲线 C2 : (x − 4)2 + y2 = 8．经过 C1上一点 P 作一条倾斜角为 45◦ 的直线 l，与 C2 交于两个不同的点 Q, R，求 |P Q| · |P R|的取值范围"
    RePI = create_MRePI_Agent()
    shared = {"image_url": image_url,
              "question": question}
    RePI.run(shared)

    print("shared完整信息如下=======================================")
    print_shared(shared)
