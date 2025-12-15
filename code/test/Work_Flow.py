import importlib
import sys
from pocketflow import Flow
from dotenv import load_dotenv
import os
from utils.tool_functions import print_shared
import inspect
from typing import Dict, Any, Optional
import pocketflow as pf
from utils.prompt_templates import REPI_EVALUATION_NODE_PROMPT, REPI_DISTILL_NODE_PROMPT
from utils.llm import call_llm_stream

class DistillNode(pf.Node):
    def prep(self, shared):
        answer = shared.get('answer', '未找到答案')
        shared['answer'] = answer
        return answer

    def exec(self, prep_res):
        prompt = REPI_DISTILL_NODE_PROMPT.format(prep_res=prep_res)
        distilled_answer = call_llm_stream(prompt)
        return distilled_answer

    def post(self, shared, prep_res, exec_res):
        print(f"💧 [DistillNode] 后处理，更新答案为: '{exec_res}'")
        shared['distilled_answer'] = exec_res


class EvaluationNode(pf.Node):
    def prep(self, shared):
        return {
            "model_answer": shared.get("distilled_answer", "NO_ANSWER_FOUND"),
            "ground_truth": shared.get("truth", "NO_TRUTH_PROVIDED"),
            "question": shared.get("question", "NO_QUESTION_FOUND")
        }

    def exec(self, prep_res):
        eval_prompt = REPI_EVALUATION_NODE_PROMPT.format(**prep_res)
        response = call_llm_stream(eval_prompt)

        return response

    def post(self, shared, prep_res, exec_res):
        shared['final_result'] = exec_res

def distill_and_answer(answer):
    distill = DistillNode()
    eval = EvaluationNode()
    answer >> distill
    distill >> eval
    return eval

class NodeContainer:
    def __init__(self, node_classes: Dict[str, Any]):
        """
        通过字典存储所有节点类
        - node_classes: 键为类名，值为类本身
        """
        self._node_classes = node_classes

    def __getattr__(self, name: str) -> Optional[Any]:
        """通过属性访问节点类"""
        return self._node_classes.get(name)

    def __contains__(self, name: str) -> bool:
        """检查是否包含指定节点"""
        return name in self._node_classes


def get_nodes(module_type: str) -> NodeContainer:
    """
    动态导入模块并返回所有节点类
    """
    module_path = f"code.{module_type}.node"

    try:
        node_module = importlib.import_module(module_path)
    except ImportError as e:
        print(f"导入模块失败: {e}")
        raise

    # 动态收集所有 Node 类（假设节点类名以 "Node" 结尾）
    node_classes = {
        name: obj
        for name, obj in inspect.getmembers(node_module)
        if inspect.isclass(obj) and name.endswith("Node")
    }

    # 校验必需节点
    required_nodes = ["ReNode", "PINode", "AnswerNode"]
    missing_nodes = [n for n in required_nodes if n not in node_classes]
    if missing_nodes:
        raise ImportError(f"模块 {module_path} 缺少必需的节点: {', '.join(missing_nodes)}")

    return NodeContainer(node_classes)


def create_RePI_Agent(node_container):
    re = node_container.ReNode()
    pi = node_container.PINode()
    answer = node_container.AnswerNode()


    re - "answer" >> answer
    re - "calculate" >> pi
    pi - "feedback" >> re

    return Flow(start=re),answer

def create_ReflectPI_Agent(node_container):
    re = node_container.ReNode()
    pi = node_container.PINode()
    answer = node_container.AnswerNode()
    reflect = node_container.ReflectNode()

    re - "calculate" >> pi
    re - "reflect" >> reflect
    re - "answer" >> answer
    pi - "feedback" >> re
    reflect - "feedback" >> re
    reflect - "answer" >> answer

    return Flow(start=re),answer

def create_DeRePI_Agent(node_container):
    decomposer = node_container.DecomposerNode()
    step_manager = node_container.StepManagerNode()
    re_node = node_container.ReNode()
    pi_node = node_container.PINode()
    answer_node = node_container.AnswerNode()

    # 定义节点之间的有向连接和条件分支
    # 1. 分解器完成后，如果成功，则启动步骤管理器
    decomposer - "execute_plan" >> step_manager

    # 2. 步骤管理器决定是处理下一步，还是结束循环
    step_manager - "process_step" >> re_node  # 如果有下一个步骤，则交给推理节点
    step_manager - "end_loop" >> answer_node  # 如果所有步骤完成，则去生成最终答案

    # 3. 推理-计算子循环
    re_node - "calculate" >> pi_node  # 如果推理结果是计算，则调用Python解释器
    re_node - "sub_task_complete" >> step_manager  # 如果子任务完成，则返回步骤管理器获取下一步
    pi_node - "feedback" >> re_node  # Python代码执行后，将结果反馈给推理节点

    return Flow(start=decomposer),answer_node

def select_flow(module_type,node_container):
    if module_type == "RePI":
        test_pipeline, answer = create_RePI_Agent(node_container)
    elif module_type == "ReflectPI":
        test_pipeline, answer = create_ReflectPI_Agent(node_container)
    elif module_type == "DeRePI":
        test_pipeline, answer = create_DeRePI_Agent(node_container)
    else:
        raise ValueError(f"未知的模块类型: {module_type}")
    distill_and_answer(answer)
    return test_pipeline

if __name__ == "__main__":
    load_dotenv()
    module_type = os.getenv("MODULE_TYPE", "default_model")

    # 动态导入节点类
    node_container = get_nodes(module_type)
    print(node_container.ReNode)
    print("🔍 识别到的节点列表：")

    # 遍历并输出所有节点
    for node_name, node_class in node_container._node_classes.items():
        print(f"- {node_name}: {node_class.__name__}")

    # 根据模块类型选择流程创建函数
    test_flow = select_flow(module_type,node_container)

    test_question = "若一个等比数列的前 4 项和为 4 ，前 8 项和为 68 ，则该等比数列的公比为 $\qquad$。"
    shared = {"question": test_question}
    # 运行流程
    test_flow.run(shared)

    print("shared完整信息如下=======================================")
    print_shared(shared)