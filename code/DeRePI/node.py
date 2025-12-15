from pocketflow import Node
from sympy.solvers.diophantine.diophantine import prime_as_sum_of_two_squares

from utils.pyinterpreter import PythonInterpreter
from utils.llm import call_llm_stream
from utils.prompt_templates import REPI_RENODE_PROMPT, DECOMPOSER_NODE_PROMPT, DEREPI_RENODE_PROMPT, \
    DEREPI_ANSWER_NODE_PROMPT
import re
import os
import json
from dotenv import load_dotenv

# 设置最大推理次数避免进入死循环
load_dotenv()
MAX_RETRY = os.getenv("MAX_RETRY")
MAX_RETRY = int(MAX_RETRY)

'''
shared状态schema

{
    question:str
    context:str
    responses:List(str)
    actions:List(str)
    codes:List(str)
    calculation_results:List(Dict)
    node_call_counts:Dict
    answer:str
}
'''


class DecomposerNode(Node):
    def prep(self, shared):
        print("🤔 [PlannerNode] 开始规划问题...")
        return shared.get('question', '')

    def exec(self, prep_res):
        question = prep_res
        if not question:
            return []

        prompt = DECOMPOSER_NODE_PROMPT.format(question=question)

        responses = call_llm_stream(prompt)

        steps = [content.strip() for content in
                 re.findall(r"<步骤\d+>([\s\S]*?)</步骤\d+>", responses)]
        steps.append("end")

        return steps

    def post(self, shared, prep_res, exec_res):
        steps = exec_res
        shared['steps'] = steps
        shared['sub_results'] = []
        shared['context'] = f"总问题: {shared['question']}\n\n"
        shared['node_call_counts'] = {}

        print(f"🗺️ [DecomposerNode] 生成计划，共 {len(steps)} 个步骤:")
        for i, task in enumerate(steps):
            print(f"  {i + 1}. {task}")

        # 如果有计划，就启动执行流程；否则直接结束
        return "execute_plan" if steps else "end"


'''
ReNode---推理节点
需要使用的数据: question + context
功能: 进行推理 和决定下一步流向:
    - 每次都对于下一步动作进行决策,选择为calculate(执行计算代码)和answer(已经得到答案,进行回答),动作放在<action></action>标签中,
    - 如果选择calculate,则编写直接可用于执行计算的python代码,并放在<code></code>标签中,
    - 如果选择answer,则将最终答案写为markdown格式并放在<answer></answer>标签中
修改的数据: responses,actions,code
'''


class ReNode(Node):
    def prep(self, shared):
        """准备输入：现在接收的是一个子任务"""
        print(f"🛠️ [SolverNode] 开始处理子任务...")

        # 从 flow 的参数中获取当前任务
        task = shared.get('current_task', '')
        context = shared.get('context', '')
        question = shared.get('question', '')

        print(f"  - 当前任务: {task}")
        print(f"  - 当前上下文长度: {len(context)}")

        return question, task, context

    def exec(self, prep_res):

        question, task, context = prep_res

        prompt = DEREPI_RENODE_PROMPT.format(question=question, task=task, context=context)

        response = call_llm_stream(prompt)

        return response

    def post(self, shared, prep_res, exec_res):
        """解析LLM输出，决定是继续计算还是完成当前子任务"""
        # ... post部分的解析逻辑与原ReNode基本相同 ...
        # ... (此处省略相同的解析代码) ...
        response = exec_res
        action_match = re.search(r'<action>(.*?)</action>', response, re.DOTALL)
        action = action_match.group(1).strip() if action_match else None
        code_match = re.search(r'<code>(.*?)</code>', response, re.DOTALL)
        code = code_match.group(1).strip() if code_match and action == 'calculate' else None
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        answer = answer_match.group(1).strip() if answer_match else None

        if code:
            # 使用正则表达式移除可选的 ```python 和末尾的 ```
            # re.DOTALL 让 . 匹配换行符
            clean_code_match = re.search(r'^(?:```(?:python)?\n)?(.*?)(?:\n```)?$', code, re.DOTALL)
            if clean_code_match:
                code = clean_code_match.group(1).strip()
        if 'codes' not in shared: shared['codes'] = []

        # 更新shared和context状态
        shared['codes'].append(code)  # 依然需要记录code给PINode

        current_context = shared.get('context', '')
        shared['context'] = current_context + f"\n\n推理步骤：\n{response}\n"

        if action == "calculate" and code:
            print(f"🛠️ [SolverNode] ➡️  子任务需要计算，转到PINode")
            return "calculate"
        elif action == "answer":
            # 子任务完成！
            task, _, __ = prep_res
            sub_result = f"任务 '{task}' 已完成, 结果: {answer}"
            print(f"✅ [SolverNode] 子任务完成: {sub_result}")

            # 将子任务结果记录到全局上下文和结果列表
            shared['sub_results'].append(sub_result)
            shared['context'] += f"\n- {sub_result}"
            return "sub_task_complete"  # 返回一个特殊信号，表示子流程结束
        else:
            # 异常处理或默认行为
            print(f"⚠️ [SolverNode] 未明确动作，默认尝试计算")
            return "calculate"


'''
PINode---Python解释器节点
需要使用的数据: code
功能: 调用工具中的python解释器类执行code,返回计算结果和提示信息,增加上下文
修改的数据: calculation_results,context
'''


class PINode(Node):
    def __init__(self, max_retries=1, wait=0):
        super().__init__(max_retries, wait)
        self.interpreter = PythonInterpreter()

    def prep(self, shared):
        print(f"🐍 [PINode] 开始预处理...")

        # 处理shared中还未保留记录的情况
        if 'calculation_results' not in shared:
            shared['calculation_results'] = []

        # 统计PINode调用次数
        if 'PINode' not in shared.get('node_call_counts', {}):
            shared['node_call_counts']['PINode'] = 0
        shared['node_call_counts']['PINode'] += 1

        # 获取code
        code = ""
        if shared.get('codes') and shared['codes'][-1] is not None:
            code = shared['codes'][-1]

        # print(f"🐍 [PINode] 代码长度: {len(code)} 字符")
        # print(f"🐍 [PINode] 调用次数: {shared['node_call_counts']['PINode']}")
        if code:
            # print(f"🐍 [PINode] 代码预览: {code[:100]}{'...' if len(code) > 100 else ''}")
            print(f"🐍 [PINode] 代码:{code}")
        return code

    def exec(self, prep_res):
        code = prep_res

        print(f"🐍 [PINode] 开始执行Python代码...")

        if not code or not code.strip():
            print(f"🐍 [PINode] ❌ 没有代码可执行")
            calculation_result = {
                'success': False,
                'output': '',
                'error': 'No code to execute'
            }
            return calculation_result

        try:
            calculation_result = self.interpreter.execute_code(code)

            if calculation_result['success']:
                print(f"🐍 [PINode] ✅ 代码执行成功")
                if calculation_result['output']:
                    print(f"🐍 [PINode] 输出: {calculation_result['output']}")
            else:
                print(f"🐍 [PINode] ❌ 代码执行失败: {calculation_result['error']}")

            return calculation_result

        except Exception as e:
            error_msg = f"Interpreter Error: {str(e)}"
            print(f"🐍 [PINode] ❌ 解释器调用异常: {error_msg}")
            calculation_result = {
                'success': False,
                'output': '',
                'error': error_msg,
            }
            return calculation_result

    def post(self, shared, prep_res, exec_res):
        print(f"🐍 [PINode] 开始后处理阶段...")

        calculation_result = exec_res

        shared['calculation_results'].append(calculation_result)

        # 更新上下文
        if 'context' not in shared:
            shared['context'] = ''

        if calculation_result['success']:
            result_info = f"\n\n计算执行成功，结果: {calculation_result['output']},请结合计算结果继续做题."
            shared['context'] += result_info
            print(f"🐍 [PINode] 上下文已更新 (成功结果: {calculation_result['output']})")
        else:
            error_info = f"\n\n计算执行失败: {calculation_result['error']},请根据错误信息修改代码,重试计算."
            shared['context'] += error_info
            print(f"🐍 [PINode] 上下文已更新 (错误信息: {calculation_result['error']})")

        # 加入action日志
        if "actions" in shared:
            shared["actions"].append('feedback')
        else:
            shared["actions"] = ['feedback']

        # 返回下一步流向 - 回到推理节点进行反馈
        print(f"🐍 [PINode] ➡️  返回推理节点 (ReNode) 进行反馈")
        return "feedback"


'''
RetrieveStepNode
功能：从一系列步骤中顺序取回步骤，供ReNode使用
'''


class RetrieveStepNode(Node):
    def prep(self, shared):
        i = shared.get('idx', '0')
        step = shared.get('steps')[i]
        return step

    def exec(self, prep_res):
        i = prep_res
        i += 1
        return i

    def post(self, shared, prep_res, exec_res):
        shared['idx'] = exec_res
        return prep_res


'''
AnswerNode
聚合所有步骤的答案，输出最终答案
'''


class AnswerNode(Node):
    def prep(self, shared):
        print("🎉 [AggregatorNode] 所有子任务完成，开始聚合答案...")
        question = shared.get('question', '')
        context = shared.get('context', '')
        return question, context

    def exec(self, prep_res):
        question, context = prep_res
        prompt = DEREPI_ANSWER_NODE_PROMPT.format(question=question, context=context)

        response = call_llm_stream(prompt)
        print("🎉 [AggregatorNode] 调用LLM生成最终答案...")
        return response

    def post(self, shared, prep_res, exec_res):
        final_answer = exec_res
        shared['answer'] = final_answer
        print(final_answer)
        return None  # 流程结束


'''
StepManager
循环控制节点
功能：控制自任务循环
'''


class StepManagerNode(Node):
    """
    循环控制器节点。
    负责迭代任务列表，决定是继续处理下一个任务还是结束循环。
    """

    def prep(self, shared):
        """准备索引和步骤列表。"""
        # 在第一次运行时初始化索引
        if 'idx' not in shared:
            shared['idx'] = 0

        idx = shared.get('idx')
        steps = shared.get('steps', [])
        return idx, steps

    def exec(self, prep_res):
        """
        检查是否还有有效的步骤。
        返回当前要执行的任务，或者返回 None 表示循环结束。
        """
        idx, steps = prep_res
        # 检查索引是否在范围内，并确保当前步骤不是结束标记
        if idx < len(steps) and steps[idx].lower() != 'end':
            return steps[idx]  # 返回当前任务
        return None  # 表示没有更多任务了

    def post(self, shared, prep_res, exec_res):
        """
        根据 exec 结果，更新共享状态并决定下一步的流向。
        """
        current_task = exec_res
        idx, _ = prep_res

        if current_task:
            # 如果有任务，则更新共享状态并准备处理
            shared['current_task'] = current_task
            shared['idx'] = idx + 1  # 索引加一，为下一次迭代做准备
            print(f"🔄 [StepManagerNode] 准备执行第 {shared['idx']} 步: '{current_task}'")
            return "process_step"  # 返回 'process_step' 动作
        else:
            # 如果没有任务，则结束循环
            print("🔄 [StepManagerNode] 所有步骤均已执行完毕。")
            return "end_loop"  # 返回 'end_loop' 动作
