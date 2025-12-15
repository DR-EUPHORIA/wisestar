from pocketflow import Node, AsyncNode
from utils.pyinterpreter import PythonInterpreter
from utils.llm import call_llm_stream, call_llm_stream_async
from utils.prompt_templates import REPI_RENODE_PROMPT
import re
import asyncio
import os
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

'''
ReNode---推理节点
需要使用的数据: question + context
功能: 进行推理 和决定下一步流向:
    - 每次都对于下一步动作进行决策,选择为calculate(执行计算代码)和answer(已经得到答案,进行回答),动作放在<action></action>标签中,
    - 如果选择calculate,则编写直接可用于执行计算的python代码,并放在<code></code>标签中,
    - 如果选择answer,则将最终答案写为markdown格式并放在<answer></answer>标签中
修改的数据: responses,actions,code
'''


class ReNode(AsyncNode):
    async def prep_async(self, shared):
        # 1. 从 self.params 获取本任务的独立数据
        task_id = self.params.get('id')
        question = self.params.get('question', '')

        # 将task_id也存入shared，方便后续节点日志打印
        shared['id'] = task_id

        print(f"🧠 [ReNode] ID: {task_id} - 开始预处理...")

        # 2. shared 字典仍然用于存储和传递 *可变的状态*
        # 如果是第一次运行，则在shared中初始化状态
        if 'context' not in shared:
            shared['context'] = ''
            shared['responses'] = []
            shared['actions'] = []
            shared['codes'] = []
            shared['node_call_counts'] = {}

        shared['node_call_counts']['ReNode'] = shared['node_call_counts'].get('ReNode', 0) + 1

        # 从 shared 获取当前的状态
        context = shared.get('context', '')

        # 3. 将所有需要的数据传递给 exec_async
        return question, context, task_id

    async def exec_async(self, prep_res):
        question, context, task_id = prep_res
        print(f"🧠 [ReNode] ID: {task_id} - 开始推理...")
        prompt = REPI_RENODE_PROMPT.format(question=question, context=context)
        response = await call_llm_stream_async(prompt)
        return response

    # 关键修改：将 def post(...) 重命名为 async def post_async(...)
    async def post_async(self, shared, prep_res, exec_res):
        task_id = shared.get('id')
        print(f"🧠 [ReNode] ID: {task_id} - 开始后处理...")
        response = exec_res

        action_match = re.search(r'<action>(.*?)</action>', response, re.DOTALL)
        action = action_match.group(1).strip() if action_match else None
        code_match = re.search(r'<code>(.*?)</code>', response, re.DOTALL)
        code = None
        if code_match:
            raw_code = code_match.group(1).strip()
            code = re.sub(r'^```python\s*|\s*```$', '', raw_code)
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        answer = answer_match.group(1).strip() if answer_match else None

        print(f"🧠 [ReNode] ID: {task_id} - 动作: {action}")

        shared['responses'].append(response)
        shared['actions'].append(action)
        shared['codes'].append(code)
        if answer is not None:
            shared['answer'] = answer
        shared['context'] += f"\n\n推理步骤：\n{response}\n"

        renode_calls = shared.get('node_call_counts', {}).get('ReNode', 0)
        if renode_calls >= MAX_RETRY and not shared.get('answer'):
            print(f"🧠 [ReNode] ID: {task_id} ⚠️  已尝试{renode_calls}次仍未得到答案，停止解答")
            shared['answer'] = "抱歉，经过多次尝试仍无法解决该问题。"
            return "answer"

        if action == "calculate" and code:
            next_step = "calculate"
        elif action == "answer":
            next_step = "answer"
        else:
            # 健壮性增强：即使LLM未按要求生成action，也根据有无代码来决定下一步
            print(f"🧠 [ReNode] ID: {task_id} ⚠️  未找到明确的 action，将根据是否有代码来决策。")
            next_step = "calculate" if code else "answer"

        print(f"🧠 [ReNode] ID: {task_id} ➡️  跳转到: {next_step}")
        return next_step


'''
PINode---Python解释器节点
需要使用的数据: code
功能: 调用工具中的python解释器类执行code,返回计算结果和提示信息,增加上下文
修改的数据: calculation_results,context
'''


class PINode(AsyncNode):
    # 为PINode增加一个可配置的超时参数
    def __init__(self, max_retries=1, wait=0, timeout_seconds=10):
        super().__init__(max_retries, wait)
        self.interpreter = PythonInterpreter()
        self.timeout_seconds = timeout_seconds  # 设置超时时间

    async def prep_async(self, shared):
        task_id = shared.get('id')
        print(f"🐍 [PINode] ID: {task_id} - 开始预处理...")
        if 'calculation_results' not in shared:
            shared['calculation_results'] = []
        shared['node_call_counts']['PINode'] = shared['node_call_counts'].get('PINode', 0) + 1
        code = shared.get('codes', [])[-1] if shared.get('codes') else ""
        if code:
            print(f"🐍 [PINode] ID: {task_id} - 待执行代码: \n---\n{code}\n---")
        return code, task_id

    async def exec_async(self, prep_res):
        code, task_id = prep_res

        if not code or not code.strip():
            print(f"🐍 [PINode] ID: {task_id} ❌ 没有代码可执行")
            return {'success': False, 'output': '', 'error': 'No code to execute'}

        print(f"🐍 [PINode] ID: {task_id} - 开始执行Python代码 (超时限制: {self.timeout_seconds}秒)...")

        try:
            # 使用 asyncio.wait_for 来包装 to_thread 调用，实现超时控制
            calculation_result = await asyncio.wait_for(
                asyncio.to_thread(self.interpreter.execute_code, code),
                timeout=self.timeout_seconds
            )
            print(f"🐍 [PINode] ID: {task_id} ✅ 代码执行完成。")
            return calculation_result
        except asyncio.TimeoutError:
            # 如果超时，捕获 TimeoutError 异常
            error_msg = f"Execution timed out after {self.timeout_seconds} seconds."
            print(f"🐍 [PINode] ID: {task_id} ❌ {error_msg}")
            return {'success': False, 'output': '', 'error': error_msg}
        except Exception as e:
            # 捕获其他可能的执行错误
            error_msg = f"Interpreter Error: {str(e)}"
            print(f"🐍 [PINode] ID: {task_id} ❌ {error_msg}")
            return {'success': False, 'output': '', 'error': error_msg}

    async def post_async(self, shared, prep_res, exec_res):
        task_id = shared.get('id')
        print(f"🐍 [PINode] ID: {task_id} - 开始后处理...")
        calculation_result = exec_res
        shared['calculation_results'].append(calculation_result)

        if calculation_result['success']:
            result_info = f"\n\n计算执行成功，结果: {calculation_result['output']},请结合计算结果继续做题."
            shared['context'] += result_info
        else:
            # 将超时或执行错误信息反馈给ReNode
            error_info = f"\n\n计算执行失败: {calculation_result['error']},请根据错误信息修改代码,重试计算或调整思路."
            shared['context'] += error_info

        shared['actions'].append('feedback')
        print(f"🐍 [PINode] ID: {task_id} ➡️  返回推理节点 (ReNode) 进行反馈")
        return "feedback"


'''
answer节点
需要的数据:answer,和一些统计信息
功能:返回答案
修改的数据:无
'''


class AnswerNode(Node):  # AnswerNode是同步的，保持不变
    def prep(self, shared):
        task_id = shared.get('id')
        print(f"📝 [AnswerNode] ID: {task_id} - 开始答案输出阶段...")

        shared['node_call_counts']['AnswerNode'] = shared['node_call_counts'].get('AnswerNode', 0) + 1

        return {
            "id": task_id,  # 传递id给exec
            "answer": shared.get('answer', "抱歉，未能找到问题的答案。"),
            "actions": shared.get('actions', []),
            "node_call_counts": shared.get('node_call_counts', {})
        }

    def exec(self, prep_res):
        task_id = prep_res['id']
        print(f"📝 [AnswerNode] ID: {task_id} - 正在格式化最终输出...")

        answer = prep_res["answer"]
        actions = prep_res["actions"]
        node_call_counts = prep_res["node_call_counts"]

        node_call_counts_without_ans = {k: v for k, v in node_call_counts.items() if k != 'AnswerNode'}

        actions_str = "\n".join([f"{idx + 1}. {act}" for idx, act in enumerate(actions)]) if actions else "(无)"
        stats_str = "节点调用次数：\n" + "".join([f"- {k}: {v}次\n" for k, v in
                                                 node_call_counts_without_ans.items()]) if node_call_counts_without_ans else "(无节点调用统计)"

        formatted_output = f"""## 最终答案
{answer}
---
### 动作历史 (Actions)
{actions_str}
---
### {stats_str}
"""
        print(f"📝 [AnswerNode] ID: {task_id} ✅ 最终输出已格式化")
        return formatted_output

    def post(self, shared, prep_res, exec_res):
        final_output = exec_res
        shared['final_output'] = final_output
        print(f"📝 [AnswerNode] ID: {shared.get('id')} 🏁 流程完成")
        return None
