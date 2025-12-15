from pocketflow import Node
from sympy.solvers.diophantine.diophantine import prime_as_sum_of_two_squares
import base64
import json
from PIL import Image
from utils.pyinterpreter import PythonInterpreter
from utils.mlm import call_llm_stream_img, call_llm_stream
from utils.prompt_templates import REPI_RENODE_PROMPT, REPI_READ_PROMPT
import re
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
    img: str(url)
}
'''


class ReadNode(Node):
    def prep(self, shared):
        '''
        初始化shared，并准备进行图片解析
        '''
        if 'responses' not in shared:
            shared['responses'] = []
        if 'actions' not in shared:
            shared['actions'] = []
        if 'codes' not in shared:
            shared['codes'] = []
        if 'context' not in shared:
            shared['context'] = ''
        if 'node_call_counts' not in shared:
            shared['node_call_counts'] = {}
        if 'image' not in shared:
            shared['image'] = []
        # 准备图片
        # 获取图像数据（确保是字符串路径或字节）
        question = shared['question']
        img = shared['image_url']
        return question, img

    def exec(self, prep_res):
        '''
        调用大模型进行图片解析
        '''
        question, image = prep_res

        print("🚀开始对图片进行解析")
        print(question, image)
        prompt = REPI_READ_PROMPT.format(image=image, question=question)
        response = call_llm_stream_img(prompt, image)
        print(response)
        return response

    def post(self, shared, prep_res, exec_res):
        response = exec_res
        shared["question"] = response

        return "process"


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
        """准备输入：从shared中提取问题和上下文信息"""
        print(f"🧠 [ReNode] 开始预处理...")

        # 统计ReNode调用次数
        if 'ReNode' not in shared['node_call_counts']:
            shared['node_call_counts']['ReNode'] = 0
        shared['node_call_counts']['ReNode'] += 1

        question = shared.get('question', '')
        context = shared.get('context', '')

        # print(f"🧠 [ReNode] 问题: {question[:50]}{'...' if len(question) > 50 else ''}")
        # print(f"🧠 [ReNode] 上下文长度: {len(context)} 字符")
        # print(f"🧠 [ReNode] 调用次数: {shared['node_call_counts']['ReNode']}")

        return question, context

    def exec(self, prep_res):
        """
        提示工程;
        引导LLM:
        1.进行动作选择
        2.编写计算代码
        3.解数学题
        """
        question, context = prep_res

        print(f"🧠 [ReNode] 开始推理...")

        prompt = REPI_RENODE_PROMPT.format(question=question, context=context)

        # 调用大模型
        response = call_llm_stream(prompt)

        print(f"🧠 [ReNode] LLM响应长度: {len(response)} 字符")

        return response

    def post(self, shared, prep_res, exec_res):
        """
        解析LLM输出，提取动作、代码和答案，并决定下一步流向
        """
        print(f"🧠 [ReNode] 开始后处理...")

        response = exec_res

        # 提取动作;re.DOTALL可以跨行(跨过"\n")匹配
        action_match = re.search(r'<action>(.*?)</action>', response, re.DOTALL)
        action = action_match.group(1).strip() if action_match else None

        # 提取代码 - 改进的代码提取
        code_match = re.search(r'<code>(.*?)</code>', response, re.DOTALL)
        code = None
        if code_match:
            raw_code = code_match.group(1).strip()
            # 去除前面的```python、```（可带空格或回车）以及后面的```
            code = re.sub(r'^```python\s*|\s*```$', '', raw_code)
            # code = re.sub(r'\s*```$', '', code, flags=re.MULTILINE)
            # code = code.strip()

        # 提取答案
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        answer = answer_match.group(1).strip() if answer_match else None

        print(f"🧠 [ReNode] - 动作: {action}")
        # print(f"   - 代码: {'有' if code else '无'} ({len(code) if code else 0} 字符)")
        # print(f"   - 答案: {'有' if answer else '无'} ({len(answer) if answer else 0} 字符)")

        # 更新shared状态
        # 将当前结果添加到List中
        shared['responses'].append(response)
        shared['actions'].append(action)
        shared['codes'].append(code)

        # 只有解析出answer才放入shared
        if answer is not None:
            shared['answer'] = answer

        # 更新上下文，添加当前推理过程(这里就简单地直接用窗口实现memory了)
        current_context = shared.get('context', '')
        shared['context'] = current_context + f"\n\n推理步骤：\n{response}\n"

        # 检查ReNode调用次数是否超过MAX_RETRY次且还没有产生答案
        renode_calls = shared.get('node_call_counts', {}).get('ReNode', 0)
        if renode_calls >= MAX_RETRY and not shared.get('answer'):
            print(f"🧠 [ReNode] ⚠️  已尝试{renode_calls}次仍未得到答案，停止解答")
            shared['answer'] = "抱歉，经过多次尝试仍无法解决该问题。问题可能过于复杂或需要更多信息。"
            print(f"🧠 [ReNode] ➡️  跳转到答案节点 (AnswerNode) - 答题失败")
            return "answer"  # 强制跳转到答案节点

        # print({code},{action})
        if action == "calculate" and code:
            next_step = "calculate"  # 进入计算节点
            print(f"🧠 [ReNode] ➡️  跳转到计算节点 (PINode)")
        elif action == "answer":
            next_step = "answer"  # 进入答案输出节点
            print(f"🧠 [ReNode] ➡️  跳转到答案节点 (AnswerNode)")
        else:
            # 如果动作不明确，根据是否有代码来判断
            if code:
                next_step = "calculate"
                print(f"🧠 [ReNode] ➡️  动作不明确但有代码，跳转到计算节点 (PINode)")
            else:
                next_step = "answer"
                print(f"🧠 [ReNode] ➡️  动作不明确且无代码，跳转到答案节点 (AnswerNode)")

        return next_step


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
answer节点
需要的数据:answer,和一些统计信息
功能:返回答案
修改的数据:无
'''


class AnswerNode(Node):
    def prep(self, shared):
        print(f"📝 [AnswerNode] 开始答案输出阶段...")

        if 'answer' not in shared:
            shared['answer'] = ""
        if 'actions' not in shared:
            shared['actions'] = []

        if 'AnswerNode' not in shared['node_call_counts']:
            shared['node_call_counts']['AnswerNode'] = 0
        shared['node_call_counts']['AnswerNode'] += 1

        answer = shared.get('answer', "")
        actions = shared.get('actions', [])

        # 格式化actions历史为有序列表
        if actions:
            actions_str = "\n".join([f"{idx + 1}. {act}" for idx, act in enumerate(actions)])
        else:
            actions_str = "(无历史动作)"

        # 输出整体格式化字符串 (你可根据需要自定义 markdown 等格式)
        formatted_output = f"""📝**最终答案**: 
{answer if answer else "(未获得答案)"} 

---
**动作历史**:
{actions_str}
"""

        print(f"📝 [AnswerNode] 格式化输出已组装")
        return formatted_output

    def exec(self, prep_res):
        # 这里无需更改，直接返回整理好的字符串即可
        return prep_res

    def post(self, shared, prep_res, exec_res):
        final_output = prep_res  # 这里 prep_res 和 exec_res 都等价, 都是格式化输出字符串
        print(f"📝 [AnswerNode] 保存最终输出并结束流程...")
        print(final_output)
        print(f"📝 [AnswerNode] 🏁 流程完成")
        return None
