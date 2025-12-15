from pocketflow import Node
from utils.pyinterpreter import PythonInterpreter
from utils.llm import call_llm_stream
from utils.prompt_templates import REFLECTPI_RENODE_PROMPT, REFLECTPI_REFLECTNODE_PROMPT
import re
import os
from dotenv import load_dotenv

# 设置最大推理次数避免进入死循环
load_dotenv()
MAX_RETRY = int(os.getenv("MAX_RETRY"))

'''
共享数据schema:
shared{
  question:str,
  context:str,
  
  responses:List(str),
  codes:List(str),
  solutions:List(str),
  
  calculation_results:List(Dict(result:float,)),
  
  reflections:List(str),
  
  actions:List(str),
  node_call_counts:Dict(node:str,num:int)

  answer:str,
}
'''

'''
1. Re节点
    a. 入参: question, context
    b. 出参: response, code, answer
    c. action: calculate, reflect, answer(达到次数限制时跳转)
    d. 功能: 进行数学题推理,编写代码用于计算
'''


class ReNode(Node):
    def prep(self, shared):
        """准备输入：从shared中提取问题和上下文信息"""
        print(f"🧠 [ReNode] 开始预处理...")

        # 初始化所有需要的列表和统计信息
        if 'context' not in shared:
            shared['context'] = ''

        if 'responses' not in shared:
            shared['responses'] = []
        if 'codes' not in shared:
            shared['codes'] = []
        if 'solutions' not in shared:
            shared['solutions'] = []

        if 'actions' not in shared:
            shared['actions'] = []
        shared.get('actions', []).append("reasoning")
        if 'node_call_counts' not in shared:
            shared['node_call_counts'] = {}

        # 统计ReNode调用次数
        if 'ReNode' not in shared['node_call_counts']:
            shared['node_call_counts']['ReNode'] = 0
        shared['node_call_counts']['ReNode'] += 1

        question = shared.get('question', '')
        context = shared.get('context', '')

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

        prompt = REFLECTPI_RENODE_PROMPT.format(question=question, context=context)

        # 调用大模型
        response = call_llm_stream(prompt)

        return response

    def post(self, shared, prep_res, exec_res):
        """
        解析LLM输出，提取动作、代码和答案，并决定下一步流向
        """
        print(f"🧠 [ReNode] 开始后处理...")

        response = exec_res
        shared['responses'].append(response)

        # 提取动作;re.DOTALL可以跨行(跨过"\n")匹配
        action_match = re.search(r'<action>(.*?)</action>', response, re.DOTALL)
        action = action_match.group(1).strip() if action_match else None

        print(f"🧠 [ReNode] - 动作: {action}")
        shared['actions'].append(action)

        # 提取代码 - 改进的代码提取
        code = None  # code要用于后面的跳转分支所以需要声明
        code_match = re.search(r'<code>(.*?)</code>', response, re.DOTALL)
        if code_match:
            raw_code = code_match.group(1).strip()
            # 去除前面的```python、```（可带空格或回车）以及后面的```
            code = re.sub(r'^```python\s*|\s*```$', '', raw_code)
            shared['codes'].append(code)
            print(f"🧠 [ReNode] - 代码: {code}")
            shared['codes'].append(code)

        shared['actions'].append(action)

        # 提取题解
        solution_match = re.search(r'<solution>(.*?)</solution>', response, re.DOTALL)
        solution = solution_match.group(1).strip() if solution_match else None
        if solution:
            shared['solutions'].append(solution)

        # 更新上下文，添加当前推理过程(这里就简单地直接用窗口实现memory了)
        current_context = shared.get('context', '')
        shared['context'] = current_context + f"\n\n推理步骤：\n{response}\n"

        # 动作选择逻辑
        # 检查ReNode调用次数是否超过MAX_RETRY次且还没有产生答案
        renode_calls = shared.get('node_call_counts', {}).get('ReNode', 0)
        if renode_calls >= MAX_RETRY and not shared.get('answer'):
            print(f"🧠 [ReNode] ⚠️  已尝试{renode_calls}次仍未得到答案，停止解答")

            shared[
                'answer'] = f"抱歉，经过多次尝试仍无法解决该问题。问题可能过于复杂或需要更多信息。返回已有解答{solution}"
            print(f"🧠 [ReNode] ➡️  跳转到答案节点 (AnswerNode) - 答题失败")
            return "answer"  # 强制跳转到答案节点

        if action == "calculate" and code:
            next_step = "calculate"  # 进入计算节点
            print(f"🧠 [ReNode] ➡️  跳转到计算节点 (PINode)")
        elif action == "reflect":
            next_step = "reflect"  # 进入答案输出节点
            print(f"🧠 [ReNode] ➡️  跳转到reflect节点 (ReflectNode)")
        else:
            # 如果动作不明确，根据是否有代码来判断
            if code:
                next_step = "calculate"
                print(f"🧠 [ReNode] ➡️  动作不明确但有代码，跳转到计算节点 (PINode)")
            else:
                next_step = "reflect"
                print(f"🧠 [ReNode] ➡️  动作不明确且无代码，跳转到reflect节点 (ReflectNode)")

        return next_step


'''
2. PI节点
    a. 入参: code
    b. 出参: calculation_result
    c. action: feedback
    d. 功能: 编译并执行python代码, 返回执行结果和提示信息; 
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
        # if code:
        #     # print(f"🐍 [PINode] 代码预览: {code[:100]}{'...' if len(code) > 100 else ''}")
        #     # print(f"🐍 [PINode] 代码:{code}")
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
3. Reflect节点
    a. 入参: response, code, answer, 
    b. 出参: reflection
    c. action: feedback, answer
    d. 功能: 类似"教师"节点, 对Re节点交出的答案和推理过程进行反思, 审阅, 进行修改, 然后决定直接作答或者让Re节点重新作答;
'''


class ReflectNode(Node):
    def prep(self, shared):
        print(f"💬[ReflectNode]预处理中...")

        question = shared.get('question')
        solution_list = shared.get('solution', [])
        solution = solution_list[-1] if solution_list else ''

        if "reflections" not in shared:
            shared["reflections"] = []

        # 统计ReflectNode调用次数
        if 'ReflectNode' not in shared.get('node_call_counts', {}):
            shared['node_call_counts']['ReflectNode'] = 0
        shared['node_call_counts']['ReflectNode'] += 1

        return question, solution

    def exec(self, prep_res):
        print(f"💬[ReflectNode]执行中...")

        prompt = REFLECTPI_REFLECTNODE_PROMPT.format(question=prep_res[0], solution=prep_res[1])
        response = call_llm_stream(prompt)

        return response

    def post(self, shared, prep_res, exec_res):
        print(f"💬[ReflectNode]后处理中...")

        response = exec_res

        # 提取动作;re.DOTALL可以跨行(跨过"\n")匹配
        action_match = re.search(r'<action>(.*?)</action>', response, re.DOTALL)
        action = action_match.group(1).strip() if action_match else None

        print(f"💬[ReflectNode] 动作: {action}")

        # 根据动作进行后续处理
        if action == 'feedback':
            # 提取反思
            reflect_match = re.search(r'<reflect>(.*?)</reflect>', response, re.DOTALL)
            reflect = reflect_match.group(1).strip() if reflect_match else None
            shared.get('reflections').append(reflect)
            shared['context'] = shared.get('context', '') + f"\n###反思结果:{reflect}\n"
        elif action == 'answer':
            # 提取答案
            answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
            answer = answer_match.group(1).strip() if answer_match else None
            if answer:
                shared['answer'] = answer
                print(f"💬[ReflectNode] 答案: {answer}")
            else:
                shared['answer'] = None
        else:
            # 对于无效动作,直接继续feedback,相当于重新让Re节点进行推理 感觉可以有更好的处理方法
            print(f"💬[ReflectNode] ❌ 无效动作: {action}")
            action = 'feedback'

        # 记录动作
        shared["actions"].append(action)

        return action


'''
4. Answer节点
    a. 结构化地展示运算结果, 推理过程, 动作序列
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
