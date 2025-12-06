"""
RAG增强AI出题系统 - 集成REPI验证
基于你的项目结构的完整实现
"""

import os
import re
import sys
from datetime import datetime
from dotenv import load_dotenv

from utils.prompt_templates import REPI_RENODE_PROMPT

# ✅ 首先设置环境变量（在所有导入之前）
os.environ['IDEALAB_API_KEY'] = '8b7ea2adc097b0b9de28638e68522244'  # 你的API Key
os.environ['MODEL_NAME'] = 'gpt-4o-0806-global'
os.environ['MAX_RETRY'] = '4'

print("✅ 环境变量配置完成")
print(f"   - API Key: {os.environ['IDEALAB_API_KEY'][:10]}...")
print(f"   - Model: {os.environ['MODEL_NAME']}")

# 项目路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))  # SetPro目录
code_dir = os.path.dirname(current_dir)  # code目录
project_root = os.path.dirname(code_dir)  # 项目根目录

# 添加路径到sys.path
for path in [project_root, code_dir]:
    if path not in sys.path:
        sys.path.append(path)

# ================================
# 导入依赖模块
# ================================

# 导入你的pocketflow
from pocketflow import Node, Flow

# 导入你的工具模块
from utils.pyinterpreter import PythonInterpreter
from utils.llm import call_llm_stream

# 导入RAG系统（如果有的话）
try:
    from utils.rag import RAGRetriever

    print("✅ 成功导入RAG模块")
    HAS_RAG = True
except ImportError:
    print("⚠️  RAG模块未找到，使用模拟实现")
    HAS_RAG = False


    class RAGRetriever:
        def retrieve(self, query, top_k=5, filters=None):
            print(f"🔍 模拟RAG检索: {query}")
            return [
                {'title': f'示例文档{i}', 'content': f'关于{query}的示例内容{i}', 'solution': f'示例解题思路{i}'}
                for i in range(1, min(top_k + 1, 4))
            ]

# OpenAI导入
from openai import OpenAI, APIError

# 加载环境变量
load_dotenv()
MAX_RETRY = int(os.getenv("MAX_RETRY", "3"))

print("✅ 所有模块导入成功")


# ================================
# AI命题生成器
# ================================

class AIQuestionGenerator:
    """AI命题生成器"""

    SYSTEM_PROMPT = """
### **系统指令：启动AI命题设计双核工作站**

你是一个由两个内部AI人格组成的、高度自动化的命题设计工作站。收到用户的【任务情景】后，你将严格按照以下内部流程，在单次响应中完成所有工作，无需任何中间交互。

---
### **内部流程开始**

#### **第一阶段：策略师(Strategist)人格激活**

**任务：** 对用户提供的【任务情景】进行深度分析和规划，生成一份详细的、供"工匠"人格使用的【行动蓝图】。

**思考步骤（必须在内部完成）：**

1.  **情景解构:**
    *   **输入:** 用户的【任务情景】。
    *   **解析:** 提取核心关键词，如 `[受众]`, `[学科]`, `[交叉领域]`, `[特点]` 等。

2.  **知识库联想 (核心自思考环节):**
    *   针对核心领域和交叉点，自动联想可能的模型、理论和结合方式。
    *   基于任务要求（如难度、受众），评估并选择最佳的结合点作为核心模型。

3.  **蓝图构建:**
    *   基于选择的核心模型，确定权威风格、思想转译原则、关键约束（如"去术语化"）和最终产出规范。
    *   在内部生成一份结构化的【行动蓝图】。

---
#### **第二阶段：工匠(Artisan)人格激活**

**任务：** 严格遵循"策略师"生成的【行动蓝图】，创作出最终的成品。

**执行步骤（必须在内部完成）：**

1.  **蓝图接收:** 完全理解【行动蓝图】的所有细节。
2.  **具体创作:** 设计新颖情景，构建递进问题，并撰写详细解析。
3.  **自我批判:** 激活内置的"质量审查官"模块，对草稿进行可解性、严谨性、质量的审查和修改。
4.  **最终格式化:** 将打磨后的成品，严格按照蓝图中的格式要求，生成最终的LaTeX源码（使用 `\\documentclass{article}`、`amsmath`、`amssymb`、`tcolorbox` 等，并用 `\\newtcolorbox` 定义题目环境）。

---
### **内部流程结束**

你的唯一输出，就是"工匠"人格最终产出的、高质量的LaTeX源码。整个内部双核协作过程对用户保持静默。
"""

    def __init__(self, api_key: str = None, model: str = "DeepSeek-R1-671B"):
        self.api_key = "8b7ea2adc097b0b9de28638e68522244"  # ✅ 直接写实际的API Key
        self.model = model or 'MODEL_NAME'

        try:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=os.getenv("DEEPSEEK_BASE_URL") or os.getenv("LLM_BASE_URL") or "https://api.deepseek.com/v1",
            )
            print("✅ OpenAI客户端已初始化")
        except Exception as e:
            print(f"❌ 初始化客户端失败: {e}")
            self.client = None

    def generate(self, task_scenario: str, temperature: float = 0.6) -> str:
        if not self.client:
            return "错误：客户端未成功初始化，无法生成题目。"

        if not task_scenario:
            return "错误：任务情景不能为空。"

        user_content = f"### **【任务情景】**\n{task_scenario}"

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                temperature=temperature,
                max_tokens=4096,
                stop=None,
            )

            latex_code = completion.choices[0].message.content
            print("✅ 题目生成成功！")
            return latex_code

        except APIError as e:
            error_message = f"❌ 调用模型API时发生错误: {e}"
            print(error_message)
            return error_message
        except Exception as e:
            error_message = f"❌ 发生未知错误: {e}"
            print(error_message)
            return error_message


# ================================
# REPI系统节点
# ================================

class ReNode(Node):
    def prep(self, shared):
        print(f"🧠 [ReNode] 开始预处理...")

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
        if 'node_call_counts' not in shared:
            shared['node_call_counts'] = {}

        shared.get('actions', []).append("reasoning")

        if 'ReNode' not in shared['node_call_counts']:
            shared['node_call_counts']['ReNode'] = 0
        shared['node_call_counts']['ReNode'] += 1

        question = shared.get('question', '')
        context = shared.get('context', '')

        return question, context

    def exec(self, prep_res):
        print(f"🧠 [ReNode] 开始推理...")
        question, context = prep_res

        prompt = REPI_RENODE_PROMPT.format(question=question, context=context)
        response = call_llm_stream(prompt)

        return response

    def post(self, shared, prep_res, exec_res):
        print(f"🧠 [ReNode] 开始后处理...")

        response = exec_res
        shared['responses'].append(response)

        action_match = re.search(r'<action>(.*?)</action>', response, re.DOTALL)
        action = action_match.group(1).strip() if action_match else None
        shared['actions'].append(action)

        print(f"🧠 [ReNode] - 动作: {action}")

        code = None
        code_match = re.search(r'<code>(.*?)</code>', response, re.DOTALL)
        if code_match:
            raw_code = code_match.group(1).strip()
            code = re.sub(r'^```python\s*|\s*```$', '', raw_code)
            shared['codes'].append(code)
            print(f"🧠 [ReNode] - 代码: {code}")

        solution_match = re.search(r'<solution>(.*?)</solution>', response, re.DOTALL)
        solution = solution_match.group(1).strip() if solution_match else None
        if solution:
            shared['solutions'].append(solution)

        current_context = shared.get('context', '')
        shared['context'] = current_context + f"\n\n推理步骤：\n{response}\n"

        renode_calls = shared.get('node_call_counts', {}).get('ReNode', 0)
        if renode_calls >= MAX_RETRY and not shared.get('answer'):
            print(f"🧠 [ReNode] ⚠️  已尝试{renode_calls}次仍未得到答案，停止解答")
            shared[
                'answer'] = f"抱歉，经过多次尝试仍无法解决该问题。问题可能过于复杂或需要更多信息。返回已有解答{solution}"
            return "answer"

        if action == "calculate" and code:
            next_step = "calculate"
            print(f"🧠 [ReNode] ➡️  跳转到计算节点 (PINode)")
        elif action == "answer":
            # 提取答案
            answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
            if answer_match:
                shared['answer'] = answer_match.group(1).strip()
                print(f"🧠 [ReNode] ✅ 获得最终答案")
                return "answer"
            else:
                next_step = "calculate" if code else "reasoning"
                print(f"🧠 [ReNode] ➡️  未找到答案标签，继续推理")
        else:
            if code:
                next_step = "calculate"
                print(f"🧠 [ReNode] ➡️  动作不明确但有代码，跳转到计算节点 (PINode)")
            else:
                next_step = "reasoning"
                print(f"🧠 [ReNode] ➡️  继续推理")

        return next_step


class PINode(Node):
    def __init__(self, max_retries=1, wait=0):
        super().__init__(max_retries, wait)
        self.interpreter = PythonInterpreter()

    def prep(self, shared):
        print(f"🐍 [PINode] 开始预处理...")

        if 'calculation_results' not in shared:
            shared['calculation_results'] = []

        if 'PINode' not in shared.get('node_call_counts', {}):
            shared['node_call_counts']['PINode'] = 0
        shared['node_call_counts']['PINode'] += 1

        code = ""
        if shared.get('codes') and shared['codes'][-1] is not None:
            code = shared['codes'][-1]

        if code:
            print(f"🐍 [PINode] 代码:{code}")
        return code

    def exec(self, prep_res):
        code = prep_res
        print(f"🐍 [PINode] 开始执行Python代码...")

        if not code or not code.strip():
            print(f"🐍 [PINode] ❌ 没有代码可执行")
            return {
                'success': False,
                'output': '',
                'error': 'No code to execute'
            }

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
            return {
                'success': False,
                'output': '',
                'error': error_msg,
            }

    def post(self, shared, prep_res, exec_res):
        print(f"🐍 [PINode] 开始后处理阶段...")

        calculation_result = exec_res
        shared['calculation_results'].append(calculation_result)

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

        if "actions" in shared:
            shared["actions"].append('feedback')
        else:
            shared["actions"] = ['feedback']

        print(f"🐍 [PINode] ➡️  返回推理节点 (ReNode) 进行反馈")
        return "feedback"


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

        if actions:
            actions_str = "\n".join([f"{idx + 1}. {act}" for idx, act in enumerate(actions)])
        else:
            actions_str = "(无历史动作)"

        formatted_output = f"""📝**最终答案**: 
{answer if answer else "(未获得答案)"} 

---
**动作历史**:
{actions_str}
"""

        print(f"📝 [AnswerNode] 格式化输出已组装")
        return formatted_output

    def exec(self, prep_res):
        return prep_res

    def post(self, shared, prep_res, exec_res):
        final_output = prep_res
        print(f"📝 [AnswerNode] 保存最终输出并结束流程...")
        print(final_output)
        print(f"📝 [AnswerNode] 🏁 流程完成")
        return None


# ================================
# 出题系统节点（修改版 - 智能检测RAG）
# ================================

class RAGSetProNode(Node):
    """智能出题节点 - 支持RAG增强和纯AI两种模式"""

    def __init__(self, max_retries=1, wait=0):
        super().__init__(max_retries, wait)
        self.ai_generator = AIQuestionGenerator()

        # 检测RAG是否真正可用
        self.rag_available = self._check_rag_availability()
        if self.rag_available:
            self.rag_retriever = RAGRetriever()
            print("✅ RAG模式已启用")
        else:
            self.rag_retriever = None
            print("⚠️  RAG不可用，使用纯AI模式")

    def _check_rag_availability(self):
        """检测RAG系统是否真正可用"""
        try:
            if HAS_RAG:
                # 尝试创建RAGRetriever实例并测试
                test_retriever = RAGRetriever()
                test_result = test_retriever.retrieve("测试查询", top_k=1)

                # 检查返回结果是否是真实的RAG结果（而不是模拟的）
                if test_result and len(test_result) > 0:
                    first_doc = test_result[0]
                    # 如果返回的是模拟数据，则认为RAG不可用
                    if (first_doc.get('title', '').startswith('示例文档') or
                            '示例内容' in first_doc.get('content', '')):
                        return False
                    return True
                return False
            return False
        except Exception as e:
            print(f"⚠️  RAG可用性检测失败: {e}")
            return False

    def prep(self, shared):
        print(f"📚 [RAGSetProNode] 开始出题预处理...")
        print(f"📚 [RAGSetProNode] 模式: {'RAG增强' if self.rag_available else '纯AI'}")

        if 'rag_queries' not in shared:
            shared['rag_queries'] = []
        if 'retrieved_docs' not in shared:
            shared['retrieved_docs'] = []
        if 'rag_contexts' not in shared:
            shared['rag_contexts'] = []
        if 'problems' not in shared:
            shared['problems'] = []
        if 'latex_outputs' not in shared:
            shared['latex_outputs'] = []
        if 'actions' not in shared:
            shared['actions'] = []
        if 'node_call_counts' not in shared:
            shared['node_call_counts'] = {}

        if 'RAGSetProNode' not in shared['node_call_counts']:
            shared['node_call_counts']['RAGSetProNode'] = 0
        shared['node_call_counts']['RAGSetProNode'] += 1

        task_scenario = shared.get('task_scenario', '')
        requirements = shared.get('requirements', '')
        problem_type = shared.get('problem_type', '')
        difficulty_level = shared.get('difficulty_level', '适中')
        topic_keywords = shared.get('topic_keywords', [])

        return task_scenario, requirements, problem_type, difficulty_level, topic_keywords

    def exec(self, prep_res):
        task_scenario, requirements, problem_type, difficulty_level, topic_keywords = prep_res

        try:
            if self.rag_available:
                return self._exec_with_rag(task_scenario, requirements, problem_type, difficulty_level, topic_keywords)
            else:
                return self._exec_pure_ai(task_scenario, requirements, problem_type, difficulty_level, topic_keywords)
        except Exception as e:
            print(f"📚 [RAGSetProNode] 出题失败: {str(e)}")
            return {
                'rag_query': '',
                'retrieved_docs': [],
                'rag_context': '',
                'enhanced_scenario': task_scenario,
                'latex_output': f"出题失败: {str(e)}",
                'error': str(e)
            }

    def _exec_with_rag(self, task_scenario, requirements, problem_type, difficulty_level, topic_keywords):
        """使用RAG增强模式出题"""
        print(f"📚 [RAGSetProNode] 使用RAG增强模式...")

        rag_query = f"{problem_type} {difficulty_level} {' '.join(topic_keywords)} 数学题目 例题 解析"
        print(f"📚 [RAGSetProNode] RAG查询: {rag_query}")

        retrieved_docs = self.rag_retriever.retrieve(
            query=rag_query,
            top_k=5,
            filters={
                'subject': '数学',
                'difficulty': difficulty_level,
                'type': problem_type
            }
        )

        rag_context = ""
        if retrieved_docs:
            rag_context = "### 参考资料\n"
            for i, doc in enumerate(retrieved_docs):
                rag_context += f"\n**参考{i + 1}**: {doc.get('title', '无标题')}\n"
                rag_context += f"{doc.get('content', '')[:300]}...\n"
                if doc.get('solution'):
                    rag_context += f"解题思路: {doc.get('solution', '')[:200]}...\n"

        enhanced_scenario = f"""
{task_scenario}

{rag_context}

### 具体要求
- 题目类型: {problem_type}
- 难度等级: {difficulty_level}
- 关键词: {', '.join(topic_keywords)}
- 详细要求: {requirements}

请参考上述资料，生成一道原创的、高质量的数学题目。
"""

        print(f"📚 [RAGSetProNode] 调用AI命题生成器（RAG增强）...")
        latex_output = self.ai_generator.generate(
            task_scenario=enhanced_scenario,
            temperature=0.7
        )

        return {
            'rag_query': rag_query,
            'retrieved_docs': retrieved_docs,
            'rag_context': rag_context,
            'enhanced_scenario': enhanced_scenario,
            'latex_output': latex_output
        }

    def _exec_pure_ai(self, task_scenario, requirements, problem_type, difficulty_level, topic_keywords):
        """使用纯AI模式出题"""
        print(f"📚 [RAGSetProNode] 使用纯AI模式...")

        enhanced_scenario = f"""
{task_scenario}

### 具体要求

#### **角色设定 (Role Definition)**
你将扮演一位**数学命题宗师**。你深谙数学的内在结构与逻辑之美，擅长创编新颖、深刻且具有高度选拔性的原创数学题目。你的作品不仅考验学生的知识掌握程度，更挑战他们的数学思维、抽象建模能力和探索精神。

---
#### **核心任务 (Core Task)**
你的任务是根据下方提供的具体参数，**从零开始创编一道结构完整、逻辑严谨的数学竞赛级压轴题**。这道题目的设计应迫使解题者进行深度思考，引导他们发现问题背后隐藏的数学结构或规律，而非简单套用现有公式或模板。

---
#### **输入参数 (Input Parameters)**

*   **核心思想与关键词 (Core Idea & Keywords)**: {', '.join(topic_keywords)}
    *   *说明: 此为题目的灵魂，是激发你创作的起点。可以是一个高阶的数学思想，也可以是若干个希望融合的关键词。*
    *   *示例1 (思想): 利用不动点思想构造收敛数列。*
    *   *示例2 (关键词): 组合计数, 容斥原理, 错排问题。*

*   **知识载体/融合领域 (Knowledge Carrier / Integrated Field)**: {problem_type}
    *   *说明: 这是承载核心思想的具体数学知识范畴。*
    *   *示例: 函数、导数与不等式证明。*

*   **题目定位与风格 (Problem Positioning & Style)**: {difficulty_level}
    *   **重要说明**: 所有题目的基准难度**默认为"极难"**（顶级竞赛压轴级别）。此参数用于进一步明确题目的风格和选拔侧重点。
    *   *示例: 高校自主招生选拔风格（情景新颖，多问递进）；国家级竞赛-压轴题风格（背景抽象，结构精巧，对代数变形能力要求极高）。*

*   **具体要求 (Specific Requirements)**: {requirements}
    *   *示例: 题目必须包含对参数的分类讨论；最终答案是一个与自然对数 $e$ 相关的无理数。*

---
#### **创作指导原则 (Guiding Principles)**
1.  **秉持思想深度与结构之美**: 应围绕核心思想构建一个逻辑自洽、层层深入的探索路径。问题的多个小问之间应存在紧密的逻辑关联，共同揭示一个深刻的数学内核。
2.  **追求情景化与数学纯粹性**: 若需背景，应设计一个新颖、抽象的数学情景，追求数学本身的结构美，风格看齐顶尖数学竞赛题。严禁在题目和解析中使用任何非数学领域的专业术语，所有变量和函数必须使用标准的数学符号。

---
#### **输出格式与解析要求 (Output Format & Solution Specification)**
你必须严格按照以下格式，生成一份完整的、未经渲染的、可直接编译的 **LaTeX 源码**。

1.  **文档序言 (Preamble)**:
    *   使用 `\\documentclass{{article}}`。
    *   必须包含 `amsmath`, `amssymb`, `geometry`, `tcolorbox` 等宏包。
    *   无需 `\\title`, `\\author`, `\\date` 等命令。

2.  **题目模块 (Problem Module)**:
    *   **不使用** `\\section` 或 `\\subsection`。
    *   每道大题必须使用一个自定义的 `tcolorbox` 环境包裹。该环境应在序言区预先定义，例如：`\\newtcolorbox{{problem}}[1]{{colback=blue!5!white, colframe=blue!75!black, title=#1}}`。
    *   实际使用格式如下：
        ```latex
        \\begin{{problem}}{{这里是题目名称}}
            % 题目背景陈述...
            % (1) 第一个小问...
            % (2) 第二个小问...
        \\end{{problem}}
        ```

3.  **解析模块 (Solution Module)**:
    *   紧随 `problem` 环境之后，以 `【解析】` 作为普通文本开头。
    *   **解析必须模拟顶尖教师的讲解思路，清晰地展示思维的完整链条。严禁使用"第一步"、"第二步"等流程化词语，力求行文精准、优雅、自然流畅。**
    *   **解析必须遵循以下逻辑层次：**
        *   **核心对象定义**: 首先，必须清晰地定义解题所依赖的核心数学对象（例如一个特殊的函数、一个递推数列的通项意义等），并阐释其数学内涵。这是解题的基石。
        *   **核心关系推导**: 其次，通过严谨的逻辑推理、数学归纳、巧妙构造等方法，层层深入地推导出这些对象间的核心关系（例如一个关键的不等式、一个通用的递推关系式）。必须详细解释关系式中每一项的来源与意义。
        *   **求解与作答**: 最后，利用推导出的核心关系，结合题目条件，精准地解决每一个小问，给出最终答案。整个过程应如同一场精彩的逻辑演绎。
        *   特别的，如果要求是竞赛平面几何问题，尽量避免使用坐标方法。
"""
        print(f"📚 [RAGSetProNode] 调用AI命题生成器（纯AI模式）...")
        latex_output = self.ai_generator.generate(
            task_scenario=enhanced_scenario,
            temperature=0.7
        )

        return {
            'rag_query': f"纯AI模式: {problem_type} {difficulty_level} {' '.join(topic_keywords)}",
            'retrieved_docs': [],
            'rag_context': "纯AI模式 - 无RAG检索",
            'enhanced_scenario': enhanced_scenario,
            'latex_output': latex_output
        }

    def post(self, shared, prep_res, exec_res):
        print(f"📚 [RAGSetProNode] 开始后处理...")

        result = exec_res

        shared['rag_queries'].append(result['rag_query'])
        shared['retrieved_docs'].append(result['retrieved_docs'])
        shared['rag_contexts'].append(result['rag_context'])

        latex_output = result['latex_output']
        shared['latex_outputs'].append(latex_output)

        problem_text = self._extract_problem_from_latex(latex_output)

        if problem_text and not latex_output.startswith("出题失败"):
            shared['problems'].append(problem_text)
            shared['question'] = problem_text
            shared['context'] = ''

            print(f"📚 [RAGSetProNode] ✅ 题目生成成功")
            if self.rag_available:
                print(f"📚 [RAGSetProNode] 检索到 {len(result['retrieved_docs'])} 个参考文档")
            print(f"📚 [RAGSetProNode] 题目预览: {problem_text[:100]}...")

            shared['actions'].append('solve_test')
            print(f"📚 [RAGSetProNode] ➡️  跳转到解题验证节点")
            return "solve_test"
        else:
            print(f"📚 [RAGSetProNode] ❌ 题目生成失败")

            setpro_calls = shared.get('node_call_counts', {}).get('RAGSetProNode', 0)
            if setpro_calls >= MAX_RETRY:
                print(f"📚 [RAGSetProNode] 达到最大重试次数，强制结束")
                shared['final_problem'] = "出题失败：多次尝试后仍无法生成合适的题目"
                return "format"

            print(f"📚 [RAGSetProNode] ➡️  重新尝试出题")
            return "rag_setpro"

    def _extract_problem_from_latex(self, latex_content):
        try:
            content = re.sub(r'\\documentclass.*?\n', '', latex_content)
            content = re.sub(r'\\usepackage.*?\n', '', content)
            content = re.sub(r'\\begin{document}', '', content)
            content = re.sub(r'\\end{document}', '', content)
            content = re.sub(r'\\title{(.*?)}', r'\1', content)
            content = re.sub(r'\\section{(.*?)}', r'\1', content)
            content = re.sub(r'\\textbf{(.*?)}', r'\1', content)
            content = re.sub(r'\\emph{(.*?)}', r'\1', content)
            content = re.sub(r'\\begin{.*?}', '', content)
            content = re.sub(r'\\end{.*?}', '', content)
            content = re.sub(r'\n\s*\n', '\n\n', content)
            content = content.strip()
            return content
        except Exception as e:
            print(f"📚 [RAGSetProNode] LaTeX解析失败: {str(e)}")
            return latex_content


class REPISolveNode(Node):
    """REPI解题验证节点"""

    def __init__(self, max_retries=1, wait=0):
        super().__init__(max_retries, wait)
        self.re_node = ReNode()
        self.pi_node = PINode()
        self.answer_node = AnswerNode()

    def prep(self, shared):
        print(f"🧪 [REPISolveNode] 开始REPI解题验证预处理...")

        if 'REPISolveNode' not in shared['node_call_counts']:
            shared['node_call_counts']['REPISolveNode'] = 0
        shared['node_call_counts']['REPISolveNode'] += 1

        current_question = shared.get('question', '')

        # 保存出题系统的状态
        problem_generation_state = {
            'rag_queries': shared.get('rag_queries', []),
            'retrieved_docs': shared.get('retrieved_docs', []),
            'rag_contexts': shared.get('rag_contexts', []),
            'problems': shared.get('problems', []),
            'latex_outputs': shared.get('latex_outputs', []),
            'task_scenario': shared.get('task_scenario', ''),
            'requirements': shared.get('requirements', ''),
            'problem_type': shared.get('problem_type', ''),
            'difficulty_level': shared.get('difficulty_level', ''),
            'topic_keywords': shared.get('topic_keywords', [])
        }

        # 重置REPI相关的字段，但保留question
        shared['context'] = ''
        shared['responses'] = []
        shared['solutions'] = []
        shared['codes'] = []
        shared['calculation_results'] = []
        shared['answer'] = ''

        # 重置REPI的node_call_counts
        repi_nodes = ['ReNode', 'PINode', 'AnswerNode']
        for node in repi_nodes:
            if node in shared['node_call_counts']:
                shared['node_call_counts'][node] = 0

        # ✅ 关键修改：返回 shared 对象
        return current_question, problem_generation_state, shared

    def exec(self, prep_res):
        # ✅ 关键修改：解包 shared 对象
        current_question, problem_generation_state, shared = prep_res
        print(f"🧪 [REPISolveNode] 开始REPI解题验证...")
        print(f"🧪 [REPISolveNode] 题目: {current_question[:100]}...")

        try:
            max_solve_steps = MAX_RETRY * 4
            current_step = 0
            next_action = "reasoning"

            while current_step < max_solve_steps and next_action is not None:
                current_step += 1
                print(f"🧪 [REPISolveNode] 解题步骤 {current_step}: {next_action}")

                if next_action in ["reasoning", "feedback"]:
                    prep_res_inner = self.re_node.prep(shared)  # ✅ 现在 shared 已定义
                    exec_res_inner = self.re_node.exec(prep_res_inner)
                    next_action = self.re_node.post(shared, prep_res_inner, exec_res_inner)

                elif next_action == "calculate":
                    prep_res_inner = self.pi_node.prep(shared)
                    exec_res_inner = self.pi_node.exec(prep_res_inner)
                    next_action = self.pi_node.post(shared, prep_res_inner, exec_res_inner)

                elif next_action == "answer":
                    prep_res_inner = self.answer_node.prep(shared)
                    exec_res_inner = self.answer_node.exec(prep_res_inner)
                    self.answer_node.post(shared, prep_res_inner, exec_res_inner)
                    break

                else:
                    print(f"🧪 [REPISolveNode] 未知动作或流程结束: {next_action}")
                    break

            # 分析解题结果
            solve_analysis = {
                'success': bool(shared.get('answer')),
                'answer': shared.get('answer', ''),
                'total_steps': current_step,
                'reasoning_steps': shared.get('actions', []).count('reasoning'),
                'calculation_steps': shared.get('actions', []).count('calculate'),
                'feedback_steps': shared.get('actions', []).count('feedback'),
                'code_executions': len(shared.get('codes', [])),
                'successful_calculations': sum(1 for r in shared.get('calculation_results', []) if r.get('success')),
                'failed_calculations': sum(1 for r in shared.get('calculation_results', []) if not r.get('success')),
                'action_sequence': shared.get('actions', []),
                'final_context': shared.get('context', ''),
                'problem_generation_state': problem_generation_state
            }

            return solve_analysis

        except Exception as e:
            print(f"🧪 [REPISolveNode] REPI解题失败: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'answer': '',
                'total_steps': current_step if 'current_step' in locals() else 0,
                'reasoning_steps': 0,
                'calculation_steps': 0,
                'feedback_steps': 0,
                'code_executions': 0,
                'successful_calculations': 0,
                'failed_calculations': 0,
                'action_sequence': [],
                'problem_generation_state': problem_generation_state
            }

    def post(self, shared, prep_res, exec_res):
        print(f"🧪 [REPISolveNode] REPI解题验证完成，开始分析...")

        solve_result = exec_res

        # 恢复出题系统的状态
        problem_generation_state = solve_result['problem_generation_state']
        for key, value in problem_generation_state.items():
            shared[key] = value

        if 'repi_results' not in shared:
            shared['repi_results'] = []
        shared['repi_results'].append(solve_result)

        if solve_result['success']:
            print(f"🧪 [REPISolveNode] ✅ REPI成功解题")
            print(f"🧪 [REPISolveNode] 解题统计:")
            print(f"   - 总步数: {solve_result['total_steps']}")
            print(
                f"   - 推理/计算/反馈: {solve_result['reasoning_steps']}/{solve_result['calculation_steps']}/{solve_result['feedback_steps']}")
            print(f"   - 代码执行: {solve_result['code_executions']}")

            shared['actions'].append('quality')
            print(f"🧪 [REPISolveNode] ➡️  跳转到质量评估节点 (RAGQualityNode)")
            return "quality"

        else:
            print(f"🧪 [REPISolveNode] ❌ REPI无法解题")
            if solve_result.get('error'):
                print(f"🧪 [REPISolveNode] 错误: {solve_result['error']}")

            shared['actions'].append('refine')
            print(f"🧪 [REPISolveNode] ➡️  跳转到改进节点 (RefineNode)")
            return "refine"


class RAGQualityNode(Node):
    """RAG质量评估节点"""

    def prep(self, shared):
        print(f"📊 [RAGQualityNode] 开始RAG质量评估预处理...")

        if 'quality_assessments' not in shared:
            shared['quality_assessments'] = []
        if 'quality_scores' not in shared:
            shared['quality_scores'] = []

        if 'RAGQualityNode' not in shared['node_call_counts']:
            shared['node_call_counts']['RAGQualityNode'] = 0
        shared['node_call_counts']['RAGQualityNode'] += 1

        current_problem = shared['problems'][-1] if shared['problems'] else ""
        repi_result = shared['repi_results'][-1] if shared['repi_results'] else {}
        rag_context = shared['rag_contexts'][-1] if shared['rag_contexts'] else ""
        retrieved_docs = shared['retrieved_docs'][-1] if shared['retrieved_docs'] else []
        requirements = shared.get('requirements', '')

        return current_problem, repi_result, rag_context, retrieved_docs, requirements

    def exec(self, prep_res):
        problem, repi_result, rag_context, retrieved_docs, requirements = prep_res
        print(f"📊 [RAGQualityNode] 开始综合质量评估...")

        # 检测是否使用了RAG模式
        is_rag_mode = not rag_context.startswith("纯AI模式")

        if is_rag_mode:
            quality_prompt = f"""
            请基于RAG参考资料和REPI解题结果，综合评估以下数学题目的质量：

            ## 生成的题目
            {problem}

            ## 出题要求
            {requirements}

            ## RAG参考信息
            检索到的参考文档数量：{len(retrieved_docs)}
            参考资料摘要：{rag_context[:500]}...

            ## REPI解题验证结果
            - 解题成功：{repi_result.get('success', False)}
            - 解题答案：{repi_result.get('answer', '无答案')[:200]}...
            - 总解题步数：{repi_result.get('total_steps', 0)}
            - 推理步数：{repi_result.get('reasoning_steps', 0)}
            - 计算步数：{repi_result.get('calculation_steps', 0)}
            - 反馈步数：{repi_result.get('feedback_steps', 0)}
            - 代码执行次数：{repi_result.get('code_executions', 0)}
            - 计算成功率：{repi_result.get('successful_calculations', 0)}/{repi_result.get('successful_calculations', 0) + repi_result.get('failed_calculations', 0)}

            ## 评估维度
            1. **RAG利用程度**：题目是否很好地利用了参考资料的知识点和结构 (1-10分)
            2. **可解性**：REPI系统是否能够成功解出 (1-10分)
            3. **复杂度合理性**：解题步数和各类操作的复杂度是否合适 (1-10分)
            4. **RAG使用效果**：检索到的资料质量和使用效果 (1-10分)
            5. **教学价值**：是否具有良好的教学和练习价值 (1-10分)

            请按以下格式输出：
            <rag_utilization_score>1-10分</rag_utilization_score>
            <solvability_score>1-10分</solvability_score>
            <complexity_score>1-10分</complexity_score>
            <rag_effectiveness_score>1-10分</rag_effectiveness_score>
            <educational_value_score>1-10分</educational_value_score>
            <overall_score>1-10分（综合评分）</overall_score>
            <strengths>题目优点</strengths>
            <weaknesses>题目缺点</weaknesses>
            <action>accept/refine</action>
            <improvement_suggestions>改进建议（如果需要）</improvement_suggestions>
            """
        else:
            quality_prompt = f"""
            请基于REPI解题结果，评估以下纯AI生成的数学题目的质量：

            ## 生成的题目
            {problem}

            ## 出题要求
            {requirements}

            ## REPI解题验证结果
            - 解题成功：{repi_result.get('success', False)}
            - 解题答案：{repi_result.get('answer', '无答案')[:200]}...
            - 总解题步数：{repi_result.get('total_steps', 0)}
            - 推理步数：{repi_result.get('reasoning_steps', 0)}
            - 计算步数：{repi_result.get('calculation_steps', 0)}

            ## 评估维度（纯AI模式）
            1. **原创性与创新性**：题目是否具有原创性，避免了常见套路 (1-10分)
            2. **可解性**：REPI系统是否能够成功解出 (1-10分)
            3. **复杂度与区分度**：以高考压轴题为基准，评估题目难度层次 (1-10分)
            4. **知识覆盖与融合**：是否有效融合多个数学知识点 (1-10分)
            5. **教学价值**：是否具有良好的教学和练习价值 (1-10分)

            ### 复杂度评分参考：
            - 1-2分：形式复杂但缺乏思维深度
            - 3-4分：准压轴题水平，常见模型应用
            - 5-6分：标准高考压轴题水平
            - 7-8分：顶尖压轴题，需要创造性思维
            - 9-10分：竞赛级难度，探索AI能力边界

            请按以下格式输出：
            <rag_utilization_score>1-10分</rag_utilization_score>
            <solvability_score>1-10分</solvability_score>
            <complexity_score>1-10分</complexity_score>
            <rag_effectiveness_score>1-10分</rag_effectiveness_score>
            <educational_value_score>1-10分</educational_value_score>
            <overall_score>1-10分（综合评分）</overall_score>
            <strengths>题目优点</strengths>
            <weaknesses>题目缺点</weaknesses>
            <action>accept/refine</action>
            <improvement_suggestions>改进建议（如果需要）</improvement_suggestions>
            """

        print(f"📊 [RAGQualityNode] 发送评估请求...")
        response = call_llm_stream(quality_prompt)

        # 调试：打印LLM响应的前500字符
        print(f"📊 [RAGQualityNode] LLM响应预览: {response[:500]}...")

        return response

    def post(self, shared, prep_res, exec_res):
        print(f"📊 [RAGQualityNode] RAG质量评估完成...")

        response = exec_res
        shared['quality_assessments'].append(response)

        scores = {}
        score_types = ['rag_utilization_score', 'solvability_score', 'complexity_score',
                       'rag_effectiveness_score', 'educational_value_score', 'overall_score']

        # 改进的评分解析
        for score_type in score_types:
            score_match = re.search(f'<{score_type}>(.*?)</{score_type}>', response, re.DOTALL)
            if score_match:
                score_str = score_match.group(1).strip()
                print(f"📊 [RAGQualityNode] 解析 {score_type}: '{score_str}'")

                try:
                    # 提取数字部分
                    number_match = re.search(r'(\d+(?:\.\d+)?)', score_str)
                    if number_match:
                        scores[score_type] = float(number_match.group(1))
                    else:
                        print(f"⚠️ [RAGQualityNode] 无法从 '{score_str}' 中提取数字")
                        scores[score_type] = 5.0  # 默认中等分数
                except Exception as e:
                    print(f"⚠️ [RAGQualityNode] 解析 {score_type} 失败: {e}")
                    scores[score_type] = 5.0
            else:
                print(f"⚠️ [RAGQualityNode] 未找到 {score_type} 标签")
                scores[score_type] = 5.0

        shared['quality_scores'].append(scores)

        action_match = re.search(r'<action>(.*?)</action>', response, re.DOTALL)
        action = action_match.group(1).strip() if action_match else "refine"

        print(f"📊 [RAGQualityNode] 评估结果:")
        print(f"   - RAG利用/原创性: {scores['rag_utilization_score']}/10")
        print(f"   - 可解性: {scores['solvability_score']}/10")
        print(f"   - 复杂度: {scores['complexity_score']}/10")
        print(f"   - RAG效果/知识覆盖: {scores['rag_effectiveness_score']}/10")
        print(f"   - 教学价值: {scores['educational_value_score']}/10")
        print(f"   - 综合评分: {scores['overall_score']}/10")
        print(f"   - 决策: {action}")

        shared['actions'].append(action)

        quality_calls = shared.get('node_call_counts', {}).get('RAGQualityNode', 0)
        if quality_calls >= MAX_RETRY:
            print(f"📊 [RAGQualityNode] 达到最大重试次数，接受当前题目")
            return "format"

        if action == "accept" or scores['overall_score'] >= 7.0:  # ✅ 设置为7.0
            print(f"📊 [RAGQualityNode] ➡️  跳转到格式化节点 (FormatNode)")
            return "format"
        else:
            print(f"📊 [RAGQualityNode] ➡️  跳转到改进节点 (RefineNode)")
            return "refine"


class RefineNode(Node):
    """改进节点"""

    def prep(self, shared):
        print(f"🔧 [RefineNode] 开始题目改进预处理...")

        if 'refinements' not in shared:
            shared['refinements'] = []

        if 'RefineNode' not in shared['node_call_counts']:
            shared['node_call_counts']['RefineNode'] = 0
        shared['node_call_counts']['RefineNode'] += 1

        current_problem = shared['problems'][-1] if shared['problems'] else ""
        repi_result = shared['repi_results'][-1] if shared['repi_results'] else {}
        quality_assessment = shared['quality_assessments'][-1] if shared['quality_assessments'] else ""
        requirements = shared.get('requirements', '')

        return current_problem, repi_result, quality_assessment, requirements

    def exec(self, prep_res):
        current_problem, repi_result, quality_assessment, requirements = prep_res
        print(f"🔧 [RefineNode] 开始基于REPI结果改进题目...")

        suggestions_match = re.search(r'<improvement_suggestions>(.*?)</improvement_suggestions>', quality_assessment,
                                      re.DOTALL)
        suggestions = suggestions_match.group(1).strip() if suggestions_match else ""

        solve_analysis = ""
        if not repi_result.get('success'):
            solve_analysis = "REPI无法解出，需要简化题目或修正错误"
        elif repi_result.get('total_steps', 0) < 3:
            solve_analysis = "解题步数过少，题目可能过于简单，需要增加复杂度"
        elif repi_result.get('total_steps', 0) > MAX_RETRY * 3:
            solve_analysis = "解题步数过多，题目可能过于复杂，需要适当简化"
        elif repi_result.get('failed_calculations', 0) > repi_result.get('successful_calculations', 0):
            solve_analysis = "计算失败率高，可能存在数据设置问题"
        else:
            solve_analysis = "解题过程基本合理，主要进行细节优化"

        refine_prompt = f"""
        请基于REPI解题分析和质量评估来改进数学题目：

        ## 原题目
        {current_problem}

        ## 出题要求
        {requirements}

        ## REPI解题分析
        {solve_analysis}

        详细解题数据：
        - 解题成功：{repi_result.get('success', False)}
        - 总步数：{repi_result.get('total_steps', 0)}
        - 推理/计算/反馈步数：{repi_result.get('reasoning_steps', 0)}/{repi_result.get('calculation_steps', 0)}/{repi_result.get('feedback_steps', 0)}
        - 计算成功/失败：{repi_result.get('successful_calculations', 0)}/{repi_result.get('failed_calculations', 0)}

        ## 质量评估建议
        {suggestions}

        ## 改进指导原则
        1. 根据REPI解题分析调整题目难度和复杂度
        2. 确保题目可解且步骤合理（建议5-{MAX_RETRY * 2}步）
        3. 保持教学价值和考查目标
        4. 优化题目描述和数据设置

        请按以下格式输出改进方案：
        <improvement_strategy>改进策略说明</improvement_strategy>
        <key_changes>关键改动点</key_changes>
        <expected_solve_steps>预期解题步数范围</expected_solve_steps>
        """

        response = call_llm_stream(refine_prompt)
        return response

    def post(self, shared, prep_res, exec_res):
        print(f"🔧 [RefineNode] 改进分析完成...")

        response = exec_res
        shared['refinements'].append(response)

        strategy_match = re.search(r'<improvement_strategy>(.*?)</improvement_strategy>', response, re.DOTALL)
        strategy = strategy_match.group(1).strip() if strategy_match else ""

        print(f"🔧 [RefineNode] 改进策略: {strategy}")

        shared['actions'].append('rag_setpro')
        print(f"🔧 [RefineNode] ➡️  返回RAG出题节点重新生成")
        return "rag_setpro"


# ================================
# 工具函数
# ================================

def save_to_file(content: str, filename: str):
    """将内容保存到文件"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"\n📄 文件已成功保存为: {os.path.abspath(filename)}")
    except Exception as e:
        print(f"❌ 保存文件时出错: {e}")


# ================================
# 主程序入口
# ================================

def main():
    """主程序函数"""
    print("=" * 80)
    print("      欢迎使用 RAG增强AI出题系统 v1.0")
    print("        (集成REPI验证系统)")
    print("=" * 80)

    # 只创建必要的节点实例
    rag_setpro_node = RAGSetProNode()
    repi_solve_node = REPISolveNode()
    rag_quality_node = RAGQualityNode()
    refine_node = RefineNode()

    print("\n请输入您的出题要求，例如：")
    print("  - 为准备清北强基计划的高中生，设计一道结合'图论'和'概率'的原创压轴题")
    print("\n输入 'quit' 或 'exit' 退出程序。")

    while True:
        try:
            user_input = input("\n>>> 请输入任务情景: ")
            if user_input.lower() in ['quit', 'exit']:
                print("\n感谢使用，再见！")
                break

            if not user_input.strip():
                continue

            problem_type = input(
                "知识载体/融合领域 （针对的具体领域），如：概率和动态规划结合，(默认: 高中数学题): ").strip() or "高中数学题"
            difficulty = input(
                "题目定位与风格（示例：江浙地区数学模拟考试压轴） （默认：高校自主招生选拔风格）: ").strip() or "高校自主招生选拔风格"
            keywords_input = input("关键词（出题的出发点，例如：错排问题的处理） (用逗号分隔，可选): ").strip()
            keywords = [k.strip() for k in keywords_input.split(",")] if keywords_input else []

            print(f"\n🚀 开始生成题目...")

            # 直接执行工作流
            shared = {
                'task_scenario': user_input,
                'requirements': "",
                'problem_type': problem_type,
                'difficulty_level': difficulty,
                'topic_keywords': keywords,
                'context': '',
                'problems': [],
                'rag_queries': [],
                'retrieved_docs': [],
                'rag_contexts': [],
                'latex_outputs': [],
                'repi_results': [],
                'quality_assessments': [],
                'quality_scores': [],
                'refinements': [],
                'node_call_counts': {},
                'actions': [],
                'final_problem': '',
                'final_latex': '',
                'final_formatted_output': ''
            }

            current_action = "rag_setpro"
            max_iterations = MAX_RETRY * 5
            iteration = 0

            while current_action and iteration < max_iterations:
                iteration += 1
                print(f"\n=== 迭代 {iteration}: {current_action} ===")

                try:
                    if current_action == "rag_setpro":
                        prep_res = rag_setpro_node.prep(shared)
                        exec_res = rag_setpro_node.exec(prep_res)
                        current_action = rag_setpro_node.post(shared, prep_res, exec_res)

                    elif current_action == "solve_test":
                        prep_res = repi_solve_node.prep(shared)
                        exec_res = repi_solve_node.exec(prep_res)
                        current_action = repi_solve_node.post(shared, prep_res, exec_res)

                    elif current_action == "quality":
                        prep_res = rag_quality_node.prep(shared)
                        exec_res = rag_quality_node.exec(prep_res)
                        current_action = rag_quality_node.post(shared, prep_res, exec_res)

                    elif current_action == "refine":
                        prep_res = refine_node.prep(shared)
                        exec_res = refine_node.exec(prep_res)
                        current_action = refine_node.post(shared, prep_res, exec_res)

                    elif current_action == "format":
                        # 直接在这里进行格式化，不需要单独的节点
                        print(f"📋 开始格式化最终输出...")

                        final_problem = shared['problems'][-1] if shared['problems'] else ""
                        final_latex = shared['latex_outputs'][-1] if shared['latex_outputs'] else ""
                        quality_scores = shared['quality_scores'][-1] if shared['quality_scores'] else {}
                        repi_result = shared['repi_results'][-1] if shared['repi_results'] else {}
                        rag_context = shared['rag_contexts'][-1] if shared['rag_contexts'] else ""

                        # 检测使用的模式
                        mode = "RAG增强模式" if not rag_context.startswith("纯AI模式") else "纯AI模式"

                        # 简单的格式化
                        result_lines = []
                        result_lines.append(f"# RAG增强AI出题系统 - 最终输出 ({mode})")
                        result_lines.append("")
                        result_lines.append("## 📝 题目内容")
                        result_lines.append(final_problem)
                        result_lines.append("")
                        result_lines.append("## 📊 质量评估")
                        result_lines.append(f"- 综合评分: {quality_scores.get('overall_score', 0)}/10")
                        result_lines.append("")
                        result_lines.append("## 🧪 REPI验证结果")
                        result_lines.append(f"- 解题状态: {'✅ 成功' if repi_result.get('success') else '❌ 失败'}")
                        result_lines.append(f"- 解题步数: {repi_result.get('total_steps', 0)}")
                        result_lines.append("")
                        result_lines.append("## 📄 LaTeX源码")
                        result_lines.append("```latex")
                        result_lines.append(final_latex)
                        result_lines.append("```")
                        result_lines.append("")
                        result_lines.append(f"---")
                        result_lines.append(f"*本题目由RAG增强AI出题系统生成 ({mode})，经过REPI系统验证*")

                        shared['final_formatted_output'] = "\n".join(result_lines)
                        break

                    else:
                        print(f"❌ 未知动作: {current_action}")
                        break

                except Exception as e:
                    print(f"❌ 执行节点时出错: {str(e)}")
                    break

            result = shared.get('final_formatted_output', '出题失败')

            print("\n" + "=" * 80)
            print("最终输出:")
            print("=" * 80)
            print(result)
            print("=" * 80)

            if not result.startswith("出题失败") and not result.startswith("出题流程超时"):
                save_choice = input("\n是否将结果保存到文件? (y/n, 默认y): ").lower()
                if save_choice in ['', 'y', 'yes']:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"rag_problem_{timestamp}.md"
                    save_to_file(result, filename)

        except (KeyboardInterrupt, EOFError):
            print("\n\n程序已中断。感谢使用，再见！")
            break
        except Exception as e:
            print(f"\n❌ 程序执行出错: {str(e)}")
            print("请重新尝试...")


if __name__ == "__main__":
    main()
