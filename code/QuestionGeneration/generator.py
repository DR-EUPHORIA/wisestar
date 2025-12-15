"""
QuestionGenerator - AI数学题目生成器

负责使用AI大模型生成高质量的数学题目
"""

import os
import re
from openai import OpenAI, APIError
from dotenv import load_dotenv

load_dotenv()


class QuestionGenerator:
    """AI数学题目生成器"""
    
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

    
    def __init__(self, api_key: str = None, model: str = None):

        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY")
        # 支持自定义兼容OpenAI的推理服务，默认使用DeepSeek公开端点
        self.base_url = os.getenv("LLM_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        self.model = model or os.getenv("DEEPSEEK_MODEL") or os.getenv("MODEL_NAME", "deepseek-chat")

        if not self.api_key:
            raise ValueError("API Key未设置，请设置 LLM_API_KEY 环境变量")

        try:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
            print(f"✅ QuestionGenerator初始化成功 (模型: {self.model})")
        except Exception as e:
            raise RuntimeError(f"初始化OpenAI客户端失败: {e}")

    def generate(self, task_scenario: str, temperature: float = 0.7) -> dict:
        """
        生成数学题目
        
        Args:
            task_scenario: 任务情景描述
            temperature: 生成温度，控制随机性
            
        Returns:
            dict: 包含latex_output和problem_text的字典
        """
        if not task_scenario or not task_scenario.strip():
            return {
                'success': False,
                'error': '任务情景不能为空',
                'latex_output': '',
                'problem_text': ''
            }
        
        user_content = f"### **【任务情景】**\n{task_scenario}"
        
        try:
            print(f"🤖 正在调用AI生成题目...")
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
            
            latex_output = completion.choices[0].message.content
            problem_text = self._extract_problem_from_latex(latex_output)
            
            print(f"✅ 题目生成成功！")
            
            return {
                'success': True,
                'latex_output': latex_output,
                'problem_text': problem_text,
                'error': None
            }
            
        except APIError as e:
            error_msg = f"调用模型API时发生错误: {e}"
            print(f"❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'latex_output': '',
                'problem_text': ''
            }
        except Exception as e:
            error_msg = f"发生未知错误: {e}"
            print(f"❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'latex_output': '',
                'problem_text': ''
            }

    def _extract_problem_from_latex(self, latex_content: str) -> str:
        """
        从LaTeX内容中提取题目文本
        
        Args:
            latex_content: LaTeX格式的内容
            
        Returns:
            str: 提取的题目文本
        """
        try:
            # 移除LaTeX文档结构
            content = re.sub(r'\\documentclass.*?\n', '', latex_content)
            content = re.sub(r'\\usepackage.*?\n', '', content)
            content = re.sub(r'\\begin{document}', '', content)
            content = re.sub(r'\\end{document}', '', content)
            
            # 简化LaTeX命令
            content = re.sub(r'\\title{(.*?)}', r'\1', content)
            content = re.sub(r'\\section{(.*?)}', r'\1', content)
            content = re.sub(r'\\textbf{(.*?)}', r'\1', content)
            content = re.sub(r'\\emph{(.*?)}', r'\1', content)
            
            # 移除环境标签
            content = re.sub(r'\\begin{.*?}', '', content)
            content = re.sub(r'\\end{.*?}', '', content)
            
            # 清理多余空行
            content = re.sub(r'\n\s*\n', '\n\n', content)
            content = content.strip()
            
            return content
        except Exception as e:
            print(f"⚠️ LaTeX解析失败: {str(e)}")
            return latex_content


if __name__ == "__main__":
    # 测试代码
    generator = QuestionGenerator()
    
    task = """
为准备高考的学生设计一道函数与导数的压轴题

### 具体要求

#### **角色设定 (Role Definition)**
你将扮演一位**数学命题宗师**。

#### **核心任务 (Core Task)**
创编一道结构完整、逻辑严谨的数学压轴题。

#### **输入参数 (Input Parameters)**

*   **核心思想与关键词**: 导数、单调性、极值
*   **知识载体/融合领域**: 函数与导数
*   **题目定位与风格**: 高考压轴题
*   **具体要求**: 需要包含参数分类讨论

#### **创作指导原则 (Guiding Principles)**
1.  秉持思想深度与结构之美
2.  追求情景化与数学纯粹性
"""
    
    result = generator.generate(task)
    
    if result['success']:
        print("\n" + "="*80)
        print("生成的题目:")
        print("="*80)
        print(result['problem_text'][:500] + "...")
        print("\n" + "="*80)
        print("LaTeX源码:")
        print("="*80)
        print(result['latex_output'][:500] + "...")
    else:
        print(f"\n生成失败: {result['error']}")
