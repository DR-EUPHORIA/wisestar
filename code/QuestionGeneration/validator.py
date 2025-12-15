"""
REPIValidator - REPI题目验证器

使用RePI系统验证生成题目的可解性
"""

import os
import re
from dotenv import load_dotenv
from utils.pyinterpreter import PythonInterpreter
from utils.llm import call_llm_stream
from utils.prompt_templates import REPI_RENODE_PROMPT

load_dotenv()
MAX_RETRY = int(os.getenv('MAX_RETRY', 3))


class ReNode:
    """推理节点 - 负责数学推理和代码编写"""
    
    def __init__(self):
        self.call_count = 0
    
    def process(self, question: str, context: str) -> dict:
        """
        处理推理任务
        
        Args:
            question: 待解决的问题
            context: 之前的推理上下文
            
        Returns:
            dict: 包含response, action, code, solution的字典
        """
        self.call_count += 1
        print(f"🧠 [ReNode] 第{self.call_count}次推理...")
        
        prompt = REPI_RENODE_PROMPT.format(question=question, context=context)
        response = call_llm_stream(prompt)
        
        # 解析动作
        action_match = re.search(r'<action>(.*?)</action>', response, re.DOTALL)
        action = action_match.group(1).strip() if action_match else None
        
        # 解析代码
        code = None
        code_match = re.search(r'<code>(.*?)</code>', response, re.DOTALL)
        if code_match:
            raw_code = code_match.group(1).strip()
            code = re.sub(r'^```python\s*|\s*```$', '', raw_code)
        
        # 解析答案
        solution = None
        solution_match = re.search(r'<solution>(.*?)</solution>', response, re.DOTALL)
        if solution_match:
            solution = solution_match.group(1).strip()
        
        # 解析最终答案
        answer = None
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()
        
        print(f"🧠 [ReNode] 动作: {action}")
        if code:
            print(f"🧠 [ReNode] 生成了计算代码")
        if answer:
            print(f"🧠 [ReNode] 得到最终答案")
        
        return {
            'response': response,
            'action': action,
            'code': code,
            'solution': solution,
            'answer': answer
        }


class PINode:
    """Python解释器节点 - 负责执行计算代码"""
    
    def __init__(self):
        self.interpreter = PythonInterpreter()
        self.call_count = 0
    
    def execute(self, code: str) -> dict:
        """
        执行Python代码
        
        Args:
            code: 待执行的代码
            
        Returns:
            dict: 包含success, output, error的字典
        """
        self.call_count += 1
        print(f"🐍 [PINode] 第{self.call_count}次执行代码...")
        
        if not code or not code.strip():
            print(f"🐍 [PINode] ❌ 没有代码可执行")
            return {
                'success': False,
                'output': '',
                'error': 'No code to execute'
            }
        
        try:
            result = self.interpreter.execute_code(code)
            
            if result['success']:
                print(f"🐍 [PINode] ✅ 代码执行成功")
                if result['output']:
                    print(f"🐍 [PINode] 输出: {result['output'][:100]}...")
            else:
                print(f"🐍 [PINode] ❌ 代码执行失败: {result['error'][:100]}...")
            
            return result
            
        except Exception as e:
            error_msg = f"Interpreter Error: {str(e)}"
            print(f"🐍 [PINode] ❌ 解释器异常: {error_msg}")
            return {
                'success': False,
                'output': '',
                'error': error_msg
            }


class REPIValidator:
    """REPI验证器 - 验证题目可解性"""
    
    def __init__(self, max_steps: int = None):
        """
        初始化验证器
        
        Args:
            max_steps: 最大解题步数，默认为MAX_RETRY * 4
        """
        self.re_node = ReNode()
        self.pi_node = PINode()
        self.max_steps = max_steps or (MAX_RETRY * 4)
        print(f"✅ REPIValidator初始化成功 (最大步数: {self.max_steps})")
    
    def validate(self, question: str) -> dict:
        """
        验证题目可解性
        
        Args:
            question: 待验证的题目
            
        Returns:
            dict: 验证结果，包含success, answer, statistics等信息
        """
        print(f"\n🧪 [REPIValidator] 开始验证题目可解性...")
        print(f"🧪 [REPIValidator] 题目: {question[:100]}...")
        
        # 初始化状态
        context = ""
        answer = None
        current_step = 0
        
        # 统计信息
        stats = {
            'total_steps': 0,
            'reasoning_steps': 0,
            'calculation_steps': 0,
            'successful_calculations': 0,
            'failed_calculations': 0,
            'action_sequence': []
        }
        
        try:
            while current_step < self.max_steps:
                current_step += 1
                stats['total_steps'] = current_step
                
                # 推理步骤
                re_result = self.re_node.process(question, context)
                stats['reasoning_steps'] += 1
                stats['action_sequence'].append(re_result['action'] or 'reasoning')
                
                # 更新上下文
                context += f"\n\n推理步骤 {current_step}：\n{re_result['response']}\n"
                
                # 检查是否得到答案
                if re_result['answer']:
                    answer = re_result['answer']
                    print(f"🧪 [REPIValidator] ✅ 成功得到答案")
                    break
                
                # 如果需要计算
                if re_result['action'] == 'calculate' and re_result['code']:
                    stats['calculation_steps'] += 1
                    
                    # 执行代码
                    pi_result = self.pi_node.execute(re_result['code'])
                    
                    if pi_result['success']:
                        stats['successful_calculations'] += 1
                        context += f"\n\n计算结果：{pi_result['output']}\n"
                    else:
                        stats['failed_calculations'] += 1
                        context += f"\n\n计算失败：{pi_result['error']}\n"
                
                # 检查是否超过重试次数
                if self.re_node.call_count >= MAX_RETRY and not answer:
                    print(f"🧪 [REPIValidator] ⚠️ 达到最大重试次数")
                    break
            
            # 生成验证结果
            success = bool(answer)
            
            result = {
                'success': success,
                'answer': answer or '',
                'statistics': stats,
                'final_context': context,
                'error': None if success else 'Failed to solve the problem'
            }
            
            # 打印统计信息
            print(f"\n🧪 [REPIValidator] 验证完成")
            print(f"   - 状态: {'✅ 成功' if success else '❌ 失败'}")
            print(f"   - 总步数: {stats['total_steps']}")
            print(f"   - 推理步数: {stats['reasoning_steps']}")
            print(f"   - 计算步数: {stats['calculation_steps']}")
            print(f"   - 计算成功率: {stats['successful_calculations']}/{stats['calculation_steps']}")
            
            return result
            
        except Exception as e:
            error_msg = f"验证过程出错: {str(e)}"
            print(f"🧪 [REPIValidator] ❌ {error_msg}")
            return {
                'success': False,
                'answer': '',
                'statistics': stats,
                'final_context': context,
                'error': error_msg
            }


if __name__ == "__main__":
    # 测试代码
    validator = REPIValidator()
    
    test_question = """
已知函数 f(x) = x^2 - 4x + 3，求：
(1) 函数的零点
(2) 函数的最小值
"""
    
    result = validator.validate(test_question)
    
    print("\n" + "="*80)
    print("验证结果:")
    print("="*80)
    print(f"成功: {result['success']}")
    print(f"答案: {result['answer']}")
    print(f"统计: {result['statistics']}")
