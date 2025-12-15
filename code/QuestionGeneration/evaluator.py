"""
QualityEvaluator - 题目质量评估器

对生成的题目进行多维度质量评估
"""

import re
from utils.llm import call_llm_stream


class QualityEvaluator:
    """题目质量评估器"""
    
    # 评分标准
    ACCEPT_THRESHOLD = 7.0  # 接受题目的最低综合评分
    
    def __init__(self, accept_threshold: float = None):
        """
        初始化评估器
        
        Args:
            accept_threshold: 接受题目的最低分数，默认7.0
        """
        self.accept_threshold = accept_threshold or self.ACCEPT_THRESHOLD
        print(f"✅ QualityEvaluator初始化成功 (接受阈值: {self.accept_threshold})")
    
    def evaluate(self, problem: str, repi_result: dict, requirements: str = "") -> dict:
        """
        评估题目质量
        
        Args:
            problem: 生成的题目文本
            repi_result: REPI验证结果
            requirements: 出题要求
            
        Returns:
            dict: 评估结果，包含scores, decision, suggestions等
        """
        print(f"\n📊 [QualityEvaluator] 开始质量评估...")
        
        # 构建评估提示词
        prompt = self._build_evaluation_prompt(problem, repi_result, requirements)
        
        # 调用LLM进行评估
        print(f"📊 [QualityEvaluator] 调用AI进行评估...")
        response = call_llm_stream(prompt)
        
        # 解析评分
        scores = self._parse_scores(response)
        
        # 解析决策和建议
        action_match = re.search(r'<action>(.*?)</action>', response, re.DOTALL)
        action = action_match.group(1).strip() if action_match else "refine"
        
        suggestions_match = re.search(r'<improvement_suggestions>(.*?)</improvement_suggestions>', 
                                     response, re.DOTALL)
        suggestions = suggestions_match.group(1).strip() if suggestions_match else ""
        
        strengths_match = re.search(r'<strengths>(.*?)</strengths>', response, re.DOTALL)
        strengths = strengths_match.group(1).strip() if strengths_match else ""
        
        weaknesses_match = re.search(r'<weaknesses>(.*?)</weaknesses>', response, re.DOTALL)
        weaknesses = weaknesses_match.group(1).strip() if weaknesses_match else ""
        
        # 综合决策
        overall_score = scores.get('overall_score', 0)
        decision = 'accept' if (action == 'accept' or overall_score >= self.accept_threshold) else 'refine'
        
        result = {
            'scores': scores,
            'decision': decision,
            'suggestions': suggestions,
            'strengths': strengths,
            'weaknesses': weaknesses,
            'raw_response': response
        }
        
        # 打印评估结果
        self._print_evaluation_result(scores, decision)
        
        return result
    
    def _build_evaluation_prompt(self, problem: str, repi_result: dict, requirements: str) -> str:
        """构建评估提示词"""
        
        stats = repi_result.get('statistics', {})
        
        prompt = f"""
请基于REPI解题结果，评估以下纯AI生成的数学题目的质量：

## 生成的题目
{problem}

## 出题要求
{requirements if requirements else "无特殊要求"}

## REPI解题验证结果
- 解题成功：{repi_result.get('success', False)}
- 解题答案：{repi_result.get('answer', '无答案')[:200]}...
- 总解题步数：{stats.get('total_steps', 0)}
- 推理步数：{stats.get('reasoning_steps', 0)}
- 计算步数：{stats.get('calculation_steps', 0)}
- 计算成功率：{stats.get('successful_calculations', 0)}/{stats.get('calculation_steps', 0)}

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
<originality_score>1-10分</originality_score>
<solvability_score>1-10分</solvability_score>
<complexity_score>1-10分</complexity_score>
<knowledge_coverage_score>1-10分</knowledge_coverage_score>
<educational_value_score>1-10分</educational_value_score>
<overall_score>1-10分（综合评分）</overall_score>
<strengths>题目优点</strengths>
<weaknesses>题目缺点</weaknesses>
<action>accept/refine</action>
<improvement_suggestions>改进建议（如果需要）</improvement_suggestions>
"""
        return prompt
    
    def _parse_scores(self, response: str) -> dict:
        """解析评分"""
        
        score_types = [
            'originality_score',
            'solvability_score', 
            'complexity_score',
            'knowledge_coverage_score',
            'educational_value_score',
            'overall_score'
        ]
        
        scores = {}
        
        for score_type in score_types:
            score_match = re.search(f'<{score_type}>(.*?)</{score_type}>', response, re.DOTALL)
            if score_match:
                score_str = score_match.group(1).strip()
                try:
                    # 提取数字
                    number_match = re.search(r'(\d+(?:\.\d+)?)', score_str)
                    if number_match:
                        scores[score_type] = float(number_match.group(1))
                    else:
                        scores[score_type] = 5.0  # 默认中等分数
                except Exception:
                    scores[score_type] = 5.0
            else:
                scores[score_type] = 5.0
        
        return scores
    
    def _print_evaluation_result(self, scores: dict, decision: str):
        """打印评估结果"""
        
        print(f"\n📊 [QualityEvaluator] 评估结果:")
        print(f"   - 原创性: {scores.get('originality_score', 0)}/10")
        print(f"   - 可解性: {scores.get('solvability_score', 0)}/10")
        print(f"   - 复杂度: {scores.get('complexity_score', 0)}/10")
        print(f"   - 知识覆盖: {scores.get('knowledge_coverage_score', 0)}/10")
        print(f"   - 教学价值: {scores.get('educational_value_score', 0)}/10")
        print(f"   - 综合评分: {scores.get('overall_score', 0)}/10")
        print(f"   - 决策: {'✅ 接受' if decision == 'accept' else '🔧 需要改进'}")


class RefineAnalyzer:
    """改进分析器"""
    
    def __init__(self):
        print(f"✅ RefineAnalyzer初始化成功")
    
    def analyze(self, problem: str, repi_result: dict, evaluation: dict, requirements: str = "") -> dict:
        """
        分析题目问题并生成改进建议
        
        Args:
            problem: 当前题目
            repi_result: REPI验证结果
            evaluation: 质量评估结果
            requirements: 出题要求
            
        Returns:
            dict: 改进分析结果
        """
        print(f"\n🔧 [RefineAnalyzer] 开始分析改进方案...")
        
        # 构建改进提示词
        prompt = self._build_refine_prompt(problem, repi_result, evaluation, requirements)
        
        # 调用LLM分析
        print(f"🔧 [RefineAnalyzer] 调用AI分析...")
        response = call_llm_stream(prompt)
        
        # 解析改进策略
        strategy_match = re.search(r'<improvement_strategy>(.*?)</improvement_strategy>', 
                                  response, re.DOTALL)
        strategy = strategy_match.group(1).strip() if strategy_match else ""
        
        changes_match = re.search(r'<key_changes>(.*?)</key_changes>', response, re.DOTALL)
        key_changes = changes_match.group(1).strip() if changes_match else ""
        
        steps_match = re.search(r'<expected_solve_steps>(.*?)</expected_solve_steps>', 
                               response, re.DOTALL)
        expected_steps = steps_match.group(1).strip() if steps_match else ""
        
        result = {
            'strategy': strategy,
            'key_changes': key_changes,
            'expected_steps': expected_steps,
            'raw_response': response
        }
        
        print(f"🔧 [RefineAnalyzer] 改进策略: {strategy[:100]}...")
        
        return result
    
    def _build_refine_prompt(self, problem: str, repi_result: dict, 
                            evaluation: dict, requirements: str) -> str:
        """构建改进提示词"""
        
        stats = repi_result.get('statistics', {})
        scores = evaluation.get('scores', {})
        suggestions = evaluation.get('suggestions', '')
        
        # 分析解题情况
        if not repi_result.get('success'):
            solve_analysis = "REPI无法解出，需要简化题目或修正错误"
        elif stats.get('total_steps', 0) < 3:
            solve_analysis = "解题步数过少，题目可能过于简单，需要增加复杂度"
        elif stats.get('total_steps', 0) > 12:
            solve_analysis = "解题步数过多，题目可能过于复杂，需要适当简化"
        elif stats.get('failed_calculations', 0) > stats.get('successful_calculations', 0):
            solve_analysis = "计算失败率高，可能存在数据设置问题"
        else:
            solve_analysis = "解题过程基本合理，主要进行细节优化"
        
        prompt = f"""
请基于REPI解题分析和质量评估来改进数学题目：

## 原题目
{problem}

## 出题要求
{requirements if requirements else "无特殊要求"}

## REPI解题分析
{solve_analysis}

详细解题数据：
- 解题成功：{repi_result.get('success', False)}
- 总步数：{stats.get('total_steps', 0)}
- 推理/计算步数：{stats.get('reasoning_steps', 0)}/{stats.get('calculation_steps', 0)}
- 计算成功/失败：{stats.get('successful_calculations', 0)}/{stats.get('failed_calculations', 0)}

## 质量评估结果
- 综合评分：{scores.get('overall_score', 0)}/10
- 原创性：{scores.get('originality_score', 0)}/10
- 可解性：{scores.get('solvability_score', 0)}/10
- 复杂度：{scores.get('complexity_score', 0)}/10

## 质量评估建议
{suggestions}

## 改进指导原则
1. 根据REPI解题分析调整题目难度和复杂度
2. 确保题目可解且步骤合理（建议5-12步）
3. 保持教学价值和考查目标
4. 优化题目描述和数据设置

请按以下格式输出改进方案：
<improvement_strategy>改进策略说明</improvement_strategy>
<key_changes>关键改动点</key_changes>
<expected_solve_steps>预期解题步数范围</expected_solve_steps>
"""
        return prompt


if __name__ == "__main__":
    # 测试代码
    evaluator = QualityEvaluator()
    
    test_problem = "已知函数 f(x) = x^2 - 4x + 3，求函数的零点和最小值。"
    
    test_repi_result = {
        'success': True,
        'answer': '零点为x=1和x=3，最小值为-1',
        'statistics': {
            'total_steps': 5,
            'reasoning_steps': 3,
            'calculation_steps': 2,
            'successful_calculations': 2,
            'failed_calculations': 0
        }
    }
    
    result = evaluator.evaluate(test_problem, test_repi_result, "高中数学题")
    
    print("\n" + "="*80)
    print("评估结果:")
    print("="*80)
    print(f"决策: {result['decision']}")
    print(f"评分: {result['scores']}")
