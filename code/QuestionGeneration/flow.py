"""
Flow - 题目生成工作流编排

整合Generator、Validator、Evaluator，实现完整的出题闭环
"""

import os
from dotenv import load_dotenv
from .generator import QuestionGenerator
from .validator import REPIValidator
from .evaluator import QualityEvaluator, RefineAnalyzer

# 导入Memory系统
try:
    from code.Memory import KnowledgeExtractor, MemoryStorage
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False
    print("⚠️ Memory模块未找到，将不记录学习历史")

load_dotenv()
MAX_RETRY = int(os.getenv('MAX_RETRY', 3))


class QuestionGenerationFlow:
    """题目生成工作流"""
    
    def __init__(self, max_iterations: int = None, enable_memory: bool = True):
        """
        初始化工作流
        
        Args:
            max_iterations: 最大迭代次数，默认为MAX_RETRY * 5
            enable_memory: 是否启用Memory记录，默认True
        """
        self.generator = QuestionGenerator()
        self.validator = REPIValidator()
        self.evaluator = QualityEvaluator()
        self.refiner = RefineAnalyzer()
        self.max_iterations = max_iterations or (MAX_RETRY * 5)
        
        # 初始化Memory系统
        self.memory_enabled = enable_memory and MEMORY_AVAILABLE
        if self.memory_enabled:
            self.extractor = KnowledgeExtractor()
            self.memory = MemoryStorage()
            print(f"✅ Memory系统已启用")
        else:
            self.extractor = None
            self.memory = None
        
        print(f"\n{'='*80}")
        print(f"✅ QuestionGenerationFlow 初始化成功")
        print(f"   - 最大迭代次数: {self.max_iterations}")
        print(f"   - Memory记录: {'启用' if self.memory_enabled else '禁用'}")
        print(f"{'='*80}\n")
    
    def run(self, config: dict) -> dict:
        """
        运行完整的题目生成流程
        
        Args:
            config: 配置字典，包含以下字段：
                - task_scenario: 任务情景描述
                - problem_type: 题目类型
                - difficulty_level: 难度级别
                - topic_keywords: 关键词列表
                - requirements: 具体要求
                
        Returns:
            dict: 包含最终题目、评估结果、验证结果等信息
        """
        print(f"\n🚀 开始题目生成流程...")
        print(f"📋 任务: {config.get('task_scenario', '')[:100]}...")
        
        # 构建完整的任务描述
        task_scenario = self._build_task_scenario(config)
        
        # 初始化状态
        iteration = 0
        current_action = "generate"
        
        # 历史记录
        history = {
            'generated_problems': [],
            'validation_results': [],
            'evaluation_results': [],
            'refinement_analyses': []
        }
        
        # 最终结果
        final_result = None
        
        while iteration < self.max_iterations and current_action:
            iteration += 1
            print(f"\n{'='*80}")
            print(f"🔄 迭代 {iteration}: {current_action}")
            print(f"{'='*80}")
            
            try:
                if current_action == "generate":
                    # 生成题目
                    gen_result = self.generator.generate(task_scenario)
                    
                    if not gen_result['success']:
                        print(f"❌ 题目生成失败: {gen_result['error']}")
                        if iteration >= MAX_RETRY:
                            final_result = {
                                'success': False,
                                'error': '多次尝试后仍无法生成题目',
                                'history': history
                            }
                            break
                        continue
                    
                    history['generated_problems'].append(gen_result)
                    current_action = "validate"
                
                elif current_action == "validate":
                    # 验证题目可解性
                    current_problem = history['generated_problems'][-1]['problem_text']
                    val_result = self.validator.validate(current_problem)
                    
                    history['validation_results'].append(val_result)
                    
                    if val_result['success']:
                        current_action = "evaluate"
                    else:
                        current_action = "refine"
                
                elif current_action == "evaluate":
                    # 评估题目质量
                    current_problem = history['generated_problems'][-1]['problem_text']
                    current_validation = history['validation_results'][-1]
                    
                    eval_result = self.evaluator.evaluate(
                        current_problem,
                        current_validation,
                        config.get('requirements', '')
                    )
                    
                    history['evaluation_results'].append(eval_result)
                    
                    if eval_result['decision'] == 'accept':
                        current_action = "finalize"
                    else:
                        current_action = "refine"
                
                elif current_action == "refine":
                    # 分析改进
                    current_problem = history['generated_problems'][-1]['problem_text']
                    current_validation = history['validation_results'][-1]
                    current_evaluation = history['evaluation_results'][-1] if history['evaluation_results'] else {}
                    
                    refine_result = self.refiner.analyze(
                        current_problem,
                        current_validation,
                        current_evaluation,
                        config.get('requirements', '')
                    )
                    
                    history['refinement_analyses'].append(refine_result)
                    
                    # 更新任务描述，加入改进建议
                    task_scenario = self._update_task_scenario(
                        config,
                        refine_result
                    )
                    
                    current_action = "generate"
                
                elif current_action == "finalize":
                    # 完成，生成最终输出
                    final_result = self._finalize_result(config, history)
                    
                    # 记录到Memory
                    if self.memory_enabled and final_result['success']:
                        self._save_to_memory(final_result, config)
                    
                    break
                
                else:
                    print(f"❌ 未知动作: {current_action}")
                    break
                    
            except Exception as e:
                print(f"❌ 执行出错: {str(e)}")
                final_result = {
                    'success': False,
                    'error': f'执行出错: {str(e)}',
                    'history': history
                }
                break
        
        # 如果达到最大迭代次数
        if iteration >= self.max_iterations and not final_result:
            print(f"\n⚠️ 达到最大迭代次数 ({self.max_iterations})")
            final_result = self._finalize_result(config, history, forced=True)
        
        return final_result or {
            'success': False,
            'error': '未知错误',
            'history': history
        }
    
    def _build_task_scenario(self, config: dict) -> str:
        """构建任务情景描述"""
        
        task_scenario = config.get('task_scenario', '')
        problem_type = config.get('problem_type', '高中数学题')
        difficulty_level = config.get('difficulty_level', '适中')
        topic_keywords = config.get('topic_keywords', [])
        requirements = config.get('requirements', '')
        
        enhanced_scenario = f"""
{task_scenario}

### 具体要求

#### **角色设定 (Role Definition)**
你将扮演一位**数学命题宗师**。你深谙数学的内在结构与逻辑之美，擅长创编新颖、深刻且具有高度选拔性的原创数学题目。

#### **核心任务 (Core Task)**
你的任务是根据下方提供的具体参数，**从零开始创编一道结构完整、逻辑严谨的数学题目**。

#### **输入参数 (Input Parameters)**

*   **核心思想与关键词 (Core Idea & Keywords)**: {', '.join(topic_keywords) if topic_keywords else '无特定关键词'}
*   **知识载体/融合领域 (Knowledge Carrier / Integrated Field)**: {problem_type}
*   **题目定位与风格 (Problem Positioning & Style)**: {difficulty_level}
*   **具体要求 (Specific Requirements)**: {requirements if requirements else '无特殊要求'}

#### **创作指导原则 (Guiding Principles)**
1.  **秉持思想深度与结构之美**: 应围绕核心思想构建一个逻辑自洽、层层深入的探索路径。
2.  **追求情景化与数学纯粹性**: 若需背景，应设计一个新颖、抽象的数学情景，追求数学本身的结构美。

#### **输出格式与解析要求 (Output Format & Solution Specification)**
你必须严格按照以下格式，生成一份完整的、未经渲染的、可直接编译的 **LaTeX 源码**。

1.  **文档序言 (Preamble)**:
    *   使用 `\\documentclass{{article}}`。
    *   必须包含 `amsmath`, `amssymb`, `geometry`, `tcolorbox` 等宏包。

2.  **题目模块 (Problem Module)**:
    *   每道大题必须使用一个自定义的 `tcolorbox` 环境包裹。

3.  **解析模块 (Solution Module)**:
    *   紧随题目之后，以 `【解析】` 作为普通文本开头。
    *   解析必须清晰地展示思维的完整链条。
"""
        return enhanced_scenario
    
    def _update_task_scenario(self, config: dict, refine_result: dict) -> str:
        """更新任务描述，加入改进建议"""
        
        base_scenario = self._build_task_scenario(config)
        
        strategy = refine_result.get('strategy', '')
        key_changes = refine_result.get('key_changes', '')
        
        updated_scenario = f"""
{base_scenario}

### 改进要求 (Refinement Requirements)

**改进策略**: {strategy}

**关键改动点**: {key_changes}

请根据以上改进要求，重新生成题目。
"""
        return updated_scenario
    
    def _finalize_result(self, config: dict, history: dict, forced: bool = False) -> dict:
        """生成最终结果"""
        
        print(f"\n📋 正在生成最终输出...")
        
        if not history['generated_problems']:
            return {
                'success': False,
                'error': '没有成功生成任何题目',
                'history': history
            }
        
        # 获取最后一次生成的题目
        final_problem = history['generated_problems'][-1]
        final_validation = history['validation_results'][-1] if history['validation_results'] else {}
        final_evaluation = history['evaluation_results'][-1] if history['evaluation_results'] else {}
        
        # 构建格式化输出
        output_lines = []
        output_lines.append("# QuestionGeneration - AI数学题目生成系统")
        output_lines.append("")
        output_lines.append(f"**生成时间**: {self._get_timestamp()}")
        output_lines.append(f"**迭代次数**: {len(history['generated_problems'])}")
        output_lines.append("")
        
        output_lines.append("## 📝 题目内容")
        output_lines.append("")
        output_lines.append(final_problem.get('problem_text', ''))
        output_lines.append("")
        
        output_lines.append("## 📊 质量评估")
        output_lines.append("")
        if final_evaluation:
            scores = final_evaluation.get('scores', {})
            output_lines.append(f"- **综合评分**: {scores.get('overall_score', 0):.1f}/10")
            output_lines.append(f"- **原创性**: {scores.get('originality_score', 0):.1f}/10")
            output_lines.append(f"- **可解性**: {scores.get('solvability_score', 0):.1f}/10")
            output_lines.append(f"- **复杂度**: {scores.get('complexity_score', 0):.1f}/10")
            output_lines.append(f"- **知识覆盖**: {scores.get('knowledge_coverage_score', 0):.1f}/10")
            output_lines.append(f"- **教学价值**: {scores.get('educational_value_score', 0):.1f}/10")
            output_lines.append("")
            
            if final_evaluation.get('strengths'):
                output_lines.append(f"**优点**: {final_evaluation['strengths']}")
                output_lines.append("")
            
            if final_evaluation.get('weaknesses'):
                output_lines.append(f"**不足**: {final_evaluation['weaknesses']}")
                output_lines.append("")
        else:
            output_lines.append("*未进行质量评估*")
            output_lines.append("")
        
        output_lines.append("## 🧪 REPI验证结果")
        output_lines.append("")
        if final_validation:
            stats = final_validation.get('statistics', {})
            output_lines.append(f"- **解题状态**: {'✅ 成功' if final_validation.get('success') else '❌ 失败'}")
            output_lines.append(f"- **解题答案**: {final_validation.get('answer', '无')[:200]}")
            output_lines.append(f"- **总步数**: {stats.get('total_steps', 0)}")
            output_lines.append(f"- **推理步数**: {stats.get('reasoning_steps', 0)}")
            output_lines.append(f"- **计算步数**: {stats.get('calculation_steps', 0)}")
            output_lines.append(f"- **计算成功率**: {stats.get('successful_calculations', 0)}/{stats.get('calculation_steps', 0)}")
            output_lines.append("")
        else:
            output_lines.append("*未进行验证*")
            output_lines.append("")
        
        output_lines.append("## 📄 LaTeX源码")
        output_lines.append("")
        output_lines.append("```latex")
        output_lines.append(final_problem.get('latex_output', ''))
        output_lines.append("```")
        output_lines.append("")
        
        output_lines.append("---")
        output_lines.append(f"*本题目由QuestionGeneration系统生成，经过{len(history['generated_problems'])}次迭代优化*")
        
        formatted_output = "\n".join(output_lines)
        
        result = {
            'success': True,
            'problem': final_problem.get('problem_text', ''),
            'latex': final_problem.get('latex_output', ''),
            'validation': final_validation,
            'evaluation': final_evaluation,
            'formatted_output': formatted_output,
            'history': history,
            'forced': forced
        }
        
        print(f"✅ 最终输出生成完成")
        
        return result
    
    def _save_to_memory(self, final_result: dict, config: dict):
        """
        保存生成的题目到Memory系统
        
        Args:
            final_result: 最终结果字典
            config: 配置字典
        """
        try:
            print(f"\n💾 正在保存到Memory系统...")
            
            # 提取知识点标签
            problem_text = final_result.get('problem', '')
            knowledge_data = self.extractor.extract(problem_text)
            
            # 获取验证结果
            validation = final_result.get('validation', {})
            stats = validation.get('statistics', {})
            
            # 构建记录
            record = {
                'question': problem_text,
                'answer': validation.get('answer', ''),
                'knowledge_tags': knowledge_data.get('tags', []),
                'primary_tag': knowledge_data.get('primary_tag', ''),
                'difficulty': config.get('difficulty_level', '未知'),
                'problem_type': config.get('problem_type', '数学题'),
                'solve_success': validation.get('success', False),
                'solve_steps': stats.get('total_steps', 0),
                'user_id': 'system',
                'source': 'QuestionGeneration',
                'metadata': {
                    'topic_keywords': config.get('topic_keywords', []),
                    'requirements': config.get('requirements', ''),
                    'latex': final_result.get('latex', ''),
                    'evaluation': final_result.get('evaluation', {}),
                    'iterations': len(final_result.get('history', {}).get('generated_problems', []))
                }
            }
            
            # 保存到Memory
            record_id = self.memory.save(record)
            
            print(f"✅ 已保存到Memory (ID: {record_id})")
            print(f"   知识点: {', '.join(knowledge_data.get('tags', [])[:3])}")
            
        except Exception as e:
            print(f"⚠️ Memory保存失败: {str(e)}")
            print(f"   题目生成成功,但未记录到学习历史")
    
    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def create_question_generation_flow(max_iterations: int = None) -> QuestionGenerationFlow:
    """
    创建题目生成工作流
    
    Args:
        max_iterations: 最大迭代次数
        
    Returns:
        QuestionGenerationFlow: 工作流实例
    """
    return QuestionGenerationFlow(max_iterations=max_iterations)


if __name__ == "__main__":
    # 测试代码
    flow = create_question_generation_flow()
    
    config = {
        'task_scenario': '为准备高考的学生设计一道函数与导数的压轴题',
        'problem_type': '函数与导数',
        'difficulty_level': '高考压轴题',
        'topic_keywords': ['导数', '单调性', '极值'],
        'requirements': '需要包含参数分类讨论'
    }
    
    result = flow.run(config)
    
    if result['success']:
        print("\n" + "="*80)
        print("最终结果:")
        print("="*80)
        print(result['formatted_output'])
    else:
        print(f"\n生成失败: {result.get('error', '未知错误')}")
