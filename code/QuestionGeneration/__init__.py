"""
QuestionGeneration - AI数学题目生成系统

这是一个独立的数学题目生成模块，支持：
- AI自动生成数学题目
- REPI系统验证题目可解性
- 质量评估和迭代改进
- LaTeX格式输出
"""

__version__ = "1.0.0"
__author__ = "WiseStar Team"

from .generator import QuestionGenerator
from .validator import REPIValidator
from .evaluator import QualityEvaluator
from .flow import create_question_generation_flow

__all__ = [
    'QuestionGenerator',
    'REPIValidator',
    'QualityEvaluator',
    'create_question_generation_flow'
]
