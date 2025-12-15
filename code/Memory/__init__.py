"""
Memory - 学习记忆系统

自动记录用户的学习历史，提取知识点标签，提供个性化推荐
"""

__version__ = "1.0.0"
__author__ = "WiseStar Team"

from .extractor import KnowledgeExtractor
from .storage import MemoryStorage
from .query import MemoryQuery
from .recommender import DailyQuestion, PersonalizedRecommender

__all__ = [
    'KnowledgeExtractor',
    'MemoryStorage',
    'MemoryQuery',
    'DailyQuestion',
    'PersonalizedRecommender'
]
