"""
Recommender - 推荐引擎

实现每日一题和个性化推荐功能
"""

import random
import sqlite3
from datetime import datetime, date
from typing import List, Dict, Optional
from .storage import MemoryStorage
from .query import MemoryQuery


class DailyQuestion:
    """每日一题管理器"""
    
    def __init__(self, storage: MemoryStorage = None):
        """
        初始化每日一题管理器
        
        Args:
            storage: MemoryStorage实例
        """
        self.storage = storage or MemoryStorage()
        self.query = MemoryQuery(self.storage)
        print("✅ DailyQuestion 初始化成功")
    
    def get_today_question(self, user_id: str = 'default') -> Optional[dict]:
        """
        获取今日题目
        
        Args:
            user_id: 用户ID
            
        Returns:
            dict: 今日题目，如果还没有则返回None
        """
        conn = sqlite3.connect(self.storage.db_path)
        cursor = conn.cursor()
        
        try:
            today = date.today().isoformat()
            
            cursor.execute("""
                SELECT question_id, completed FROM daily_questions
                WHERE date = ? AND user_id = ?
            """, (today, user_id))
            
            row = cursor.fetchone()
            
            if row:
                question = self.storage.get_by_id(row[0])
                if question:
                    question['completed'] = bool(row[1])
                return question
            
            return None
            
        finally:
            conn.close()
    
    def generate_daily_question(self, user_id: str = 'default', strategy: str = 'balanced') -> dict:
        """
        生成今日题目
        
        Args:
            user_id: 用户ID
            strategy: 推荐策略
                - 'balanced': 平衡模式（综合考虑）
                - 'weak': 针对薄弱点
                - 'review': 复习模式
                - 'random': 随机模式
                
        Returns:
            dict: 今日题目
        """
        # 检查今天是否已经有题目
        existing = self.get_today_question(user_id)
        if existing:
            print(f"📅 [DailyQuestion] 今日题目已存在")
            return existing
        
        # 根据策略选择题目
        if strategy == 'weak':
            question = self._select_weak_point_question(user_id)
        elif strategy == 'review':
            question = self._select_review_question(user_id)
        elif strategy == 'random':
            question = self._select_random_question(user_id)
        else:  # balanced
            question = self._select_balanced_question(user_id)
        
        if not question:
            print(f"⚠️ [DailyQuestion] 没有找到合适的题目")
            return None
        
        # 保存到每日一题记录
        self._save_daily_question(question['id'], user_id)
        
        print(f"✅ [DailyQuestion] 生成今日题目成功")
        question['completed'] = False
        
        return question
    
    def _select_weak_point_question(self, user_id: str) -> Optional[dict]:
        """选择薄弱知识点的题目"""
        weak_points = self.query.get_weak_points(user_id, 3)
        
        if not weak_points:
            return self._select_random_question(user_id)
        
        # 随机选择一个薄弱知识点
        weak_tag = random.choice(weak_points)['tag']
        
        # 查找该知识点的题目
        questions = self.query.search_by_tags([weak_tag], user_id, 10)
        
        if questions:
            return random.choice(questions)
        
        return self._select_random_question(user_id)
    
    def _select_review_question(self, user_id: str) -> Optional[dict]:
        """选择复习题目（之前做过的题）"""
        recent = self.query.get_recent_questions(20, user_id)
        
        if recent:
            return random.choice(recent)
        
        return self._select_random_question(user_id)
    
    def _select_random_question(self, user_id: str) -> Optional[dict]:
        """随机选择题目"""
        recent = self.query.get_recent_questions(50, user_id)
        
        if recent:
            return random.choice(recent)
        
        return None
    
    def _select_balanced_question(self, user_id: str) -> Optional[dict]:
        """平衡模式选择题目"""
        # 70%概率选择薄弱点，30%概率复习
        if random.random() < 0.7:
            return self._select_weak_point_question(user_id)
        else:
            return self._select_review_question(user_id)
    
    def _save_daily_question(self, question_id: str, user_id: str):
        """保存每日一题记录"""
        conn = sqlite3.connect(self.storage.db_path)
        cursor = conn.cursor()
        
        try:
            today = date.today().isoformat()
            
            cursor.execute("""
                INSERT OR REPLACE INTO daily_questions (date, question_id, user_id, completed)
                VALUES (?, ?, ?, 0)
            """, (today, question_id, user_id))
            
            conn.commit()
            
        finally:
            conn.close()
    
    def mark_completed(self, user_id: str = 'default'):
        """标记今日题目为已完成"""
        conn = sqlite3.connect(self.storage.db_path)
        cursor = conn.cursor()
        
        try:
            today = date.today().isoformat()
            
            cursor.execute("""
                UPDATE daily_questions
                SET completed = 1
                WHERE date = ? AND user_id = ?
            """, (today, user_id))
            
            conn.commit()
            print(f"✅ [DailyQuestion] 标记今日题目为已完成")
            
        finally:
            conn.close()


class PersonalizedRecommender:
    """个性化推荐引擎"""
    
    def __init__(self, storage: MemoryStorage = None):
        """
        初始化推荐引擎
        
        Args:
            storage: MemoryStorage实例
        """
        self.storage = storage or MemoryStorage()
        self.query = MemoryQuery(self.storage)
        print("✅ PersonalizedRecommender 初始化成功")
    
    def recommend(self, user_id: str = 'default', limit: int = 5, 
                 strategy: str = 'adaptive') -> List[dict]:
        """
        个性化推荐题目
        
        Args:
            user_id: 用户ID
            limit: 推荐数量
            strategy: 推荐策略
                - 'adaptive': 自适应（根据用户水平）
                - 'weak_focus': 专注薄弱点
                - 'diverse': 多样化推荐
                - 'similar': 相似题目推荐
                
        Returns:
            list: 推荐的题目列表
        """
        if strategy == 'weak_focus':
            return self._recommend_weak_focus(user_id, limit)
        elif strategy == 'diverse':
            return self._recommend_diverse(user_id, limit)
        elif strategy == 'similar':
            return self._recommend_similar(user_id, limit)
        else:  # adaptive
            return self._recommend_adaptive(user_id, limit)
    
    def _recommend_adaptive(self, user_id: str, limit: int) -> List[dict]:
        """自适应推荐"""
        stats = self.query.get_statistics(user_id)
        success_rate = stats['success_rate']
        
        recommendations = []
        
        # 根据成功率调整策略
        if success_rate < 0.5:
            # 成功率低，推荐简单题和薄弱点
            weak_questions = self._recommend_weak_focus(user_id, limit // 2)
            recommendations.extend(weak_questions)
            
            # 补充一些已掌握的知识点题目（增强信心）
            mastered = self.query.get_mastered_points(user_id, 2)
            if mastered:
                for point in mastered:
                    questions = self.query.search_by_tags([point['tag']], user_id, 1)
                    recommendations.extend(questions)
        
        elif success_rate > 0.8:
            # 成功率高，推荐挑战性题目
            diverse_questions = self._recommend_diverse(user_id, limit)
            recommendations.extend(diverse_questions)
        
        else:
            # 中等水平，平衡推荐
            weak_questions = self._recommend_weak_focus(user_id, limit // 2)
            diverse_questions = self._recommend_diverse(user_id, limit - len(weak_questions))
            recommendations.extend(weak_questions)
            recommendations.extend(diverse_questions)
        
        # 去重并限制数量
        seen_ids = set()
        unique_recommendations = []
        for q in recommendations:
            if q['id'] not in seen_ids:
                seen_ids.add(q['id'])
                unique_recommendations.append(q)
                if len(unique_recommendations) >= limit:
                    break
        
        return unique_recommendations
    
    def _recommend_weak_focus(self, user_id: str, limit: int) -> List[dict]:
        """专注薄弱点推荐"""
        weak_points = self.query.get_weak_points(user_id, 5)
        
        if not weak_points:
            return self.query.get_recent_questions(limit, user_id)
        
        recommendations = []
        
        for point in weak_points:
            questions = self.query.search_by_tags([point['tag']], user_id, 2)
            recommendations.extend(questions)
            
            if len(recommendations) >= limit:
                break
        
        return recommendations[:limit]
    
    def _recommend_diverse(self, user_id: str, limit: int) -> List[dict]:
        """多样化推荐"""
        stats = self.query.get_statistics(user_id)
        tag_distribution = stats['tag_distribution']
        
        if not tag_distribution:
            return self.query.get_recent_questions(limit, user_id)
        
        # 选择不同的知识点
        all_tags = list(tag_distribution.keys())
        selected_tags = random.sample(all_tags, min(limit, len(all_tags)))
        
        recommendations = []
        
        for tag in selected_tags:
            questions = self.query.search_by_tags([tag], user_id, 1)
            recommendations.extend(questions)
        
        return recommendations[:limit]
    
    def _recommend_similar(self, user_id: str, limit: int) -> List[dict]:
        """相似题目推荐（基于最近做的题）"""
        recent = self.query.get_recent_questions(1, user_id)
        
        if not recent:
            return []
        
        last_question = recent[0]
        tags = last_question.get('knowledge_tags', [])
        
        if not tags:
            return []
        
        # 查找相似题目
        similar = self.query.search_by_tags(tags, user_id, limit + 1)
        
        # 排除最近做的题目
        similar = [q for q in similar if q['id'] != last_question['id']]
        
        return similar[:limit]
    
    def calculate_similarity(self, tags1: List[str], tags2: List[str]) -> float:
        """
        计算两个题目的相似度（基于知识点标签）
        
        Args:
            tags1: 题目1的知识点标签
            tags2: 题目2的知识点标签
            
        Returns:
            float: 相似度（0-1之间）
        """
        set1 = set(tags1)
        set2 = set(tags2)
        
        if not set1 or not set2:
            return 0.0
        
        # Jaccard相似度
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def find_similar_questions(self, question_id: str, limit: int = 5) -> List[dict]:
        """
        查找相似题目
        
        Args:
            question_id: 题目ID
            limit: 返回数量
            
        Returns:
            list: 相似题目列表（按相似度排序）
        """
        target_question = self.storage.get_by_id(question_id)
        
        if not target_question:
            return []
        
        target_tags = target_question.get('knowledge_tags', [])
        
        if not target_tags:
            return []
        
        # 获取所有题目
        all_questions = self.query.get_recent_questions(100, target_question.get('user_id', 'default'))
        
        # 计算相似度
        similarities = []
        for q in all_questions:
            if q['id'] == question_id:
                continue
            
            similarity = self.calculate_similarity(target_tags, q.get('knowledge_tags', []))
            
            if similarity > 0:
                similarities.append((q, similarity))
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return [q for q, _ in similarities[:limit]]


if __name__ == "__main__":
    # 测试代码
    daily = DailyQuestion()
    recommender = PersonalizedRecommender()
    
    # 生成每日一题
    today_question = daily.generate_daily_question(strategy='balanced')
    
    if today_question:
        print("\n" + "="*80)
        print("📅 今日一题:")
        print("="*80)
        print(f"题目: {today_question['question'][:100]}...")
        print(f"知识点: {today_question.get('knowledge_tags', [])}")
    
    # 个性化推荐
    recommendations = recommender.recommend(limit=5, strategy='adaptive')
    
    print("\n" + "="*80)
    print("💡 个性化推荐:")
    print("="*80)
    for i, q in enumerate(recommendations, 1):
        print(f"{i}. {q['question'][:80]}...")
        print(f"   知识点: {q.get('knowledge_tags', [])}")
