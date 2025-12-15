"""
MemoryQuery - 记忆查询接口

提供丰富的查询和统计功能
"""

from typing import List, Dict, Optional
from .storage import MemoryStorage


class MemoryQuery:
    """记忆查询管理器"""
    
    def __init__(self, storage: MemoryStorage = None):
        """
        初始化查询器
        
        Args:
            storage: MemoryStorage实例，如果不提供则创建新实例
        """
        self.storage = storage or MemoryStorage()
        print("✅ MemoryQuery 初始化成功")
    
    def get_recent_questions(self, limit: int = 10, user_id: str = 'default') -> List[dict]:
        """
        获取最近的题目
        
        Args:
            limit: 返回数量
            user_id: 用户ID
            
        Returns:
            list: 题目记录列表
        """
        return self.storage.get_recent(limit, user_id)
    
    def search_by_tags(self, tags: List[str], user_id: str = 'default', limit: int = 10) -> List[dict]:
        """
        根据知识点标签搜索
        
        Args:
            tags: 知识点标签列表
            user_id: 用户ID
            limit: 返回数量
            
        Returns:
            list: 匹配的题目列表
        """
        return self.storage.get_by_tags(tags, user_id, limit)
    
    def search_by_keyword(self, keyword: str, user_id: str = 'default', limit: int = 10) -> List[dict]:
        """
        根据关键词搜索题目
        
        Args:
            keyword: 搜索关键词
            user_id: 用户ID
            limit: 返回数量
            
        Returns:
            list: 匹配的题目列表
        """
        import sqlite3
        
        conn = sqlite3.connect(self.storage.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT id FROM question_history
                WHERE user_id = ? AND question LIKE ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (user_id, f'%{keyword}%', limit))
            
            ids = [row[0] for row in cursor.fetchall()]
            return [self.storage.get_by_id(id) for id in ids]
            
        finally:
            conn.close()
    
    def get_statistics(self, user_id: str = 'default') -> dict:
        """
        获取统计信息
        
        Args:
            user_id: 用户ID
            
        Returns:
            dict: 统计信息
        """
        return self.storage.get_statistics(user_id)
    
    def get_weak_points(self, user_id: str = 'default', limit: int = 5) -> List[dict]:
        """
        获取薄弱知识点（错误率高的知识点）
        
        Args:
            user_id: 用户ID
            limit: 返回数量
            
        Returns:
            list: [{'tag': '知识点', 'total': 总数, 'failed': 失败数, 'fail_rate': 失败率}]
        """
        import sqlite3
        
        conn = sqlite3.connect(self.storage.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT 
                    kt.tag,
                    COUNT(*) as total,
                    SUM(CASE WHEN qh.solve_success = 0 THEN 1 ELSE 0 END) as failed,
                    SUM(CASE WHEN qh.solve_success = 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as fail_rate
                FROM knowledge_tags kt
                JOIN question_history qh ON kt.question_id = qh.id
                WHERE qh.user_id = ? AND qh.solve_success IS NOT NULL
                GROUP BY kt.tag
                HAVING COUNT(*) >= 2
                ORDER BY fail_rate DESC, total DESC
                LIMIT ?
            """, (user_id, limit))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'tag': row[0],
                    'total': row[1],
                    'failed': row[2],
                    'fail_rate': row[3]
                })
            
            return results
            
        finally:
            conn.close()
    
    def get_mastered_points(self, user_id: str = 'default', limit: int = 5) -> List[dict]:
        """
        获取已掌握的知识点（成功率高的知识点）
        
        Args:
            user_id: 用户ID
            limit: 返回数量
            
        Returns:
            list: [{'tag': '知识点', 'total': 总数, 'success': 成功数, 'success_rate': 成功率}]
        """
        import sqlite3
        
        conn = sqlite3.connect(self.storage.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT 
                    kt.tag,
                    COUNT(*) as total,
                    SUM(CASE WHEN qh.solve_success = 1 THEN 1 ELSE 0 END) as success,
                    SUM(CASE WHEN qh.solve_success = 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as success_rate
                FROM knowledge_tags kt
                JOIN question_history qh ON kt.question_id = qh.id
                WHERE qh.user_id = ? AND qh.solve_success IS NOT NULL
                GROUP BY kt.tag
                HAVING COUNT(*) >= 3
                ORDER BY success_rate DESC, total DESC
                LIMIT ?
            """, (user_id, limit))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'tag': row[0],
                    'total': row[1],
                    'success': row[2],
                    'success_rate': row[3]
                })
            
            return results
            
        finally:
            conn.close()
    
    def get_learning_progress(self, user_id: str = 'default', days: int = 7) -> dict:
        """
        获取学习进度（最近N天）
        
        Args:
            user_id: 用户ID
            days: 天数
            
        Returns:
            dict: 学习进度信息
        """
        import sqlite3
        from datetime import datetime, timedelta
        
        conn = sqlite3.connect(self.storage.db_path)
        cursor = conn.cursor()
        
        try:
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            # 每日题目数
            cursor.execute("""
                SELECT DATE(timestamp) as date, COUNT(*) as count
                FROM question_history
                WHERE user_id = ? AND DATE(timestamp) >= ?
                GROUP BY DATE(timestamp)
                ORDER BY date
            """, (user_id, start_date))
            
            daily_counts = {row[0]: row[1] for row in cursor.fetchall()}
            
            # 总体统计
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN solve_success = 1 THEN 1 ELSE 0 END) as success,
                    AVG(solve_steps) as avg_steps
                FROM question_history
                WHERE user_id = ? AND DATE(timestamp) >= ?
            """, (user_id, start_date))
            
            row = cursor.fetchone()
            
            return {
                'period': f'最近{days}天',
                'daily_counts': daily_counts,
                'total_questions': row[0] or 0,
                'success_count': row[1] or 0,
                'avg_steps': row[2] or 0,
                'success_rate': (row[1] / row[0]) if row[0] else 0
            }
            
        finally:
            conn.close()
    
    def get_wrong_questions(self, user_id: str = 'default', limit: int = 10) -> List[dict]:
        """
        获取错题（解题失败的题目）
        
        Args:
            user_id: 用户ID
            limit: 返回数量
            
        Returns:
            list: 错题列表
        """
        import sqlite3
        
        conn = sqlite3.connect(self.storage.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT id FROM question_history
                WHERE user_id = ? AND solve_success = 0
                ORDER BY timestamp DESC
                LIMIT ?
            """, (user_id, limit))
            
            ids = [row[0] for row in cursor.fetchall()]
            return [self.storage.get_by_id(id) for id in ids]
            
        finally:
            conn.close()
    
    def generate_report(self, user_id: str = 'default') -> str:
        """
        生成学习报告
        
        Args:
            user_id: 用户ID
            
        Returns:
            str: Markdown格式的学习报告
        """
        stats = self.get_statistics(user_id)
        weak_points = self.get_weak_points(user_id, 5)
        mastered_points = self.get_mastered_points(user_id, 5)
        progress = self.get_learning_progress(user_id, 7)
        
        report_lines = []
        report_lines.append("# 📊 学习报告")
        report_lines.append("")
        report_lines.append(f"**用户**: {user_id}")
        report_lines.append(f"**生成时间**: {self._get_timestamp()}")
        report_lines.append("")
        
        report_lines.append("## 📈 总体统计")
        report_lines.append("")
        report_lines.append(f"- **总题目数**: {stats['total_questions']}")
        report_lines.append(f"- **成功率**: {stats['success_rate']:.1%}")
        report_lines.append("")
        
        report_lines.append("## 📚 知识点分布")
        report_lines.append("")
        for tag, count in sorted(stats['tag_distribution'].items(), key=lambda x: x[1], reverse=True)[:10]:
            report_lines.append(f"- **{tag}**: {count}题")
        report_lines.append("")
        
        if weak_points:
            report_lines.append("## ⚠️ 薄弱知识点")
            report_lines.append("")
            for point in weak_points:
                report_lines.append(f"- **{point['tag']}**: 错误率 {point['fail_rate']:.1%} ({point['failed']}/{point['total']})")
            report_lines.append("")
        
        if mastered_points:
            report_lines.append("## ✅ 已掌握知识点")
            report_lines.append("")
            for point in mastered_points:
                report_lines.append(f"- **{point['tag']}**: 成功率 {point['success_rate']:.1%} ({point['success']}/{point['total']})")
            report_lines.append("")
        
        report_lines.append("## 📅 最近7天学习进度")
        report_lines.append("")
        report_lines.append(f"- **总题目数**: {progress['total_questions']}")
        report_lines.append(f"- **成功数**: {progress['success_count']}")
        report_lines.append(f"- **平均步数**: {progress['avg_steps']:.1f}")
        report_lines.append(f"- **成功率**: {progress['success_rate']:.1%}")
        report_lines.append("")
        
        return "\n".join(report_lines)
    
    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    # 测试代码
    query = MemoryQuery()
    
    # 获取统计信息
    stats = query.get_statistics()
    print("\n" + "="*80)
    print("统计信息:")
    print("="*80)
    print(f"总题目数: {stats['total_questions']}")
    print(f"成功率: {stats['success_rate']:.1%}")
    
    # 生成报告
    report = query.generate_report()
    print("\n" + "="*80)
    print("学习报告:")
    print("="*80)
    print(report)
