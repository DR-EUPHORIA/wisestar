"""
MemoryStorage - 记忆存储层

使用SQLite存储学习历史记录
"""

import os
import sqlite3
import json
import uuid
from datetime import datetime
from typing import Optional, List, Dict


class MemoryStorage:
    """记忆存储管理器"""
    
    def __init__(self, db_path: str = None):
        """
        初始化存储
        
        Args:
            db_path: 数据库文件路径，默认为 output/memory/memory.db
        """
        if db_path is None:
            # 默认路径
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            db_dir = os.path.join(project_root, "output", "memory")
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, "memory.db")
        
        self.db_path = db_path
        self.conn = None
        
        # 初始化数据库
        self._init_database()
        
        print(f"✅ MemoryStorage 初始化成功")
        print(f"   数据库路径: {self.db_path}")
    
    def _init_database(self):
        """初始化数据库表结构"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建题目历史表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS question_history (
                id TEXT PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                question TEXT NOT NULL,
                answer TEXT,
                difficulty TEXT,
                problem_type TEXT,
                solve_success BOOLEAN,
                solve_steps INTEGER,
                user_id TEXT DEFAULT 'default',
                source TEXT,
                metadata TEXT
            )
        """)
        
        # 创建知识点标签表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_id TEXT NOT NULL,
                tag TEXT NOT NULL,
                is_primary BOOLEAN DEFAULT 0,
                importance INTEGER DEFAULT 1,
                FOREIGN KEY (question_id) REFERENCES question_history(id)
            )
        """)
        
        # 创建每日一题记录表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE UNIQUE NOT NULL,
                question_id TEXT NOT NULL,
                user_id TEXT DEFAULT 'default',
                completed BOOLEAN DEFAULT 0,
                FOREIGN KEY (question_id) REFERENCES question_history(id)
            )
        """)
        
        # 创建索引
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON question_history(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tags ON knowledge_tags(tag)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_user ON question_history(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_date ON daily_questions(date)")
        
        conn.commit()
        conn.close()
    
    def save(self, record: dict) -> str:
        """
        保存一条学习记录
        
        Args:
            record: {
                'question': str,
                'answer': str (optional),
                'knowledge_tags': list,
                'primary_tag': str (optional),
                'difficulty': str (optional),
                'problem_type': str (optional),
                'solve_success': bool (optional),
                'solve_steps': int (optional),
                'user_id': str (optional),
                'source': str (optional)
            }
            
        Returns:
            str: 记录ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # 生成ID
            record_id = str(uuid.uuid4())
            
            # 插入主记录
            cursor.execute("""
                INSERT INTO question_history 
                (id, question, answer, difficulty, problem_type, solve_success, solve_steps, user_id, source, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record_id,
                record.get('question', ''),
                record.get('answer'),
                record.get('difficulty'),
                record.get('problem_type'),
                record.get('solve_success'),
                record.get('solve_steps'),
                record.get('user_id', 'default'),
                record.get('source', 'unknown'),
                json.dumps(record.get('metadata', {}))
            ))
            
            # 插入知识点标签
            tags = record.get('knowledge_tags', [])
            primary_tag = record.get('primary_tag', '')
            
            for i, tag in enumerate(tags):
                is_primary = (tag == primary_tag)
                importance = len(tags) - i  # 重要性递减
                
                cursor.execute("""
                    INSERT INTO knowledge_tags (question_id, tag, is_primary, importance)
                    VALUES (?, ?, ?, ?)
                """, (record_id, tag, is_primary, importance))
            
            conn.commit()
            print(f"✅ [MemoryStorage] 保存成功 (ID: {record_id[:8]}...)")
            
            return record_id
            
        except Exception as e:
            conn.rollback()
            print(f"❌ [MemoryStorage] 保存失败: {e}")
            raise
        finally:
            conn.close()
    
    def get_by_id(self, record_id: str) -> Optional[dict]:
        """根据ID获取记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # 获取主记录
            cursor.execute("""
                SELECT * FROM question_history WHERE id = ?
            """, (record_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            # 获取知识点标签
            cursor.execute("""
                SELECT tag, is_primary, importance 
                FROM knowledge_tags 
                WHERE question_id = ?
                ORDER BY importance DESC
            """, (record_id,))
            
            tags_rows = cursor.fetchall()
            
            # 构建结果
            result = {
                'id': row[0],
                'timestamp': row[1],
                'question': row[2],
                'answer': row[3],
                'difficulty': row[4],
                'problem_type': row[5],
                'solve_success': bool(row[6]),
                'solve_steps': row[7],
                'user_id': row[8],
                'source': row[9],
                'metadata': json.loads(row[10]) if row[10] else {},
                'knowledge_tags': [tag[0] for tag in tags_rows],
                'primary_tag': next((tag[0] for tag in tags_rows if tag[1]), None)
            }
            
            return result
            
        finally:
            conn.close()
    
    def get_recent(self, limit: int = 10, user_id: str = 'default') -> List[dict]:
        """获取最近的记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT id FROM question_history 
                WHERE user_id = ?
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (user_id, limit))
            
            ids = [row[0] for row in cursor.fetchall()]
            
            return [self.get_by_id(id) for id in ids]
            
        finally:
            conn.close()
    
    def get_by_tags(self, tags: List[str], user_id: str = 'default', limit: int = 10) -> List[dict]:
        """根据知识点标签查询"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            placeholders = ','.join(['?' for _ in tags])
            
            cursor.execute(f"""
                SELECT DISTINCT qh.id, COUNT(kt.tag) as match_count
                FROM question_history qh
                JOIN knowledge_tags kt ON qh.id = kt.question_id
                WHERE kt.tag IN ({placeholders}) AND qh.user_id = ?
                GROUP BY qh.id
                ORDER BY match_count DESC, qh.timestamp DESC
                LIMIT ?
            """, (*tags, user_id, limit))
            
            ids = [row[0] for row in cursor.fetchall()]
            
            return [self.get_by_id(id) for id in ids]
            
        finally:
            conn.close()
    
    def get_statistics(self, user_id: str = 'default') -> dict:
        """获取统计信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # 总题目数
            cursor.execute("""
                SELECT COUNT(*) FROM question_history WHERE user_id = ?
            """, (user_id,))
            total_questions = cursor.fetchone()[0]
            
            # 知识点分布
            cursor.execute("""
                SELECT kt.tag, COUNT(*) as count
                FROM knowledge_tags kt
                JOIN question_history qh ON kt.question_id = qh.id
                WHERE qh.user_id = ?
                GROUP BY kt.tag
                ORDER BY count DESC
            """, (user_id,))
            tag_distribution = {row[0]: row[1] for row in cursor.fetchall()}
            
            # 难度分布
            cursor.execute("""
                SELECT difficulty, COUNT(*) as count
                FROM question_history
                WHERE user_id = ? AND difficulty IS NOT NULL
                GROUP BY difficulty
            """, (user_id,))
            difficulty_distribution = {row[0]: row[1] for row in cursor.fetchall()}
            
            # 成功率
            cursor.execute("""
                SELECT 
                    SUM(CASE WHEN solve_success = 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as success_rate
                FROM question_history
                WHERE user_id = ? AND solve_success IS NOT NULL
            """, (user_id,))
            success_rate = cursor.fetchone()[0] or 0.0
            
            return {
                'total_questions': total_questions,
                'tag_distribution': tag_distribution,
                'difficulty_distribution': difficulty_distribution,
                'success_rate': success_rate
            }
            
        finally:
            conn.close()
    
    def clear_all(self, user_id: str = None):
        """清空所有记录（慎用）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            if user_id:
                # 只删除特定用户的记录
                cursor.execute("DELETE FROM question_history WHERE user_id = ?", (user_id,))
            else:
                # 删除所有记录
                cursor.execute("DELETE FROM question_history")
                cursor.execute("DELETE FROM knowledge_tags")
                cursor.execute("DELETE FROM daily_questions")
            
            conn.commit()
            print(f"✅ [MemoryStorage] 清空完成")
            
        finally:
            conn.close()


if __name__ == "__main__":
    # 测试代码
    storage = MemoryStorage()
    
    # 保存测试记录
    test_record = {
        'question': '已知函数 f(x) = x^2 - 4x + 3，求函数的零点',
        'answer': 'x = 1 或 x = 3',
        'knowledge_tags': ['函数', '零点', '一元二次方程'],
        'primary_tag': '零点',
        'difficulty': '简单',
        'problem_type': '函数',
        'solve_success': True,
        'solve_steps': 3,
        'source': 'test'
    }
    
    record_id = storage.save(test_record)
    
    # 查询
    record = storage.get_by_id(record_id)
    print("\n" + "="*80)
    print("查询结果:")
    print("="*80)
    print(f"题目: {record['question']}")
    print(f"知识点: {record['knowledge_tags']}")
    
    # 统计
    stats = storage.get_statistics()
    print("\n" + "="*80)
    print("统计信息:")
    print("="*80)
    print(f"总题目数: {stats['total_questions']}")
    print(f"知识点分布: {stats['tag_distribution']}")
