from datetime import datetime, timedelta
from typing import Dict, List
from memory_storage import memory_storage

class StatisticsCalculator:
    def __init__(self):
        self.memory = memory_storage
    
    def get_weekly_data(self) -> List[Dict]:
        records = self.memory.get_records()
        
        weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        weekly_data = []
        
        for i, day in enumerate(weekdays):
            solved = 3 + (i * 2) % 5
            generated = 1 + (i * 3) % 4
            weekly_data.append({
                "day": day,
                "solved": solved,
                "generated": generated
            })
        
        return weekly_data
    
    def get_knowledge_data(self) -> List[Dict]:
        records = self.memory.get_records()
        
        tag_counts = {}
        for record in records:
            for tag in record.get("tags", []):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        colors = ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444', '#6b7280']
        knowledge_data = []
        
        sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
        
        for i, (tag, count) in enumerate(sorted_tags[:6]):
            knowledge_data.append({
                "name": tag,
                "value": count,
                "color": colors[i % len(colors)]
            })
        
        if len(sorted_tags) > 6:
            other_count = sum(count for _, count in sorted_tags[6:])
            knowledge_data.append({
                "name": "其他",
                "value": other_count,
                "color": colors[-1]
            })
        
        return knowledge_data
    
    def get_difficulty_data(self) -> List[Dict]:
        records = self.memory.get_records()
        
        difficulty_counts = {"简单": 0, "中等": 0, "困难": 0}
        
        for record in records:
            difficulty = record.get("difficulty", "中等")
            if difficulty in difficulty_counts:
                difficulty_counts[difficulty] += 1
        
        return [
            {"name": "简单", "value": difficulty_counts["简单"], "color": "#10b981"},
            {"name": "中等", "value": difficulty_counts["中等"], "color": "#f59e0b"},
            {"name": "困难", "value": difficulty_counts["困难"], "color": "#ef4444"}
        ]
    
    def get_progress_data(self) -> List[Dict]:
        months = ["1月", "2月", "3月", "4月", "5月"]
        progress_data = []
        
        base_rate = 65
        for i, month in enumerate(months):
            rate = base_rate + i * 2 + (i % 3)
            progress_data.append({
                "month": month,
                "rate": rate
            })
        
        return progress_data
    
    def get_summary(self) -> Dict:
        stats = self.memory.get_statistics()
        records = self.memory.get_records()
        
        now = datetime.now()
        week_ago = now - timedelta(days=7)
        
        weekly_count = 0
        for record in records:
            try:
                record_time = datetime.strptime(record.get("timestamp", ""), "%Y-%m-%d %H:%M:%S")
                if record_time >= week_ago:
                    weekly_count += 1
            except:
                pass
        
        streak_days = 15
        
        return {
            "success_rate": stats["success_rate"],
            "weekly_count": weekly_count,
            "mastered_topics": len(stats["mastered_points"]),
            "streak_days": streak_days
        }
    
    def get_all_statistics(self) -> Dict:
        return {
            "weekly_data": self.get_weekly_data(),
            "knowledge_data": self.get_knowledge_data(),
            "difficulty_data": self.get_difficulty_data(),
            "progress_data": self.get_progress_data(),
            "success_rate": self.get_summary()["success_rate"],
            "weekly_count": self.get_summary()["weekly_count"],
            "mastered_topics": self.get_summary()["mastered_topics"],
            "streak_days": self.get_summary()["streak_days"]
        }

statistics_calculator = StatisticsCalculator()
