import json
import os
from datetime import datetime
from typing import List, Dict, Optional

class MemoryStorage:
    def __init__(self, storage_file="data/memory_records.json"):
        self.storage_file = storage_file
        self._ensure_storage_exists()
        self._init_sample_data()
    
    def _ensure_storage_exists(self):
        os.makedirs(os.path.dirname(self.storage_file), exist_ok=True)
        if not os.path.exists(self.storage_file):
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump({"records": []}, f, ensure_ascii=False, indent=2)
    
    def _init_sample_data(self):
        data = self._load_data()
        if len(data["records"]) == 0:
            sample_records = [
                {
                    "id": 1,
                    "question": "求函数 f(x) = x³ - 3x + 1 在区间 [-2, 2] 上的最大值和最小值",
                    "answer": "最大值为 3，最小值为 -1",
                    "tags": ["函数", "导数"],
                    "difficulty": "中等",
                    "success": True,
                    "steps": 8,
                    "timestamp": "2025-01-20 14:30:25",
                    "time_used": "45秒"
                },
                {
                    "id": 2,
                    "question": "已知等比数列前4项和为4，前8项和为68，求公比",
                    "answer": "公比 q = 2",
                    "tags": ["数列"],
                    "difficulty": "简单",
                    "success": True,
                    "steps": 6,
                    "timestamp": "2025-01-20 13:15:10",
                    "time_used": "32秒"
                },
                {
                    "id": 3,
                    "question": "证明：在正方体中，异面直线所成角的范围",
                    "answer": "解题失败",
                    "tags": ["立体几何"],
                    "difficulty": "困难",
                    "success": False,
                    "steps": 12,
                    "timestamp": "2025-01-20 11:45:33",
                    "time_used": "120秒"
                },
                {
                    "id": 4,
                    "question": "已知 sin(α) = 3/5，α ∈ (π/2, π)，求 cos(α) 的值",
                    "answer": "cos(α) = -4/5",
                    "tags": ["三角函数"],
                    "difficulty": "简单",
                    "success": True,
                    "steps": 4,
                    "timestamp": "2025-01-20 10:20:15",
                    "time_used": "18秒"
                },
                {
                    "id": 5,
                    "question": "求不等式 |x-1| + |x+2| < 5 的解集",
                    "answer": "解集为 (-3, 2)",
                    "tags": ["不等式"],
                    "difficulty": "中等",
                    "success": True,
                    "steps": 7,
                    "timestamp": "2025-01-19 16:45:20",
                    "time_used": "38秒"
                },
                {
                    "id": 6,
                    "question": "求椭圆 x²/16 + y²/9 = 1 的焦点坐标",
                    "answer": "焦点坐标为 (±√7, 0)",
                    "tags": ["解析几何"],
                    "difficulty": "简单",
                    "success": True,
                    "steps": 3,
                    "timestamp": "2025-01-19 15:30:45",
                    "time_used": "15秒"
                },
                {
                    "id": 7,
                    "question": "已知向量 a=(1,2)，b=(3,4)，求 a·b 和 |a+b|",
                    "answer": "a·b = 11，|a+b| = √29",
                    "tags": ["向量"],
                    "difficulty": "简单",
                    "success": True,
                    "steps": 5,
                    "timestamp": "2025-01-19 14:10:30",
                    "time_used": "22秒"
                },
                {
                    "id": 8,
                    "question": "求复数 z = (1+i)/(1-i) 的模和辐角主值",
                    "answer": "解题失败",
                    "tags": ["复数"],
                    "difficulty": "中等",
                    "success": False,
                    "steps": 6,
                    "timestamp": "2025-01-19 12:55:18",
                    "time_used": "55秒"
                },
                {
                    "id": 9,
                    "question": "求函数 f(x) = e^x - x 的单调区间",
                    "answer": "在 (-∞, 0) 上单调递减，在 (0, +∞) 上单调递增",
                    "tags": ["函数", "导数"],
                    "difficulty": "中等",
                    "success": True,
                    "steps": 6,
                    "timestamp": "2025-01-19 11:20:40",
                    "time_used": "35秒"
                },
                {
                    "id": 10,
                    "question": "已知数列 {an} 满足 a1=1，an+1=2an+1，求通项公式",
                    "answer": "an = 2^n - 1",
                    "tags": ["数列"],
                    "difficulty": "中等",
                    "success": True,
                    "steps": 8,
                    "timestamp": "2025-01-19 10:05:25",
                    "time_used": "42秒"
                },
                {
                    "id": 11,
                    "question": "证明：对于任意正实数 a, b，有 (a+b)/2 ≥ √(ab)",
                    "answer": "证明成功（均值不等式）",
                    "tags": ["不等式"],
                    "difficulty": "简单",
                    "success": True,
                    "steps": 5,
                    "timestamp": "2025-01-18 16:30:15",
                    "time_used": "28秒"
                },
                {
                    "id": 12,
                    "question": "求三棱锥 P-ABC 的体积，已知底面积为 S，高为 h",
                    "answer": "V = (1/3)Sh",
                    "tags": ["立体几何"],
                    "difficulty": "简单",
                    "success": True,
                    "steps": 3,
                    "timestamp": "2025-01-18 15:15:50",
                    "time_used": "12秒"
                },
                {
                    "id": 13,
                    "question": "求概率：从5个红球和3个白球中随机取2个，恰好1红1白的概率",
                    "answer": "P = 15/28",
                    "tags": ["概率统计"],
                    "difficulty": "中等",
                    "success": True,
                    "steps": 6,
                    "timestamp": "2025-01-18 14:20:35",
                    "time_used": "30秒"
                },
                {
                    "id": 14,
                    "question": "求双曲线 x²/4 - y²/9 = 1 的渐近线方程",
                    "answer": "y = ±(3/2)x",
                    "tags": ["解析几何"],
                    "difficulty": "简单",
                    "success": True,
                    "steps": 4,
                    "timestamp": "2025-01-18 13:10:20",
                    "time_used": "20秒"
                },
                {
                    "id": 15,
                    "question": "求三角形面积，已知两边 a=3, b=4，夹角 C=60°",
                    "answer": "S = 3√3",
                    "tags": ["三角函数"],
                    "difficulty": "简单",
                    "success": True,
                    "steps": 3,
                    "timestamp": "2025-01-18 11:45:10",
                    "time_used": "15秒"
                },
                {
                    "id": 16,
                    "question": "求极限 lim(x→0) (sin x)/x",
                    "answer": "极限值为 1",
                    "tags": ["函数", "极限"],
                    "difficulty": "中等",
                    "success": True,
                    "steps": 5,
                    "timestamp": "2025-01-17 16:55:30",
                    "time_used": "25秒"
                },
                {
                    "id": 17,
                    "question": "求矩阵 [[1,2],[3,4]] 的行列式",
                    "answer": "det = -2",
                    "tags": ["线性代数"],
                    "difficulty": "简单",
                    "success": True,
                    "steps": 2,
                    "timestamp": "2025-01-17 15:30:45",
                    "time_used": "10秒"
                },
                {
                    "id": 18,
                    "question": "求抛物线 y² = 4x 的焦点和准线",
                    "answer": "焦点 (1, 0)，准线 x = -1",
                    "tags": ["解析几何"],
                    "difficulty": "简单",
                    "success": True,
                    "steps": 3,
                    "timestamp": "2025-01-17 14:15:20",
                    "time_used": "18秒"
                },
                {
                    "id": 19,
                    "question": "求二项式 (x+y)^5 展开式中 x²y³ 的系数",
                    "answer": "系数为 10",
                    "tags": ["组合数学"],
                    "difficulty": "中等",
                    "success": True,
                    "steps": 4,
                    "timestamp": "2025-01-17 13:20:15",
                    "time_used": "22秒"
                },
                {
                    "id": 20,
                    "question": "求函数 f(x) = x³ - 3x² + 2 的拐点",
                    "answer": "拐点为 (1, 0)",
                    "tags": ["函数", "导数"],
                    "difficulty": "中等",
                    "success": True,
                    "steps": 7,
                    "timestamp": "2025-01-17 11:50:40",
                    "time_used": "40秒"
                }
            ]
            data["records"] = sample_records
            self._save_data(data)
    
    def _load_data(self) -> Dict:
        with open(self.storage_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _save_data(self, data: Dict):
        with open(self.storage_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def add_record(self, record: Dict) -> int:
        data = self._load_data()
        record["id"] = len(data["records"]) + 1
        record["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data["records"].append(record)
        self._save_data(data)
        return record["id"]
    
    def get_records(self, tag: Optional[str] = None, difficulty: Optional[str] = None) -> List[Dict]:
        data = self._load_data()
        records = data["records"]
        
        if tag:
            records = [r for r in records if tag in r.get("tags", [])]
        
        if difficulty:
            records = [r for r in records if r.get("difficulty") == difficulty]
        
        return sorted(records, key=lambda x: x.get("timestamp", ""), reverse=True)
    
    def get_statistics(self) -> Dict:
        data = self._load_data()
        records = data["records"]
        
        if not records:
            return {
                "total": 0,
                "success_rate": 0.0,
                "weak_points": [],
                "mastered_points": []
            }
        
        total = len(records)
        success_count = sum(1 for r in records if r.get("success", False))
        success_rate = success_count / total if total > 0 else 0.0
        
        tag_stats = {}
        for record in records:
            for tag in record.get("tags", []):
                if tag not in tag_stats:
                    tag_stats[tag] = {"total": 0, "success": 0}
                tag_stats[tag]["total"] += 1
                if record.get("success", False):
                    tag_stats[tag]["success"] += 1
        
        weak_points = []
        mastered_points = []
        
        for tag, stats in tag_stats.items():
            rate = stats["success"] / stats["total"] if stats["total"] > 0 else 0
            if stats["total"] >= 2:
                if rate < 0.6:
                    weak_points.append(tag)
                elif rate >= 0.8:
                    mastered_points.append(tag)
        
        return {
            "total": total,
            "success_rate": success_rate,
            "weak_points": weak_points,
            "mastered_points": mastered_points
        }

memory_storage = MemoryStorage()
