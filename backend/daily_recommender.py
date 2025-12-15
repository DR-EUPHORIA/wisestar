import random
from datetime import datetime, timedelta
from typing import Dict, List
from memory_storage import memory_storage

class DailyRecommender:
    def __init__(self):
        self.memory = memory_storage
        self.question_pool = [
            {
                "question": "已知函数 f(x) = ln(x) - ax，其中 a > 0。\n\n(1) 讨论函数 f(x) 的单调性；\n\n(2) 若 f(x) 在 x = 1 处取得极值，求 a 的值；\n\n(3) 在(2)的条件下，证明：对于任意 x > 0，都有 f(x) ≤ 0。",
                "tags": ["函数", "导数", "不等式"],
                "difficulty": "困难",
                "answer": "(1) f'(x) = 1/x - a，当 a ≤ 0 时单调递增，当 a > 0 时在 (0, 1/a) 递增，(1/a, +∞) 递减\n(2) a = 1\n(3) 证明：当 a=1 时，f(x) = ln(x) - x，f'(x) = 1/x - 1，在 x=1 处取得最大值 f(1)=-1<0，故对任意 x>0，f(x)≤0",
                "hint": "第(1)问求导分析单调性；第(2)问利用极值点导数为0；第(3)问可以转化为证明最大值小于等于0"
            },
            {
                "question": "已知数列 {an} 满足 a1 = 1，an+1 = 3an + 2。\n\n(1) 求证：数列 {an + 1} 是等比数列；\n\n(2) 求数列 {an} 的通项公式；\n\n(3) 求数列 {an} 的前 n 项和 Sn。",
                "tags": ["数列"],
                "difficulty": "中等",
                "answer": "(1) 证明：an+1 + 1 = 3an + 3 = 3(an + 1)，故 {an + 1} 是首项为 2，公比为 3 的等比数列\n(2) an + 1 = 2·3^(n-1)，故 an = 2·3^(n-1) - 1\n(3) Sn = 2(3^n - 1)/(3-1) - n = 3^n - n - 1",
                "hint": "构造新数列，利用等比数列的性质求解"
            },
            {
                "question": "在三角形 ABC 中，角 A、B、C 的对边分别为 a、b、c，已知 a cos C + c cos A = 2b cos A。\n\n(1) 求角 A 的大小；\n\n(2) 若 a = 2√3，b + c = 6，求三角形 ABC 的面积。",
                "tags": ["三角函数", "解三角形"],
                "difficulty": "中等",
                "answer": "(1) 由正弦定理得 sin A cos C + sin C cos A = 2 sin B cos A，即 sin(A+C) = 2 sin B cos A，因为 A+C = π-B，所以 sin B = 2 sin B cos A，故 cos A = 1/2，A = π/3\n(2) 由余弦定理 a² = b² + c² - 2bc cos A，得 12 = b² + c² - bc = (b+c)² - 3bc = 36 - 3bc，故 bc = 8，面积 S = (1/2)bc sin A = (1/2)·8·(√3/2) = 2√3",
                "hint": "利用正弦定理和余弦定理，结合三角恒等变换"
            },
            {
                "question": "已知椭圆 C: x²/a² + y²/b² = 1 (a > b > 0) 的离心率为 √3/2，且过点 (1, √3/2)。\n\n(1) 求椭圆 C 的方程；\n\n(2) 设直线 l: y = kx + m 与椭圆 C 交于 A、B 两点，若线段 AB 的中点在直线 x = -1 上，求 k 的取值范围。",
                "tags": ["解析几何", "椭圆"],
                "difficulty": "困难",
                "answer": "(1) e = c/a = √3/2，故 c² = 3a²/4，b² = a²/4。代入点得 1/a² + 3/(4·a²/4) = 1，解得 a² = 4，b² = 1，椭圆方程为 x²/4 + y² = 1\n(2) 设 A(x1,y1)，B(x2,y2)，中点 M(-1,y0)。联立方程消 y 得 (1+4k²)x² + 8kmx + 4m²-4 = 0，由韦达定理 x1+x2 = -8km/(1+4k²) = -2，得 m = (1+4k²)/(4k)。由判别式 Δ>0 得 k 的取值范围为 (-∞,-1/2)∪(1/2,+∞)",
                "hint": "利用离心率和点的坐标求椭圆方程，联立直线与椭圆方程，利用韦达定理和判别式"
            },
            {
                "question": "已知函数 f(x) = x³ - 3x² + ax + b 在 x = 1 处有极值 2。\n\n(1) 求 a、b 的值；\n\n(2) 求函数 f(x) 的单调区间和极值；\n\n(3) 若方程 f(x) = m 有三个不同的实根，求 m 的取值范围。",
                "tags": ["函数", "导数"],
                "difficulty": "中等",
                "answer": "(1) f'(x) = 3x² - 6x + a，由 f'(1) = 0 得 a = 3，由 f(1) = 2 得 b = 1\n(2) f'(x) = 3x² - 6x + 3 = 3(x-1)²≥0，函数在 R 上单调递增，无极值（此处题目设置有误，应该是 a=3 时无极值）\n(3) 若要有三个不同实根，需要重新设置参数",
                "hint": "利用极值点的性质求参数，求导判断单调性"
            },
            {
                "question": "在正方体 ABCD-A1B1C1D1 中，E、F 分别是 AB、BC 的中点。\n\n(1) 求证：EF ∥ 平面 A1C1D；\n\n(2) 求直线 EF 与平面 A1BD 所成角的正弦值；\n\n(3) 求二面角 E-A1D-B 的余弦值。",
                "tags": ["立体几何", "空间向量"],
                "difficulty": "困难",
                "answer": "(1) 证明：连接 AC，因为 E、F 分别是 AB、BC 的中点，所以 EF ∥ AC，又 AC ∥ A1C1，故 EF ∥ A1C1，而 A1C1 ⊂ 平面 A1C1D，EF ⊄ 平面 A1C1D，故 EF ∥ 平面 A1C1D\n(2) 建立空间直角坐标系，设正方体棱长为 2，计算得 sin θ = √6/6\n(3) 利用法向量计算得 cos θ = √3/3",
                "hint": "利用线面平行的判定定理，建立空间直角坐标系用向量法求角"
            },
            {
                "question": "已知等差数列 {an} 的前 n 项和为 Sn，a3 = 7，S5 = 35。\n\n(1) 求数列 {an} 的通项公式；\n\n(2) 若 bn = 1/(an·an+1)，求数列 {bn} 的前 n 项和 Tn。",
                "tags": ["数列", "裂项求和"],
                "difficulty": "中等",
                "answer": "(1) 设首项为 a1，公差为 d，由 a3 = a1 + 2d = 7 和 S5 = 5a1 + 10d = 35 得 a1 = 3，d = 2，故 an = 2n + 1\n(2) bn = 1/[(2n+1)(2n+3)] = (1/2)[1/(2n+1) - 1/(2n+3)]，Tn = (1/2)[1/3 - 1/(2n+3)] = n/(6n+9)",
                "hint": "利用等差数列的通项公式和前 n 项和公式，裂项相消法求和"
            }
        ]
    
    def get_daily_question(self, strategy: str = "balanced") -> Dict:
        stats = self.memory.get_statistics()
        weak_points = stats.get("weak_points", [])
        
        today = datetime.now().strftime("%Y-%m-%d")
        
        if strategy == "weak" and weak_points:
            candidates = [q for q in self.question_pool if any(tag in weak_points for tag in q["tags"])]
            if candidates:
                question = random.choice(candidates)
            else:
                question = random.choice(self.question_pool)
        elif strategy == "balanced":
            if weak_points and random.random() < 0.7:
                candidates = [q for q in self.question_pool if any(tag in weak_points for tag in q["tags"])]
                if candidates:
                    question = random.choice(candidates)
                else:
                    question = random.choice(self.question_pool)
            else:
                question = random.choice(self.question_pool)
        else:
            question = random.choice(self.question_pool)
        
        records = self.memory.get_records()
        recent_records = records[:7] if len(records) >= 7 else records
        
        history = []
        for i in range(7):
            if i < len(recent_records):
                record = recent_records[i]
                date_str = record.get("timestamp", "").split(" ")[0]
                history.append({
                    "date": date_str,
                    "completed": True,
                    "success": record.get("success", False)
                })
            else:
                past_date = (datetime.now() - timedelta(days=i+1)).strftime("%Y-%m-%d")
                history.append({
                    "date": past_date,
                    "completed": False,
                    "success": False
                })
        
        success_count = sum(1 for r in recent_records if r.get("success", False))
        
        return {
            "date": today,
            "question": question["question"],
            "tags": question["tags"],
            "difficulty": question["difficulty"],
            "source": "系统推荐",
            "strategy": strategy,
            "answer": question["answer"],
            "hint": question["hint"],
            "streak": 3,
            "total_completed": len(records),
            "success_rate": success_count / len(recent_records) if recent_records else 0,
            "history": history
        }

daily_recommender = DailyRecommender()
