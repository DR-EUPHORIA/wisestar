# Memory - 学习记忆系统

## 📖 模块简介

Memory是WiseStar-MathAgent项目的**学习记忆与个性化推荐系统**，能够自动记录用户的解题历史、提取知识点标签、分析学习薄弱点，并提供智能推荐功能。

### 核心功能

- **🏷️ 知识点提取**：基于LLM自动提取题目的知识点标签
- **💾 学习历史记录**：持久化存储解题记录到SQLite数据库
- **📊 统计分析**：提供丰富的学习统计和薄弱点分析
- **📅 每日一题**：智能生成每日推荐题目
- **💡 个性化推荐**：基于学习历史的多策略推荐引擎
- **📈 学习报告**：自动生成Markdown格式的学习报告

---

## 🏗️ 架构设计

### 模块组成

```
code/Memory/
├── __init__.py           # 模块初始化和导出
├── extractor.py          # 知识点提取器
├── storage.py            # 数据存储层
├── query.py              # 查询接口
├── recommender.py        # 推荐引擎
├── main.py               # CLI交互界面
└── README.md             # 本文档
```

### 数据库Schema

Memory使用SQLite作为持久化存储，包含以下表结构：

#### 1. question_history（题目历史）

| 字段 | 类型 | 说明 |
|------|------|------|
| id | TEXT PRIMARY KEY | 记录唯一ID（UUID） |
| timestamp | TEXT | 记录时间戳 |
| question | TEXT | 题目内容 |
| answer | TEXT | 答案内容 |
| difficulty | TEXT | 难度级别 |
| problem_type | TEXT | 题目类型 |
| solve_success | INTEGER | 解题是否成功（0/1） |
| solve_steps | INTEGER | 解题步数 |
| user_id | TEXT | 用户ID |
| source | TEXT | 来源（RePI/QuestionGeneration等） |
| metadata | TEXT | 元数据（JSON格式） |

#### 2. knowledge_tags（知识点标签）

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER PRIMARY KEY | 自增ID |
| question_id | TEXT | 关联的题目ID |
| tag | TEXT | 知识点标签 |
| is_primary | INTEGER | 是否为主要知识点（0/1） |
| importance | REAL | 重要性权重（0-1） |

#### 3. daily_questions（每日一题）

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER PRIMARY KEY | 自增ID |
| date | TEXT | 日期（YYYY-MM-DD） |
| question_id | TEXT | 题目ID |
| user_id | TEXT | 用户ID |
| completed | INTEGER | 是否完成（0/1） |

---

## 🔧 核心组件详解

### 1. KnowledgeExtractor（知识点提取器）

**文件**：`extractor.py`

**功能**：使用LLM从题目文本中提取知识点标签

**核心方法**：

```python
def extract(self, question: str) -> dict:
    """
    提取知识点标签
    
    Args:
        question: 题目文本
        
    Returns:
        dict: {
            'tags': ['知识点1', '知识点2', ...],
            'primary_tag': '主要知识点',
            'difficulty': '难度级别',
            'category': '题目类别'
        }
    """
```

**提取策略**：

1. **LLM提取**：使用结构化Prompt引导LLM提取3-5个知识点标签
2. **关键词回退**：如果LLM提取失败，使用关键词匹配作为备选方案
3. **标签规范化**：统一标签格式，避免重复

**常见知识点参考**：

- 代数：方程、不等式、函数、数列
- 几何：平面几何、立体几何、解析几何
- 概率统计：概率、统计、排列组合
- 微积分：导数、积分、极限
- 其他：三角函数、向量、复数、数论

---

### 2. MemoryStorage（数据存储层）

**文件**：`storage.py`

**功能**：提供SQLite数据库的CRUD操作

**核心方法**：

```python
def save(self, record: dict) -> str:
    """保存学习记录，返回记录ID"""

def get_by_id(self, record_id: str) -> Optional[dict]:
    """根据ID获取记录"""

def get_recent(self, limit: int = 10, user_id: str = 'default') -> List[dict]:
    """获取最近的记录"""

def get_by_tags(self, tags: List[str], user_id: str = 'default', limit: int = 10) -> List[dict]:
    """根据知识点标签搜索"""

def get_statistics(self, user_id: str = 'default') -> dict:
    """获取统计信息"""
```

**记录格式**：

```python
record = {
    'question': '题目内容',
    'answer': '答案内容',
    'knowledge_tags': ['知识点1', '知识点2'],
    'primary_tag': '主要知识点',
    'difficulty': '难度级别',
    'problem_type': '题目类型',
    'solve_success': True/False,
    'solve_steps': 10,
    'user_id': 'default',
    'source': 'RePI',
    'metadata': {...}  # 额外信息
}
```

---

### 3. MemoryQuery（查询接口）

**文件**：`query.py`

**功能**：提供丰富的查询和统计功能

**核心方法**：

```python
def get_weak_points(self, user_id: str = 'default', limit: int = 5) -> List[dict]:
    """
    获取薄弱知识点（错误率高的知识点）
    
    Returns:
        [{'tag': '知识点', 'total': 总数, 'failed': 失败数, 'fail_rate': 失败率}]
    """

def get_mastered_points(self, user_id: str = 'default', limit: int = 5) -> List[dict]:
    """
    获取已掌握的知识点（成功率高的知识点）
    
    Returns:
        [{'tag': '知识点', 'total': 总数, 'success': 成功数, 'success_rate': 成功率}]
    """

def get_learning_progress(self, user_id: str = 'default', days: int = 7) -> dict:
    """获取最近N天的学习进度"""

def get_wrong_questions(self, user_id: str = 'default', limit: int = 10) -> List[dict]:
    """获取错题列表"""

def generate_report(self, user_id: str = 'default') -> str:
    """生成Markdown格式的学习报告"""
```

---

### 4. DailyQuestion（每日一题）

**文件**：`recommender.py`

**功能**：管理每日一题功能

**核心方法**：

```python
def get_today_question(self, user_id: str = 'default') -> Optional[dict]:
    """获取今日题目"""

def generate_daily_question(self, user_id: str = 'default', strategy: str = 'balanced') -> dict:
    """
    生成今日题目
    
    Args:
        strategy: 推荐策略
            - 'balanced': 平衡模式（70%薄弱点 + 30%复习）
            - 'weak': 针对薄弱点
            - 'review': 复习模式
            - 'random': 随机模式
    """

def mark_completed(self, user_id: str = 'default'):
    """标记今日题目为已完成"""
```

---

### 5. PersonalizedRecommender（个性化推荐引擎）

**文件**：`recommender.py`

**功能**：基于学习历史的智能推荐

**核心方法**：

```python
def recommend(self, user_id: str = 'default', limit: int = 5, strategy: str = 'adaptive') -> List[dict]:
    """
    个性化推荐题目
    
    Args:
        strategy: 推荐策略
            - 'adaptive': 自适应（根据成功率调整）
            - 'weak_focus': 专注薄弱点
            - 'diverse': 多样化推荐
            - 'similar': 相似题目推荐
    """

def calculate_similarity(self, tags1: List[str], tags2: List[str]) -> float:
    """计算题目相似度（Jaccard相似度）"""

def find_similar_questions(self, question_id: str, limit: int = 5) -> List[dict]:
    """查找相似题目"""
```

**推荐策略详解**：

#### adaptive（自适应推荐）

根据用户成功率动态调整：

- **成功率 < 50%**：推荐简单题 + 薄弱点题目，增强信心
- **成功率 > 80%**：推荐挑战性题目，提升难度
- **成功率 50%-80%**：平衡推荐（薄弱点 + 多样化）

#### weak_focus（薄弱点专注）

专注于错误率高的知识点，帮助用户攻克难点。

#### diverse（多样化推荐）

从不同知识点中选择题目，扩展知识面。

#### similar（相似题目推荐）

基于最近做的题目，推荐相似题目进行巩固练习。

---

## 🚀 使用指南

### 1. 基础使用

#### 在代码中集成Memory

```python
from code.Memory import KnowledgeExtractor, MemoryStorage

# 初始化
extractor = KnowledgeExtractor()
memory = MemoryStorage()

# 提取知识点
question = "求函数 f(x) = x^2 + 2x + 1 的最小值"
knowledge_data = extractor.extract(question)
print(knowledge_data)
# {'tags': ['函数', '二次函数', '最值'], 'primary_tag': '二次函数', ...}

# 保存记录
record = {
    'question': question,
    'answer': '最小值为0',
    'knowledge_tags': knowledge_data['tags'],
    'primary_tag': knowledge_data['primary_tag'],
    'difficulty': '简单',
    'problem_type': '函数',
    'solve_success': True,
    'solve_steps': 5,
    'user_id': 'default',
    'source': 'manual'
}
record_id = memory.save(record)
print(f"已保存，ID: {record_id}")
```

#### 查询和统计

```python
from code.Memory import MemoryQuery

query = MemoryQuery()

# 获取统计信息
stats = query.get_statistics('default')
print(f"总题目数: {stats['total_questions']}")
print(f"成功率: {stats['success_rate']:.1%}")

# 获取薄弱知识点
weak_points = query.get_weak_points('default', 5)
for point in weak_points:
    print(f"{point['tag']}: 错误率 {point['fail_rate']:.1%}")

# 获取最近题目
recent = query.get_recent_questions(10, 'default')
for q in recent:
    print(q['question'][:50])
```

#### 推荐功能

```python
from code.Memory import DailyQuestion, PersonalizedRecommender

# 每日一题
daily = DailyQuestion()
today_q = daily.generate_daily_question('default', strategy='balanced')
print(f"今日一题: {today_q['question']}")

# 个性化推荐
recommender = PersonalizedRecommender()
recommendations = recommender.recommend('default', limit=5, strategy='adaptive')
for i, q in enumerate(recommendations, 1):
    print(f"{i}. {q['question'][:50]}")
```

---

### 2. CLI交互界面

Memory提供了完整的命令行交互界面：

```bash
python -m code.Memory.main
```

**可用命令**：

| 命令 | 说明 | 示例 |
|------|------|------|
| `stats` | 查看学习统计 | `stats` |
| `recent [N]` | 查看最近N道题目 | `recent 10` |
| `search <关键词>` | 搜索题目 | `search 函数` |
| `tags <标签1,标签2>` | 根据知识点搜索 | `tags 导数,极值` |
| `weak` | 查看薄弱知识点 | `weak` |
| `mastered` | 查看已掌握知识点 | `mastered` |
| `wrong` | 查看错题 | `wrong` |
| `daily` | 获取今日一题 | `daily` |
| `recommend [策略]` | 个性化推荐 | `recommend adaptive` |
| `report` | 生成学习报告 | `report` |
| `help` | 显示帮助信息 | `help` |
| `quit/exit` | 退出程序 | `quit` |

**使用示例**：

```
>>> stats
================================================================================
📊 学习统计
================================================================================
总题目数: 25
成功率: 76.0%

知识点分布（Top 10）:
  - 函数: 8题
  - 导数: 6题
  - 不等式: 5题
  ...

>>> weak
================================================================================
⚠️  薄弱知识点
================================================================================
  - 立体几何: 错误率 60.0% (3/5)
  - 数列: 错误率 50.0% (2/4)
  ...

>>> recommend adaptive
================================================================================
💡 个性化推荐 (策略: adaptive)
================================================================================
为您推荐 5 道题目:
1. [2024-01-15] 求数列的通项公式...
   知识点: 数列, 递推关系
2. ...
```

---

### 3. 与其他模块集成

#### 在RePI中自动记录

Memory已经集成到RePI系统中，会自动记录每次解题：

```python
from code.RePI.flow import create_RePI_Agent

# 创建Agent（默认启用Memory）
agent = create_RePI_Agent(enable_memory=True)

# 解题
shared = {"question": "你的题目", "user_id": "user123"}
agent.run(shared)

# 解题完成后会自动保存到Memory
```

#### 在QuestionGeneration中自动记录

QuestionGeneration也已集成Memory：

```python
from code.QuestionGeneration import create_question_generation_flow

# 创建工作流（默认启用Memory）
flow = create_question_generation_flow()

config = {
    'task_scenario': '为高考学生设计一道函数题',
    'problem_type': '函数与导数',
    'difficulty_level': '中等',
    'topic_keywords': ['导数', '单调性'],
    'requirements': '需要参数讨论'
}

result = flow.run(config)
# 生成的题目会自动保存到Memory
```

---

## 📊 数据分析示例

### 生成学习报告

```python
from code.Memory import MemoryQuery

query = MemoryQuery()
report = query.generate_report('default')

# 保存报告
with open('learning_report.md', 'w', encoding='utf-8') as f:
    f.write(report)
```

**报告示例**：

```markdown
# 📊 学习报告

**用户**: default
**生成时间**: 2024-01-20 15:30:00

## 📈 总体统计

- **总题目数**: 50
- **成功率**: 78.0%

## 📚 知识点分布

- **函数**: 15题
- **导数**: 12题
- **不等式**: 10题
...

## ⚠️ 薄弱知识点

- **立体几何**: 错误率 55.0% (6/11)
- **数列**: 错误率 45.0% (5/11)
...

## ✅ 已掌握知识点

- **函数**: 成功率 93.3% (14/15)
- **三角函数**: 成功率 87.5% (7/8)
...

## 📅 最近7天学习进度

- **总题目数**: 12
- **成功数**: 9
- **平均步数**: 8.5
- **成功率**: 75.0%
```

---

## 🔍 高级功能

### 1. 相似题目查找

基于知识点标签的Jaccard相似度：

```python
from code.Memory import PersonalizedRecommender

recommender = PersonalizedRecommender()

# 查找与某题相似的题目
similar = recommender.find_similar_questions('question_id_123', limit=5)

for q in similar:
    print(f"相似题目: {q['question'][:50]}")
    print(f"知识点: {q['knowledge_tags']}")
```

### 2. 自定义推荐策略

可以扩展`PersonalizedRecommender`类实现自定义推荐策略：

```python
class MyRecommender(PersonalizedRecommender):
    def _recommend_custom(self, user_id: str, limit: int) -> List[dict]:
        # 自定义推荐逻辑
        pass
```

### 3. 多用户支持

Memory支持多用户，通过`user_id`区分：

```python
# 用户A的记录
record_a = {..., 'user_id': 'user_a'}
memory.save(record_a)

# 用户B的记录
record_b = {..., 'user_id': 'user_b'}
memory.save(record_b)

# 分别查询
stats_a = query.get_statistics('user_a')
stats_b = query.get_statistics('user_b')
```

---

## 🛠️ 配置说明

### 数据库位置

默认数据库路径：`output/memory/learning_history.db`

可以通过环境变量或初始化参数修改：

```python
# 方法1：环境变量
import os
os.environ['MEMORY_DB_PATH'] = '/path/to/your/db.db'

# 方法2：初始化参数
storage = MemoryStorage(db_path='/path/to/your/db.db')
```

### LLM配置

知识点提取使用的LLM配置继承自项目的`utils.llm`模块，可以通过`.env`文件配置：

```env
LLM_MODEL=qwen2.5-max
LLM_TEMPERATURE=0.7
```

---

## 📝 最佳实践

### 1. 记录粒度

建议每次完整解题后记录一次，避免过于频繁的记录导致数据冗余。

### 2. 知识点标签规范

- 使用统一的知识点命名（如"函数"而非"函数题"）
- 避免过于细化的标签（如"二次函数的最值"可简化为"二次函数"+"最值"）
- 控制标签数量在3-5个

### 3. 推荐策略选择

- **初学者**：使用`weak_focus`策略，专注薄弱点
- **进阶学习**：使用`adaptive`策略，自动调整难度
- **考前复习**：使用`diverse`策略，全面覆盖知识点
- **巩固练习**：使用`similar`策略，重复练习相似题型

### 4. 定期生成报告

建议每周生成一次学习报告，了解学习进度和薄弱点。

---

## 🐛 故障排查

### 问题1：Memory模块导入失败

**现象**：`ImportError: No module named 'code.Memory'`

**解决**：确保项目根目录在Python路径中：

```python
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
```

### 问题2：数据库锁定

**现象**：`sqlite3.OperationalError: database is locked`

**解决**：确保同一时间只有一个进程访问数据库，或使用连接池。

### 问题3：知识点提取失败

**现象**：提取的知识点为空或不准确

**解决**：
1. 检查LLM配置是否正确
2. 查看LLM响应日志
3. 使用关键词回退机制

---

## 🔄 版本历史

### v1.0.0 (2024-01-20)

- ✅ 初始版本发布
- ✅ 实现知识点提取
- ✅ 实现数据存储和查询
- ✅ 实现每日一题功能
- ✅ 实现个性化推荐引擎
- ✅ 集成到RePI和QuestionGeneration
- ✅ 提供CLI交互界面

---

## 📚 相关文档

- [项目主README](../../README.md)
- [RePI模块文档](../RePI/README.md)
- [QuestionGeneration模块文档](../QuestionGeneration/README.md)

---

## 🤝 贡献指南

欢迎贡献代码和建议！可以扩展的方向：

1. **更多推荐策略**：如基于协同过滤的推荐
2. **可视化界面**：Web界面展示学习统计
3. **导出功能**：导出学习数据为Excel/CSV
4. **知识图谱**：构建知识点之间的关联关系
5. **学习曲线分析**：分析学习进度趋势

---

## 📧 联系方式

如有问题或建议，请联系项目维护者。

---

**Memory - 让学习更智能，让进步可追踪** 🚀
