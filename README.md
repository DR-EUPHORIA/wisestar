# WiseStar-MathAgent

![项目Logo](images/logo.png)

---

## 智多星项目-数学智能体

一个完整的数学问题求解和题目生成系统，包含多种Agent架构和工作流实现。

## 📚 项目结构

### 核心模块

#### 1. **数学解题系统** (`code/RePI`, `code/ReflectPI`, `code/DeRePI` 等)

多种用于数学解题的Agent系统：

- **RePI**: 基础的推理-计算循环Agent
- **ReflectPI**: 带反思能力的RePI系统
- **DeRePI**: 带任务分解的RePI系统
- **MRePI**: 支持多模态输入的RePI
- **ReflectRPI**: 结合RAG的完整解题系统

#### 2. **题目生成系统** (`code/QuestionGeneration`) ⭐ 新增

完整的AI数学题目自动生成系统：

- **QuestionGenerator**: AI题目生成器
- **REPIValidator**: REPI可解性验证器
- **QualityEvaluator**: 多维度质量评估器
- **RefineAnalyzer**: 自动改进分析器

详见：[QuestionGeneration模块文档](code/QuestionGeneration/README.md)

#### 3. **学习记忆系统** (`code/Memory`) ⭐ 新增

智能学习历史记录与个性化推荐系统：

- **KnowledgeExtractor**: 自动提取知识点标签
- **MemoryStorage**: SQLite持久化存储
- **MemoryQuery**: 丰富的查询和统计接口
- **DailyQuestion**: 每日一题推荐
- **PersonalizedRecommender**: 多策略个性化推荐引擎

详见：[Memory模块文档](code/Memory/README.md)

#### 4. **评测系统** (`code/Evaluation`)

自动评估Agent生成答案的正确性

#### 5. **路由系统** (`code/Routing`)

根据题型自动选择合适的Agent

### 工具模块 (`utils/`)

- `llm.py`: LLM调用接口（同步/异步/流式）
- `pyinterpreter.py`: Python代码安全执行器
- `prompt_templates.py`: 提示词模板库
- `tool_functions.py`: 通用辅助函数

### 数据集 (`data/`)

- **AIME数据集**: 美国数学邀请赛题目（1983-2025年）
- **高考数据集**: 中国高考数学题（按难度分类）

### 输出目录 (`output/`)

- `evaluation/`: 各Agent评测结果
- `question_generation/`: 生成的题目输出 ⭐ 新增
- `memory/`: 学习历史数据库 ⭐ 新增
- `Blue_book_JSON/`: 数学竞赛题库

## 🚀 快速开始

### 环境配置

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 配置环境变量（创建 `.env` 文件）：

```bash
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
MAX_RETRY=3
CONCURRENCY_LIMIT=5
TIMEOUT=30
```

### 启动 Web 界面（前后端）

1. **启动后端 API**（需在 `backend/` 目录下或指定路径运行，否则会提示找不到 `app.py`）：
   ```bash
   cd backend
   pip install -r requirements.txt
   cp .env.example .env  # 填入 DEEPSEEK_API_KEY 等
   python app.py         # 或从仓库根目录执行：python backend/app.py
   ```
   后端默认监听 `http://localhost:8000`。

2. **启动前端开发服务器**（如果没有 `anpm`，请使用常规的 `npm`/`pnpm`）：
   ```bash
   cd frontend
   npm install           # 或 pnpm install
   npm run dev           # 或 pnpm dev
   ```
   前端默认监听 `http://localhost:3048`，使用 HashRouter 路由，示例：`http://localhost:3048/#/dashboard`。

### 使用题目生成系统

#### 命令行交互模式

```bash
python -m code.QuestionGeneration.main
```

#### Python API调用

```python
from code.QuestionGeneration import create_question_generation_flow

# 创建工作流
flow = create_question_generation_flow()

# 配置出题参数
config = {
    'task_scenario': '为准备高考的学生设计一道函数与导数的压轴题',
    'problem_type': '函数与导数',
    'difficulty_level': '高考压轴题',
    'topic_keywords': ['导数', '不等式', '参数讨论'],
    'requirements': '需要包含参数分类讨论'
}

# 运行工作流
result = flow.run(config)

# 获取结果
if result['success']:
    print(result['formatted_output'])
```

### 使用解题系统

```python
from code.RePI.flow import create_RePI_Agent

# 创建RePI Agent（默认启用Memory记录）
agent = create_RePI_Agent(enable_memory=True)

# 解题
shared = {"question": "若一个等比数列的前4项和为4，前8项和为68，则该等比数列的公比为"}
agent.run(shared)

# 获取答案
print(shared.get('answer'))
# 解题记录会自动保存到Memory系统
```

### 使用学习记忆系统

#### 命令行交互模式

```bash
python -m code.Memory.main
```

#### Python API调用

```python
from code.Memory import MemoryQuery, DailyQuestion, PersonalizedRecommender

# 查询学习统计
query = MemoryQuery()
stats = query.get_statistics('default')
print(f"总题目数: {stats['total_questions']}")
print(f"成功率: {stats['success_rate']:.1%}")

# 获取薄弱知识点
weak_points = query.get_weak_points('default', 5)
for point in weak_points:
    print(f"{point['tag']}: 错误率 {point['fail_rate']:.1%}")

# 每日一题
daily = DailyQuestion()
today_q = daily.generate_daily_question('default', strategy='balanced')
print(f"今日一题: {today_q['question']}")

# 个性化推荐
recommender = PersonalizedRecommender()
recommendations = recommender.recommend('default', limit=5, strategy='adaptive')
for i, q in enumerate(recommendations, 1):
    print(f"{i}. {q['question'][:50]}...")
```

## 📖 详细文档

- [QuestionGeneration模块文档](code/QuestionGeneration/README.md) - 题目生成系统完整说明
- [Memory模块文档](code/Memory/README.md) - 学习记忆系统完整说明 ⭐ 新增
- [RePI模块文档](code/RePI/README.md) - 基础解题系统
- [ReflectPI模块文档](code/ReflectPI/README.md) - 反思解题系统
- [Evaluation模块文档](code/Evaluation/README.md) - 评测系统

## 🎯 核心特性

### 题目生成系统

- ✅ **闭环自优化**: 生成→验证→评估→改进的完整闭环
- ✅ **纯AI原创**: 无需RAG，完全基于AI生成
- ✅ **REPI验证**: 自动验证题目可解性
- ✅ **质量评估**: 多维度评分（原创性、可解性、复杂度等）
- ✅ **自动改进**: 根据验证和评估结果自动迭代优化
- ✅ **LaTeX输出**: 生成可直接编译的LaTeX源码

### 学习记忆系统 ⭐ 新增

- ✅ **知识点提取**: 基于LLM自动提取题目知识点标签
- ✅ **学习历史**: SQLite持久化存储解题记录
- ✅ **统计分析**: 成功率、薄弱点、已掌握知识点分析
- ✅ **每日一题**: 智能推荐每日练习题目
- ✅ **个性化推荐**: 多策略推荐引擎（自适应、薄弱点专注、多样化等）
- ✅ **学习报告**: 自动生成Markdown格式学习报告
- ✅ **自动集成**: 与RePI和QuestionGeneration无缝集成

### 解题系统

- ✅ **多种Agent架构**: RePI、ReflectPI、DeRePI等
- ✅ **Python代码执行**: 支持数值计算和符号运算
- ✅ **反思机制**: 自我检查和修正
- ✅ **任务分解**: 复杂问题分步求解
- ✅ **多模态支持**: 处理包含图像的题目
- ✅ **Memory集成**: 自动记录解题历史到学习记忆系统

## 🔧 技术架构

- **工作流引擎**: PocketFlow（节点式工作流）
- **LLM平台**: 兼容 OpenAI 接口的公共推理服务（默认使用 OpenAI GPT 模型）
- **数学计算**: NumPy、Pandas、SymPy
- **异步处理**: asyncio、aiohttp

## 📊 数据标签规范

所有数据集包含以下字段：

- `id`: 题目编号
- `question`: 题目文本
- `ground_truth`/`answer`: 标准答案
- `type`: 题目类型（可选）

## 🎓 使用场景

1. **教育机构**: 批量生成练习题、模拟考题，追踪学生学习进度
2. **在线教育平台**: 动态生成个性化题目，智能推荐薄弱知识点练习
3. **教师备课**: 快速生成高质量教学素材，分析学生薄弱点
4. **竞赛培训**: 生成竞赛级难度题目，针对性训练
5. **自主学习**: 生成针对性练习题，每日一题巩固学习
6. **数学研究**: 自动解题和验证，记录研究历史
7. **学习分析**: 生成学习报告，可视化学习进度和知识点掌握情况

## 📝 版本历史

### v1.2.0 (2025-01) - 最新版本

- ✨ 新增Memory学习记忆系统
- ✨ 自动提取知识点标签
- ✨ 学习历史持久化存储
- ✨ 每日一题和个性化推荐
- ✨ 学习统计和薄弱点分析
- ✨ 与RePI和QuestionGeneration无缝集成
- 📊 支持生成学习报告

### v1.1.0 (2025-01)

- ✨ 新增QuestionGeneration题目生成系统
- ✨ 完全移除RAG依赖，实现纯AI模式
- ✨ 集成REPI验证和质量评估
- ✨ 支持自动迭代改进
- 🔧 重构项目结构，模块化设计

### v1.0.0 (2024-12)

- 初始版本发布
- 实现RePI、ReflectPI等解题系统
- 支持RAG检索增强

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

## 📄 许可证

本项目遵循MIT许可证。

## 👥 团队

WiseStar Team - 智多星项目组  
淘天集团 - 用户&内容技术团队

## 📧 联系方式

如有问题或建议，请通过项目Issue反馈。

---

**注意**: 本项目默认使用公开的 OpenAI API，需要配置相应的 API Key 或其他兼容 OpenAI 协议的推理服务。
