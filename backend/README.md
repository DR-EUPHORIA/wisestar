# WiseStar-MathAgent 后端API服务

## 快速开始

### 1. 安装依赖

```bash
cd backend
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env` 并填入你的API Key：

```bash
cp .env.example .env
```

编辑 `.env` 文件（默认基于公共可用的 DeepSeek 端点，如网络受限可替换为镜像或代理）：
```
LLM_API_KEY=your_deepseek_api_key
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
```

### 3. 启动服务

```bash
python app.py
```

> 如果你在仓库根目录执行，可直接运行 `python backend/app.py`；请确保命令的工作目录包含 `app.py` 文件，否则会出现 “can't open file '.../app.py'” 的路径错误。

服务将在 `http://localhost:8000` 启动

### 4. 查看API文档

访问 `http://localhost:8000/docs` 查看自动生成的API文档

## API端点

### 1. POST /solve - 解题
```json
{
  "question": "题目内容"
}
```

### 2. POST /generate - 生成题目
```json
{
  "difficulty_level": "中等",
  "problem_type": "函数",
  "topic_keywords": ["导数", "不等式"],
  "requirements": "需要参数讨论"
}
```

### 3. GET /statistics - 统计数据
返回学习统计信息

### 4. GET /memory - 学习记录
支持参数：`tag`, `difficulty`

### 5. GET /daily - 每日一题
返回今日推荐题目

### 6. POST /daily/submit - 提交答案
```json
{
  "questionId": 1,
  "answer": "答案内容"
}
```

### 7. POST /plot/execute - 执行Python代码
```json
{
  "code": "import matplotlib.pyplot as plt..."
}
```

### 8. POST /plot/generate - AI生成绘图代码
```json
{
  "description": "绘制一个圆和一条直线"
}
```

## 项目结构

```
backend/
├── app.py              # FastAPI主应用
├── config.py           # 配置文件
├── llm_client.py       # LLM调用客户端
├── python_executor.py  # Python代码执行器
├── prompts.py          # Prompt模板
├── requirements.txt    # Python依赖
├── .env.example        # 环境变量示例
└── README.md           # 本文档
```

## 注意事项

1. **API Key安全**：不要将 `.env` 文件提交到版本控制
2. **CORS配置**：已配置允许前端跨域访问
3. **代码执行安全**：Python执行器有安全限制，禁止危险操作
4. **超时设置**：LLM调用默认超时60秒
5. **错误处理**：所有API都有完整的错误处理机制

## 开发建议

- 使用 `uvicorn app:app --reload` 启动开发模式
- 查看 `/docs` 端点测试API
- 查看日志排查问题
- 根据需要调整 `config.py` 中的配置
