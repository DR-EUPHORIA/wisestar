FROM python:3.11-slim

# 1. 工作目录
WORKDIR /app

# 2. 只先复制 backend 的依赖文件，方便分层缓存
COPY backend/requirements.txt /app/backend/requirements.txt

# 3. 安装后端依赖（用你 backend 里那份 requirements）
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# 4. 复制整个项目（至少要有 backend 和 config/.env 所需内容）
COPY . /app

# 5. 如果需要 .env，可以用 --env-file 传进去，这里先不写死
# ENV PYTHONUNBUFFERED=1

# 6. 启动后端（如果你之前是 python backend/app.py，就保持一致）
CMD ["python", "backend/app.py"]
