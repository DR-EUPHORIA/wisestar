## RePI

Re=reasoning  
PI=python interpreter

RePI=ReAct结构将Act节点换为PI节点；

- Re节点用于进行数学推理和计算代码编写
- PI节点用于执行python代码并进行计算

Flow流向:
![img.png](../../images/RePI.png)

batchrun.py：按顺序解答指定数据集中的题目，并与题目答案进行比较，详见文件中注释

后端Agent使用:

1. 在`.env`文件中设置 `DEEPSEEK_API_KEY`（或 `LLM_API_KEY`）为自己的 DeepSeek Key，`DEEPSEEK_BASE_URL=https://api.deepseek.com/v1`，`MODEL_NAME`/`DEEPSEEK_MODEL=deepseek-chat`。
2. 可直接在flow.py下进行整体流程测试