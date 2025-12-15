# RePI+Reflect

在RePI（推理&python代码解释器）的基础上引入反思节点，并且将路由选择权转交给Reflect;  
本质上是对RePI中Re节点功能的进一步拆分,增加用于反思的token开销;  
也可以理解为在Reflect Agent流程中,为Re节点增加了调用python解释器的能力;

![img.png](../../images/ReflectPI.png)

## 使用

1. 在 `.env` 文件中设置 `DEEPSEEK_API_KEY`（或 `LLM_API_KEY`）为自己的 DeepSeek Key，`DEEPSEEK_BASE_URL=https://api.deepseek.com/v1`，`MODEL_NAME`/`DEEPSEEK_MODEL=deepseek-chat`。
2. 可接在flow.py下进行整体流程测试