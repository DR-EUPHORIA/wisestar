# 导入所需的库
import os
from dotenv import load_dotenv
from openai import OpenAI  # 注意：这里我们导入 OpenAI 库

# 从 .env 文件加载环境变量
load_dotenv()

def call_xai_grok_stream(prompt: str) -> str:
    """
    使用流式方式调用 xAI Grok 平台的 API。

    Args:
        prompt: 发送给模型的用户提示。

    Returns:
        模型返回的完整内容。
    """
    try:
        # --- 关键修改：初始化客户端以指向 xAI API ---
        client = OpenAI(
            # 1. 指定 xAI 的 API 端点
            base_url="https://api.x.ai/v1",
            # 2. 从环境变量读取你的 xAI API 密钥
            api_key=os.getenv("XAI_API_KEY"),
        )

        # 设置 stream=True 来开启流式响应
        stream = client.chat.completions.create(
            # 3. 指定 xAI 的模型名称，目前是 "grok-1"
            model="grok-3-mini",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            stream=True,
        )

        # 拼接并打印流式输出内容
        full_content = ""
        print("xAI Grok Stream Response: ", end="")
        for chunk in stream:
            # 获取增量内容并进行判断
            delta_content = chunk.choices[0].delta.content
            if delta_content:
                print(delta_content, end="", flush=True)  # 实时输出
                full_content += delta_content

        print()  # 结束后换行
        return full_content

    except Exception as e:
        print(f"调用 xAI Grok API 时发生错误: {e}")
        return ""


# 主程序入口
if __name__ == "__main__":
    # 我们来问一个 Grok 模型可能更擅长回答的、具有实时性的问题
    call_xai_grok_stream("给我讲个关于程序员的笑话")
    # 或者
    # call_xai_grok_stream("What are the latest hot topics on X (Twitter) right now?")