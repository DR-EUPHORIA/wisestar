from openai import OpenAI, OpenAIError
import os
from dotenv import load_dotenv

load_dotenv()


def _get_api_key():
    return os.getenv('DEEPSEEK_API_KEY') or os.getenv('LLM_API_KEY') or os.getenv('OPENAI_API_KEY')


def _get_base_url():
    return os.getenv('DEEPSEEK_BASE_URL') or os.getenv('LLM_BASE_URL') or os.getenv('OPENAI_BASE_URL') or "https://api.deepseek.com/v1"


def _get_model():
    return os.getenv('DEEPSEEK_MODEL') or os.getenv('MODEL_NAME') or "deepseek-chat"


def call_llm_stream(prompt):
    client = OpenAI(
        api_key=_get_api_key(),
        base_url=_get_base_url(),
    )

    # 设置 stream=True 开启流式响应
    response = client.chat.completions.create(
        model=_get_model(),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        stream=True
    )

    # 拼接流式输出内容
    full_content = ""
    for chunk in response:
        # chunk.choices[0].delta.content 可能为 None（stream首块只有role），需判断
        delta = chunk.choices[0].delta
        if hasattr(delta, "content") and delta.content:
            print(delta.content, end="", flush=True)  # 实时输出
            full_content += delta.content
    print()  # 换行（可选）
    return full_content

def call_llm_stream_img(prompt: str, url: str):
    client = OpenAI(
        api_key=_get_api_key(),
        base_url=_get_base_url(),
    )

    # 正确的 messages 结构
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": { "url": url }     # ← 这里只写 url 变量
                },
                {
                    "type": "text",
                    "text": prompt                  # ← 直接用 prompt
                }
            ]
        }
    ]

    try:
        response = client.chat.completions.create(
            model=_get_model(),
            messages=messages,
            temperature=0.7,
            stream=True
        )

        full_content = ""
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content:
                print(delta.content, end="", flush=True)
                full_content += delta.content
        print()
        return full_content

    except OpenAIError as e:
        # 如果还是报“多模态数据无效”，把完整错误信息打出来
        print("调用失败：", e)
        raise


if __name__ == "__main__":
    img_url = "https://gitee.com/hwangpengyu/math-picture/raw/master/image/10.png"
    prompt = "这是一个数学几何题的图形部分，请分析这个图片，总结位置关系，强调：用中文输出！"
    call_llm_stream_img(
        prompt,
        img_url
    )