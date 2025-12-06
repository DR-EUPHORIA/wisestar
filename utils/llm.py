from openai import OpenAI, AsyncOpenAI
import os
from dotenv import load_dotenv
import asyncio

load_dotenv()


def _get_api_key():
    return os.getenv('DEEPSEEK_API_KEY') or os.getenv('LLM_API_KEY') or os.getenv('OPENAI_API_KEY')


def _get_base_url():
    return os.getenv('DEEPSEEK_BASE_URL') or os.getenv('LLM_BASE_URL') or os.getenv('OPENAI_BASE_URL') or "https://api.deepseek.com/v1"


def _get_model_name(fallback: str = "deepseek-chat"):
    return os.getenv('DEEPSEEK_MODEL') or os.getenv('MODEL_NAME') or fallback


def call_llm(prompt, temperature=0.7, thinking=True):
    client = OpenAI(
        api_key=_get_api_key(),
        base_url=_get_base_url(),
    )
    response = client.chat.completions.create(
        model=_get_model_name(),
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        extra_body={"enable_thinking": thinking},
    )

    return response.choices[0].message.content


def call_llm_stream(prompt, model_name=None):
    client = OpenAI(
        api_key=_get_api_key(),
        base_url=_get_base_url(),
    )

    # 设置 stream=True 开启流式响应
    response = client.chat.completions.create(
        model=model_name if model_name else _get_model_name(),
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
            # print(delta.content, end="", flush=True)  # 实时输出
            full_content += delta.content
    # print()  # 换行（可选）
    return full_content


async def call_llm_async(prompt):
    # 2. 使用 AsyncOpenAI 客户端
    client = AsyncOpenAI(
        api_key=_get_api_key(),
        base_url=_get_base_url(),
    )
    # 3. 使用 await 等待异步调用完成
    response = await client.chat.completions.create(
        model=_get_model_name(),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content


async def call_llm_stream_async(prompt):
    client = AsyncOpenAI(
        api_key=_get_api_key(),
        base_url=_get_base_url(),
    )

    # 设置 stream=True 开启流式响应
    response = await client.chat.completions.create(
        model=_get_model_name(),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        stream=True
    )

    # 拼接流式输出内容
    full_content = ""
    async for chunk in response:
        # chunk.choices[0].delta.content 可能为 None（stream首块只有role），需判断
        delta = chunk.choices[0].delta
        if hasattr(delta, "content") and delta.content:
            print(delta.content, end="", flush=True)  # 实时输出
            full_content += delta.content
    print()  # 换行（可选）
    return full_content


if __name__ == "__main__":
    # print(call_llm("给我讲个笑话,要有趣一点"))
    print(call_llm_stream("你是谁"))

    # 调用异步函数call_llm_stream_async
    # asyncio.run(call_llm_stream_async("你是谁"))
