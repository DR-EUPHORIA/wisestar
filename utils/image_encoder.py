import os
from typing import Optional
from openai import OpenAI


def _resolve_base_url(base_url: Optional[str] = None) -> str:
    return base_url or os.getenv('DEEPSEEK_BASE_URL') or os.getenv('LLM_BASE_URL') or os.getenv('OPENAI_BASE_URL') or "https://api.deepseek.com/v1"


class ImageEncoder:
    def __init__(self, api_key: str, base_url: str | None = None):
        self.client = OpenAI(
            api_key=api_key,
            base_url=_resolve_base_url(base_url)
        )

    def encode_image_with_qwen(self, image_url: str, question_text: str) -> Optional[str]:
        """util
        使用 qwen-vl-max 模型对图像和题目文本进行分析，返回自然语言描述。

        参数:
            image_url (str): 图像的 URL 地址
            question_text (str): 题目文本内容

        返回:
            str: 模型返回的描述文本，失败时返回 None
        """
        prompt = (
            f"你是一位数学领域的专家，现在有一些带图像的几何题目需要你去做分析。\n"
            f"请根据题目条件去解析几何图像的内容。\n"
            f"题目是：{question_text}\n"
            f"图像URL是：{image_url}\n"
            f"请给出10000 token 以内的自然语言描述，只分析题目条件和图像特征。"
        )

        try:
            completion = self.client.chat.completions.create(
                model="gpt-4o-mini-vision",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"调用模型失败: {e}")
            return None

    def encode_image_to_lean(self, image_url: str, text: str) -> Optional[str]:
        """util
        使用 qwen-vl-max 模型对图像和题目文本进行分析，返回lean语言描述。

        参数:
            image_url (str): 图像的 URL 地址
            question_text (str): 题目文本内容

        返回:
            str: 模型返回的描述文本，失败时返回 None
        """
        prompt = (f"你是一位数学领域的专家，现在有一些带图像的几何题目需要你去做分析"
                  f"，我需要你根据题目的条件去解析几何图像的内容，题目是{text}，"
                  f"图像url已经上传，请用lean语言描述整道题目，限制输出在2000token以内，"
                  f"你只需重构题目，不需要解答,只输出lean语言描述，注视可以用中文")
        try:
            completion = self.client.chat.completions.create(
                model="gpt-4o-mini-vision",
                messages=[{"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"{image_url}"}},
                    {"type": "text", "text": f"{prompt}"},
                ]}]
            )
            answer = completion.choices[0].message.content
            return answer
        except Exception as e:
            print(f"调用模型失败: {e}")
            return None
