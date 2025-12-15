from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv('OPENROUTER_API'),
)

# Set stream=True to receive a stream of events
stream = client.chat.completions.create(
  model="x-ai/grok-4-07-09",
  messages=[
    {
      "role": "system",
      "content": "你是一个高效精准的数学竞赛解题程序。对于我提出的任何数学竞赛题，请直接、清晰地提供以下内容： \n 1.  **最终答案:** 将最终答案放在最前面，并加粗显示。 \n 2.  **核心步骤:** 简洁明了地列出关键的解题步骤。 \n 3.  **公式使用:** 所有数学公式和符号都必须使用 LaTeX 格式。 \n \n 不要有任何多余的寒暄、分析或解释。直接给出核心解答即可。"
    },
    {
      "role": "user",
      "content": "已知 $m = \frac{e^x}{x} - \ln x - \frac{1}{x}$ 的两根为 $x_1$ 和 $x_2$，且 $x_1 < x_2$。 \n 求证：$x_1 + \ln x_2 < m - e + 2$"
    }
  ],
  stream=True,
)

# Iterate over the stream to print each chunk of the response
for chunk in stream:
  if chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")

print() # for a final newline