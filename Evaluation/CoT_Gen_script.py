import json
import base64
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

#参数
api_key = 'd1585aadfbd38ffe09194fe87456f648'
MODEL_NAME = "gemini-2.5-flash-06-17"
#自行替换存储位置
save_dir = '/Users/yjlmacbook/Documents/智多星项目/gemini-0716'
os.makedirs(save_dir, exist_ok=True)

prompt = """
你是一名专业的数学答题专家，擅长根据图片中呈现的问题及其对应的简洁答案，将其拓展为步骤清晰、逻辑严谨的详细解题过程。

你的任务是：

1. 识别并理解题目内容与要求，即使图像文字不够完整，也尽量做出合理推断；
2. 基于提供的简洁答案，反向还原完整的解题思路和推理过程；
3. 分步骤书写解答过程，每一步都包含必要的公式、定理或运算依据；
4. 语言通俗易懂，适合中学生或大学生阅读理解；
5. 控制在一千字以内，确保内容精炼、重点突出、不啰嗦；
6. 请将我提供的图片内容（可能包括题目截图、草稿纸等）转化为结构清晰、便于学习的数学解答。

最后应将回答记录为json格式，第一行为"question",第二行为"answer"。
"""

def img_to_base64_str(filepath):
    with open(filepath, 'rb') as f:
        img_bytes = f.read()
    mime = 'jpeg' if filepath.lower().endswith('jpg') or filepath.lower().endswith('jpeg') else 'png'
    return f'data:image/{mime};base64,' + base64.b64encode(img_bytes).decode('utf-8')

def process_problem(problem_id, img_paths, client):
    imgs = [img_to_base64_str(path) for path in img_paths]
    content = [{"type": "text", "text": prompt}]
    for pic_b64 in imgs:
        content.append({"type": "image_url", "image_url": {"url": pic_b64}})
    messages = [{"role": "user", "content": content}]
    # 调用API
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages
    )
    result = completion.choices[0].message.content
    try:
        # 去掉 markdown 包裹
        if result.startswith("```json"):
            result = result[7:]
        if result.startswith("```"):
            result = result[3:]
        result = result.strip('`\n ')
        data = json.loads(result)
    except Exception as e:
        print(f'【{problem_id}】返回内容无法标准JSON解析，原文如下：\n', result)
        data = None
    # 写文件
    if data is not None:
        save_path = os.path.join(save_dir, f'{problem_id}.json')
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"题目{problem_id} 已保存至: {save_path}")
    else:
        print(f"题目{problem_id} 结果未被保存。")
    return problem_id, data is not None

if __name__ == '__main__':
    # 自行替换路径
    problems = [
        (' IMO5', ['/Users/yjlmacbook/Desktop/IMO5.png']),
        # 可继续添加...
    ]

    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL") or os.getenv("LLM_BASE_URL") or "https://api.deepseek.com/v1"
    )
#max_worker为并发数，可自行设置。默认2
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        for prob_id, img_paths in problems:
            futures.append(executor.submit(process_problem, prob_id, img_paths, client))
        for future in as_completed(futures):
            prob_id, success = future.result()
            if not success:
                print(f"题目{prob_id} 处理失败！")

