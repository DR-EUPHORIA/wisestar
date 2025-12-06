import json
import time
import os
from openai import OpenAI

# 配置参数
api_key = 'd1585aadfbd38ffe09194fe87456f648'
MODEL_NAME = "gemini-2.5-pro-06-17"
base_url = os.getenv("DEEPSEEK_BASE_URL") or os.getenv("LLM_BASE_URL") or "https://api.deepseek.com/v1"
# 自己改路径/名字
input_file = "/Users/yjlmacbook/Documents/智多星项目/gemini-0716/第二章模型解答_2.json"
output_file = "stepmath_gold_output.json"

SCORE_PROMPT = """
你是一名专业的数学评分专家，擅长按照解题过程客观地评价数学题目的回复质量。数学题目共有三种类型，分别是计算题、证明题、开放题。
现在，给你提供一个数学问题和一个参考答案，请首先将回复内容按照推理步骤进行划分，并确保划分出的每个推理步骤都是最细粒度的，
如果是计算题的话最终答案一般为划分步骤中的最后一个推理步骤。然后，请依次判断每一个划分出的推理步骤是否正确，正确则为1，错误则为0。
紧接着，根据如下的计算公式计算出这个回复的最终得分，假设划分出的推理步骤共有n步，则计算题的最终得分S为：
S=6*(前n-1步中正确的推理步骤)/(n-1)+4*第n个推理步骤得分，
证明题和开放题的最终得分S为：S=10*(n步中正确的推理步骤)/n。
最终得分需要进行四舍五入取值，仅保留整数位。最后，请输出这道题目中所有的错误链，错误链由划分出的错误的推理步骤序号组成，如(3)-(4)-(6)。
请注意：
1.最终得分应该在0-10分之间；
2.如果中间某个步骤单独来看是正确的，但由于之前的推理步骤出错导致这个推理步骤的正确没有意义，此时这个步骤的得分为0；
3.错误链应包含没有意义的推理步骤，且应列举所有的错误链使其可以构成错误树；
4.请在分析完毕之后，另起一行，返回一个标准json格式的答案，如：
{{"(1)具体的推理步骤1...": 1, "(2)具体的推理步骤2...": 0, ..., "(n)具体的推理步骤n...": 0, "最终得分": 7, "错误链": "(3)-(4)-(6), (5)-(6)"}}
现在，请开始。

题目：
{question}

参考答案：
{answer}
"""

def evaluate_one(item, client, delay=1.2):
    """对单题自动评分并返回评分结构"""
    q = item['question']
    a = item['answer']
    prompt = SCORE_PROMPT.format(question=q.strip(), answer=a.strip())
    messages = [{"role": "user", "content": prompt}]
    try:
        r = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0
        )
        result = r.choices[0].message.content
        # 规范清理多余包裹
        if result.startswith("```json"):
            result = result[7:]
        if result.startswith("```"):
            result = result[3:]
        result = result.strip('`\n ')
        # 截取第一个大括号内JSON
        left = result.find("{")
        right = result.rfind("}")
        if left != -1 and right != -1:
            json_res = result[left:right+1]
            eval_json = json.loads(json_res)
        else:
            raise Exception("未提取到标准JSON: {}".format(result))
        print(f"√ {item.get('id', '?')} 评分完成")
    except Exception as e:
        print(f"× {item.get('id', '?')} 评分异常：{e}")
        eval_json = {"解析异常": 0}
    time.sleep(delay)
    return eval_json

def main():
    # 题库
    with open(input_file, 'r', encoding='utf-8') as fin:
        items = json.load(fin)
    # 初始化模型
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    #逐条批量评分
    result_list = []
    for item in items:
        eval_result = evaluate_one(item, client)
        item['评分结果'] = eval_result
        result_list.append(item)
    # 输出到output.json（或者自己改名字
    with open(output_file, 'w', encoding='utf-8') as fout:
        json.dump(result_list, fout, ensure_ascii=False, indent=2)
    print("全部评分完成，输出见", output_file)

import re

def safe_json_loads(result):
    # 去掉markdown包裹
    if result.startswith("```"):
        result = result.strip('` \n')
        if result.startswith('json'):
            result = result[4:].strip()
    # 尝试修正属性名
    # 替换 'key':  为 "key":
    result = re.sub(r"\'([^\']+)\'\s*:", r'"\1":', result)
    # 替换 (n) 或其他无引号key为 "key":
    result = re.sub(r"\(\d+\)[^:]*:", lambda m: '"{}":'.format(m.group(0)[:-1].strip()), result)
    # 全部属性名（key）用"双引号"包起来
    # 替换剩余不合法key，举例如需，可更进一步写规则
    try:
        return json.loads(result)
    except Exception as e:
        print("二次修正失败，原内容：", result)
        raise e

if __name__ == '__main__':
    main()
