from pocketflow import Node, Flow
from code.RePI.node import ReNode, PINode, AnswerNode
from utils.llm import call_llm_stream as call_llm
from dotenv import load_dotenv
from utils.prompt_templates import REPI_DISTILL_NODE_PROMPT, REPI_EVALUATION_NODE_PROMPT
import json
import csv
import pandas as pd
import os

'''
自动化测试
功能：调用agent顺序做数据集中的题目，并与答案进行对比
输入：题目数据集（包含id、题目、答案）
输出：一份csv，列为：id、agent给出的答案answer、题目答案truth、对比结果final
切换数据集请修改：filename、input_filepath
'''

'''
DistillNode---提取节点
需要使用的数据: final_output
功能: 从完整解题过程final_output中提取出最终答案
'''


class DistillNode(Node):
    def prep(self, shared):
        solve = shared.get('answer', 'none')
        return solve

    def exec(self, prep_res):
        prompt = REPI_DISTILL_NODE_PROMPT.format(prep_res=prep_res)
        response = call_llm(prompt)
        return response

    def post(self, shared, prep_res, exec_res):
        shared['answer'] = exec_res


'''
EvaluationNode---比对节点
需要使用的数据: answer, truth
功能: 对比模型输出的答案与标准答案的一致性
'''


class EvaluationNode(Node):
    def prep(self, shared):
        # 准备评估需要的所有数据
        return {
            "model_answer": shared.get("answer", "none"),
            "ground_truth": shared.get("truth", "none")
        }

    def exec(self, prep_res):
        model_answer = prep_res["model_answer"]
        ground_truth = prep_res["ground_truth"]

        # 如果任何一个答案无效，直接返回
        if model_answer == "none" or ground_truth == "none":
            return "Skipped - Missing Answer"

        eval_prompt = REPI_EVALUATION_NODE_PROMPT.format(model_answer=model_answer, ground_truth=ground_truth)

        return call_llm(eval_prompt)

    def post(self, shared, prep_res, exec_res):
        # 将评估结果存入shared
        shared['final_result'] = exec_res


def create_Distill_Flow():
    re = ReNode()
    pi = PINode()
    answer = AnswerNode()
    distill = DistillNode()
    eval = EvaluationNode()

    re - "answer" >> answer
    re - "calculate" >> pi
    pi - "feedback" >> re
    answer >> distill >> eval

    return Flow(start=re)


if __name__ == '__main__':
    # 1. 初始化
    load_dotenv()
    model_name = os.getenv("MODEL_NAME")
    test_flow = create_Distill_Flow()  # 假设你使用了包含EvaluationNode的完整流程
    filename = os.getenv("FILE_NAME")
    input_filepath = f'../../data/{filename}.json'
    output_filename = f'../../output_data/{filename}_对比结果_{model_name}.csv'

    # 2. 断点续传逻辑
    processed_ids = set()
    output_file_exists = os.path.exists(output_filename)
    if output_file_exists:
        print(f"检测到输出文件 '{output_filename}'，正在恢复进度...")
        try:
            with open(output_filename, 'r', encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                header = next(reader, None)  # 安全地读取表头
                if header:  # 确保文件不是空的
                    for row in reader:
                        if row:  # 避免空行导致 IndexError
                            processed_ids.add(row[0])  # 添加的已经是字符串ID
            print(f"成功恢复！已处理 {len(processed_ids)} 个条目。")
        except Exception as e:
            print(f"读取现有CSV文件时出错: {e}，将从头开始。")
            processed_ids = set()  # 出错时重置
            output_file_exists = False  # 视作出错的文件不存在，以便重新写入表头

    # 3. 加载数据集
    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            datasets = json.load(f)
    except Exception as e:
        print(f"加载数据集 '{input_filepath}' 失败: {e}")
        exit()

    # 4. 高效的文件写入
    with open(output_filename, 'a', encoding='utf-8-sig', newline='') as f_output:
        writer = csv.writer(f_output)

        # 如果是新文件或恢复进度失败，写入表头
        if not output_file_exists:
            writer.writerow(['id', 'problem', 'answer', 'truth', 'final'])

        # 5. 主循环
        for i, data in enumerate(datasets):
            # --- 核心修复点 ---
            # 统一将ID处理为字符串类型，以匹配从CSV中读取的ID类型
            current_id = str(data.get("id", f"no-id-{i}"))

            if current_id in processed_ids:
                print(f"ID: {current_id} 已处理，跳过。")
                continue

            print(f"正在处理第 {i + 1}/{len(datasets)} 项, ID: {current_id}...")

            try:
                shared = {
                    "question": data.get("question（纯文本）", "N/A"),
                    "truth": data.get('ground_truth', "N/A")
                }

                test_flow.run(shared)

                result_row = [
                    current_id,
                    shared.get("question", "N/A"),
                    shared.get("answer", "ERROR"),
                    shared.get("truth", "N/A"),
                    shared.get("final_result", "EVAL_ERROR")
                ]
                writer.writerow(result_row)
                f_output.flush()  # 实时写入磁盘

            except Exception as e:
                print(f"处理 ID: {current_id} 时发生严重错误: {e}")
                writer.writerow([
                    current_id,
                    data.get("question（纯文本）", "N/A"),
                    'FATAL_ERROR',
                    data.get('ground_truth', 'N/A'),
                    str(e)
                ])
                f_output.flush()

    print(f"处理完成！结果已保存至 '{output_filename}'。")
