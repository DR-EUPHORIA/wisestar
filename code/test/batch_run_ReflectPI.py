import pocketflow as pf
import openai
import os
import json
import csv
import re
import sys
from dotenv import load_dotenv
from code.ReflectPI.node import ReNode, PINode, ReflectNode, AnswerNode
from utils.llm import call_llm_stream
from utils.pyinterpreter import PythonInterpreter
from utils.prompt_templates import REFLECTPI_RENODE_PROMPT, REFLECTPI_REFLECTNODE_PROMPT, REPI_EVALUATION_NODE_PROMPT, \
    REPI_DISTILL_NODE_PROMPT

# ==============================================================================
# 0. 路径和环境设置 (关键修改)
# ==============================================================================
# 通过计算脚本的绝对路径来确保无论从哪里执行，都能正确找到项目根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

# 将项目根目录添加到系统路径，以便能成功导入 utils 模块
sys.path.append(PROJECT_ROOT)

# 从项目根目录加载 .env 文件
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

MODEL_NAME = os.getenv("MODEL_NAME")
MAX_RETRY = int(os.getenv("MAX_RETRY", "3"))


class DistillNode(pf.Node):
    def prep(self, shared):
        """准备从shared中提取原始答案"""
        print("💧 [DistillNode] 预处理，提取原始答案...")
        answer = shared.get('answer', '未找到答案')
        # 保存原始答案以供调试
        shared['answer'] = answer
        return answer

    def exec(self, prep_res):
        """调用LLM进行答案提纯"""
        print("💧 [DistillNode] 执行提纯...")
        prompt = REPI_DISTILL_NODE_PROMPT.format(prep_res=prep_res)
        distilled_answer = call_llm_stream(prompt)
        return distilled_answer

    def post(self, shared, prep_res, exec_res):
        print(f"💧 [DistillNode] 后处理，更新答案为: '{exec_res}'")
        shared['distilled_answer'] = exec_res


class EvaluationNode(pf.Node):
    def prep(self, shared):
        return {
            "model_answer": shared.get("distilled_answer", "NO_ANSWER_FOUND"),
            "ground_truth": shared.get("truth", "NO_TRUTH_PROVIDED"),
            "question": shared.get("question", "NO_QUESTION_FOUND")
        }

    def exec(self, prep_res):
        # 使用导入的提示词
        eval_prompt = REPI_EVALUATION_NODE_PROMPT.format(
            model_answer=prep_res["model_answer"],
            ground_truth=prep_res["ground_truth"],
            question=prep_res["question"]
        )
        response = call_llm_stream(eval_prompt)
        if '不一致' in response:
            return '不一致'
        else:
            return '一致'

        return f'EVAL_FORMAT_ERROR: {response}'

    def post(self, shared, prep_res, exec_res):
        shared['final_result'] = exec_res


def create_full_test_pipeline():
    """
    创建从解题到评估的完整自动化测试流程。
    """
    # 实例化所有节点
    re = ReNode()
    pi = PINode()
    reflect = ReflectNode()
    answer = AnswerNode()
    distill = DistillNode()
    evaluation = EvaluationNode()

    # 定义 ReflectPI Agent 内部的循环逻辑
    re - "calculate" >> pi
    re - "reflect" >> reflect
    re - "answer" >> answer
    pi - "feedback" >> re
    reflect - "feedback" >> re
    reflect - "answer" >> answer

    # 将 Agent 的出口连接到后续处理节点，形成一个完整的流水线
    answer >> distill >> evaluation

    # 返回以 ReNode 为起点的完整 Flow
    return pf.Flow(start=re)


# ==============================================================================
# 4. 批处理主程序 (循环逻辑已简化)
# ==============================================================================

if __name__ == '__main__':
    print("🚀 开始自动化测试流程...")

    test_pipeline = create_full_test_pipeline()
    filename = os.getenv("FILE_NAME", "default_dataset")
    model_name = os.getenv("MODEL_NAME", "default_dataset")

    output_dir = os.path.join(PROJECT_ROOT, 'output_data')
    os.makedirs(output_dir, exist_ok=True)
    base_output_filename = os.path.join(output_dir, f'{filename}_ReflectPI_{MODEL_NAME}_对比结果')
    output_csv_filename = f'{base_output_filename}.csv'
    output_json_filename = f'{base_output_filename}.json'

    print(f"📂 数据集路径: {os.path.join(PROJECT_ROOT, 'data', f'{filename}.json')}")
    print(f"📄 CSV输出路径: {output_csv_filename}")
    print(f"📄 JSON输出路径: {output_json_filename}")
    print(f"🚀 模型使用:{model_name}")
    processed_ids = set()
    if os.path.exists(output_csv_filename):
        print(f"检测到输出文件 '{os.path.basename(output_csv_filename)}'，恢复进度...")
        try:
            with open(output_csv_filename, 'r', encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if row: processed_ids.add(row[0])
            print(f"✅ 成功恢复！已处理 {len(processed_ids)} 个条目。")
        except Exception as e:
            print(f"⚠️ 读取CSV文件时出错: {e}")
            processed_ids = set()

    try:
        with open(os.path.join(PROJECT_ROOT, 'data', f'{filename}.json'), 'r', encoding='utf-8') as f:
            datasets = json.load(f)
        print(f"📚 数据集加载成功，共 {len(datasets)} 条。")
    except Exception as e:
        print(f"❌ 加载数据集失败: {e}")
        exit()

    json_results = []
    with open(output_csv_filename, 'a', encoding='utf-8-sig', newline='') as f_csv:
        writer = csv.writer(f_csv)

        # --- CSV列名修改点 ---
        if not processed_ids:
            writer.writerow(['id', 'problem', 'answer', 'truth', 'final'])

        for i, data in enumerate(datasets):
            current_id = str(data.get("id", f"no-id-{i}"))
            if current_id in processed_ids:
                print(f"⏭️  ID: {current_id} 已处理，跳过。")
                continue

            print("-" * 60)
            print(f"⚙️  正在处理第 {i + 1}/{len(datasets)} 项, ID: {current_id}...")

            try:
                shared = {
                    "question": data.get("question（纯文本）", data.get("question", "N/A")),
                    "truth": str(data.get('ground_truth', "N/A"))
                }

                test_pipeline.run(shared)

                # --- CSV行数据修改点 ---
                result_row = [
                    current_id,
                    shared.get("question", "N/A"),
                    shared.get("answer", "DISTILL_ERROR"),  # 'answer' 列使用提纯后的答案
                    shared.get("truth", "N/A"),  # 'truth' 列
                    shared.get("final_result", "EVAL_ERROR")  # 'final' 列
                ]
                writer.writerow(result_row)
                f_csv.flush()

                json_results.append(shared.copy())

                print(f"🏁 ID: {current_id} 完整流程结束。最终结果: {shared.get('final_result', 'EVAL_ERROR')}")

            except Exception as e:
                print(f"❌ 处理 ID: {current_id} 时发生严重错误: {e}")
                # --- 错误行格式修改点 ---
                writer.writerow([
                    current_id,
                    data.get("question（纯文本）", "N/A"),
                    'FATAL_ERROR',
                    data.get('ground_truth', 'N/A'),
                    str(e)
                ])
                f_csv.flush()

    if json_results:
        print("-" * 60)
        print(f"💾 正在将 {len(json_results)} 条详细结果写入JSON文件...")
        with open(output_json_filename, 'w', encoding='utf-8') as f_json:
            json.dump(json_results, f_json, ensure_ascii=False, indent=2)
        print(f"💾 JSON文件 '{output_json_filename}' 保存成功。")

    print("-" * 60)
    print(f"🎉 全部处理完成！结果已保存至 '{output_csv_filename}' 和 '{output_json_filename}'。")
