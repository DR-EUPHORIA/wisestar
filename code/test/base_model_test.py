import os
import re
import json
import csv
import time
import threading
import sys
import concurrent.futures
from dotenv import load_dotenv
from utils.llm import call_llm_stream  # 直接调用 LLM
from utils.prompt_templates import REPI_EVALUATION_NODE_PROMPT

# ==============================================================================
# 0. 路径和环境设置
# ==============================================================================
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

# ==============================================================================
# 1. 配置加载
# ==============================================================================
MODEL_NAME = os.getenv("MODEL_NAME", "default_model")
FILENAME = os.getenv("FILE_NAME")
MAX_WORKERS = int(os.getenv("CONCURRENCY_LIMIT", "4"))


# 修改后的评估函数
def Evaluation(model_answer, question, ground_truth):
    eval_prompt = REPI_EVALUATION_NODE_PROMPT

    final_response = call_llm_stream(eval_prompt)

    return final_response


def extract_answer(text):
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)

    return match.group(1).strip() if match else None


def process_item_and_write_csv(data_item, f_csv, csv_writer, csv_lock):
    """
    直接调用 LLM 处理单个数据项并写入 CSV
    """
    thread_id = threading.get_ident()
    current_id = str(data_item.get("id", f"no-id-{int(time.time())}"))

    print(f"[{thread_id}] ⚙️  开始处理 ID: {current_id}...")

    question = data_item.get("question", data_item.get("question", "N/A"))
    truth = str(data_item.get('answer', "N/A"))

    # 直接调用 LLM 生成答案
    try:
        prompt = f"""你是一名严谨的数学竞赛解题助手。
            -----------------------------
            【任务】
            给出题目完整解答过程，并在最后单独给出答案。
            【格式要求（极其重要）】
            1. 先输出推理/计算过程。
            2. 最后仅有一行写答案，且必须严格放在 <answer></answer> 标签中，标签里只放答案本身，不能含有额外空格或说明。
            示例：
            推理过程……
            <answer>42</answer>
            -----------------------------
            题目：
            {question}
            """

        model_answer = call_llm_stream(prompt)


    except Exception as e:
        model_answer = f"LLM_ERROR: {str(e)}"

    print(model_answer)
    final_answer = extract_answer(model_answer)
    print("🛫最终提取后的答案为：")
    print(final_answer)

    # final_result = Evaluation(model_answer, question, truth)

    result_row = [
        current_id,
        question,
        final_answer,
        truth
    ]

    # 线程安全写入 CSV
    with csv_lock:
        csv_writer.writerow(result_row)
        f_csv.flush()  # 立即写入磁盘

    print(f"[{thread_id}] ✅ ID: {current_id} 处理成功。")


def main():
    start_time = time.time()
    print("🚀 开始自动化并发测试流程...")

    # --- 文件路径设置 ---
    output_dir = os.path.join(PROJECT_ROOT, 'output_data', 'base_model_output')
    os.makedirs(output_dir, exist_ok=True)
    output_csv_filename = os.path.join(output_dir, f'{FILENAME}_{MODEL_NAME}_basemodel结果.csv')

    print(f"📂 数据集路径: {os.path.join(PROJECT_ROOT, 'data', 'AIME', f'{FILENAME}.json')}")
    print(f"📄 CSV输出路径: {output_csv_filename}")
    print(f"🚀 模型: {MODEL_NAME}, 并发数: {MAX_WORKERS}")

    # --- 断点续传逻辑 ---
    processed_ids = set()
    if os.path.exists(output_csv_filename):
        print("🔄 检测到输出文件，恢复进度...")
        try:
            with open(output_csv_filename, 'r', encoding='utf-8-sig') as f_read:
                reader = csv.reader(f_read)
                header = next(reader, None)
                if header:
                    for row in reader:
                        if row and row[0]: processed_ids.add(row[0])
            print(f"✅ 成功恢复！已处理 {len(processed_ids)} 个条目。")
        except Exception as e:
            print(f"⚠️ 读取CSV文件恢复进度时出错: {e}")
            processed_ids = set()

    # --- 加载并过滤数据集 ---
    try:
        with open(os.path.join(PROJECT_ROOT, 'data', 'AIME', f'{FILENAME}.json'), 'r', encoding='utf-8') as f:
            all_datasets = json.load(f)
        tasks_to_run = [
            data for i, data in enumerate(all_datasets)
            if str(data.get("id", f"no-id-{i}")) not in processed_ids
        ]
        print(f"📚 数据集加载成功。共 {len(all_datasets)} 条，需处理 {len(tasks_to_run)} 条新任务。")
    except Exception as e:
        print(f"❌ 加载数据集失败: {e}")
        sys.exit(1)

    if not tasks_to_run:
        print("🎉 无新任务需要处理。程序结束。")
        sys.exit(0)

    # --- 并发执行与写入 ---
    with open(output_csv_filename, 'a', encoding='utf-8-sig', newline='') as f_csv:
        writer = csv.writer(f_csv)

        f_csv.seek(0, os.SEEK_END)
        if f_csv.tell() == 0:
            writer.writerow(['id', 'problem', 'model_answer', 'truth'])
            f_csv.flush()

        csv_writer_lock = threading.Lock()

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(process_item_and_write_csv, data, f_csv, writer, csv_writer_lock)
                for data in tasks_to_run
            ]

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    print(f"💥 [Main Thread] 一个工作线程奔溃: {exc}")

    end_time = time.time()
    print("-" * 60)
    print(f"🎉 全部处理完成！总耗时: {end_time - start_time:.2f} 秒。")
    print(f"📄 结果已全部保存至: '{output_csv_filename}'")


if __name__ == "__main__":
    main()
