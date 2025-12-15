import pocketflow as pf
import os
import json
import csv
import sys
import time
import threading  # 导入threading模块
import concurrent.futures  # 导入并发库
from dotenv import load_dotenv
# 添加工作流
from Work_Flow import get_nodes, select_flow

# ==============================================================================
# 0. 路径和环境设置 (与原版相同)
# ==============================================================================
# 通过计算脚本的绝对路径来确保无论从哪里执行，都能正确找到项目根目录
try:
    # __file__ 在正常 Python 执行时可用
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # 在交互式环境（如Jupyter）中，__file__ 不存在，使用当前工作目录
    SCRIPT_DIR = os.getcwd()

PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

# 将项目根目录添加到系统路径，以便能成功导入 utils 模块
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# 从项目根目录加载 .env 文件
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))


# ==============================================================================
# 1. 配置加载 (新增并发相关配置)
# ==============================================================================
MODEL_NAME = os.getenv("MODEL_NAME", "default_model")
MAX_RETRY = int(os.getenv("MAX_RETRY", "3"))
FILENAME = os.getenv("FILE_NAME")
# 新增：从环境变量配置最大并发工作线程数，默认为4
MAX_WORKERS = int(os.getenv("CONCURRENCY_LIMIT", "4"))


# 动态导入相应的节点进行处理
# ==============================================================================
# 2. 节点与流程定义 (与原版基本相同)
# ==============================================================================


def process_item_and_write_csv(data_item, f_csv, csv_writer, csv_lock,module_type):
    """
    处理单个数据项，并线程安全地将其结果立即写入并刷新到CSV文件。

    Args:
        data_item (dict): 数据集中的一个元素。
        f_csv (file object): 文件句柄，用于调用 flush()。
        csv_writer: csv.writer 对象。
        csv_lock (threading.Lock): 用于保护CSV写入操作的锁。
    """
    thread_id = threading.get_ident()
    current_id = str(data_item.get("id", f"no-id-{int(time.time())}"))

    print(f"[{thread_id}] ⚙️  开始处理 ID: {current_id}...")

    node_container = get_nodes(module_type)
    # 选择工作流
    test_pipeline = select_flow(module_type,node_container)


    # test_pipeline = create_full_test_pipeline()
    shared = {
        "id": current_id,
        "question": data_item.get("question（纯文本）", data_item.get("question", "N/A")),
        "truth": str(data_item.get('ground_truth', "N/A")),
        "img_url": data_item.get("img_url")
    }
    start_time = time.time()  # 🔍 记录开始时间

    try:
        test_pipeline.run(shared)
        result_row = [
            current_id,
            shared.get("question"),
            shared.get("answer", "No_answer"),
            shared.get("distilled_answer", "DISTILL_ERROR"),
            shared.get("truth"), shared.get("final_result", "EVAL_ERROR"),
            shared.get("img_url", "NO_img")
        ]
        print(f"[{thread_id}] ✅ ID: {current_id} 处理成功。")
    except Exception as e:
        print(f"[{thread_id}] ❌ ID: {current_id} 发生严重错误: {e}")
        result_row = [
            current_id, shared.get("question"), 'FATAL_ERROR',
            shared.get("truth"), str(e)
        ]
    end_time = time.time()
    duration = end_time - start_time
    # 添加处理时间到结果行
    result_row.append(f"{duration:.4f}")  # 保留4位小数

    # --- 线程安全地、立即写入并刷新到磁盘 ---
    # 这是实现中断安全的关键
    with csv_lock:
        csv_writer.writerow(result_row)
        f_csv.flush()  # <--- THE FIX: 强制将缓冲区写入磁盘

def main():
    start_time = time.time()
    print("🚀 开始自动化并发测试流程...")

    # --- 文件路径设置 ---
    output_dir = os.path.join(PROJECT_ROOT, 'output_data')
    module_type = os.getenv("MODULE_TYPE", "default_model")

    os.makedirs(output_dir, exist_ok=True)
    base_output_filename = os.path.join(output_dir, f'{FILENAME}_{module_type}_{MODEL_NAME}_对比结果')
    output_csv_filename = f'{base_output_filename}.csv'

    print(f"📂 数据集路径: {os.path.join(PROJECT_ROOT, 'data', f'{FILENAME}.json')}")
    print(f"📄 CSV输出路径: {output_csv_filename}")
    print(f"🚀 模型: {MODEL_NAME}, 并发数: {MAX_WORKERS},框架:{module_type}" )

    # --- 断点续传逻辑 (不变) ---
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

    # --- 加载并过滤数据集 (不变) ---
    try:
        with open(os.path.join(PROJECT_ROOT, 'data', f'{FILENAME}.json'), 'r', encoding='utf-8') as f:
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

    # --- 并发执行与写入 (修复) ---
    with open(output_csv_filename, 'a', encoding='utf-8-sig', newline='') as f_csv:
        writer = csv.writer(f_csv)

        f_csv.seek(0, os.SEEK_END)
        if f_csv.tell() == 0:
            writer.writerow(['id', 'problem', 'answer','distilled_answer', 'truth', 'final_result','processing_time'])
            f_csv.flush()   # 写入表头后也最好刷新一下

        csv_writer_lock = threading.Lock()

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 修改点：将 f_csv 文件句柄也传递给工作函数
            futures = [
                executor.submit(process_item_and_write_csv, data, f_csv, writer, csv_writer_lock, module_type)
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


