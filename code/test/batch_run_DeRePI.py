import pocketflow as pf
import openai  # 遵循规范，导入openai
import os
import json
import csv
import sys
import time
import threading
import concurrent.futures
from dotenv import load_dotenv

# ==============================================================================
# 0. 路径和环境设置
# 确保无论从哪里执行，都能正确找到项目根目录和模块
# ==============================================================================
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

# 假设此脚本位于项目根目录下的 'scripts' 文件夹中
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../..'))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# 从项目根目录加载 .env 文件
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

# ==============================================================================
# 1. 配置加载与OpenAI客户端初始化
# ==============================================================================
MODEL_NAME = os.getenv("MODEL_NAME", "default_model")
MAX_RETRY = int(os.getenv("MAX_RETRY", "3"))
FILENAME = os.getenv("FILE_NAME", "math_test")  # 默认数据集文件名
MAX_WORKERS = int(os.getenv("CONCURRENCY_LIMIT", "4"))  # 从环境变量配置并发数

# ==============================================================================
# 2. 节点与流程定义
# 导入你已有的节点，并添加评估节点
# ==============================================================================
# 从你的项目中导入节点定义
from code.DeRePI.node import DecomposerNode, ReNode, PINode, AnswerNode
from utils.llm import call_llm_stream  # 假设这个函数存在且能用
from utils.prompt_templates import REPI_EVALUATION_NODE_PROMPT  # 需要评估prompt


class EvaluationNode(pf.Node):
    """
    评估节点，用于比较模型输出和标准答案。
    """

    def prep(self, shared):
        # 从shared中获取问题、模型答案和真实答案
        return {
            "model_answer": shared.get("answer", "NO_ANSWER_FOUND"),  # DeRePI的最终答案在'answer'字段
            "ground_truth": shared.get("truth", "NO_TRUTH_PROVIDED"),
            "question": shared.get("question", "NO_QUESTION_FOUND")
        }

    def exec(self, prep_res):
        # 使用大模型进行评估打分
        eval_prompt = REPI_EVALUATION_NODE_PROMPT.format(**prep_res)
        # 假设call_llm_stream内部处理了API调用逻辑
        response = call_llm_stream(eval_prompt)  # 将client传递进去
        return response

    def post(self, shared, prep_res, exec_res):
        # 将最终评估结果存入shared
        shared['final_result'] = exec_res


def create_derepi_test_pipeline():
    """
    工厂函数：创建从分解、执行到最终评估的完整DeRePI测试流程。
    这确保了每个并发线程都使用自己独立的流程实例，避免状态冲突。
    """
    # 1. 实例化所有需要的节点
    decomposer_node = DecomposerNode()
    re_node = ReNode(max_retries=MAX_RETRY)
    pi_node = PINode()
    answer_node = AnswerNode()
    evaluation_node = EvaluationNode()  # 新增评估节点

    # 2. 定义子流程：ReNode 和 PINode 之间的推理-计算循环
    re_node.next(pi_node, action="calculate")
    pi_node.next(re_node, action="feedback")

    # ReNode在子任务完成后会返回 "sub_task_complete"，
    # 这将结束当前子任务在BatchFlow中的执行。

    # 3. 创建批处理流程 (BatchFlow) 来执行所有子任务
    task_executor_flow = pf.BatchFlow()

    # 绑定数据准备方法到BatchFlow实例
    def task_executor_prep(self, shared):
        steps = shared.get('steps', [])
        # 当分解出0个步骤时，steps可能是['end']，我们需要过滤掉
        valid_steps = [s for s in steps if s.lower() != 'end']
        print(f"🔄 [BatchFlow] 准备执行 {len(valid_steps)} 个子任务...")
        return [{'task': step} for step in valid_steps]

    task_executor_flow.prep = task_executor_prep.__get__(task_executor_flow, pf.BatchFlow)

    # 设置批处理流程的起点
    task_executor_flow.start(re_node)

    # 4. 构建主流程 (Main Flow)
    main_flow = pf.Flow()
    main_flow.start(decomposer_node)

    # 5. 连接主流程的各个阶段
    # 分解器完成后，如果需要执行计划，则启动批处理流程
    decomposer_node.next(task_executor_flow, action="execute_plan")

    # 如果分解器直接结束(例如问题太简单)，则直接跳到回答节点
    # 注意：你的DecomposerNode实现中，如果没有步骤会返回'end'，但没有定义end的流向
    # 我们这里将其导向AnswerNode，让它尝试基于现有信息回答
    decomposer_node.next(answer_node, action="end")

    # 批处理流程完成后，流向回答节点进行答案整合
    task_executor_flow.next(answer_node)

    # 回答节点完成后，流向评估节点进行最终评估
    answer_node.next(evaluation_node)

    return main_flow


def process_item_and_write_csv(data_item, f_csv, csv_writer, csv_lock):
    """
    处理单个数据项，并线程安全地将其结果立即写入并刷新到CSV文件。
    这是并发处理的核心工作函数。
    """
    thread_id = threading.get_ident()
    current_id = str(data_item.get("id", f"no-id-{int(time.time())}"))

    print(f"[{thread_id}] ⚙️  开始处理 ID: {current_id}...")

    # 为当前任务创建一个全新的、独立的流程实例
    test_pipeline = create_derepi_test_pipeline()

    # 初始化当前任务的共享状态
    shared = {
        "id": current_id,
        "question": data_item.get("question（纯文本）", data_item.get("question", "N/A")),
        "truth": str(data_item.get('ground_truth', "N/A")),
        'responses': [],
        'actions': [],
        'codes': [],
        'calculation_results': [],
        'node_call_counts': {},
        'answer': ''
    }

    try:
        # 运行完整的测试流程
        test_pipeline.run(shared)
        result_row = [
            current_id,
            shared.get("question"),
            shared.get("answer", "ANSWER_ERROR"),  # 最终答案
            shared.get("truth"),
            shared.get("final_result", "EVAL_ERROR")  # 评估结果
        ]
        print(f"[{thread_id}] ✅ ID: {current_id} 处理成功。")
    except Exception as e:
        print(f"[{thread_id}] ❌ ID: {current_id} 发生严重错误: {e}")
        result_row = [
            current_id,
            shared.get("question"),
            'FATAL_ERROR',
            shared.get("truth"),
            str(e)
        ]

    # --- 线程安全地、立即写入并刷新到磁盘（实现中断安全的关键） ---
    with csv_lock:
        csv_writer.writerow(result_row)
        f_csv.flush()  # 强制将缓冲区内容写入磁盘


# ==============================================================================
# 5. 并发批处理主程序
# ==============================================================================
if __name__ == '__main__':
    start_time = time.time()
    print("🚀 开始DeRePI智能体自动化并发测试流程...")

    # --- 文件路径设置 ---
    output_dir = os.path.join(PROJECT_ROOT, 'output_data')
    os.makedirs(output_dir, exist_ok=True)
    output_csv_filename = os.path.join(output_dir, f'{FILENAME}_DeRePI_{MODEL_NAME}_对比结果.csv')

    print(f"📂 数据集路径: {os.path.join(PROJECT_ROOT, 'data', f'{FILENAME}.json')}")
    print(f"📄 CSV输出路径: {output_csv_filename}")
    print(f"🤖 模型: {MODEL_NAME}, ⚙️ 最大并发数: {MAX_WORKERS}")

    # --- 断点续传逻辑 ---
    processed_ids = set()
    if os.path.exists(output_csv_filename):
        print("🔄 检测到输出文件，正在恢复进度...")
        try:
            with open(output_csv_filename, 'r', encoding='utf-8-sig') as f_read:
                reader = csv.reader(f_read)
                header = next(reader, None)
                if header:
                    for row in reader:
                        if row and row[0]:
                            processed_ids.add(row[0])
            print(f"✅ 成功恢复！已处理 {len(processed_ids)} 个条目。")
        except Exception as e:
            print(f"⚠️ 读取CSV文件恢复进度时出错: {e}。将重新开始。")
            processed_ids = set()

    # --- 加载并过滤数据集 ---
    try:
        with open(os.path.join(PROJECT_ROOT, 'data', f'{FILENAME}.json'), 'r', encoding='utf-8') as f:
            all_datasets = json.load(f)
        tasks_to_run = [
            data for i, data in enumerate(all_datasets)
            if str(data.get("id", f"no-id-{i}")) not in processed_ids
        ]
        print(f"📚 数据集加载成功。共 {len(all_datasets)} 条，需处理 {len(tasks_to_run)} 条新任务。")
    except FileNotFoundError:
        print(f"❌ 错误：找不到数据集文件 at {os.path.join(PROJECT_ROOT, 'data', f'{FILENAME}.json')}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 加载数据集失败: {e}")
        sys.exit(1)

    if not tasks_to_run:
        print("🎉 无新任务需要处理。程序结束。")
        sys.exit(0)

    # --- 并发执行与写入 ---
    with open(output_csv_filename, 'a', encoding='utf-8-sig', newline='') as f_csv:
        writer = csv.writer(f_csv)

        # 如果文件是新建的，写入表头
        if not processed_ids:
            writer.writerow(['id', 'problem', 'model_answer', 'truth', 'final_result'])
            f_csv.flush()

        # 创建一个锁来保护对CSV文件的写入操作
        csv_writer_lock = threading.Lock()

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 提交所有待处理任务到线程池
            futures = [
                executor.submit(process_item_and_write_csv, data, f_csv, writer, csv_writer_lock)
                for data in tasks_to_run
            ]

            # 等待所有任务完成
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()  # 获取结果，如果任务中发生异常，这里会重新抛出
                except Exception as exc:
                    print(f"💥 [主线程] 一个工作线程奔溃: {exc}")

    end_time = time.time()
    print("-" * 60)
    print(f"🎉 全部处理完成！总耗时: {end_time - start_time:.2f} 秒。")
    print(f"📄 结果已全部保存至: '{output_csv_filename}'")
