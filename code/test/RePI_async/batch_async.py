from pocketflow import Node, Flow, AsyncNode, AsyncParallelBatchFlow
from node_async import ReNode,PINode,AnswerNode
from utils.llm import call_llm_stream_async as call_llm_async
from utils.prompt_templates import REPI_DISTILL_NODE_PROMPT
from dotenv import load_dotenv
import asyncio
import json
import csv
import pandas as pd
import os

CONCURRENCY_LIMIT = os.getenv('CONCURRENCY_LIMIT')
CONCURRENCY_LIMIT = int(CONCURRENCY_LIMIT)

TIMEOUT = os.getenv('TIMEOUT')
TIMEOUT=int(TIMEOUT)

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

class DistillNode(AsyncNode):
    async def prep_async(self, shared):
        solve = shared.get('answer','none')
        return solve

    async def exec_async(self, prep_res):
        prompt = REPI_DISTILL_NODE_PROMPT.format(prep_res=prep_res)
        response = await call_llm_async(prompt)
        return response

    async def post_async(self, shared, prep_res, exec_res):
        shared['answer'] = exec_res

'''
EvaluationNode---比对节点
需要使用的数据: answer, truth
功能: 对比模型输出的答案与标准答案的一致性
'''

class EvaluationNode(AsyncNode):
    async def prep_async(self, shared):
        # 准备评估需要的所有数据
        return {
            "model_answer": shared.get("answer", "none"),
            "ground_truth": self.params.get("truth", "none")
        }

    async def exec_async(self, prep_res):
        model_answer = prep_res["model_answer"]
        ground_truth = prep_res["ground_truth"]

        # 如果任何一个答案无效，直接返回
        if model_answer == "none" or ground_truth == "none":
            return "Skipped - Missing Answer"

        eval_prompt = f'''
###任务
检测两个答案是否一致

###待检测的答案
答案1: {model_answer}
答案2: {ground_truth}

###案例
-输入： sqrt(2)、根号2
-输出： 一致

-输入：（1）3 （2）不相等 、 第一小问：3，第二小问：想等
-输出：（1）一致，(2)不一致

-输入：C.线段AD的长度为12  、  C
-输出：一致

###注意事项
你的输出只能为“一致”或“不一致”（可以包含小问的一致、不一致），不包含多余的解释
'''

        return await call_llm_async(eval_prompt)

    async def post_async(self, shared, prep_res, exec_res):
        # 将评估结果存入shared
        shared['final_result'] = exec_res

'''
SaveResultNode---保存节点
功能: 保存异步批处理流中单个任务的结果
'''

class SaveResultNode(Node):
    """
    这个节点是同步的，负责将单个任务的结果安全地写入CSV文件。
    它在每个并行流程的最后被调用。
    """

    def prep(self, shared):
        # 从 shared 中准备好要写入一行的数据
        # 通过 self.params 获取由 Flow 传递过来的全局参数
        return {
            "output_filename": self.params.get("output_filename"),
            "row_data": [
                self.params.get("id", "N/A"),
                self.params.get("question", "N/A"),
                shared.get("answer", "ERROR"),
                self.params.get("truth", "N/A"),
                shared.get("final_result", "EVAL_ERROR")
            ]
        }

    def exec(self, prep_res):
        filename = prep_res["output_filename"]
        row = prep_res["row_data"]

        # 使用'a'模式追加写入
        with open(filename, 'a', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

        # 在控制台打印保存确认信息，便于追踪进度
        print(f"✅ [SaveResultNode] ID: {row[0]} - 结果已保存。")
        return "saved"


class TestAutomationFlow(AsyncParallelBatchFlow):
    """
    一个健壮的异步并行批处理流程，用于自动化测试Agent。
    核心特性：
    1.  **状态隔离**: 每个并行任务都有其独立的`shared`状态，防止数据交叉污染。
    2.  **并发控制**: 使用Semaphore限制同时运行的任务数量，避免API过载。
    3.  **断点续传**: 自动跳过输出文件中已存在的ID，可以随时中断和恢复。
    """

    def __init__(self, start_node, input_filepath, output_filename, concurrency_limit=CONCURRENCY_LIMIT):
        super().__init__(start=start_node)
        self.input_filepath = input_filepath
        self.output_filename = output_filename
        # 设置全局参数，供所有子任务中的节点（如SaveResultNode）访问
        self.set_params({"output_filename": self.output_filename})
        # 创建一个Semaphore实例用于并发控制
        self.semaphore = asyncio.Semaphore(concurrency_limit)
        print(f"🚦 并发控制器已启动，限制为 {concurrency_limit} 个并行任务。")

    async def prep_async(self, shared):
        """
        在流程开始前运行，负责加载所有任务并根据输出文件过滤掉已完成的任务。
        """
        print("🚀 [Flow Prep] 开始加载数据并过滤已处理项...")
        processed_ids = set()
        if os.path.exists(self.output_filename):
            try:
                # 使用pandas读取更健壮，能处理空文件等情况
                df = pd.read_csv(self.output_filename, dtype={'id': str})
                if 'id' in df.columns:
                    processed_ids = set(df['id'].dropna())
                print(f"🔍 已找到 {len(processed_ids)} 个已处理的ID。")
            except pd.errors.EmptyDataError:
                print(f"📋 输出文件 '{self.output_filename}' 为空，将处理所有任务。")
            except Exception as e:
                print(f"⚠️ 读取现有CSV文件时出错: {e}，将处理所有任务。")

        try:
            with open(self.input_filepath, 'r', encoding='utf-8') as f:
                datasets = json.load(f)
        except Exception as e:
            print(f"❌ 加载数据集 '{self.input_filepath}' 失败: {e}")
            return []  # 返回空列表以停止流程

        # 准备要并行处理的任务参数列表
        tasks_to_run = []
        for data in datasets:
            current_id = str(data.get("id"))
            if current_id not in processed_ids:
                tasks_to_run.append({
                    "id": current_id,
                    "question": data.get("question（纯文本）", "N/A"),
                    "truth": data.get('ground_truth', "N/A")
                })

        print(f"⚡️ 准备了 {len(tasks_to_run)} 个新任务进行并行处理。")
        return tasks_to_run  # 返回任务参数列表

    async def _orch_async(self, shared, params=None):
        """
        重写的核心编排器，接收一个独立的shared字典和任务参数。
        """
        # 将任务特有的参数（如id, question）注入到这个任务私有的shared状态中
        if params:
            shared.update(params)

        curr = self.start_node
        last_action = None

        # flow的全局参数（如output_filename）通过p传递
        p = self.params.copy()
        if params:
            p.update(params)

        while curr:
            curr.set_params(p)
            if isinstance(curr, AsyncNode):
                last_action = await curr._run_async(shared)
            else:
                last_action = curr._run(shared)
            curr = self.get_next_node(curr, last_action)
        return last_action

    async def _run_task_with_semaphore(self, task_params):
        """
        一个包装器，为每个任务创建独立的shared字典，并使用semaphore进行并发控制。
        """
        async with self.semaphore:
            # 关键：为每个任务创建一个全新的、干净的 shared 字典
            # task_params 包含了 id, question, truth 等信息
            return await self._orch_async({}, {**self.params, **task_params})

    async def _run_async(self, shared):
        """
        流程的入口。它获取任务列表，并为每个任务启动一个带状态隔离和并发控制的流程。
        """
        tasks_params_list = await self.prep_async(shared) or []

        # 使用 asyncio.gather 并发运行所有包装后的任务
        await asyncio.gather(*(self._run_task_with_semaphore(params) for params in tasks_params_list))

        return await self.post_async(shared, tasks_params_list, None)


def create_async_test_flow(input_path, output_path):
    """
    工厂函数：创建并配置自动测试的完整工作流。
    """
    # 1. 实例化所有需要的节点
    re_node = ReNode()
    # 在这里为PINode设置超时时间
    pi_node = PINode(timeout_seconds=TIMEOUT)
    distill_node = DistillNode()
    evaluation_node = EvaluationNode()
    saver_node = SaveResultNode()

    # 2. 编排节点流程 (保持不变)
    re_node - "calculate" >> pi_node
    pi_node - "feedback" >> re_node
    re_node - "answer" >> distill_node
    distill_node >> evaluation_node >> saver_node

    # 3. 创建并返回配置好的 Flow 实例
    return TestAutomationFlow(
        start_node=re_node,
        input_filepath=input_path,
        output_filename=output_path,
        concurrency_limit=CONCURRENCY_LIMIT
    )

def sort_csv_by_id(file_path: str) -> None:
    """
    读取指定 CSV 文件，按 id 升序排序，同时过滤空行和字段不完整的行。

    :param file_path: CSV 文件路径
    """
    if not os.path.exists(file_path):
        print(f"⚠️ 文件 '{file_path}' 不存在，跳过排序。")
        return

    print(f"🔄 正在对 '{file_path}' 按 id 排序...")

    try:
        with open(file_path, 'r', encoding='utf-8-sig', newline='') as f:
            reader = csv.reader(f)
            header = next(reader)  # 读取表头
            rows = list(reader)    # 读取所有数据行

        # 过滤空行和字段不足的行
        valid_rows = []
        empty_rows_count = 0
        invalid_rows_count = 0

        for row in rows:
            if not row:  # 空行（如读取到空列表[]）
                empty_rows_count += 1
            elif len(row) < 5:  # 字段不足
                invalid_rows_count += 1
            else:
                valid_rows.append(row)

        # 输出过滤信息
        if empty_rows_count > 0:
            print(f"⚠️ 检测到并删除了 {empty_rows_count} 行空行。")
        if invalid_rows_count > 0:
            print(f"⚠️ 检测到并删除了 {invalid_rows_count} 行字段不完整的数据。")

        # 按 id 排序
        valid_rows.sort(key=lambda x: int(x[0]))

        # 覆盖写入原文件
        with open(file_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)  # 写入表头
            writer.writerows(valid_rows)  # 写入排序后的有效行

        print(f"✅ 已成功按 id 排序并保存至 '{file_path}'")

    except Exception as e:
        print(f"❌ 排序过程中发生错误：{e}")

# --- 步骤 4: 重构主程序以调用异步流程 ---

async def main():
    """主异步函数，负责设置和启动流程"""
    # 初始化文件名
    load_dotenv()
    filename = os.getenv("FILE_NAME")
    modelname = os.getenv(("MODEL_NAME"))
    input_filepath = f'../../../data/{filename}.json'
    output_filename = f'../../../output_data/{filename}_RePI_{modelname}_对比结果.csv'

    # 检查并写入CSV表头（如果文件不存在）
    # 这是一个一次性的设置操作，在所有并行任务开始前完成
    if not os.path.exists(output_filename):
        print(f"📋 输出文件 '{output_filename}' 不存在，正在创建并写入表头...")
        with open(output_filename, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'problem', 'answer', 'truth', 'final'])

    # 创建异步流程实例
    test_flow = create_async_test_flow(input_filepath, output_filename)

    print("\n--- 🚀 开始异步并行批处理 ---")
    # 使用 run_async 启动整个流程，初始shared为空
    # Flow的prep_async会负责加载所有数据
    await test_flow.run_async({})
    print("\n--- ✅ 异步并行批处理完成 ---")

    sort_csv_by_id(output_filename)

if __name__ == '__main__':
    # 使用 asyncio.run 来执行顶层的异步主函数
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 用户中断了程序。")