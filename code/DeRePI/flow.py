from code.test.RePI_async.node_async import MAX_RETRY
from node import ReNode, PINode, AnswerNode, DecomposerNode, StepManagerNode

import pocketflow as pf
import os

MAX_RETRY = int(os.getenv('MAX_RETRY', 3))

decomposer = DecomposerNode()
step_manager = StepManagerNode()
re_node = ReNode()
pi_node = PINode()
answer_node = AnswerNode()

# 创建一个Flow实例
agent_flow = pf.Flow()

# 定义工作流的起始节点
agent_flow.start(decomposer)

# 定义节点之间的有向连接和条件分支
# 1. 分解器完成后，如果成功，则启动步骤管理器
decomposer - "execute_plan" >> step_manager

# 2. 步骤管理器决定是处理下一步，还是结束循环
step_manager - "process_step" >> re_node  # 如果有下一个步骤，则交给推理节点
step_manager - "end_loop" >> answer_node  # 如果所有步骤完成，则去生成最终答案

# 3. 推理-计算子循环
re_node - "calculate" >> pi_node  # 如果推理结果是计算，则调用Python解释器
re_node - "sub_task_complete" >> step_manager  # 如果子任务完成，则返回步骤管理器获取下一步
pi_node - "feedback" >> re_node  # Python代码执行后，将结果反馈给推理节点

# `decomposer` 在没有步骤时返回 "end"，`answer_node` 返回 `None`。
# 在这两种情况下，由于没有定义后续节点，工作流将自动终止。

print("✅ [PocketFlow] 工作流构建完成！")

# -------------------------------------------------------------------------
# 4. 执行工作流
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # 初始化共享状态（Shared State）
    # 这是工作流中所有节点都可以访问和修改的中央数据存储。
    shared_state = {
        "question": "若一个等比数列的前 4 项和为 4 ，前 8 项和为 68 ，则该等比数列的公比为",
        # 其他状态将由节点在运行时动态添加
    }

    print("\n================== 工作流开始 ==================")
    print(f"初始问题: {shared_state['question']}")

    # 运行工作流
    # run方法会从起始节点开始，根据每个节点的返回值和我们定义的连接，自动地在节点间流转。
    agent_flow.run(shared_state)
