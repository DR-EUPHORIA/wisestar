import asyncio, warnings, copy, time


class BaseNode:
    """工作流节点的基类，定义了节点的基本结构和行为"""

    def __init__(self):
        # 初始化节点参数和后继节点字典
        self.params, self.successors = {}, {}

    def set_params(self, params):
        """设置节点参数"""
        self.params = params

    def next(self, node, action="default"):
        """设置后继节点，支持条件转移"""
        if action in self.successors:
            warnings.warn(f"Overwriting successor for action '{action}'")
        self.successors[action] = node
        return node

    def prep(self, shared):
        """准备阶段：在执行前进行数据准备，子类可重写"""
        pass

    def exec(self, prep_res):
        """执行阶段：核心业务逻辑，子类必须实现"""
        pass

    def post(self, shared, prep_res, exec_res):
        """后处理阶段：执行后的清理或结果处理，子类可重写"""
        pass

    def _exec(self, prep_res):
        """内部执行方法，可被子类重写以添加额外逻辑"""
        return self.exec(prep_res)

    def _run(self, shared):
        """内部运行方法：按prep->exec->post顺序执行"""
        p = self.prep(shared)
        e = self._exec(p)
        return self.post(shared, p, e)

    def run(self, shared):
        """公共运行接口，单节点执行时使用"""
        if self.successors:
            warnings.warn("Node won't run successors. Use Flow.")
        return self._run(shared)

    def __rshift__(self, other):
        """重载 >> 操作符，用于连接节点：node1 >> node2"""
        return self.next(other)

    def __sub__(self, action):
        """重载 - 操作符，用于条件转移：node1 - "condition" >> node2"""
        if isinstance(action, str):
            return _ConditionalTransition(self, action)
        raise TypeError("Action must be a string")


class _ConditionalTransition:
    """条件转移的辅助类，用于支持 node1 - "action" >> node2 语法"""

    def __init__(self, src, action):
        self.src, self.action = src, action

    def __rshift__(self, tgt):
        """完成条件转移的连接"""
        return self.src.next(tgt, self.action)


class Node(BaseNode):
    """标准节点类，支持重试机制"""

    def __init__(self, max_retries=1, wait=0):
        super().__init__()
        self.max_retries, self.wait = max_retries, wait

    def exec_fallback(self, prep_res, exc):
        """重试失败后的回退处理，默认抛出异常"""
        raise exc

    def _exec(self, prep_res):
        """带重试机制的执行方法"""
        for self.cur_retry in range(self.max_retries):
            try:
                return self.exec(prep_res)
            except Exception as e:
                if self.cur_retry == self.max_retries - 1:
                    return self.exec_fallback(prep_res, e)
                if self.wait > 0:
                    time.sleep(self.wait)


class BatchNode(Node):
    """批处理节点，对输入的每个项目执行相同的逻辑"""

    def _exec(self, items):
        """对每个输入项目执行父类的_exec方法"""
        return [super(BatchNode, self)._exec(i) for i in (items or [])]


class Flow(BaseNode):
    """工作流类，管理多个节点的执行顺序"""

    def __init__(self, start=None):
        super().__init__()
        self.start_node = start

    def start(self, start):
        """设置起始节点"""
        self.start_node = start
        return start

    def get_next_node(self, curr, action):
        """根据当前节点和动作获取下一个节点"""
        nxt = curr.successors.get(action or "default")
        if not nxt and curr.successors:
            warnings.warn(f"Flow ends: '{action}' not found in {list(curr.successors)}")
        return nxt

    def _orch(self, shared, params=None):
        """工作流编排：按顺序执行节点直到结束"""
        curr, p, last_action = copy.copy(self.start_node), (params or {**self.params}), None
        while curr:
            curr.set_params(p)
            last_action = curr._run(shared)
            curr = copy.copy(self.get_next_node(curr, last_action))
        return last_action

    def _run(self, shared):
        """工作流运行方法"""
        p = self.prep(shared)
        o = self._orch(shared)
        return self.post(shared, p, o)

    def post(self, shared, prep_res, exec_res):
        """默认返回执行结果"""
        return exec_res


class BatchFlow(Flow):
    """批处理工作流，对多个参数集合分别执行工作流"""

    def _run(self, shared):
        pr = self.prep(shared) or []
        # 为每个批处理参数执行工作流
        for bp in pr:
            self._orch(shared, {**self.params, **bp})
        return self.post(shared, pr, None)


class AsyncNode(Node):
    """异步节点类，支持异步执行"""

    async def prep_async(self, shared):
        """异步准备阶段"""
        pass

    async def exec_async(self, prep_res):
        """异步执行阶段，子类必须实现"""
        pass

    async def exec_fallback_async(self, prep_res, exc):
        """异步重试失败回退"""
        raise exc

    async def post_async(self, shared, prep_res, exec_res):
        """异步后处理阶段"""
        pass

    async def _exec(self, prep_res):
        """带重试机制的异步执行"""
        for i in range(self.max_retries):
            try:
                return await self.exec_async(prep_res)
            except Exception as e:
                if i == self.max_retries - 1:
                    return await self.exec_fallback_async(prep_res, e)
                if self.wait > 0:
                    await asyncio.sleep(self.wait)

    async def run_async(self, shared):
        """异步运行接口"""
        if self.successors:
            warnings.warn("Node won't run successors. Use AsyncFlow.")
        return await self._run_async(shared)

    async def _run_async(self, shared):
        """异步内部运行方法"""
        p = await self.prep_async(shared)
        e = await self._exec(p)
        return await self.post_async(shared, p, e)

    def _run(self, shared):
        """禁用同步运行"""
        raise RuntimeError("Use run_async.")


class AsyncBatchNode(AsyncNode, BatchNode):
    """异步批处理节点，顺序处理每个项目"""

    async def _exec(self, items):
        """顺序异步执行每个项目"""
        return [await super(AsyncBatchNode, self)._exec(i) for i in items]


class AsyncParallelBatchNode(AsyncNode, BatchNode):
    """异步并行批处理节点，并行处理所有项目"""

    async def _exec(self, items):
        """并行异步执行所有项目"""
        return await asyncio.gather(*(super(AsyncParallelBatchNode, self)._exec(i) for i in items))


class AsyncFlow(Flow, AsyncNode):
    """异步工作流类"""

    async def _orch_async(self, shared, params=None):
        """异步工作流编排"""
        curr, p, last_action = copy.copy(self.start_node), (params or {**self.params}), None
        while curr:
            curr.set_params(p)
            # 根据节点类型选择同步或异步执行
            last_action = await curr._run_async(shared) if isinstance(curr, AsyncNode) else curr._run(shared)
            curr = copy.copy(self.get_next_node(curr, last_action))
        return last_action

    async def _run_async(self, shared):
        """异步工作流运行"""
        p = await self.prep_async(shared)
        o = await self._orch_async(shared)
        return await self.post_async(shared, p, o)

    async def post_async(self, shared, prep_res, exec_res):
        """异步后处理"""
        return exec_res


class AsyncBatchFlow(AsyncFlow, BatchFlow):
    """异步批处理工作流，顺序执行多个工作流实例"""

    async def _run_async(self, shared):
        pr = await self.prep_async(shared) or []
        # 顺序执行每个批处理参数的工作流
        for bp in pr:
            await self._orch_async(shared, {**self.params, **bp})
        return await self.post_async(shared, pr, None)


class AsyncParallelBatchFlow(AsyncFlow, BatchFlow):
    """异步并行批处理工作流，并行执行多个工作流实例"""

    async def _run_async(self, shared):
        pr = await self.prep_async(shared) or []
        # 并行执行所有批处理参数的工作流
        await asyncio.gather(*(self._orch_async(shared, {**self.params, **bp}) for bp in pr))
        return await self.post_async(shared, pr, None)
