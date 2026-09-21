"""
Ch8-g: 分叉/回溯（Fork / Time Travel）

【学习目标】
- 学会查看状态历史
- 学会从历史状态重新执行（时间旅行）

【使用场景】
- 回溯到之前的状态重新尝试
- 探索不同的执行路径
- 调试和分析问题

【核心方法】
- graph.get_state_history(config): 获取状态历史
- graph.invoke(None, historical_config): 从历史状态继续执行

【状态历史】
每次状态变化都会被记录，可以回溯到任意时间点。
"""

# ============ 导入 ============
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver


# ============ 主函数 ============
def main():
    """
    演示如何查看历史状态和从历史状态重新执行
    """
    # 创建图
    builder = StateGraph()
    # 实际使用需要添加节点和边
    graph = builder.compile(checkpointer=MemorySaver())

    config = {"configurable": {"thread_id": "1"}}

    # ============ 获取状态历史 ============
    # get_state_history() 返回一个迭代器
    # 最新的状态在前面
    history = [state for state in graph.get_state_history(config)]

    print(f"历史状态数量: {len(history)}")

    # ============ 从历史状态重新执行 ============
    # 假设我们想回到第 3 个历史状态（index=2）
    if len(history) >= 3:
        historical_state = history[2]

        # 使用历史状态的 config 来恢复执行
        # 这会从那个时间点重新开始
        result = graph.invoke(None, historical_state.config)

        print("从历史状态重新执行的结果:", result)
    else:
        print("历史状态不足 3 个，无法演示回溯")


# ============ 运行 ============
if __name__ == "__main__":
    main()
