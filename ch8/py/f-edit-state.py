"""
Ch8-f: 编辑状态（Edit State）

【学习目标】
- 学会查看和修改图的状态
- 了解 get_state() 和 update_state() 方法

【使用场景】
- 在中断时检查状态
- 修改工具调用的参数
- 注入额外信息
- 调试和测试

【核心方法】
- graph.get_state(config): 获取当前状态
- graph.update_state(config, update): 更新状态
"""

# ============ 导入 ============
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver


# ============ 主函数 ============
def main():
    """
    演示如何查看和修改状态
    """
    # 创建图
    builder = StateGraph()
    # 实际使用需要添加节点和边
    graph = builder.compile(checkpointer=MemorySaver())

    config = {"configurable": {"thread_id": "1"}}

    # ============ 获取当前状态 ============
    state = graph.get_state(config)
    print("当前状态:", state)
    # state.values: 状态的值
    # state.next: 下一个要执行的节点
    # state.config: 配置信息

    # ============ 更新状态 ============
    # 定义要更新的内容
    # 这个字典会与当前状态合并
    update = {
        # 例如: 修改消息、添加数据等
        # "messages": [new_message],
        # "custom_field": new_value,
    }

    # 执行更新
    graph.update_state(config, update)
    print("状态已更新")

    # 更新后可以继续执行
    # graph.invoke(None, config)


# ============ 运行 ============
if __name__ == "__main__":
    main()
