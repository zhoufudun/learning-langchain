from dotenv import load_dotenv
load_dotenv()
"""
Ch8-b: 流式输出

【学习目标】
- 学会用 stream() 方法获取流式输出
- 了解 stream_mode 参数

【流式输出的好处】
- 不用等完整响应，边生成边显示
- 更好的用户体验（打字机效果）
- 可以提前中断

【stream_mode 选项】
- "values": 返回每次更新后的完整状态
- "updates": 只返回变化的部分（更省带宽）
"""

# ============ 导入 ============
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph


# ============ 创建简单的图（示例框架）============
def create_simple_graph():
    """
    创建一个简单的图用于演示

    注意: 这只是一个框架，实际使用需要添加节点和边
    """
    builder = StateGraph()
    # 实际使用时需要:
    # builder.add_node("node_name", node_function)
    # builder.add_edge(START, "node_name")
    return builder.compile()


graph = create_simple_graph()

# ============ 准备输入 ============
input = {
    "messages": [
        HumanMessage(
            "How old was the 30th president of the United States when he died?"
        )
    ]
}

# ============ 流式输出 ============
# stream_mode="updates" 只返回变化的部分
# 比 "values" 模式更省带宽
for c in graph.stream(input, stream_mode="updates"):
    print(c)
    # 每次循环会输出一个节点的更新
    # 例如: {'node_name': {'messages': [...]}}
