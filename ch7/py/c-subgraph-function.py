"""
Ch7-c: 子图（函数包装方式）

【学习目标】
- 学会用函数包装子图
- 处理父图和子图状态完全不同的情况

【与 b-subgraph-direct.py 的区别】
- 直接嵌入: 父图和子图共享某些键
- 函数包装: 父图和子图状态完全独立，用函数做转换

【本例场景】
- 父图状态: {foo: str}
- 子图状态: {bar: str, baz: str}
- 没有共享键！需要手动转换

【流程】
1. 父图状态 {foo: "hello"}
2. 函数转换: foo → bar
3. 子图处理: bar + "baz" → bar
4. 函数转换: bar → foo
5. 父图状态 {foo: "hellobaz"}
"""

# ============ 导入 ============
from typing import TypedDict
from langgraph.graph import START, StateGraph


# ============ 定义状态类型 ============
class State(TypedDict):
    """父图状态"""
    foo: str


class SubgraphState(TypedDict):
    """
    子图状态

    注意: 没有任何键与父图共享
    """
    bar: str
    baz: str


# ============ 定义子图节点 ============
def subgraph_node(state: SubgraphState):
    """子图节点: 在 bar 后追加 'baz'"""
    return {"bar": state["bar"] + "baz"}


# ============ 构建子图 ============
subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node("subgraph_node", subgraph_node)
subgraph_builder.add_edge(START, "subgraph_node")
subgraph = subgraph_builder.compile()


# ============ 定义包装函数 ============
def node(state: State):
    """
    包装函数: 负责状态转换

    1. 把父图状态转换成子图状态
    2. 调用子图
    3. 把子图结果转换回父图状态
    """
    # 父图状态 → 子图状态
    # state["foo"] → {"bar": state["foo"]}
    response = subgraph.invoke({"bar": state["foo"]})

    # 子图结果 → 父图状态
    # response["bar"] → {"foo": response["bar"]}
    return {"foo": response["bar"]}


# ============ 构建父图 ============
builder = StateGraph(State)

# 用函数作为节点（不是直接用子图）
builder.add_node("node", node)
builder.add_edge(START, "node")

# 编译
graph = builder.compile()


# ============ 运行示例 ============
initial_state = {"foo": "hello"}
result = graph.invoke(initial_state)

print(f"输入: {initial_state}")
print(f"输出: {result}")
# 输出: {'foo': 'hellobaz'}
#
# 流程:
# 1. foo="hello" → 转换 → bar="hello"
# 2. 子图处理 → bar="hellobaz"
# 3. 转换回 → foo="hellobaz"
