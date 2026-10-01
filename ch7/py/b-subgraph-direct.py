from dotenv import load_dotenv
load_dotenv()
"""
Ch7-b: 子图（直接嵌入方式）

【学习目标】
- 理解子图的概念
- 学会用共享键在父图和子图之间传递数据

【什么是子图？】
- 把一个图作为节点嵌入另一个图
- 用于组织复杂的逻辑
- 类似于函数调用函数

【本例的方式: 直接嵌入】
- 子图和父图共享某些状态键
- 子图编译后直接作为节点添加

【状态共享】
- 父图 State: {foo: str}
- 子图 SubgraphState: {foo: str, bar: str}
- foo 是共享键，子图可以读写它
"""

# ============ 导入 ============
from typing import TypedDict
from langgraph.graph import START, StateGraph


# ============ 定义状态类型 ============
class State(TypedDict):
    """父图状态"""
    foo: str  # 这个键与子图共享


class SubgraphState(TypedDict):
    """子图状态"""
    foo: str  # 与父图共享的键
    bar: str  # 子图私有的键


# ============ 定义子图节点 ============
def subgraph_node(state: SubgraphState):
    """
    子图节点

    可以通过共享的 "foo" 键与父图通信
    修改 foo 的值会传递回父图
    """
    return {"foo": state["foo"] + "bar"}


# ============ 构建子图 ============
subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node("subgraph_node", subgraph_node)
subgraph_builder.add_edge(START, "subgraph_node")
# compile() 编译成可执行的图
subgraph = subgraph_builder.compile()


# ============ 构建父图 ============
builder = StateGraph(State)

# 直接把编译好的子图作为节点添加
# LangGraph 会自动处理共享键的传递
builder.add_node("subgraph", subgraph)
builder.add_edge(START, "subgraph")

# 编译父图
graph = builder.compile()


# ============ 运行示例 ============
initial_state = {"foo": "hello"}
result = graph.invoke(initial_state)

print(f"输入: {initial_state}")
print(f"输出: {result}")
# 输出: {'foo': 'hellobar'}
# 子图在 foo 后面追加了 "bar"
