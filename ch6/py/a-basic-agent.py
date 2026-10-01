from dotenv import load_dotenv
load_dotenv()
"""
Ch6-a: 基础 Agent（带工具的智能体）

【学习目标】
- 理解 Agent 的概念
- 学会定义和使用工具（Tool）
- 学会用 @tool 装饰器创建工具

【什么是 Agent？】
- Agent = LLM + Tools
- LLM 决定是否需要使用工具
- 如果需要，调用工具获取信息
- 循环直到得到最终答案

【流程图】
        ┌────────────────────────────┐
        ↓                            │
START → model → [需要工具?] → tools ─┘
          │
          └→ [不需要] → END

【新概念】
- @tool 装饰器: 把函数变成工具
- bind_tools(): 让模型知道有哪些工具可用
- ToolNode: 执行工具的节点
- tools_condition: 判断是否需要调用工具
"""

# ============ 导入 ============
import ast  # 用于安全地执行数学表达式
from typing import Annotated, TypedDict

from langchain_community.tools import DuckDuckGoSearchRun  # 搜索工具
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool  # @tool 装饰器
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition  # 预置组件


# ============ 定义工具 ============
@tool
def calculator(query: str) -> str:
    """
    A simple calculator tool. Input should be a mathematical expression.

    【@tool 装饰器】
    - 把普通函数变成 LangChain 工具
    - 函数的文档字符串会告诉 LLM 这个工具的用途
    - LLM 根据描述决定什么时候使用这个工具
    """
    # ast.literal_eval 安全地计算数学表达式
    # 比 eval() 更安全，只能执行字面量表达式
    return ast.literal_eval(query)


# 使用预置的搜索工具
search = DuckDuckGoSearchRun()

# 工具列表
tools = [search, calculator]

# ============ 创建模型并绑定工具 ============
# bind_tools() 让模型知道有哪些工具可用
# 模型会在需要时生成工具调用请求
model = ChatOpenAI(temperature=0.1).bind_tools(tools)


# ============ 定义状态 ============
class State(TypedDict):
    messages: Annotated[list, add_messages]


# ============ 定义模型节点 ============
def model_node(state: State) -> State:
    """
    模型节点：调用 LLM

    LLM 会根据问题决定:
    - 直接回答
    - 或者调用工具获取更多信息
    """
    res = model.invoke(state["messages"])
    return {"messages": res}


# ============ 构建图 ============
builder = StateGraph(State)

# 添加节点
builder.add_node("model", model_node)
builder.add_node("tools", ToolNode(tools))  # ToolNode 自动执行工具

# 添加边
builder.add_edge(START, "model")

# 条件边：判断是否需要调用工具
# tools_condition 会检查模型输出是否包含工具调用
# 如果有 → 跳转到 "tools" 节点
# 如果没有 → 结束（END）
builder.add_conditional_edges("model", tools_condition)

# 工具执行完后，回到模型节点（让模型处理工具的结果）
builder.add_edge("tools", "model")

# 编译
graph = builder.compile()

# ============ 运行示例 ============
# 这个问题需要搜索（找到第 30 任总统是谁，以及他的生卒年份）
# 然后需要计算（算年龄）
input = {
    "messages": [
        HumanMessage(
            "How old was the 30th president of the United States when he died?"
        )
    ]
}

print("查询:", input["messages"][0].content)
print()
for c in graph.stream(input):
    print(c)
