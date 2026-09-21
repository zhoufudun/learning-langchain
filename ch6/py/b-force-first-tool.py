"""
Ch6-b: 强制首次使用工具

【学习目标】
- 学会强制 Agent 第一步必须使用指定工具
- 理解 ToolCall 的结构

【问题场景】
有时候我们希望 Agent 第一步必须先搜索信息，
而不是直接回答（可能会产生幻觉）。

【解决方案】
创建一个 first_model 节点，手动构造工具调用，
强制第一步就执行搜索。

【流程图】
START → first_model → tools → model → [需要工具?] → tools
                                 │                    ↑
                                 │                    │
                                 └────────────────────┘
                                 └→ [不需要] → END

【Python 语法】
- uuid4().hex: 生成 32 位的唯一 ID 字符串
- ToolCall: 表示一次工具调用的数据结构
"""

# ============ 导入 ============
import ast
from typing import Annotated, TypedDict
from uuid import uuid4  # 生成唯一 ID

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import AIMessage, HumanMessage, ToolCall
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


# ============ 定义工具 ============
@tool
def calculator(query: str) -> str:
    """A simple calculator tool. Input should be a mathematical expression."""
    return ast.literal_eval(query)


search = DuckDuckGoSearchRun()
tools = [search, calculator]
model = ChatOpenAI(temperature=0.1).bind_tools(tools)


# ============ 定义状态 ============
class State(TypedDict):
    messages: Annotated[list, add_messages]


# ============ 定义节点 ============
def model_node(state: State) -> State:
    """普通模型节点"""
    res = model.invoke(state["messages"])
    return {"messages": res}


def first_model(state: State) -> State:
    """
    强制首次搜索的节点

    手动构造一个工具调用，强制第一步执行搜索
    """
    # 获取用户的查询
    query = state["messages"][-1].content

    # 手动创建 ToolCall
    # ToolCall 是 LangChain 中表示工具调用的数据结构
    search_tool_call = ToolCall(
        name="duckduckgo_search",  # 工具名称（必须和实际工具名匹配）
        args={"query": query},     # 工具参数
        id=uuid4().hex             # 唯一 ID（用于关联工具结果）
    )

    # 返回一个包含工具调用的 AIMessage
    # content="" 因为这不是文本回复
    # tool_calls=[...] 包含要执行的工具调用
    return {"messages": AIMessage(content="", tool_calls=[search_tool_call])}


# ============ 构建图 ============
builder = StateGraph(State)

# 添加节点
builder.add_node("first_model", first_model)  # 强制搜索节点
builder.add_node("model", model_node)         # 普通模型节点
builder.add_node("tools", ToolNode(tools))

# 添加边
builder.add_edge(START, "first_model")  # 从 first_model 开始（而不是 model）
builder.add_edge("first_model", "tools")  # first_model → tools（强制执行搜索）
builder.add_conditional_edges("model", tools_condition)  # model 后根据情况选择
builder.add_edge("tools", "model")  # 工具执行后回到 model

# 编译
graph = builder.compile()

# ============ 运行示例 ============
input = {
    "messages": [
        HumanMessage(
            "How old was the 30th president of the United States when he died?"
        )
    ]
}

print("强制首次搜索的 Agent")
print("查询:", input["messages"][0].content)
print()
for c in graph.stream(input):
    print(c)
