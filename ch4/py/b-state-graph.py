from dotenv import load_dotenv
load_dotenv()
"""
Ch4-b: 状态图（StateGraph）- LangGraph 基础

【学习目标】
- 理解 LangGraph 的核心概念
- 学会用 StateGraph 构建有状态的应用

【什么是 LangGraph？】
- LangChain 的状态管理框架
- 用图（Graph）来描述应用流程
- 自动管理状态传递

【核心概念】
- State: 状态类型定义（包含哪些数据）
- Node: 节点（处理函数）
- Edge: 边（节点之间的连接）
- START/END: 特殊节点，表示开始和结束

【Python 语法 - 重要】
- TypedDict: 带类型的字典，定义字典的结构
- Annotated[类型, 注解]: 给类型添加额外信息
- add_messages: LangGraph 的消息累加函数
"""

# ============ 导入 ============
from typing import Annotated, TypedDict
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.memory import MemorySaver


# ============ 定义状态结构 ============
class State(TypedDict):
    """
    状态类型定义

    【Python 语法】
    - TypedDict: 定义字典应该有哪些键，以及值的类型
    - Annotated[list, add_messages]:
      - list: 基础类型是列表
      - add_messages: 注解，告诉 LangGraph 如何更新这个字段
        当返回新消息时，会追加到列表而不是替换
    """
    messages: Annotated[list, add_messages]


# ============ 创建状态图 ============
# StateGraph 需要知道状态的类型
builder = StateGraph(State)

model = ChatOpenAI()


# ============ 定义节点函数 ============
def chatbot(state: State):
    """
    聊天机器人节点

    参数:
        state: 当前状态，包含 messages 列表

    返回:
        更新后的状态（只返回需要更新的部分）
        因为用了 add_messages，返回的消息会追加到列表
    """
    # 调用模型，传入所有历史消息
    answer = model.invoke(state["messages"])
    # 返回新消息，会追加到 messages 列表
    return {"messages": [answer]}


# ============ 构建图 ============
# 添加节点
builder.add_node("chatbot", chatbot)

# 添加边（连接）
builder.add_edge(START, "chatbot")  # 开始 → chatbot
builder.add_edge("chatbot", END)    # chatbot → 结束

# 编译图
graph = builder.compile()

# ============ 运行图 ============
# 输入初始状态
input = {"messages": [HumanMessage("hi!")]}

# stream() 流式输出每个节点的结果
for chunk in graph.stream(input):
    print(chunk)
# 输出: {'chatbot': {'messages': [AIMessage(content='Hello! How can I help you today?')]}}
