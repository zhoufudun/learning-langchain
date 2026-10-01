from dotenv import load_dotenv
load_dotenv()
"""
Ch5-a: 基础聊天机器人（LangGraph 版）

【学习目标】
- 用 LangGraph 构建简单的聊天机器人
- 理解 StateGraph 的基本用法

【这个文件和 Ch4-b 类似，但更精简】
- 定义状态（State）
- 定义节点（chatbot 函数）
- 构建图（添加节点和边）
- 编译并运行

【LangGraph 核心概念回顾】
- State: 状态，在节点之间传递的数据
- Node: 节点，处理状态的函数
- Edge: 边，定义节点之间的连接关系
"""

# ============ 导入 ============
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# ============ 创建模型 ============
model = ChatOpenAI()


# ============ 定义状态类型 ============
class State(TypedDict):
    """
    状态类型定义

    messages 字段使用 add_messages 注解:
    - 当节点返回新消息时，会追加到列表
    - 而不是替换整个列表
    """
    messages: Annotated[list, add_messages]


# ============ 定义节点函数 ============
def chatbot(state: State):
    """
    聊天机器人节点

    1. 接收当前状态（包含历史消息）
    2. 调用模型生成回复
    3. 返回新消息（会追加到 messages 列表）
    """
    answer = model.invoke(state["messages"])
    return {"messages": [answer]}


# ============ 构建图 ============
builder = StateGraph(State)

# 添加节点
builder.add_node("chatbot", chatbot)

# 添加边
builder.add_edge(START, "chatbot")  # 开始 → chatbot
builder.add_edge("chatbot", END)    # chatbot → 结束

# 编译图
graph = builder.compile()

# ============ 运行示例 ============
input = {"messages": [HumanMessage("hi!")]}

# stream() 流式输出每个节点的结果
for chunk in graph.stream(input):
    print(chunk)
# 输出: {'chatbot': {'messages': [AIMessage(content='Hello! How can I assist you today?')]}}
