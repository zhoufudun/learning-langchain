from dotenv import load_dotenv
load_dotenv()
"""
Ch4-c: 持久化记忆（跨请求保持对话）

【学习目标】
- 理解 checkpointer 的作用
- 学会用 thread_id 区分不同对话

【问题场景】
b-state-graph.py 的记忆只在单次运行中有效。
用户第二次调用时，AI 不记得第一次的对话。

【解决方案】
- 使用 checkpointer（检查点）保存状态
- 用 thread_id 区分不同的对话线程
- 同一个 thread_id 的对话会保持上下文

【MemorySaver vs 其他 checkpointer】
- MemorySaver: 保存在内存，程序重启后丢失
- SqliteSaver: 保存到 SQLite 数据库
- PostgresSaver: 保存到 PostgreSQL
"""

# ============ 导入 ============
from typing import Annotated, TypedDict
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.memory import MemorySaver  # 内存检查点


# ============ 定义状态 ============
class State(TypedDict):
    messages: Annotated[list, add_messages]


# ============ 构建图 ============
builder = StateGraph(State)
model = ChatOpenAI()


def chatbot(state: State):
    answer = model.invoke(state["messages"])
    return {"messages": [answer]}


builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

# ============ 添加持久化 ============
# compile() 时传入 checkpointer 参数
# MemorySaver() 会在内存中保存每次对话的状态
graph = builder.compile(checkpointer=MemorySaver())

# ============ 使用 thread_id 区分对话 ============
# configurable 是配置字典
# thread_id 是对话的唯一标识
thread1 = {"configurable": {"thread_id": "1"}}

# ============ 第一次对话 ============
result_1 = graph.invoke(
    {"messages": [HumanMessage("hi, my name is Jack!")]},
    thread1  # 传入配置
)
print("第一次:", result_1["messages"][-1].content)

# ============ 第二次对话（同一线程）============
# 因为用了同一个 thread_id，AI 会记得 Jack 这个名字
result_2 = graph.invoke(
    {"messages": [HumanMessage("what is my name?")]},
    thread1
)
print("第二次:", result_2["messages"][-1].content)
# 输出类似: "Your name is Jack!"

# ============ 查看完整状态 ============
# get_state() 获取指定线程的当前状态
state = graph.get_state(thread1)
print("\n完整状态:")
print(f"消息数量: {len(state.values['messages'])}")
