"""
Ch7-d: Supervisor 模式（监督者协调多个 Agent）

【学习目标】
- 理解 Supervisor 模式
- 学会用一个 Agent 协调多个专业 Agent

【什么是 Supervisor 模式？】
- Supervisor（监督者）: 决定下一步由哪个 Agent 执行
- Workers（工人）: 各自负责专业任务
- 类似于团队领导分配任务给成员

【流程图】
        ┌───────────────────────────────────┐
        ↓                                   │
START → supervisor → [选择下一个?] → researcher ─┤
              │                         ↓      │
              │                      coder ────┘
              │
              └→ [FINISH] → END

【本例的 Agent】
- supervisor: 决定任务分配
- researcher: 负责研究和分析
- coder: 负责编写代码
"""

# ============ 导入 ============
from typing import Literal
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START
from pydantic import BaseModel


# ============ 定义 Supervisor 的输出结构 ============
class SupervisorDecision(BaseModel):
    """
    Supervisor 的决策

    【Python 语法】
    Literal["a", "b", "c"]: 值只能是这几个之一
    """
    next: Literal["researcher", "coder", "FINISH"]


# ============ 创建模型 ============
model = ChatOpenAI(model="gpt-4", temperature=0)
# with_structured_output 让模型返回 SupervisorDecision 对象
model = model.with_structured_output(SupervisorDecision)

# 可用的 Agent 列表
agents = ["researcher", "coder"]

# ============ Supervisor 的提示词 ============
system_prompt_part_1 = f"""You are a supervisor tasked with managing a conversation between the
following workers: {agents}. Given the following user request,
respond with the worker to act next. Each worker will perform a
task and respond with their results and status. When finished,
respond with FINISH."""

system_prompt_part_2 = f"""Given the conversation above, who should act next? Or should we FINISH? Select one of: {", ".join(agents)}, FINISH"""


# ============ 定义 Supervisor 节点 ============
def supervisor(state):
    """
    Supervisor 节点: 决定下一步由谁执行

    返回 SupervisorDecision 对象，包含 next 字段
    """
    messages = [
        ("system", system_prompt_part_1),
        *state["messages"],
        ("system", system_prompt_part_2),
    ]
    return model.invoke(messages)


# ============ 定义 Agent 状态 ============
class AgentState(MessagesState):
    """
    Agent 状态

    MessagesState 是 LangGraph 预定义的状态类，包含 messages 字段
    我们添加 next 字段来记录下一步要执行的 Agent
    """
    next: Literal["researcher", "coder", "FINISH"]


# ============ 定义 Worker Agent ============
def researcher(state: AgentState):
    """研究员 Agent: 负责分析和研究"""
    response = model.invoke(
        [
            {
                "role": "system",
                "content": "You are a research assistant. Analyze the request and provide relevant information.",
            },
            {"role": "user", "content": state["messages"][0].content},
        ]
    )
    return {"messages": [response]}


def coder(state: AgentState):
    """程序员 Agent: 负责编写代码"""
    response = model.invoke(
        [
            {
                "role": "system",
                "content": "You are a coding assistant. Implement the requested functionality.",
            },
            {"role": "user", "content": state["messages"][0].content},
        ]
    )
    return {"messages": [response]}


# ============ 构建图 ============
builder = StateGraph(AgentState)

# 添加节点
builder.add_node("supervisor", supervisor)
builder.add_node("researcher", researcher)
builder.add_node("coder", coder)

# 添加边
builder.add_edge(START, "supervisor")

# 条件边: 根据 supervisor 的决策选择下一个节点
# lambda state: state["next"] 返回 "researcher"、"coder" 或 "FINISH"
# "FINISH" 会自动映射到 END
builder.add_conditional_edges("supervisor", lambda state: state["next"])

# Worker 完成后回到 Supervisor
builder.add_edge("researcher", "supervisor")
builder.add_edge("coder", "supervisor")

# 编译
graph = builder.compile()


# ============ 运行示例 ============
initial_state = {
    "messages": [
        {
            "role": "user",
            "content": "I need help analyzing some data and creating a visualization.",
        }
    ],
    "next": "supervisor",
}

print("Supervisor 模式演示")
print("任务: 分析数据并创建可视化")
print()

for output in graph.stream(initial_state):
    print(f"\n决策: {output.get('next', 'N/A')}")
    if output.get("messages"):
        print(f"回复: {output['messages'][-1].content[:100]}...")
