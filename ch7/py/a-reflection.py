"""
Ch7-a: 反思模式（Reflection）

【学习目标】
- 理解 Reflection（反思）的概念
- 学会让 LLM 自我批评和改进

【什么是 Reflection？】
- 让 LLM 生成内容
- 然后让另一个 LLM（或同一个）评价和批评
- 根据反馈改进内容
- 循环直到满意

【流程图】
START → generate → [消息数>6?] ─→ END
           ↑           │
           │           ↓
           └── reflect ←┘

【关键技巧】
- 角色互换: 让 AI 的输出变成 Human 输入
- 让 LLM "看到"自己的输出并进行批评
"""

# ============ 导入 ============
from typing import Annotated, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# ============ 创建模型 ============
model = ChatOpenAI()


# ============ 定义状态 ============
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ============ 定义提示词 ============
# 生成器的提示词: 写文章
generate_prompt = SystemMessage(
    "You are an essay assistant tasked with writing excellent 3-paragraph essays."
    " Generate the best essay possible for the user's request."
    " If the user provides critique, respond with a revised version of your previous attempts."
)

# 反思者的提示词: 评价文章
reflection_prompt = SystemMessage(
    "You are a teacher grading an essay submission. Generate critique and recommendations for the user's submission."
    " Provide detailed recommendations, including requests for length, depth, style, etc."
)


# ============ 生成节点 ============
def generate(state: State) -> State:
    """生成文章（或根据反馈修改）"""
    answer = model.invoke([generate_prompt] + state["messages"])
    return {"messages": [answer]}


# ============ 反思节点 ============
def reflect(state: State) -> State:
    """
    反思/批评节点

    【关键技巧】
    把消息角色互换:
    - AI 的输出 → 变成 Human 输入（让批评者"看到"生成的文章）
    - Human 的输入 → 变成 AI 输出
    这样批评者就能对文章进行评价
    """
    # 角色映射: AI ↔ Human
    cls_map = {AIMessage: HumanMessage, HumanMessage: AIMessage}

    # 第一条消息是原始请求，保持不变
    # 后续消息进行角色互换
    translated = [reflection_prompt, state["messages"][0]] + [
        cls_map[msg.__class__](content=msg.content)
        for msg in state["messages"][1:]
    ]

    answer = model.invoke(translated)

    # 把批评作为 Human 反馈返回给生成器
    return {"messages": [HumanMessage(content=answer.content)]}


# ============ 终止条件 ============
def should_continue(state: State):
    """
    判断是否继续循环

    每次循环产生 2 条消息（生成 + 反思）
    6 条消息 = 3 次迭代，然后停止
    """
    if len(state["messages"]) > 6:
        return END
    else:
        return "reflect"


# ============ 构建图 ============
builder = StateGraph(State)

# 添加节点
builder.add_node("generate", generate)
builder.add_node("reflect", reflect)

# 添加边
builder.add_edge(START, "generate")
builder.add_conditional_edges("generate", should_continue)  # 生成后判断是否继续
builder.add_edge("reflect", "generate")  # 反思后继续生成

# 编译
graph = builder.compile()

# ============ 运行示例 ============
initial_state = {
    "messages": [
        HumanMessage(
            content="Write an essay about the relevance of 'The Little Prince' today."
        )
    ]
}

print("反思模式演示")
print("主题: The Little Prince 的现代意义")
print()

for output in graph.stream(initial_state):
    message_type = "generate" if "generate" in output else "reflect"
    content = output[message_type]["messages"][-1].content
    print(f"\n[{message_type.upper()}] {content[:100]}...")
