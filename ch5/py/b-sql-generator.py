from dotenv import load_dotenv
load_dotenv()
"""
Ch5-b: SQL 生成器（多节点图）

【学习目标】
- 构建包含多个节点的图
- 理解如何在节点之间传递数据
- 学会定义输入/输出类型

【流程图】
START → generate_sql → explain_sql → END

用户问题 → 生成 SQL → 解释 SQL → 输出

【新概念】
- Input/Output 类型: 定义图的输入输出结构
- 多个节点: 每个节点处理不同的任务
- 不同的 temperature: 生成 SQL 用低温度，解释用高温度
"""

# ============ 导入 ============
from typing import Annotated, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# ============ 创建两个模型 ============
# 低温度：输出更确定，适合生成代码/SQL
model_low_temp = ChatOpenAI(temperature=0.1)
# 高温度：输出更多样，适合自然语言解释
model_high_temp = ChatOpenAI(temperature=0.7)


# ============ 定义状态类型 ============
class State(TypedDict):
    """完整状态，包含中间数据"""
    messages: Annotated[list, add_messages]  # 对话历史
    user_query: str      # 用户的问题
    sql_query: str       # 生成的 SQL（中间结果）
    sql_explanation: str # SQL 解释（最终结果）


class Input(TypedDict):
    """图的输入类型"""
    user_query: str


class Output(TypedDict):
    """图的输出类型"""
    sql_query: str
    sql_explanation: str


# ============ 节点1: 生成 SQL ============
generate_prompt = SystemMessage(
    "You are a helpful data analyst, who generates SQL queries for users based on their questions."
)


def generate_sql(state: State) -> State:
    """
    生成 SQL 查询

    【Python 语法】
    - *state["messages"]: 星号展开列表
      [a, b, c] → a, b, c
    """
    user_message = HumanMessage(state["user_query"])
    # 组装消息: 系统提示 + 历史消息 + 用户问题
    messages = [generate_prompt, *state["messages"], user_message]
    res = model_low_temp.invoke(messages)
    return {
        "sql_query": res.content,
        # 更新对话历史
        "messages": [user_message, res],
    }


# ============ 节点2: 解释 SQL ============
explain_prompt = SystemMessage(
    "You are a helpful data analyst, who explains SQL queries to users."
)


def explain_sql(state: State) -> State:
    """
    解释 SQL 查询

    此时 state["messages"] 已经包含了用户问题和生成的 SQL
    """
    messages = [
        explain_prompt,
        *state["messages"],  # 包含用户问题和 SQL
    ]
    res = model_high_temp.invoke(messages)
    return {
        "sql_explanation": res.content,
        "messages": res,
    }


# ============ 构建图 ============
# 指定 input 和 output 类型
builder = StateGraph(State, input=Input, output=Output)

# 添加节点
builder.add_node("generate_sql", generate_sql)
builder.add_node("explain_sql", explain_sql)

# 添加边（定义执行顺序）
builder.add_edge(START, "generate_sql")
builder.add_edge("generate_sql", "explain_sql")
builder.add_edge("explain_sql", END)

# 编译
graph = builder.compile()

# ============ 运行示例 ============
result = graph.invoke({"user_query": "What is the total sales for each product?"})
print("SQL 查询:")
print(result["sql_query"])
print("
SQL 解释:")
print(result["sql_explanation"])
