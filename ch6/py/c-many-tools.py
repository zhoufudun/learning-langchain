"""
Ch6-c: 动态工具选择（工具太多时）

【学习目标】
- 解决工具数量过多的问题
- 学会用向量检索动态选择工具

【问题场景】
当有几十上百个工具时:
- 每次都把所有工具传给 LLM 很浪费 token
- LLM 可能被太多选项搞混

【解决方案】
1. 把所有工具的描述存入向量库
2. 根据用户查询，检索最相关的工具
3. 只把相关工具绑定给模型

【流程图】
START → select_tools → model → [需要工具?] → tools
              │           │                    ↑
              │           │                    │
              │           └────────────────────┘
              │           └→ [不需要] → END
              │
              └→ 检索最相关的工具

【Python 语法】
- 列表推导式: [tool for tool in tools if tool.name in names]
"""

# ============ 导入 ============
import ast
from typing import Annotated, TypedDict

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_core.vectorstores.in_memory import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

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

# ============ 创建工具描述的向量库 ============
embeddings = OpenAIEmbeddings()
model = ChatOpenAI(temperature=0.1)

# 把每个工具的描述存成文档
# metadata 中保存工具名称，用于后续检索
tools_retriever = InMemoryVectorStore.from_documents(
    [
        Document(
            tool.description,                 # 文档内容: 工具描述
            metadata={"name": tool.name}      # 元数据: 工具名称
        )
        for tool in tools
    ],
    embeddings,
).as_retriever()


# ============ 定义状态 ============
class State(TypedDict):
    messages: Annotated[list, add_messages]
    selected_tools: list[str]  # 新增: 存储选中的工具名称


# ============ 定义节点 ============
def model_node(state: State) -> State:
    """
    模型节点

    只绑定 selected_tools 中的工具，而不是所有工具
    """
    # 根据 selected_tools 过滤工具列表
    selected_tools = [
        tool for tool in tools
        if tool.name in state["selected_tools"]
    ]
    # 动态绑定工具
    res = model.bind_tools(selected_tools).invoke(state["messages"])
    return {"messages": res}


def select_tools(state: State) -> State:
    """
    工具选择节点

    根据用户查询，检索最相关的工具
    """
    query = state["messages"][-1].content

    # 用向量检索找到最相关的工具文档
    tool_docs = tools_retriever.invoke(query)

    # 提取工具名称
    # doc.metadata["name"] 是我们之前存入的工具名称
    selected_names = [doc.metadata["name"] for doc in tool_docs]

    return {"selected_tools": selected_names}


# ============ 构建图 ============
builder = StateGraph(State)

# 添加节点
builder.add_node("select_tools", select_tools)  # 新增: 工具选择节点
builder.add_node("model", model_node)
builder.add_node("tools", ToolNode(tools))

# 添加边
builder.add_edge(START, "select_tools")  # 先选择工具
builder.add_edge("select_tools", "model")  # 然后调用模型
builder.add_conditional_edges("model", tools_condition)
builder.add_edge("tools", "model")

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

print("动态工具选择 Agent")
print("查询:", input["messages"][0].content)
print()
for c in graph.stream(input):
    print(c)
