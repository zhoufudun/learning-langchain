from dotenv import load_dotenv
load_dotenv()
"""
Ch5-c: 多 RAG 聊天机器人（条件路由）

【学习目标】
- 构建带条件分支的图
- 学会用 add_conditional_edges 实现路由
- 理解如何根据状态选择不同的执行路径

【流程图】
                    ┌→ retrieve_medical_records ──┐
START → router ────┤                              ├→ generate_answer → END
                    └→ retrieve_insurance_faqs ───┘

【场景】
医疗问答系统:
- 病历相关问题 → 检索病历数据库
- 保险相关问题 → 检索保险 FAQ 数据库

【新概念】
- Literal["a", "b"]: 值只能是 "a" 或 "b"
- add_conditional_edges: 根据函数返回值决定下一个节点
"""

# ============ 导入 ============
from typing import Annotated, Literal, TypedDict
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.vectorstores.in_memory import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# ============ 准备模型和嵌入 ============
embeddings = OpenAIEmbeddings()
model_low_temp = ChatOpenAI(temperature=0.1)   # 用于路由决策
model_high_temp = ChatOpenAI(temperature=0.7)  # 用于生成回答


# ============ 定义状态类型 ============
class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_query: str
    # Literal 限制值只能是这两个之一
    domain: Literal["records", "insurance"]
    documents: list[Document]
    answer: str


class Input(TypedDict):
    user_query: str


class Output(TypedDict):
    documents: list[Document]
    answer: str


# ============ 准备示例数据 ============
sample_docs = [
    Document(page_content="Patient medical record...", metadata={"domain": "records"}),
    Document(page_content="Insurance policy details...", metadata={"domain": "insurance"}),
]

# 创建两个独立的向量库
medical_records_store = InMemoryVectorStore.from_documents(sample_docs, embeddings)
medical_records_retriever = medical_records_store.as_retriever()

insurance_faqs_store = InMemoryVectorStore.from_documents(sample_docs, embeddings)
insurance_faqs_retriever = insurance_faqs_store.as_retriever()


# ============ 节点1: 路由器 ============
router_prompt = SystemMessage(
    """You need to decide which domain to route the user query to. You have two domains to choose from:
- records: contains medical records of the patient, such as diagnosis, treatment, and prescriptions.
- insurance: contains frequently asked questions about insurance policies, claims, and coverage.

Output only the domain name."""
)


def router_node(state: State) -> State:
    """
    路由节点: 决定查询应该去哪个检索器
    """
    user_message = HumanMessage(state["user_query"])
    messages = [router_prompt, *state["messages"], user_message]
    res = model_low_temp.invoke(messages)
    return {
        "domain": res.content,  # "records" 或 "insurance"
        "messages": [user_message, res],
    }


# ============ 路由函数 ============
def pick_retriever(state: State) -> Literal["retrieve_medical_records", "retrieve_insurance_faqs"]:
    """
    根据 domain 返回下一个节点的名称

    【重要】
    返回值必须是 add_conditional_edges 中定义的节点名称
    """
    if state["domain"] == "records":
        return "retrieve_medical_records"
    else:
        return "retrieve_insurance_faqs"


# ============ 节点2a: 检索病历 ============
def retrieve_medical_records(state: State) -> State:
    documents = medical_records_retriever.invoke(state["user_query"])
    return {"documents": documents}


# ============ 节点2b: 检索保险 FAQ ============
def retrieve_insurance_faqs(state: State) -> State:
    documents = insurance_faqs_retriever.invoke(state["user_query"])
    return {"documents": documents}


# ============ 节点3: 生成回答 ============
medical_records_prompt = SystemMessage(
    "You are a helpful medical chatbot, who answers questions based on the patient's medical records, such as diagnosis, treatment, and prescriptions."
)

insurance_faqs_prompt = SystemMessage(
    "You are a helpful medical insurance chatbot, who answers frequently asked questions about insurance policies, claims, and coverage."
)


def generate_answer(state: State) -> State:
    """
    根据检索到的文档生成回答
    根据 domain 选择不同的系统提示词
    """
    if state["domain"] == "records":
        prompt = medical_records_prompt
    else:
        prompt = insurance_faqs_prompt

    messages = [
        prompt,
        *state["messages"],
        HumanMessage(f"Documents: {state['documents']}"),
    ]
    res = model_high_temp.invoke(messages)
    return {
        "answer": res.content,
        "messages": res,
    }


# ============ 构建图 ============
builder = StateGraph(State, input=Input, output=Output)

# 添加节点
builder.add_node("router", router_node)
builder.add_node("retrieve_medical_records", retrieve_medical_records)
builder.add_node("retrieve_insurance_faqs", retrieve_insurance_faqs)
builder.add_node("generate_answer", generate_answer)

# 添加边
builder.add_edge(START, "router")

# 条件边: 根据 pick_retriever 的返回值决定下一个节点
# pick_retriever 返回 "retrieve_medical_records" → 执行该节点
# pick_retriever 返回 "retrieve_insurance_faqs" → 执行该节点
builder.add_conditional_edges("router", pick_retriever)

# 两个检索节点都连接到 generate_answer
builder.add_edge("retrieve_medical_records", "generate_answer")
builder.add_edge("retrieve_insurance_faqs", "generate_answer")
builder.add_edge("generate_answer", END)

# 编译
graph = builder.compile()

# ============ 运行示例 ============
input = {"user_query": "Am I covered for COVID-19 treatment?"}
print("查询:", input["user_query"])
print()
for chunk in graph.stream(input):
    print(chunk)
