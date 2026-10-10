"""
Ch1-g: ChatPromptTemplate + Model（聊天模板 + 模型）

【学习目标】
- 完整流程：聊天模板 → 填充变量 → 调用模型 → 获取回答

【流程图】
ChatPromptTemplate  →  invoke({变量})  →  消息列表  →  model.invoke()  →  回答
"""

# ============ 导入 ============
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()
# ============ 可复用的组件 ============

# 聊天模板：定义对话结构
template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            'Answer the question based on the context below. If the question cannot be answered using the information provided, answer with "I don\'t know".',
        ),
        ("human", "Context: {context}"),
        ("human", "Question: {question}"),
    ]
)

# 模型
model = ChatOpenAI(model="deepseek-v4-flash-0731")

# ============ 单次调用 ============

# 第一步：填充模板，生成消息列表
prompt = template.invoke(
    {
        "context": "The most recent advancements in NLP are being driven by Large Language Models (LLMs). These models outperform their smaller counterparts and have become invaluable for developers who are creating applications with NLP capabilities. Developers can tap into these models through Hugging Face's `transformers` library, or by utilizing OpenAI and Cohere's offerings through the `openai` and `cohere` libraries, respectively.",
        "question": "Which model providers offer LLMs?",
    }
)

# 第二步：把消息列表传给模型
# 打印模型的回答
print(model.invoke(prompt))
