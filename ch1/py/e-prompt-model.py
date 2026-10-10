"""
Ch1-e: PromptTemplate + Model（模板 + 模型调用）

【学习目标】
- 了解如何组合模板和模型
- 理解"可复用"的概念：模板和模型创建一次，可以多次使用

【流程】
1. 创建模板（可复用）
2. 创建模型（可复用）
3. 用模板生成提示词
4. 用模型生成回答
"""

# ============ 导入 ============

from dotenv import load_dotenv

load_dotenv()  # 加载.env

from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# ============ 可复用的组件 ============
# 这两个对象创建一次，可以反复使用

# 模板：定义提示词的结构
template = PromptTemplate.from_template("""Answer the question based on the context below. If the question cannot be answered using the information provided, answer with "I don't know".

Context: {context}

Question: {question}

Answer: """)

# 模型：指定使用哪个 LLM
model = ChatOpenAI(model="deepseek-v4-flash-0731")

# ============ 单次使用 ============
# 下面是使用模板和模型的一次调用

# 第一步：用模板生成提示词
prompt = template.invoke(
    {
        "context": "The most recent advancements in NLP are being driven by Large Language Models (LLMs). These models outperform their smaller counterparts and have become invaluable for developers who are creating applications with NLP capabilities. Developers can tap into these models through Hugging Face's `transformers` library, or by utilizing OpenAI and Cohere's offerings through the `openai` and `cohere` libraries, respectively.",
        "question": "Which model providers offer LLMs?",
    }
)

# 第二步：把提示词传给模型，获取回答
response = model.invoke(prompt)

resp = model.invoke(prompt)

# 打印模型的回答（AIMessage 对象）
print(resp)
