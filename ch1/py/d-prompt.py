"""
Ch1-d: PromptTemplate 提示词模板（不调用 LLM）

【学习目标】
- 了解 PromptTemplate 的作用：复用提示词结构
- 了解如何用 {变量名} 定义占位符
- 了解如何用字典填充模板

【LangChain 概念】
- PromptTemplate: 提示词模板，可以复用
- {context}, {question}: 占位符，运行时用实际值替换
- invoke(): 填充模板，返回完整的提示词字符串

【Python 语法】
- 三引号字符串 \"""...\""": 多行字符串
- {key: value}: 字典（dict），键值对的集合
- f-string 类似概念: {变量} 会被替换成实际值
"""

from dotenv import load_dotenv
load_dotenv()

# ============ 导入 ============
from langchain_core.prompts import PromptTemplate

# ============ 创建模板 ============
# from_template() 从字符串创建模板
# {context} 和 {question} 是占位符，之后用实际值替换
template = PromptTemplate.from_template("""Answer the question based on the context below. If the question cannot be answered using the information provided, answer with "I don't know".

Context: {context}

Question: {question}

Answer: """)

# ============ 填充模板 ============
# invoke() 传入字典，字典的 key 对应模板中的占位符
# 返回填充后的完整字符串
response = template.invoke(
    {
        # "context" 对应模板中的 {context}
        "context": "The most recent advancements in NLP are being driven by Large Language Models (LLMs). These models outperform their smaller counterparts and have become invaluable for developers who are creating applications with NLP capabilities. Developers can tap into these models through Hugging Face's `transformers` library, or by utilizing OpenAI and Cohere's offerings through the `openai` and `cohere` libraries, respectively.",
        # "question" 对应模板中的 {question}
        "question": "Which model providers offer LLMs?",
    }
)

# 打印填充后的完整提示词（注意：这里没有调用 LLM，只是生成了提示词）
print(f"result: {response}")
