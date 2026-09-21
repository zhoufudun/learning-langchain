"""
Ch1-f: ChatPromptTemplate 聊天提示词模板

【学习目标】
- 了解 ChatPromptTemplate 和 PromptTemplate 的区别
- ChatPromptTemplate 可以定义多条消息（system/human/ai）
- PromptTemplate 只是一个字符串模板

【LangChain 概念】
- ChatPromptTemplate.from_messages(): 从消息列表创建模板
- ("system", "..."):  系统消息
- ("human", "..."):   用户消息
- ("ai", "..."):      AI消息（用于few-shot示例）

【Python 语法】
- ("a", "b"): 元组（tuple），不可变的序列
- [(元组1), (元组2)]: 元组的列表
"""

# ============ 导入 ============
from langchain_core.prompts import ChatPromptTemplate

# ============ 创建聊天模板 ============
# from_messages() 接收一个消息列表
# 每条消息是一个元组: (角色, 内容)
template = ChatPromptTemplate.from_messages(
    [
        # 第一条消息：系统指令
        (
            "system",
            'Answer the question based on the context below. If the question cannot be answered using the information provided, answer with "I don\'t know".',
        ),
        # 第二条消息：用户提供上下文（包含 {context} 占位符）
        ("human", "Context: {context}"),
        # 第三条消息：用户提问（包含 {question} 占位符）
        ("human", "Question: {question}"),
    ]
)

# ============ 填充模板 ============
# invoke() 会把 {context} 和 {question} 替换成实际值
# 返回的是 ChatPromptValue，包含格式化后的消息列表
response = template.invoke(
    {
        "context": "The most recent advancements in NLP are being driven by Large Language Models (LLMs). These models outperform their smaller counterparts and have become invaluable for developers who are creating applications with NLP capabilities. Developers can tap into these models through Hugging Face's `transformers` library, or by utilizing OpenAI and Cohere's offerings through the `openai` and `cohere` libraries, respectively.",
        "question": "Which model providers offer LLMs?",
    }
)

# 打印格式化后的消息（注意：这里还没调用 LLM）
print(response)
