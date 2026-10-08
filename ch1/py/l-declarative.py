"""
Ch1-l: 声明式编程（用 | 管道符连接组件）

【学习目标】
- 了解 LangChain 最核心的概念：链式调用
- 了解 | 管道符的用法

【Python 语法 - 管道符】
- | 在 Python 里通常是"按位或"运算符
- 但 LangChain 重载了这个运算符，让它变成"连接"的意思
- template | model 表示：template 的输出作为 model 的输入

【声明式 vs 命令式】
- 命令式（k-imperative.py）：一步步写代码
- 声明式：用 | 把组件"声明"式地连接起来

【这是 LangChain 最重要的概念！】
- 几乎所有 LangChain 应用都用 | 来组装组件
- 组件可以是：PromptTemplate、Model、OutputParser、Retriever 等
"""
from dotenv import load_dotenv
load_dotenv()


# ============ 导入 ============
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI

# ============ 准备组件 ============
# 模板
messages = ChatPromptTemplate.from_messages([("system", "You are a helpful assistant."), ("human", "{question}"), ])

# 模型
model = ChatOpenAI(model="deepseek-chat")

# ============ 用 | 连接组件 ============
# 这一行是 LangChain 的核心！
# template | model 创建一个"链"
# 输入 → template（生成提示词）→ model（生成回答）→ 输出
chatbot = messages | model

# ============ 使用链 ============
# invoke(): 同步调用
response = chatbot.invoke({"question": "1+1=?"})
print(response.content)

# ============ 流式输出 ============
# stream(): 流式调用，同样可以用
print("\n--- 流式输出 ---")
for streamResult in chatbot.stream({"question","1+1=?"}):
    # part 是 AIMessageChunk
    print(streamResult.content, end="")
