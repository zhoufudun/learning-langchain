"""
Ch1-c: 使用 SystemMessage 设定 AI 角色

【学习目标】
- 了解 SystemMessage（系统消息）的作用
- 了解如何组合多条消息

【LangChain 概念】
- SystemMessage: 系统指令，用来设定 AI 的行为和角色
- HumanMessage: 用户的问题
- 消息顺序: 通常是 [SystemMessage, HumanMessage, ...]

【Python 语法】
- [元素1, 元素2]: 创建包含多个元素的列表
"""

# ============ 导入 ============
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai.chat_models import ChatOpenAI

# ============ 创建模型 ============
model = ChatOpenAI(model="deepseek-v4-flash-0731")

# ============ 创建消息 ============
# SystemMessage: 告诉 AI 它是什么角色、应该怎么回答
# 这里设定它回答时要加三个感叹号
system_msg= SystemMessage("You are a helpful assistant that responds to questions with three exclamation marks.")

# HumanMessage: 用户的问题
human_msg = HumanMessage("What is the capital of France?")

# ============ 调用模型 ============
# 把两条消息组成列表传给模型
# 模型会先读 system 指令，再回答 human 的问题
response = model.invoke([system_msg,human_msg])
print(response.content)  # 输出会是: Paris!!!
