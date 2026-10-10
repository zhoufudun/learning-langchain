"""
Ch1-b: 使用 HumanMessage 包装用户消息

【学习目标】
- 了解 LangChain 的消息类型（HumanMessage）
- 了解如何用列表传递消息

【Python 语法】
- [元素]: 创建只有一个元素的列表（list）
- 列表用方括号 [] 表示，可以包含任意类型的对象

【LangChain 概念】
- HumanMessage: 代表用户发送的消息
- 聊天模型接收"消息列表"，不是单纯的字符串
"""

# ============ 导入 ============

from dotenv import load_dotenv
load_dotenv()

from langchain_openai.chat_models import ChatOpenAI
# HumanMessage: 用户消息类
from langchain_core.messages import HumanMessage

# ============ 创建模型 ============
# 不指定 model 参数时，默认使用 gpt-3.5-turbo
model = ChatOpenAI(model="deepseek-v4-flash-0731")

# ============ 创建消息列表 ============
# HumanMessage("问题") - 创建一个用户消息对象
# [HumanMessage(...)] - 把消息放进列表里
# 为什么要用列表？因为对话可能有多条消息（system、human、ai...）
prompt = [HumanMessage("天空什么颜色")]

# ============ 调用模型 ============
# invoke() 传入消息列表
response = model.invoke(prompt)

# 打印回答内容
print(response.content)
