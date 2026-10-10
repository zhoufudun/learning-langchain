from dotenv import load_dotenv
load_dotenv()
"""
Ch4-d: 消息裁剪（控制上下文长度）

【学习目标】
- 理解为什么需要裁剪消息
- 学会使用 trim_messages 函数

【问题场景】
- 对话越来越长，超出 LLM 的 token 限制
- 太长的历史也浪费钱（按 token 计费）
- 需要只保留最近的、最重要的消息

【trim_messages 参数】
- max_tokens: 最大 token 数
- strategy: "last"（保留最后）或 "first"（保留最前）
- token_counter: 用于计算 token 的模型
- include_system: 是否保留系统消息
- allow_partial: 是否允许截断单条消息
- start_on: 从哪种类型的消息开始保留
"""

# ============ 导入 ============
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    trim_messages,  # 消息裁剪函数
)
from langchain_openai import ChatOpenAI

# ============ 准备示例消息 ============
# 模拟一个较长的对话历史
messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]
print(f"原始消息数: {len(messages)}")

# ============ 创建裁剪器 ============
trimmer = trim_messages(
    max_tokens=65,                       # 最多保留 65 个 token
    strategy="last",                     # 保留最后的消息（最近的对话）
    token_counter=ChatOpenAI(model="deepseek-v4-flash-0731"),  # 用这个模型计算 token
    include_system=True,                 # 保留系统消息（重要！）
    allow_partial=False,                 # 不允许截断单条消息
    start_on="human",                    # 保留的消息从 human 开始
)

# ============ 应用裁剪 ============
trimmed = trimmer.invoke(messages)
print(f"裁剪后消息数: {len(trimmed)}")
print("\n裁剪后的消息:")
for msg in trimmed:
    print(f"  {msg.__class__.__name__}: {msg.content}")

# 输出类似:
#   SystemMessage: you're a good assistant  （系统消息保留）
#   HumanMessage: having fun?               （最近的对话）
#   AIMessage: yes!
