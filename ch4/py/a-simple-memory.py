from dotenv import load_dotenv
load_dotenv()
"""
Ch4-a: 简单记忆（手动传递历史消息）

【学习目标】
- 理解对话历史的作用
- 学会用 placeholder 传递消息历史

【什么是记忆？】
- 让 AI 记住之前的对话内容
- 没有记忆：每次对话都是独立的
- 有记忆：AI 知道你之前说了什么

【实现方式】
- 最简单的方式：手动把历史消息传给模型
- 用 placeholder 占位符接收消息列表

【Python 语法】
- ("human", "内容"): 元组，表示用户消息
- ("ai", "内容"): 元组，表示 AI 消息
- ("placeholder", "{变量}"): 占位符，会被替换成消息列表
"""

# ============ 导入 ============
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# ============ 创建带占位符的提示词模板 ============
prompt = ChatPromptTemplate.from_messages([
    # 系统消息：设定 AI 的角色
    ("system", "You are a helpful assistant. Answer all questions to the best of         your ability."),
    # placeholder 占位符：会被替换成 {messages} 的值
    # 这里用于插入历史对话
    ("placeholder", "{messages}"),
])

model = ChatOpenAI()

# 用 | 连接提示词和模型
chain = prompt | model

# ============ 调用时传入历史消息 ============
# messages 是一个列表，包含之前的对话
response = chain.invoke({
    "messages": [
        # 第一轮对话
        ("human", "Translate this sentence from English to French: I love programming."),
        ("ai", "J'adore programmer."),
        # 第二轮对话（引用了第一轮的内容）
        ("human", "What did you just say?"),
    ],
})

# AI 会知道"你刚才说了什么"指的是上一条翻译
print(response.content)
# 输出类似: "I said 'J'adore programmer.' which means 'I love programming.' in French."
