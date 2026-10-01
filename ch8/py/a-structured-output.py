from dotenv import load_dotenv
load_dotenv()
"""
Ch8-a: 结构化输出

【学习目标】
- 复习 with_structured_output 的用法
- 让 LLM 返回固定格式的数据

【使用场景】
- 需要解析 LLM 输出时
- 需要固定格式的数据时（如 JSON）
- API 响应需要特定结构时

【Python 语法】
- BaseModel: Pydantic 的基类
- Field(description=...): 字段描述，帮助 LLM 理解字段含义
"""

# ============ 导入 ============
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


# ============ 定义输出结构 ============
class Joke(BaseModel):
    """
    笑话的数据结构

    LLM 会根据这个结构返回数据
    """
    setup: str = Field(description="The setup of the joke")     # 笑话的铺垫
    punchline: str = Field(description="The punchline to the joke")  # 笑话的梗


# ============ 创建结构化输出的模型 ============
model = ChatOpenAI(model="gpt-4o", temperature=0)

# with_structured_output() 让模型返回 Joke 对象
model = model.with_structured_output(Joke)

# ============ 调用模型 ============
result = model.invoke("Tell me a joke about cats")

# result 是 Joke 对象，不是字符串
print(f"类型: {type(result)}")
print(f"Setup: {result.setup}")
print(f"Punchline: {result.punchline}")

# 输出类似:
# 类型: <class '__main__.Joke'>
# Setup: Why don't cats play poker in the jungle?
# Punchline: Too many cheetahs!
