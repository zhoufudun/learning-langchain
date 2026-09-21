"""
Ch1-h: 结构化输出（让 LLM 返回固定格式的数据）

【学习目标】
- 了解如何让 LLM 返回 Python 对象而不是纯文本
- 了解 Pydantic 的 BaseModel 用法

【Python 语法 - 重要】
- class 类名(BaseModel): 定义一个数据类（继承自 BaseModel）
- answer: str - 类型注解，表示 answer 字段是字符串类型
- \"""文档字符串\""" - 类或字段的说明文档

【LangChain 概念】
- with_structured_output(类): 让模型返回指定类的对象
- 模型会自动解析回答，填充到对象的字段中
"""

# ============ 导入 ============
from langchain_openai import ChatOpenAI
# BaseModel: Pydantic 的基类，用于定义数据结构
from pydantic import BaseModel


# ============ 定义输出结构 ============
# 继承 BaseModel，定义我们希望 LLM 返回的数据结构
class AnswerWithJustification(BaseModel):
    """An answer to the user's question along with justification for the answer."""
    # 这是类的文档字符串，描述这个类是干什么的

    answer: str
    """The answer to the user's question"""
    # answer: str 表示有一个叫 answer 的字段，类型是字符串
    # 下面的文档字符串描述这个字段的含义

    justification: str
    """Justification for the answer"""
    # justification: str 表示有一个叫 justification 的字段，类型是字符串


# ============ 创建结构化输出的模型 ============
llm = ChatOpenAI(model="gpt-3.5", temperature=0)

# with_structured_output() 让模型返回 AnswerWithJustification 对象
# 而不是普通的文本字符串
structured_llm = llm.with_structured_output(AnswerWithJustification)

# ============ 调用模型 ============
# 返回的 response 是 AnswerWithJustification 对象
# 可以用 response.answer 和 response.justification 访问字段
response = structured_llm.invoke(
    "What weighs more, a pound of bricks or a pound of feathers")

print(response)
# 输出类似: AnswerWithJustification(answer='They weigh the same', justification='A pound is a pound...')
