"""
Ch1-h: 结构化输出（让 LLM 返回固定格式的数据）

【学习目标】
- 了解如何让 LLM 返回 Python 对象而不是纯文本
- 了解 Pydantic 的 BaseModel 用法

【Python 语法 - 重要】
- class 类名(BaseModel): 定义一个数据类（继承自 BaseModel）
- answer: str - 类型注解，表示 answer 字段是字符串类型
- \"""文档字符串\""" - 类或字段的说明文档

【注意】
- with_structured_output() 依赖 OpenAI 的 response_format 功能
- DeepSeek 目前不支持这个功能
- 本文件提供两种方案：
  1. 方案A：用提示词让 LLM 返回 JSON，手动解析
  2. 方案B：用 OpenAI 的模型（需要 OpenAI API key）
"""

# ============ 导入 ============
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel
import json


# ============ 定义输出结构 ============
class AnswerWithJustification(BaseModel):
    """
    回答的数据结构

    【Python 语法】
    - answer: str 表示有一个叫 answer 的字段，类型是字符串
    - 冒号后面是类型注解，不是赋值
    """
    answer: str        # 答案
    justification: str # 理由


# ============================================================
# 方案 A：用提示词让 LLM 返回 JSON（DeepSeek 可用）
# ============================================================
print("=" * 50)
print("方案 A：用提示词返回 JSON（DeepSeek 可用）")
print("=" * 50)

llm = ChatOpenAI(model="deepseek-chat", temperature=0)

# 提示词：明确告诉 LLM 返回 JSON 格式
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个助手。请用 JSON 格式回答问题。
JSON 格式必须包含两个字段：
- answer: 你的答案
- justification: 你的理由

只输出 JSON，不要输出其他内容。
示例：{{"answer": "2", "justification": "因为1+1=2"}}"""),
    ("human", "{question}")
])

# 创建链
chain = prompt | llm

# 调用
response = chain.invoke({"question": "1+1等于什么？"})
print("原始输出:", response.content)

# 手动解析 JSON
try:
    # 提取 JSON 部分（有时 LLM 会加一些额外文字）
    content = response.content.strip()
    # 如果输出被 ```json 包裹，去掉它
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]

    data = json.loads(content)
    result = AnswerWithJustification(**data)
    print(f"答案: {result.answer}")
    print(f"理由: {result.justification}")
except Exception as e:
    print(f"解析失败: {e}")


# ============================================================
# 方案 B：用 JsonOutputParser（更优雅的方式）
# ============================================================
print("\n" + "=" * 50)
print("方案 B：用 JsonOutputParser")
print("=" * 50)

# JsonOutputParser 会自动解析 JSON
parser = JsonOutputParser(pydantic_object=AnswerWithJustification)

prompt_b = ChatPromptTemplate.from_messages([
    ("system", "你是一个助手。{format_instructions}"),
    ("human", "{question}")
])

# 把解析器的格式说明加入提示词
chain_b = prompt_b | llm | parser

try:
    result_b = chain_b.invoke({
        "question": "天空为什么是蓝色的？",
        "format_instructions": parser.get_format_instructions()
    })
    print(f"结果类型: {type(result_b)}")
    print(f"答案: {result_b.get('answer', 'N/A')}")
    print(f"理由: {result_b.get('justification', 'N/A')}")
except Exception as e:
    print(f"解析失败: {e}")


# ============================================================
# 方案 C：如果你有 OpenAI API key（原生支持）
# ============================================================
# 取消下面的注释即可使用（需要设置 OPENAI_API_KEY 环境变量指向 OpenAI）
#
# print("\n" + "=" * 50)
# print("方案 C：OpenAI 原生结构化输出")
# print("=" * 50)
#
# from langchain_openai import ChatOpenAI
#
# # 注意：这里不设置 base_url，使用 OpenAI 官方 API
# openai_llm = ChatOpenAI(
#     model="gpt-3.5-turbo",
#     temperature=0,
#     # 如果你的 OPENAI_API_KEY 指向 DeepSeek，需要另外设置
#     # api_key="你的 OpenAI API key",
#     # base_url="https://api.openai.com/v1"  # OpenAI 官方地址
# )
#
# structured_llm = openai_llm.with_structured_output(AnswerWithJustification)
# response_c = structured_llm.invoke("1+1等于什么？")
# print(f"答案: {response_c.answer}")
# print(f"理由: {response_c.justification}")
