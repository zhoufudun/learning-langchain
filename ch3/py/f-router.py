from dotenv import load_dotenv
load_dotenv()
"""
Ch3-f: 查询路由（Query Routing）

【学习目标】
- 理解查询路由的作用
- 学会用 LLM 分类查询并路由到不同处理链

【什么是查询路由？】
根据用户查询的内容，自动选择最合适的处理方式。
例如:
- Python 相关问题 → 查询 Python 文档
- JavaScript 相关问题 → 查询 JS 文档
- 数学问题 → 使用计算器
- 一般问题 → 使用通用 LLM

【Python 语法 - 重要】
- Literal["a", "b"]: 类型注解，值只能是 "a" 或 "b"
- Field(...): Pydantic 字段配置，... 表示必填
- RunnableLambda: 把普通函数包装成 LangChain 的 Runnable
"""

# ============ 导入 ============
from typing import Literal  # 用于定义字面量类型
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda


# ============ 定义路由数据模型 ============
class RouteQuery(BaseModel):
    """
    路由查询的数据模型

    【Python 语法】
    - BaseModel: Pydantic 的基类
    - Literal["python_docs", "js_docs"]: 值只能是这两个之一
    - Field(...): 字段配置
      - ...: 必填（三个点是 Python 的省略号对象）
      - description: 字段描述，帮助 LLM 理解如何填写
    """
    datasource: Literal["python_docs", "js_docs"] = Field(
        ...,  # 必填
        description="Given a user question, choose which datasource would be most relevant for answering their question",
    )


# ============ 创建结构化输出的 LLM ============
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# with_structured_output() 让 LLM 返回 RouteQuery 对象
# LLM 会自动解析回答并填充 datasource 字段
structured_llm = llm.with_structured_output(RouteQuery)

# ============ 路由提示词 ============
system = """You are an expert at routing a user question to the appropriate data source. Based on the programming language the question is referring to, route it to the relevant data source."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}")
    ]
)

# 路由链: 提示词 → 结构化 LLM
router = prompt | structured_llm

# ============ 测试路由 ============
question = """Why doesn't the following code work:
from langchain_core.prompts
import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages(["human", "speak in {language}"])
prompt.invoke("french") """

result = router.invoke({"question": question})
print("路由结果:", result)
# 输出: RouteQuery(datasource='python_docs')


# ============ 根据路由结果执行不同逻辑 ============
def choose_route(result):
    """
    根据路由结果选择不同的处理链

    【Python 语法】
    - result.datasource: 访问对象的属性
    - .lower(): 转换为小写
    - in: 判断是否包含
    """
    if "python_docs" in result.datasource.lower():
        return "chain for python_docs"  # 这里返回实际的处理链
    else:
        return "chain for js_docs"


# ============ 完整路由链 ============
# RunnableLambda 把普通函数包装成 Runnable
# 这样就可以用 | 连接了
full_chain = router | RunnableLambda(choose_route)

result = full_chain.invoke({"question": question})
print("选择的路由:", result)
