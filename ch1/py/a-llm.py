"""
Ch1-a: 最简单的 LLM 调用

【学习目标】
- 了解如何创建一个 LLM 模型对象
- 了解如何用 invoke() 方法调用模型
- 了解如何用 dotenv 加载环境变量

【Python 语法】
- from ... import ...: 从模块导入类/函数
- 函数名(): 调用函数
- 变量 = 类名(...): 创建对象（实例化）
- 对象.方法(): 调用对象的方法
- print(): 打印输出
"""

# ============ 加载环境变量 ============
# dotenv 库用于从 .env 文件加载环境变量
# .env 文件里写着 OPENAI_API_KEY=xxx，load_dotenv() 会把它加载到环境变量中
from dotenv import load_dotenv
load_dotenv()  # 执行加载，之后代码就能读取 .env 里的变量了

# ============ 导入模型类 ============
# ChatOpenAI 是 LangChain 对 OpenAI 聊天模型的封装
# 也兼容 DeepSeek 等兼容 OpenAI API 的服务
from langchain_openai.chat_models import ChatOpenAI

# ============ 创建模型对象 ============
# model="deepseek-v4-flash-0731" 指定使用 DeepSeek 的模型
# API key 和 base_url 会从环境变量自动读取（OPENAI_API_KEY, OPENAI_API_BASE）
model = ChatOpenAI(model="deepseek-v4-flash-0731")

# ============ 调用模型 ============
# invoke("问题") 是最基础的调用方式
# 传入字符串，返回 AIMessage 对象
response = model.invoke("你是谁")

# response.content 是模型返回的文本内容
print(response.content)
