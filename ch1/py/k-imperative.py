"""
Ch1-k: 命令式编程（用 @chain 装饰器）

【学习目标】
- 了解如何用函数封装 LangChain 流程
- 了解 @chain 装饰器的作用

【Python 语法 - 装饰器】
- @chain 是装饰器语法
- 装饰器放在函数定义上方，会"增强"这个函数
- @chain 让普通函数获得 invoke()、stream()、batch() 等方法

【命令式 vs 声明式】
- 命令式：一步步写代码，像写菜谱（先做这个，再做那个）
- 声明式：用 | 管道符连接组件（见 l-declarative.py）
"""

# ============ 导入 ============
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
# chain 装饰器，把普通函数变成 Runnable
from langchain_core.runnables import chain

# ============ 准备组件 ============
# 模板
template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{question}"),
    ]
)

# 模型
model = ChatOpenAI(model="gpt-3.5-turbo")


# ============ 用 @chain 定义流程 ============
@chain  # 这个装饰器让 chatbot 函数变成 Runnable 对象
def chatbot(values):
    """
    values: 字典，包含 {question: "问题内容"}
    """
    # 第一步：用模板生成提示词
    prompt = template.invoke(values)
    # 第二步：用模型生成回答
    return model.invoke(prompt)


# ============ 使用 ============
# 因为有 @chain 装饰器，chatbot 现在有了 invoke() 方法
response = chatbot.invoke({"question": "Which model providers offer LLMs?"})
print(response.content)
