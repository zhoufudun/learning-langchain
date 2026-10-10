"""
Ch1-ka: 流式输出（打字机效果）

【学习目标】
- 了解如何实现流式输出
- 了解 yield 关键字的用法

【Python 语法 - yield】
- yield 用于生成器函数
- 普通函数用 return 返回一个值就结束
- 生成器函数用 yield 可以"产出"多个值，每次产出一个
- for 循环可以遍历生成器产出的值

【流式输出的好处】
- 不用等完整回答，边生成边显示
- 用户体验更好（像 ChatGPT 的打字效果）
"""
from dotenv import load_dotenv
load_dotenv()


# ============ 导入 ============
from langchain_core.runnables import chain
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ============ 准备组件 ============
model = ChatOpenAI(model="deepseek-v4-flash-0731")

template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human","{question}")
    ]
)

# ============ 流式输出的 chatbot ============
@chain
def chatbot2(values):
    """
    这个函数用 yield 而不是 return
    所以它是一个生成器函数
    """
    prompt = template.invoke(values)
    # model.stream() 返回一个迭代器，逐个产出 token
    for token in model.stream(prompt):
        # yield 把每个 token "产出"给调用者
        yield token


# ============ 使用流式输出 ============
# chatbot.stream() 返回生成器，用 for 循环逐个获取 token

for result in chatbot2.stream({"question": "Which model providers offer LLMs?"}):
    # result 是 AIMessageChunk，包含一小段文本
    print(result.content, end="")  # end="" 不换行，实现打字机效果