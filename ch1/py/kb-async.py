"""
Ch1-kb: 异步编程（async/await）

【学习目标】
- 了解 Python 异步编程的基本概念
- 了解 async def 和 await 的用法

【Python 语法 - 异步】
- async def 函数名(): 定义异步函数（协程）
- await 表达式: 等待异步操作完成
- asyncio.run(): 运行异步函数

【为什么要用异步？】
- 同步：调用 API 时程序"卡住"等待
- 异步：调用 API 时可以去做其他事，API 返回后再继续
- 适合需要同时处理多个请求的场景

【注意】
- 异步函数只能在异步函数里调用
- 普通函数不能直接调用异步函数
- 需要用 asyncio.run() 来启动异步代码
"""
from dotenv import load_dotenv
load_dotenv()


# ============ 导入 ============
from langchain_core.runnables import chain
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ============ 准备组件 ============
template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{question}"),
    ]
)

model = ChatOpenAI(model="gpt-3.5-turbo")


# ============ 异步 chatbot ============
@chain
async def chatbot(values):
    """
    async def: 定义异步函数
    await: 等待异步操作完成

    ainvoke() 是 invoke() 的异步版本
    """
    # await template.ainvoke(): 异步填充模板
    prompt = await template.ainvoke(values)
    # await model.ainvoke(): 异步调用模型
    return await model.ainvoke(prompt)


# ============ 主函数 ============
async def main():
    """异步主函数"""
    # 异步调用 chatbot
    return await chatbot.ainvoke({"question": "Which model providers offer LLMs?"})


# ============ 运行 ============
# __name__ == "__main__" 确保只在直接运行时执行（不是被导入时）
if __name__ == "__main__":
    import asyncio
    # asyncio.run() 是运行异步代码的入口
    print(asyncio.run(main()))
