from dotenv import load_dotenv
load_dotenv()
"""
Ch8-c: 中断执行（Interrupt）

【学习目标】
- 学会在执行过程中中断图
- 了解 asyncio.Event 的用法

【使用场景】
- 用户取消操作
- 超时自动中断
- 需要人工确认才能继续

【Python 语法 - 重要】
- async def: 定义异步函数
- await: 等待异步操作完成
- asyncio.Event(): 事件对象，用于线程间通信
- event.set(): 设置事件（触发中断）
- event.is_set(): 检查事件是否被设置
- aclosing(): 确保异步生成器正确关闭
"""

# ============ 导入 ============
import asyncio
from contextlib import aclosing  # 确保异步资源正确关闭

from langchain.schema import HumanMessage
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver


# ============ 主函数 ============
async def main():
    """
    演示如何中断图的执行
    """
    # 创建图
    builder = StateGraph()
    # 实际使用需要添加节点和边
    graph = builder.compile(checkpointer=MemorySaver())

    # 创建事件对象，用于触发中断
    event = asyncio.Event()

    # 准备输入
    input = {
        "messages": [
            HumanMessage(
                "How old was the 30th president of the United States when he died?"
            )
        ]
    }

    config = {"configurable": {"thread_id": "1"}}

    # ============ 流式处理（可中断）============
    # aclosing() 确保即使中断也能正确关闭流
    async with aclosing(graph.astream(input, config)) as stream:
        async for chunk in stream:
            # 检查是否需要中断
            if event.is_set():
                print("中断执行!")
                break
            else:
                print(chunk)  # 处理输出

    # ============ 模拟 2 秒后触发中断 ============
    await asyncio.sleep(2)
    event.set()  # 设置事件，触发中断


# ============ 运行 ============
if __name__ == "__main__":
    asyncio.run(main())
