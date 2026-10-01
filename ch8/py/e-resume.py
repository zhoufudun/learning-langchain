from dotenv import load_dotenv
load_dotenv()
"""
Ch8-e: 恢复执行（Resume）

【学习目标】
- 学会从中断点恢复执行
- 了解 invoke(None, config) 的用法

【与 d-authorize.py 的关系】
- d-authorize: 在节点前暂停
- e-resume: 从暂停处恢复执行

【恢复执行的方法】
- graph.invoke(None, config)
- graph.astream(None, config)
- 传入 None 表示"继续上次的执行"
"""

# ============ 导入 ============
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver


# ============ 主函数 ============
async def main():
    """
    演示从中断点恢复执行
    """
    # 创建图
    builder = StateGraph()
    # 实际使用需要添加节点和边
    graph = builder.compile(checkpointer=MemorySaver())

    config = {"configurable": {"thread_id": "1"}}

    # ============ 恢复执行 ============
    # 假设之前的执行在 tools 节点前暂停了
    # 现在人工确认后，继续执行

    # 传入 None 作为输入，表示"继续上次的执行"
    # config 必须与之前相同（相同的 thread_id）
    output = graph.astream(None, config, interrupt_before=["tools"])

    async for c in output:
        print(c)


# ============ 运行 ============
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
