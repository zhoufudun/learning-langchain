from dotenv import load_dotenv
load_dotenv()
"""
Ch8-d: 授权/人工确认（Human-in-the-Loop）

【学习目标】
- 学会在特定节点前暂停等待确认
- 了解 interrupt_before 参数

【使用场景】
- 执行危险操作前需要人工确认
- 调用外部 API 前需要审批
- 敏感数据操作前需要授权

【关键参数】
- interrupt_before=["tools"]: 在执行 tools 节点前暂停
- 图会暂停并保存状态
- 等待人工确认后才能继续

【流程】
1. 执行到 tools 节点前暂停
2. 人工审核工具调用
3. 确认后继续执行（用 resume 或 invoke(None, config)）
"""

# ============ 导入 ============
from langchain.schema import HumanMessage
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver


# ============ 主函数 ============
async def main():
    """
    演示在 tools 节点前暂停等待授权
    """
    # 创建图
    builder = StateGraph()
    # 实际使用需要添加节点和边（包括 tools 节点）
    graph = builder.compile(checkpointer=MemorySaver())  # 需要 checkpointer 保存状态

    # 准备输入
    input = {
        "messages": [
            HumanMessage(
                "How old was the 30th president of the United States when he died?"
            )
        ]
    }

    config = {"configurable": {"thread_id": "1"}}

    # ============ 带中断点的流式执行 ============
    # interrupt_before=["tools"] 表示在 tools 节点前暂停
    output = graph.astream(input, config, interrupt_before=["tools"])

    async for c in output:
        print(c)
        # 当图暂停时，输出会停止
        # 此时可以检查即将执行的工具调用
        # 人工确认后再继续

    # 注意: 确认后需要调用 graph.astream(None, config) 继续执行
    # 或者使用 e-resume.py 中的方法


# ============ 运行 ============
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
