"""
Ch4-f: 消息合并（合并连续的同类型消息）

【学习目标】
- 学会用 merge_message_runs 合并消息
- 理解什么时候需要合并消息

【使用场景】
有时候会出现连续的同类型消息:
- 用户连续发了两条消息
- AI 的回答被分成了两部分
- 多个系统指令

合并可以:
- 减少消息数量，节省 token
- 让对话结构更清晰

【Python 语法】
- content 可以是字符串，也可以是列表（多模态内容）
- {"type": "text", "text": "..."}: 文本类型的内容块
"""

# ============ 导入 ============
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    merge_message_runs,  # 消息合并函数
)

# ============ 准备示例消息 ============
# 注意：有连续的同类型消息
messages = [
    # 两条连续的 SystemMessage
    SystemMessage(content="you're a good assistant."),
    SystemMessage(content="you always respond with a joke."),
    # 两条连续的 HumanMessage
    HumanMessage(
        # content 可以是列表（多模态格式）
        content=[{"type": "text", "text": "i wonder why it's called langchain"}]
    ),
    HumanMessage(content="and who is harrison chasing anyways"),
    # 两条连续的 AIMessage
    AIMessage(
        content='Well, I guess they thought "WordRope" and "SentenceString" just didn\'t have the same ring to it!'
    ),
    AIMessage(
        content="Why, he's probably chasing after the last cup of coffee in the office!"
    ),
]
print(f"合并前消息数: {len(messages)}")

# ============ 合并消息 ============
# 连续的同类型消息会被合并成一条
merged = merge_message_runs(messages)
print(f"合并后消息数: {len(merged)}")  # 6 → 3

print("\n合并后的消息:")
for msg in merged:
    print(f"\n{msg.__class__.__name__}:")
    print(f"  {msg.content}")

# 输出:
# SystemMessage:
#   you're a good assistant.
#   you always respond with a joke.
#
# HumanMessage:
#   [{'type': 'text', 'text': "i wonder why it's called langchain"}, 'and who is harrison chasing anyways']
#
# AIMessage:
#   Well, I guess they thought "WordRope"...
#   Why, he's probably chasing after...
