"""
Ch4-e: 消息过滤（按条件筛选消息）

【学习目标】
- 学会用 filter_messages 筛选消息
- 理解消息的 id、name、type 属性

【使用场景】
- 只保留用户消息（用于日志）
- 排除示例消息（few-shot 示例不需要传给后续处理）
- 按 ID 精确过滤

【filter_messages 参数】
- include_types: 只包含这些类型（"human"、"ai"、"system"）
- exclude_types: 排除这些类型
- include_names: 只包含这些 name
- exclude_names: 排除这些 name
- include_ids: 只包含这些 id
- exclude_ids: 排除这些 id

【Python 语法】
- id="1": 消息的唯一标识
- name="bob": 消息的发送者名称
"""

# ============ 导入 ============
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    filter_messages,  # 消息过滤函数
)

# ============ 准备示例消息 ============
# 每条消息都有 id 和 name 属性
messages = [
    SystemMessage(content="you are a good assistant", id="1"),
    HumanMessage(content="example input", id="2", name="example_user"),
    AIMessage(content="example output", id="3", name="example_assistant"),
    HumanMessage(content="real input", id="4", name="bob"),
    AIMessage(content="real output", id="5", name="alice"),
]

# ============ 按类型过滤 ============
# 只保留 human 类型的消息
human_messages = filter_messages(messages, include_types="human")
print("Human messages:")
for msg in human_messages:
    print(f"  {msg.content}")
# 输出: example input, real input

# ============ 按名称排除 ============
# 排除示例用户和示例助手的消息
excluded_names = filter_messages(
    messages,
    exclude_names=["example_user", "example_assistant"]
)
print("\n排除示例消息:")
for msg in excluded_names:
    print(f"  {msg.__class__.__name__}: {msg.content}")
# 输出: system + bob + alice 的消息

# ============ 组合过滤 ============
# 同时按类型和 ID 过滤
filtered_messages = filter_messages(
    messages,
    include_types=["human", "ai"],  # 只要 human 和 ai
    exclude_ids=["3"]                # 排除 id="3" 的消息
)
print("\n组合过滤 (human+ai, 排除 id=3):")
for msg in filtered_messages:
    print(f"  [{msg.id}] {msg.__class__.__name__}: {msg.content}")
