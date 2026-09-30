"""
Ch1-j: 三种调用方式（invoke / batch / stream）

【学习目标】
- invoke(): 单个调用，等待完整回答
- batch(): 批量调用，一次处理多个问题
- stream(): 流式调用，逐字输出（打字机效果）

【Python 语法】
- for token in ...: for 循环遍历
- yield: 生成器，stream() 会逐个产出 token
"""
from dotenv import load_dotenv
load_dotenv()


# ============ 导入 ============
from langchain_openai.chat_models import ChatOpenAI

model = ChatOpenAI(model="deepseek-chat")

# ============ 方式一：invoke() 单个调用 ============
# 最常用，传入一个问题，等待完整回答
completion = model.invoke("Hi there!")
invoke = model.invoke("hello!")
print(f"result: {invoke}")
print("invoke 结果:", completion.content)
# 输出完整的回答，如 "Hi!"

# ============ 方式二：batch() 批量调用 ============
# 一次传入多个问题（列表），返回多个回答（列表）
# 适合需要批量处理的场景
completions = model.batch(["Hi there!", "Bye!"])
print("batch 结果:", [c.content for c in completions])

batch = model.batch(["hello", "1+1等于多少？"])
print(f"batch result: {batch}")
# 输出: ['Hi!', 'See you!']

# ============ 方式三：stream() 流式调用 ============
# 不等完整回答，逐个 token 输出
# 适合需要"打字机效果"的场景
print("stream 结果:")
for token in model.stream("Bye!"):
    # token 是 AIMessageChunk，每次只包含一小段文本
    print(token.content, end="")  # end="" 让输出不换行
    # 输出效果: G o o d b y e !
print()  # 最后换行

for token in model.stream("解释相对论"):
    print(token.content, end="")# end="" 让输出不换行

