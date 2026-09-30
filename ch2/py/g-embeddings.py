"""
Ch2-g: 文本嵌入（Embeddings）

【学习目标】
- 了解什么是 Embedding（嵌入/向量）
- 学会把文本转换成向量

【什么是 Embedding？】
- 把文本转换成数字向量（一串数字）
- 语义相近的文本，向量也相近
- 用于计算文本相似度、向量检索等

【例子】
- "猫" → [0.1, 0.5, 0.3, ...]
- "小猫" → [0.1, 0.6, 0.3, ...]  （相近）
- "汽车" → [0.9, 0.1, 0.7, ...]  （差异大）

【Python 语法】
- 列表推导式: [x for x in items] 遍历并生成新列表
"""
from dotenv import load_dotenv
load_dotenv()


# ============ 导入 ============
from langchain_openai import OpenAIEmbeddings

# ============ 创建嵌入模型 ============
# text-embedding-3-small 是 OpenAI 的嵌入模型
# 也可以用其他嵌入模型，如 HuggingFace 的模型
model = OpenAIEmbeddings(model="text-embedding-3-small")

# ============ 批量嵌入文档 ============
# embed_documents() 把多个文本转换成向量
# 输入: 字符串列表
# 输出: 向量列表（每个向量是一个浮点数列表）
embeddings = model.embed_documents([
    "Hi there!",
    "Oh, hello!",
    "What's your name?",
    "My friends call me World",
    "Hello World!"
])

# ============ 查看结果 ============
print(f"生成了 {len(embeddings)} 个向量")
print(f"每个向量的维度: {len(embeddings[0])}")
print(f"第一个向量的前 5 个值: {embeddings[0][:5]}")
# OpenAI 的 text-embedding-3-small 生成 1536 维向量
