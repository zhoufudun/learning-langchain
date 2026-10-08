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

【重要说明】
DeepSeek 目前只提供 Chat API，不提供 Embedding API
所以我们使用 HuggingFace 的免费本地模型

【依赖安装】
pip install sentence-transformers

【Python 语法】
- 列表推导式: [x for x in items] 遍历并生成新列表
"""
from dotenv import load_dotenv
load_dotenv()

# ============ 导入 ============
# HuggingFaceEmbeddings 使用本地模型，不需要 API key
from langchain_huggingface import HuggingFaceEmbeddings

# ============ 创建嵌入模型 ============
# all-MiniLM-L6-v2 是一个轻量级的嵌入模型
# - 首次运行会自动下载模型（约 90MB）
# - 之后会使用本地缓存
# - 完全免费，不需要 API key
ai_embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

# 【其他可选模型】
# - "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # 多语言支持，包括中文
# - "BAAI/bge-small-zh-v1.5"  # 中文优化模型
# - "BAAI/bge-small-en-v1.5"  # 英文优化模型

# ============ 批量嵌入文档 ============
# embed_documents() 把多个文本转换成向量
# 输入: 字符串列表
# 输出: 向量列表（每个向量是一个浮点数列表）
texts = [
    "Hi there!",
    "Oh, hello!",
    "What's your name?",
    "My friends call me World",
    "Hello World!"
]
embeddings = ai_embeddings.embed_documents(texts)

# ============ 查看结果 ============
print(f"生成了 {len(embeddings)} 个向量")
print(f"每个向量的维度: {len(embeddings[0])}")

# 显示每个文本和对应的向量（前5个值）
# zip() 把两个列表配对: [(text1, emb1), (text2, emb2), ...]
# enumerate() 同时获取索引 i
print("\n文本 → 向量（前5个值）:")
for i, (text, emb) in enumerate(zip(texts, embeddings)):
    print(f"  [{i}] \"{text}\" → {emb[:5]}...")

# ============ 单个查询嵌入 ============
# embed_query() 用于嵌入单个查询文本
query_text = "hello"
query_embedding = ai_embeddings.embed_query(query_text)
print(f"\n查询文本: \"{query_text}\"")
print(f"查询向量维度: {len(query_embedding)}")

# ============ 计算相似度，找出最匹配的文本 ============
# 【原理】
# - 余弦相似度: 两个向量夹角越小，相似度越高
# - 范围: -1 到 1，1 表示完全相同，0 表示无关
import numpy as np

def cosine_similarity(vec1, vec2):
    """计算两个向量的余弦相似度"""
    # np.dot() 计算点积
    # np.linalg.norm() 计算向量的模（长度）
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# 计算查询与每个文档的相似度
print(f"\n与查询 \"{query_text}\" 的相似度:")
similarities = []
for i, (text, emb) in enumerate(zip(texts, embeddings)):
    sim = cosine_similarity(query_embedding, emb)
    similarities.append((sim, text))
    print(f"  [{i}] {sim:.4f} - \"{text}\"")

# 按相似度排序，找出最匹配的
# sorted() 排序，key=lambda x: x[0] 按第一个元素（相似度）排序
# reverse=True 从高到低排序
sorted_results = sorted(similarities, key=lambda x: x[0], reverse=True)

print(f"\n🎯 最匹配的文本: \"{sorted_results[0][1]}\"")
print(f"   相似度: {sorted_results[0][0]:.4f}")
