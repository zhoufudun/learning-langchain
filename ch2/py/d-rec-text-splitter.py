"""
Ch2-d: 递归字符文本分割器

【学习目标】
- 了解为什么需要分割文本
- 学会使用 RecursiveCharacterTextSplitter

【为什么要分割？】
- LLM 有 token 限制，不能一次处理太长的文本
- 向量检索时，小块文本更精确
- 分割后每块大小相近，便于处理

【LangChain 概念】
- RecursiveCharacterTextSplitter: 最常用的分割器
- chunk_size: 每块的最大字符数
- chunk_overlap: 相邻块之间重叠的字符数（保持上下文连贯）

【Python 语法】
- 方法链: loader.load() 返回的结果可以直接传给下一个方法
"""
from dotenv import load_dotenv
load_dotenv()


# ============ 导入 ============
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# ============ 第一步：加载文档 ============
text_loader = TextLoader('../.././test2.txt', encoding="utf-8")
docs = text_loader.load()  # docs 是 Document 列表

# ============ 第二步：创建分割器 ============
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,    # 每块最多 1000 个字符
    chunk_overlap=200   # 相邻块重叠 200 个字符
)
# 重叠的作用: 避免句子被切断，保持上下文连贯
# 例如: 块1 结尾 "...机器学习是" + 块2 开头 "机器学习是一种..."

# ============ 第三步：分割文档 ============
# split_documents() 接收 Document 列表，返回分割后的 Document 列表
splitted_docs = text_splitter.split_documents(docs)

# ============ 查看结果 ============
# 原来 1 个大文档 → 分割成多个小文档
print(f"原文档数: {len(docs)}, 分割后: {len(splitted_docs)}")
for doc in splitted_docs:
    print(doc, end="\n\n")
