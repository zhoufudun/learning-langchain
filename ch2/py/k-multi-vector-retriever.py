"""
Ch2-k: 多向量检索器（用摘要检索，返回原文）

【学习目标】
- 了解 MultiVectorRetriever 的作用
- 学会用摘要做检索，返回完整文档

【问题场景】
- 长文档直接检索效果不好
- 用摘要检索更精准，但用户需要完整文档

【解决方案】
- 生成文档摘要，用摘要建索引
- 检索时：找到最相关的摘要 → 返回对应的原文档

【流程图】
原文档 → 生成摘要 → 摘要存入向量库（用于检索）
         ↓
       原文档存入文档库（用于返回）
         ↓
查询 → 在摘要中搜索 → 找到匹配的摘要 → 返回对应的原文档

【Python 语法】
- lambda x: x.page_content: 匿名函数，等价于 def f(x): return x.page_content
- enumerate(): 遍历时同时获取索引和元素
- zip(): 把两个列表配对
"""
from typing import Sequence

from dotenv import load_dotenv

load_dotenv()

# ============ 导入 ============
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
# DeepSeek 没有 Embedding API，改用智谱 AI
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
# 新版 LangChain 把 MultiVectorRetriever 移到了 langchain_classic
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_classic.storage import InMemoryStore
import uuid


# ============ 分批嵌入的包装类 ============
class BatchingZhipuEmbeddings(ZhipuAIEmbeddings):
    """智谱嵌入，自动分批（单批最多 64 条）"""
    batch_size: int = 64

    def embed_documents(self, texts):
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            all_embeddings.extend(super().embed_documents(batch))
        return all_embeddings


# ============ 配置 ============
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
collection_name = "summaries"
embeddings_model = BatchingZhipuEmbeddings(model="embedding-3")

# ============ 加载并分割文档 ============
loader = TextLoader("../../test2.txt", encoding="utf-8")
docs = loader.load()
print("原文档长度:", len(docs[0].page_content))

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs_chunks = splitter.split_documents(docs)
print(f"分割成 {len(docs_chunks)} 个块")

# ============ 创建摘要生成链 ============
prompt_text = "Summarize the following document:\n\n{doc}"
template = ChatPromptTemplate.from_template(prompt_text)
deepseek = ChatOpenAI(temperature=0, model="claude-sonnet-4-6")

# 【链的构建详解】
# LangChain 用 | 管道符把多个组件串联成链
#
# summarize_chain 的数据流：
#   输入 Document 对象
#        ↓
#   {"doc": lambda x: x.page_content}  ← 第1步：提取文档内容，构建字典
#        ↓ 输出 {"doc": "文档内容..."}
#   template                           ← 第2步：把字典填入模板
#        ↓ 输出 ChatPromptValue 对象
#   llm                                ← 第3步：调用 LLM 生成摘要
#        ↓ 输出 AIMessage 对象
#   StrOutputParser()                  ← 第4步：提取 AIMessage 的文本内容
#        ↓ 输出 字符串（摘要）
#
# 【第1步详解：字典 + lambda】
# {"doc": lambda x: x.page_content} 是一个 RunnableParallel
# - 键 "doc" 对应模板中的 {doc} 占位符
# - 值是一个 lambda 函数，从输入对象中提取 page_content 属性
# - 等价于：
#   def extract_content(x):
#       return x.page_content
#   {"doc": extract_content}
#
# 【lambda 语法】
# lambda 参数: 返回值
# lambda x: x.page_content  等价于  def f(x): return x.page_content
# summarize_chain = {
#     "doc": lambda x: x.page_content  # 从 Document 对象提取文本内容
# } | template | deepseek | StrOutputParser()

docContentParser = {"doc": lambda x: x.page_content}  # 从 Document 对象提取文本内容

summarize_chain = docContentParser | template | deepseek | StrOutputParser()

# ============ 批量生成摘要 ============
# batch() 批量处理，max_concurrency=5 表示最多 5 个并发
summaries = summarize_chain.batch(docs_chunks, {"max_concurrency": 50})
print(f"生成了 {len(summaries)} 个摘要")

# ============ 创建向量库和文档库 ============
# 向量库: 存储摘要的向量（用于搜索）
vectorstore = PGVector(
    embeddings=embeddings_model,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)

# 文档库: 存储原始文档（用于返回）
originDocStore = InMemoryStore()  # 内存存储，生产环境应该用持久化存储

# ============ 创建多向量检索器 ============
id_key = "doc_id"  # 用于关联摘要和原文档的字段名(理解为 key)

retriever = MultiVectorRetriever(
    vectorstore=vectorstore,  # 摘要向量库
    docstore=originDocStore,  # 原文档库
    id_key=id_key,  # 关联字段
)

# ============ 关联摘要和原文档 ============
# 【为什么需要 ID？】
# 摘要和原文档是分开存储的：
# - 摘要 → 向量库（用于搜索）
# - 原文档 → 文档库（用于返回）
# 需要一个 ID 把它们关联起来：搜索到摘要后，通过 ID 找到对应的原文档
#
# 【列表推导式 + uuid 详解】
# doc_ids = [str(uuid.uuid4()) for _ in docs_chunks]
#
# - uuid.uuid4(): 生成一个全局唯一的 ID
#   例如: "550e8400-e29b-41d4-a716-446655440000"
# - str(...): 把 UUID 对象转成字符串
# - for _ in docs_chunks: 遍历 docs_chunks，_ 表示不需要使用这个变量
#   只是想循环 len(docs_chunks) 次
# - 最终生成一个列表，长度等于 docs_chunks 的长度
#
# 【等价写法】
# doc_ids = []
# for _ in docs_chunks:
#     doc_ids.append(str(uuid.uuid4()))
#
# 【下划线 _ 的含义】
# Python 惯例：用 _ 表示"这个变量我不关心"
# 这里只需要循环次数，不需要使用 docs_chunks 中的具体元素
doc_ids = [str(uuid.uuid4()) for _ in docs_chunks]

# 【创建摘要文档详解】
# 把每个摘要包装成 Document 对象，并通过 metadata 关联到原文档
#
# 【列表推导式结构】
# [表达式 for 变量 in 可迭代对象]
#
# 这里的表达式是 Document(...)，变量是 (i, s)
#
# 【enumerate() 函数】
# enumerate(summaries) 返回 (索引, 元素) 的配对
# 例如 summaries = ["摘要A", "摘要B", "摘要C"]
# enumerate(summaries) → [(0, "摘要A"), (1, "摘要B"), (2, "摘要C")]
#
# 【for i, s in enumerate(...)】
# 这是元组解包（tuple unpacking）
# 把 (0, "摘要A") 拆成 i=0, s="摘要A"
#
# 【metadata 的作用】
# {id_key: doc_ids[i]} 即 {"doc_id": "某个UUID"}
# 这个 doc_id 和原文档的 ID 相同，用于关联
# 搜索时：找到摘要 → 读取 metadata["doc_id"] → 用这个 ID 去文档库取原文档
#
# 【等价写法】
# summary_docs = []
# for i, s in enumerate(summaries):
#     doc = Document(
#         page_content=s,
#         metadata={id_key: doc_ids[i]}
#     )
#     summary_docs.append(doc)
# summary_docs = [
#     Document(
#         page_content=s,                    # 摘要文本
#         metadata={id_key: doc_ids[i]}      # 通过 doc_id 关联到原文档
#     )
#     for i, s in enumerate(summaries)       # i 是索引，s 是摘要内容
# ]

summary_docs = []
for index, summary_text in enumerate(summaries):  # 遍历 summaries（摘要字符串列表）
    doc = Document(
        page_content=summary_text,
        metadata={id_key: doc_ids[index]}
    )
    summary_docs.append(doc)

# 摘要存入向量库
# 【重要】add_documents 需要提供 ids 参数，否则 PGVector 会报 null id 错误
# 直接用 doc_ids 作为数据库记录 ID（同时也存在 metadata 里用于关联原文档）
retriever.vectorstore.add_documents(summary_docs, ids=doc_ids)

# 【原文档存入文档库】
#
# 【zip() 函数详解】
# zip() 把多个列表"配对"成元组，像拉链一样扣在一起：
#
# doc_ids   = ["id-001",  "id-002",  "id-003"]
# docs_chunks = [  DocA,      DocB,      DocC  ]
#                   ↓          ↓          ↓
# zip(...)  → [("id-001", DocA), ("id-002", DocB), ("id-003", DocC)]
#
# 【list() 的作用】
# zip() 返回的是迭代器（惰性求值），需要 list() 转成列表
#
# 【mset() 的作用】
# mset = "multi set"，批量存储键值对
# 格式要求：[(key1, value1), (key2, value2), ...]
# 存储后可以通过 key 取回对应的 value：
#   docstore.mget(["id-001"]) → [DocA]
#
# 【整体流程】
# 1. zip() 把 ID 和文档配对
# 2. list() 转成列表
# 3. mset() 批量存入文档库
# 之后检索时：通过摘要的 metadata["doc_id"] 找到 ID，再用 ID 取出原文档
retriever.docstore.mset(list(zip(doc_ids, docs_chunks)))

# ============ 测试检索 ============
query = "chapter on philosophy"

# 直接在向量库搜索（返回摘要）
sub_docs = retriever.vectorstore.similarity_search(query, k=2)
print("\n摘要搜索结果:")
print(sub_docs[0].page_content[:100] + "...")
print(f"摘要长度: {len(sub_docs[0].page_content)}")

# 用检索器搜索（返回原文档）
retrieved_docs = retriever.invoke(query)
print("\n原文档检索结果:")
print(f"原文档长度: {len(retrieved_docs[0].page_content)}")
# 返回的是完整的原文档，不是摘要
