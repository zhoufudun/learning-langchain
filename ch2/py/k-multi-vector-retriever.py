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
from dotenv import load_dotenv
load_dotenv()


# ============ 导入 ============
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_postgres.vectorstores import PGVector
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
import uuid

# ============ 配置 ============
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
collection_name = "summaries"
embeddings_model = OpenAIEmbeddings()

# ============ 加载并分割文档 ============
loader = TextLoader("./test.txt", encoding="utf-8")
docs = loader.load()
print("原文档长度:", len(docs[0].page_content))

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)
print(f"分割成 {len(chunks)} 个块")

# ============ 创建摘要生成链 ============
prompt_text = "Summarize the following document:\n\n{doc}"
prompt = ChatPromptTemplate.from_template(prompt_text)
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

# 链: 提取内容 → 生成提示词 → 调用 LLM → 解析为字符串
summarize_chain = {
    "doc": lambda x: x.page_content  # lambda 匿名函数，提取文档内容
} | prompt | llm | StrOutputParser()

# ============ 批量生成摘要 ============
# batch() 批量处理，max_concurrency=5 表示最多 5 个并发
summaries = summarize_chain.batch(chunks, {"max_concurrency": 5})
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
store = InMemoryStore()  # 内存存储，生产环境应该用持久化存储

# ============ 创建多向量检索器 ============
id_key = "doc_id"  # 用于关联摘要和原文档的字段名

retriever = MultiVectorRetriever(
    vectorstore=vectorstore,  # 摘要向量库
    docstore=store,           # 原文档库
    id_key=id_key,            # 关联字段
)

# ============ 关联摘要和原文档 ============
# 为每个块生成唯一 ID
doc_ids = [str(uuid.uuid4()) for _ in chunks]

# 创建摘要文档，通过 doc_id 关联到原文档
summary_docs = [
    Document(
        page_content=s,           # 摘要内容
        metadata={id_key: doc_ids[i]}  # 关联到原文档的 ID
    )
    for i, s in enumerate(summaries)  # enumerate 同时获取索引 i 和摘要 s
]

# 摘要存入向量库
retriever.vectorstore.add_documents(summary_docs)

# 原文档存入文档库
# zip(doc_ids, chunks) 把 ID 和文档配对: [(id1, chunk1), (id2, chunk2), ...]
# mset() 批量存储键值对
retriever.docstore.mset(list(zip(doc_ids, chunks)))

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
