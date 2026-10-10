from dotenv import load_dotenv
load_dotenv()
"""
Ch3-h: 自查询检索（Self-Query Retrieval）

【学习目标】
- 理解自查询检索的原理
- 学会从自然语言中提取过滤条件

【什么是自查询检索？】
让 LLM 自动从用户问题中提取:
1. 语义查询（用于向量搜索）
2. 元数据过滤条件（用于精确过滤）

【例子】
用户问: "推荐一部评分高于 8.5 的科幻电影"
LLM 提取:
- 语义查询: "科幻电影"
- 过滤条件: rating > 8.5, genre = "science fiction"

【依赖安装】
pip install lark

【Python 语法】
- AttributeInfo: 描述可过滤的字段信息
- Document: LangChain 的文档对象，包含 page_content 和 metadata
"""

# ============ 导入 ============
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_openai import ChatOpenAI
# DeepSeek 没有 Embedding API，改用智谱 AI
from langchain_community.embeddings import ZhipuAIEmbeddings


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
from langchain_postgres.vectorstores import PGVector
from langchain_core.documents import Document

# ============ 准备数据 ============
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"

# 电影数据：每个文档有内容和元数据
docs = [
    Document(
        page_content="A bunch of scientists bring back dinosaurs and mayhem breaks loose",
        metadata={"year": 1993, "rating": 7.7, "genre": "science fiction"},
    ),
    Document(
        page_content="Leo DiCaprio gets lost in a dream within a dream within a dream within a ...",
        metadata={"year": 2010, "director": "Christopher Nolan", "rating": 8.2},
    ),
    Document(
        page_content="A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea",
        metadata={"year": 2006, "director": "Satoshi Kon", "rating": 8.6},
    ),
    Document(
        page_content="A bunch of normal-sized women are supremely wholesome and some men pine after them",
        metadata={"year": 2019, "director": "Greta Gerwig", "rating": 8.3},
    ),
    Document(
        page_content="Toys come alive and have a blast doing so",
        metadata={"year": 1995, "genre": "animated"},
    ),
    Document(
        page_content="Three men walk into the Zone, three men walk out of the Zone",
        metadata={
            "year": 1979,
            "director": "Andrei Tarkovsky",
            "genre": "thriller",
            "rating": 9.9,
        },
    ),
]

# 创建向量库
embeddings_model = BatchingZhipuEmbeddings(model="embedding-3")
vectorstore = PGVector.from_documents(
    docs, embeddings_model, connection=connection)

# ============ 定义可过滤的字段 ============
# 告诉 LLM 有哪些字段可以用于过滤
fields = [
    AttributeInfo(
        name="genre",           # 字段名
        description="The genre of the movie",  # 描述（帮助 LLM 理解）
        type="string or list[string]",  # 数据类型
    ),
    AttributeInfo(
        name="year",
        description="The year the movie was released",
        type="integer",
    ),
    AttributeInfo(
        name="director",
        description="The name of the movie director",
        type="string",
    ),
    AttributeInfo(
        name="rating",
        description="A 1-10 rating for the movie",
        type="float",
    ),
]

# 文档内容的描述
description = "Brief summary of a movie"

# ============ 创建自查询检索器 ============
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# from_llm() 创建自查询检索器
# LLM 会自动从用户查询中提取过滤条件
retriever = SelfQueryRetriever.from_llm(
    llm,           # 用于解析查询的 LLM
    vectorstore,   # 向量库
    description,   # 文档描述
    fields         # 可过滤的字段
)

# ============ 测试：只有过滤条件 ============
print("查询: 评分高于 8.5 的电影")
results = retriever.invoke("I want to watch a movie rated higher than 8.5")
for doc in results:
    print(f"  - {doc.page_content[:50]}... (rating: {doc.metadata.get('rating')})")

print()

# ============ 测试：语义查询 + 过滤条件 ============
print("查询: 评分高于 8.5 的科幻电影")
results = retriever.invoke("What's a highly rated (above 8.5) science fiction film?")
for doc in results:
    print(f"  - {doc.page_content[:50]}... (rating: {doc.metadata.get('rating')}, genre: {doc.metadata.get('genre')})")
