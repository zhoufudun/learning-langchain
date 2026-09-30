"""
Ch2-i: 向量数据库（PGVector）

【学习目标】
- 了解向量数据库的作用
- 学会用 PGVector 存储和检索向量

【前置准备】
1. 安装 Docker: https://docs.docker.com/get-docker/
2. 安装依赖: pip install -qU langchain_postgres
3. 启动 PostgreSQL 容器:

docker run \\
    --name pgvector-container \\
    -e POSTGRES_USER=langchain \\
    -e POSTGRES_PASSWORD=langchain \\
    -e POSTGRES_DB=langchain \\
    -p 6024:5432 \\
    -d pgvector/pgvector:pg16

【向量数据库的作用】
- 存储文本的向量表示
- 快速进行相似度搜索（找最相近的文本）
- 是 RAG 的核心组件

【Python 语法】
- uuid.uuid4(): 生成全局唯一 ID
- str(): 把对象转换成字符串
"""
from dotenv import load_dotenv
load_dotenv()


# ============ 导入 ============
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector
from langchain_core.documents import Document
import uuid

# ============ 数据库连接 ============
# 连接字符串格式: postgresql+psycopg://用户名:密码@主机:端口/数据库名
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"

# ============ 第一步：加载并分割文档 ============
raw_documents = TextLoader('./test.txt', encoding="utf-8").load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(raw_documents)

# ============ 第二步：创建嵌入模型 ============
embeddings_model = OpenAIEmbeddings()

# ============ 第三步：存入向量数据库 ============
# from_documents() 会自动:
# 1. 生成每个文档的向量
# 2. 存入 PostgreSQL 数据库
db = PGVector.from_documents(
    documents,           # 文档列表
    embeddings_model,    # 嵌入模型
    connection=connection  # 数据库连接
)

# ============ 第四步：相似度搜索 ============
# similarity_search() 找出与查询最相似的 k 个文档
results = db.similarity_search("query", k=4)  # k=4 表示返回 4 个结果
print("搜索结果:", results)

# ============ 添加新文档 ============
print("\n--- 添加新文档 ---")
# 为新文档生成唯一 ID
ids = [str(uuid.uuid4()), str(uuid.uuid4())]

# add_documents() 添加新文档到数据库
db.add_documents(
    [
        Document(
            page_content="there are cats in the pond",  # 文档内容
            metadata={"location": "pond", "topic": "animals"},  # 元数据
        ),
        Document(
            page_content="ducks are also found in the pond",
            metadata={"location": "pond", "topic": "animals"},
        ),
    ],
    ids=ids,  # 指定文档 ID
)

print("文档添加成功，获取到的文档数:", len(db.get_by_ids(ids)))

# ============ 删除文档 ============
print("\n--- 删除文档 ---")
db.delete({"ids": ids})  # 按 ID 删除
print("文档删除成功，获取到的文档数:", len(db.get_by_ids(ids)))
