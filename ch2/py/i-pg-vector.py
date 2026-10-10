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
from typing import List

from dotenv import load_dotenv
load_dotenv()


# ============ 导入 ============
from langchain_community.document_loaders import TextLoader
# DeepSeek 没有 Embedding API，改用智谱 AI
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector
from langchain_core.documents import Document
import uuid

# 继承ZhipuAIEmbeddings
class BatchZhipuEmbeddings(ZhipuAIEmbeddings):
    """
    自动分批（单批最多 64 条）
    """
    # 类属性：可在创建对象时覆盖，如 BatchingZhipuEmbeddings(batch_size=32)
    batch_size: int = 64

    def embed_documents(self, texts: list[str]):
        """重写父类方法：把 texts 拆成小批，逐批调用 API"""
        all_embeddings = []
        for i in range(0,len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            all_embeddings.extend(super().embed_documents(batch))
        return all_embeddings

# ============ 数据库连接 ============
# 连接字符串格式: postgresql+psycopg://用户名:密码@主机:端口/数据库名
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"

# ============ 第一步：加载并分割文档 ============
raw_documents = TextLoader('../../test.txt', encoding="utf-8").load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
documents = text_splitter.split_documents(raw_documents)

# ============ 第二步：创建嵌入模型 ============
# 用我们自定义的类，自动分批（避免智谱 64 条限制）
zhipu_embeddings_model = BatchZhipuEmbeddings(model="embedding-3") # 构造参数

# ============ 第三步：存入向量数据库 ============
# from_documents() 会自动:
# 1. 生成每个文档的向量
# 2. 存入 PostgreSQL 数据库
db_pgvector = PGVector.from_documents(
    documents=documents,  # 文档列表
    embedding=zhipu_embeddings_model,  # 嵌入模型
    connection=connection  # 数据库连接
)

# ============ 第四步：相似度搜索 ============
# similarity_search() 找出与查询最相似的 k 个文档
similarity_search = db_pgvector.similarity_search("query", k=4)

# k=4 表示返回 4 个结果
print(f"搜索结果:{similarity_search}")

# ============ 添加新文档 ============
print("\n--- 添加新文档 ---")
# 为新文档生成唯一 ID
ids = [str(uuid.uuid4()), str(uuid.uuid4())]

# add_documents() 添加新文档到数据库
add_documents = db_pgvector.add_documents(
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

print("文档添加成功，获取到的文档数:", len(db_pgvector.get_by_ids(ids)))

# ============ 删除文档 ============
print("\n--- 删除文档 ---")
db_pgvector.delete({"ids": ids})  # 按 ID 删除
print("文档删除成功，获取到的文档数:", len(db_pgvector.get_by_ids(ids)))
