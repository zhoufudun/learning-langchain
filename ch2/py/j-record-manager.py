"""
Ch2-j: 记录管理器（防止重复索引）

【学习目标】
- 了解为什么需要记录管理器
- 学会用 index() 函数增量更新向量库

【问题场景】
- 每次运行都重新索引所有文档？太浪费
- 文档更新了，如何只更新变化的部分？
- 如何避免重复插入相同的文档？

【解决方案】
- RecordManager: 记录哪些文档已经索引过
- index() 函数: 智能判断增删改

【cleanup 参数】
- "incremental": 增量更新，保留旧的，只添加新的/更新变化的
- "full": 全量更新，删除所有旧的，插入所有新的

【注意】
新版 LangChain 中 SQLRecordManager 已移除
这里使用 InMemoryRecordManager 演示（数据存在内存，重启后丢失）
生产环境需要使用持久化的 RecordManager
"""
from dotenv import load_dotenv
load_dotenv()


# ============ 导入 ============
from langchain_core.indexing import InMemoryRecordManager, index
from langchain_postgres.vectorstores import PGVector
# DeepSeek 没有 Embedding API，改用智谱 AI
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_core.documents import Document


# ============ 分批嵌入的包装类 ============
# 智谱 API 限制单次最多 64 条，需要自动分批
class BatchingZhipuEmbeddings(ZhipuAIEmbeddings):
    """智谱嵌入，自动分批（单批最多 64 条）"""
    batch_size: int = 64

    def embed_documents(self, texts:list[str]):
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            all_embeddings.extend(super().embed_documents(batch))
        return all_embeddings


# ============ 配置 ============
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
collection_name = "my_docs"       # 集合名称
namespace = "my_docs_namespace"   # 命名空间（用于记录管理）
zhipu_embeddings_model = BatchingZhipuEmbeddings(model="embedding-3")

# ============ 创建向量库 ============
# 【JSONB 说明】
# use_jsonb=True 让元数据以 JSONB 格式存储在 PostgreSQL 中
# JSONB 是 PostgreSQL 的二进制 JSON 类型，优点：
# - 灵活：每个文档的 metadata 可以有不同字段
# - 可查询：支持 SQL 查询元数据，如 WHERE metadata->>'source' = 'cats.txt'
# - 高效：二进制格式，比纯文本 JSON 查询更快
vectorstore = PGVector(
    embeddings=zhipu_embeddings_model,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,  # 使用 JSONB 存储元数据（推荐）
)

# ============ 创建记录管理器 ============
# InMemoryRecordManager: 内存中记录索引状态
# namespace: 命名空间，用于隔离不同的索引
record_manager = InMemoryRecordManager(namespace=namespace)
# ============ 准备文档 ============
docs = [
    Document(
        page_content='there are cats in the pond',
        metadata={"id": 1, "source": "cats.txt"}
    ),
    Document(
        page_content='ducks are also found in the pond',
        metadata={"id": 2, "source": "ducks.txt"}
    ),
]

# ============ 第一次索引 ============
# 【index() 函数详解】
# index() 是 LangChain 的智能索引函数，会自动判断：
# - 新文档 → 添加到向量库
# - 已存在且未变 → 跳过
# - 已存在但内容变了 → 删除旧的，添加新的
# - 向量库有但本次没传 → 根据 cleanup 模式决定是否删除
#
# 【参数说明】
# - docs: 要索引的文档列表
# - record_manager: 记录管理器，追踪哪些文档已索引
# - vectorstore: 向量数据库，存储文档向量
# - cleanup: 清理模式
#   - "incremental": 增量模式，只处理本次传入的文档，不影响其他文档
#   - "full": 全量模式，删除所有不在本次列表中的旧文档
#   - None: 不清理，只添加新的
# - source_id_key: 指定用 metadata 中的哪个字段作为文档唯一标识
#   这里用 "source"，即 metadata={"source": "cats.txt"} 中的 source 值
#
# 【返回值】
# 返回字典，记录本次操作的统计：
# {'num_added': 添加数, 'num_updated': 更新数, 'num_deleted': 删除数, 'num_skipped': 跳过数}
index_1 = index(
    docs,                      # 要索引的文档列表
    record_manager,            # 记录管理器（追踪索引状态）
    vectorstore,               # 向量库（存储向量）
    cleanup="incremental",     # 增量模式（只处理变化的）
    source_id_key="source",    # 用 metadata["source"] 作为唯一标识
)

print("第一次索引:", index_1)
# 输出: {'num_added': 2, 'num_updated': 0, 'num_deleted': 0, 'num_skipped': 0}
# 解读: 2 个新文档被添加

# ============ 第二次索引（相同文档）============
# 因为文档没变，所以会跳过
index_2 = index(
    docs,
    record_manager,
    vectorstore,
    cleanup="incremental",
    source_id_key="source",
)
print("第二次索引:", index_2)
# 输出: {'num_added': 0, 'num_updated': 0, 'num_deleted': 0, 'num_skipped': 2}

# ============ 修改文档后再索引 ============
docs[1].page_content = "I just modified this document!"

index_3 = index(
    docs,
    record_manager,
    vectorstore,
    cleanup="incremental",
    source_id_key="source",
)
print("第三次索引:", index_3)
# 输出: {'num_added': 1, 'num_updated': 0, 'num_deleted': 1, 'num_skipped': 1}
# 旧版本被删除，新版本被添加
