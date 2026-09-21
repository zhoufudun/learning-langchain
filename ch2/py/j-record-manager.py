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
- SQLRecordManager: 记录哪些文档已经索引过
- index() 函数: 智能判断增删改

【cleanup 参数】
- "incremental": 增量更新，保留旧的，只添加新的/更新变化的
- "full": 全量更新，删除所有旧的，插入所有新的
"""

# ============ 导入 ============
from langchain.indexes import SQLRecordManager, index
from langchain_postgres.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document

# ============ 配置 ============
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
collection_name = "my_docs"       # 集合名称
namespace = "my_docs_namespace"   # 命名空间（用于记录管理）
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# ============ 创建向量库 ============
vectorstore = PGVector(
    embeddings=embeddings_model,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,  # 使用 JSONB 存储元数据
)

# ============ 创建记录管理器 ============
# 记录管理器追踪哪些文档已经索引过
record_manager = SQLRecordManager(
    namespace,  # 命名空间，用于隔离不同的索引
    db_url="postgresql+psycopg://langchain:langchain@localhost:6024/langchain",
)

# 首次运行需要创建表结构
record_manager.create_schema()

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
index_1 = index(
    docs,                      # 要索引的文档
    record_manager,            # 记录管理器
    vectorstore,               # 向量库
    cleanup="incremental",     # 增量模式
    source_id_key="source",    # 用 source 字段作为文档唯一标识
)
print("第一次索引:", index_1)
# 输出: {'num_added': 2, 'num_updated': 0, 'num_deleted': 0, 'num_skipped': 0}

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
docs[0].page_content = "I just modified this document!"

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
