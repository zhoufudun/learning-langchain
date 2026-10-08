"""
Ch2-h: 完整流程：加载 → 分割 → 嵌入

【学习目标】
- 掌握文档处理的完整流程
- 这是 RAG（检索增强生成）的前置步骤

【流程图】
文件 → TextLoader → 文档 → TextSplitter → 小块 → Embeddings → 向量

【Python 语法】
- 列表推导式: [chunk.page_content for chunk in chunks]
  等价于:
  result = []
  for chunk in chunks:
      result.append(chunk.page_content)
"""
from dotenv import load_dotenv
load_dotenv()


# ============ 导入 ============
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 智谱 AI 的 Embedding 模型
from langchain_community.embeddings import ZhipuAIEmbeddings

# ============ 第一步：加载文档 ============
loader = TextLoader('../.././test.txt', encoding="utf-8")
doc = loader.load()  # 返回 [Document(...)]
print(f"加载了 {len(doc)} 个文档")

# ============ 第二步：分割文档 ============
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,    # 每块最多 1000 字符
    chunk_overlap=200   # 重叠 200 字符
)
chunks = splitter.split_documents(doc)
print(f"分割成 {len(chunks)} 个小块")

# ============ 第三步：生成嵌入向量 ============
# 智谱 Embedding 模型
# 需要设置环境变量 ZHIPUAI_API_KEY，或在这里传入 api_key 参数
# 模型选择: embedding-2 或 embedding-3
ai_embeddings = ZhipuAIEmbeddings(model="embedding-3")

# embed_documents() 需要字符串列表
# [chunk.page_content for chunk in chunks] 提取每个块的文本内容
texts = [chunk.page_content for chunk in chunks]

# texts = []
# for chunk in chunks:
#     texts.extend(chunk.page_content)

# 智谱 API 限制：每次最多 64 条，需要分批处理
BATCH_SIZE = 64
embeddings = []

# range(0, 200, 64)  →  生成: 0, 64, 128, 192
# # 循环过程：
# i = 0    → texts[0:64]    # 第 1-64 条
# i = 64   → texts[64:128]  # 第 65-128 条
# i = 128  → texts[128:192] # 第 129-192 条
# i = 192  → texts[192:256] # 第 193-200 条（不足64条也没关系）

for i in range(0, len(texts), BATCH_SIZE):
    # texts[i:i+BATCH_SIZE] 切片，取第 i 到 i+64 条
    batch = texts[i:i+BATCH_SIZE]
    batch_embeddings = ai_embeddings.embed_documents(batch)
    embeddings.extend(batch_embeddings)  # extend() 把列表元素逐个添加
    print(f"  已处理 {min(i+BATCH_SIZE, len(texts))}/{len(texts)} 条")

print(f"生成了 {len(embeddings)} 个向量")
print(f"向量维度: {len(embeddings[0])}")

query_text = "hello"
query_embedding = ai_embeddings.embed_query(query_text)
print(f"\n查询文本: \"{query_text}\"")
print(f"查询向量维度: {len(query_embedding)}")

# 接下来可以把这些向量存入向量数据库（见下一个文件）
