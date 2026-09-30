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
from langchain_openai import OpenAIEmbeddings

# ============ 第一步：加载文档 ============
loader = TextLoader("./test.txt", encoding="utf-8")
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
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# embed_documents() 需要字符串列表
# [chunk.page_content for chunk in chunks] 提取每个块的文本内容
embeddings = embeddings_model.embed_documents(
    [chunk.page_content for chunk in chunks]
)

# ============ 查看结果 ============
print(f"生成了 {len(embeddings)} 个向量")
print(f"向量维度: {len(embeddings[0])}")

# 接下来可以把这些向量存入向量数据库（见下一个文件）
