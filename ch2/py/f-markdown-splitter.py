"""
Ch2-f: Markdown 分割器

【学习目标】
- 了解如何按 Markdown 的标题、段落分割
- 学会给文档添加元数据

【Markdown 分割特点】
- 按标题（#、##、###）分割
- 按代码块（```）分割
- 保持文档结构的完整性
"""
from dotenv import load_dotenv
load_dotenv()


# ============ 导入 ============
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)

# ============ 准备 Markdown 文本 ============
markdown_text = """ # 🦜🔗 LangChain ⚡ Building applications with LLMs through composability ⚡ ## Quick Install ```bash pip install langchain ``` As an open source project in a rapidly developing field, we are extremely open     to contributions. """

# ============ 创建 Markdown 分割器 ============
md_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,  # 指定 Markdown 语言
    chunk_size=60,               # 每块最大字符数
    chunk_overlap=0
)

# ============ 分割并添加元数据 ============
# create_documents() 的第二个参数可以传入元数据列表
# 元数据会附加到每个生成的 Document 上
md_docs = md_splitter.create_documents(
    [markdown_text],                           # 第一个参数: 文本列表
    [{"source": "https://www.langchain.com"}]  # 第二个参数: 对应的元数据列表
)

# ============ 查看结果 ============
print(md_docs)
# 每个 Document 的 metadata 都会包含 source 字段
