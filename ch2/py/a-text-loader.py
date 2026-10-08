"""
Ch2-a: 文本文件加载器

【学习目标】
- 了解 Document Loader 的概念
- 学会从本地文本文件加载内容

【LangChain 概念】
- Document: LangChain 中的文档对象，包含 page_content（内容）和 metadata（元数据）
- Loader: 加载器，从各种来源（文件、网页、数据库）加载文档
- TextLoader: 专门加载纯文本文件的加载器

【Python 语法】
- './test.txt': 相对路径，表示当前目录下的 test.txt 文件
- encoding="utf-8": 指定文件编码，中文文件一般用 utf-8
"""
from dotenv import load_dotenv
load_dotenv()

# ============ 导入 ============
# TextLoader 来自 langchain_community 包
from langchain_community.document_loaders import TextLoader

# ============ 创建加载器 ============
# TextLoader 需要指定文件路径
# encoding="utf-8" 确保能正确读取中文
text_loader = TextLoader('../.././test.txt', encoding="utf-8")

# ============ 加载文档 ============
# load() 方法返回 Document 对象的列表
# 即使只有一个文件，也返回列表
docsList = text_loader.load()

# ============ 查看结果 ============
# docs 是列表，每个元素是 Document 对象
# Document 有两个属性:
#   - page_content: 文档内容（字符串）
#   - metadata: 元数据（字典），如文件路径、来源等
for doc in docsList:
    print(doc.metadata)
    print(doc.page_content)
# 输出类似: [Document(page_content='文件内容...', metadata={'source': './test.txt'})]
