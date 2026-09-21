"""
Ch2-c: PDF 文件加载器

【学习目标】
- 学会从 PDF 文件加载内容
- 了解 PDF 加载是按页拆分的

【依赖安装】
pip install pypdf

【LangChain 概念】
- PyPDFLoader: 从 PDF 文件加载内容
- 每一页会变成一个独立的 Document
- metadata 中会包含页码信息
"""

# ============ 导入 ============
from langchain_community.document_loaders import PyPDFLoader

# ============ 创建加载器 ============
# 传入 PDF 文件路径
loader = PyPDFLoader('./test.pdf')

# ============ 加载文档 ============
# load() 返回的列表，每个元素是 PDF 的一页
# pages[0] 是第一页，pages[1] 是第二页...
pages = loader.load()

# ============ 查看结果 ============
# 每个 Document 的 metadata 包含:
#   - source: 文件路径
#   - page: 页码（从 0 开始）
print(pages)
# 例如: [Document(page_content='第一页内容', metadata={'source': './test.pdf', 'page': 0}), ...]
