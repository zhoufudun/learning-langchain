"""
Ch2-b: 网页加载器

【学习目标】
- 学会从网页 URL 加载内容
- 了解 WebBaseLoader 的用法

【依赖安装】
pip install beautifulsoup4

【LangChain 概念】
- WebBaseLoader: 从网页 URL 加载内容
- 底层使用 BeautifulSoup 解析 HTML
- 会自动提取网页的文本内容，去掉 HTML 标签
"""
from dotenv import load_dotenv
load_dotenv()


# ============ 导入 ============
from langchain_community.document_loaders import WebBaseLoader

# ============ 创建加载器 ============
# 传入网页 URL
base_loader = WebBaseLoader('https://www.langchain.com/')

# ============ 加载文档 ============
# load() 会:
# 1. 请求这个 URL
# 2. 解析 HTML
# 3. 提取文本内容
# 4. 返回Document列表
# docs = base_loader.docs ()
docs  = base_loader.lazy_load()

# ============ 查看结果 ============
# docs[0].page_content 是网页的文本内容
# docs[0].metadata 包含 source（URL）等信息
for doc in docs:
    print(doc.metadata)
    print(doc.page_content)
