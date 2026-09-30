"""
Ch2-e: 代码分割器（按编程语言语法分割）

【学习目标】
- 了解如何按编程语言的语法结构分割代码
- 学会使用 from_language() 方法

【为什么代码需要特殊分割？】
- 普通分割可能把函数切成两半
- 代码分割器会按函数、类等语法边界分割
- 保持代码块的完整性

【支持的语言】
Language.PYTHON, Language.JS, Language.JAVA, Language.GO,
Language.MARKDOWN, Language.HTML, Language.CSS 等
"""
from dotenv import load_dotenv
load_dotenv()


# ============ 导入 ============
from langchain_text_splitters import (
    Language,                         # 语言枚举
    RecursiveCharacterTextSplitter,
)

# ============ 准备代码文本 ============
PYTHON_CODE = """ def hello_world(): print("Hello, World!") # Call the function hello_world() """

# ============ 创建代码分割器 ============
# from_language() 根据编程语言创建分割器
# 它知道 Python 的函数定义、类定义等语法结构
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,  # 指定语言
    chunk_size=50,             # 每块最大字符数
    chunk_overlap=0            # 不重叠
)

# ============ 分割代码 ============
# create_documents() 从字符串列表创建 Document 列表
# 与 split_documents() 不同：
#   - split_documents(): 输入是 Document 列表
#   - create_documents(): 输入是字符串列表
python_docs = python_splitter.create_documents([PYTHON_CODE])

# ============ 查看结果 ============
print(python_docs)
# 分割器会尽量在函数边界分割，而不是在函数中间切断
