"""
Ch1-i: 输出解析器（把字符串解析成列表）

【学习目标】
- 了解输出解析器的作用
- CommaSeparatedListOutputParser: 把逗号分隔的字符串变成 Python 列表

【Python 语法】
- 字符串 "a, b, c" 是纯文本
- 列表 ["a", "b", "c"] 是 Python 数据结构，可以遍历、索引

【使用场景】
- LLM 返回 "apple, banana, cherry"
- 解析器把它变成 ["apple", "banana", "cherry"]
- 这样你就可以用 for 循环处理每个元素了
"""

# ============ 导入 ============
from dotenv import load_dotenv
load_dotenv()

# CommaSeparatedListOutputParser: 逗号分隔列表解析器
from langchain_core.output_parsers import CommaSeparatedListOutputParser

# ============ 创建解析器 ============
parser = CommaSeparatedListOutputParser()

# ============ 解析字符串 ============
# invoke() 把逗号分隔的字符串解析成 Python 列表
response = parser.invoke("apple, banana, cherry")

print(response)
# 输出: ['apple', 'banana', 'cherry']
# 注意：输出是列表，不是字符串了

# 现在可以这样使用:
# for fruit in response:
#     print(fruit)
