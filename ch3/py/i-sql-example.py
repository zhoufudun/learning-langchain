from dotenv import load_dotenv
load_dotenv()
"""
Ch3-i: SQL 查询（Text-to-SQL）

【学习目标】
- 理解如何用 LLM 将自然语言转换为 SQL
- 学会使用 LangChain 的 SQL 工具

【Text-to-SQL 流程】
用户问题 → LLM 生成 SQL → 执行 SQL → 返回结果

【前置准备】
下载并创建 Chinook 数据库（音乐商店示例数据库）:

```bash
curl -s https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql | sqlite3 Chinook.db
```

把 Chinook.db 放在代码同目录下。

【安全警告】
让 LLM 直接执行 SQL 有安全风险！
生产环境需要:
- 限制可执行的 SQL 类型（只读）
- 使用权限受限的数据库用户
- 对生成的 SQL 进行审核
"""

# ============ 导入 ============
from langchain_community.tools import QuerySQLDatabaseTool  # 执行 SQL 的工具
from langchain_community.utilities import SQLDatabase       # 数据库连接
# 直接从定义所在模块导入；PyCharm 可以更容易跳转到 query.py 中的源码。
from langchain_classic.chains.sql_database.query import create_sql_query_chain # 创建 SQL 生成链
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate

# ============ 连接数据库 ============
# from_uri() 从连接字符串创建数据库对象
# sqlite:///Chinook.db 表示当前目录下的 Chinook.db 文件
db = SQLDatabase.from_uri("sqlite:///Chinook.db")

# 查看数据库有哪些表
print("数据库表:", db.get_usable_table_names())
# 输出: ['Album', 'Artist', 'Customer', 'Employee', 'Genre', 'Invoice', ...]

# ============ 创建 SQL 生成链 ============
llm = ChatOpenAI(model="gpt-5.6-luna", temperature=0)

# create_sql_query_chain() 创建一个链:
# 输入: 自然语言问题
# 输出: SQL 查询语句

# create_sql_query_chain 会自动填入三个必需变量：
# - {input}: 用户的自然语言问题
# - {table_info}: 数据库表、字段等结构信息，LLM 据此写 SQL
# - {top_k}: 最多返回多少条记录
# 自定义模板必须保留它们，否则链不知道把这些信息放在哪里。
template = ChatPromptTemplate.from_template(
    """You are a SQLite expert. Return only one valid SQL query.
Do not add explanations, labels such as SQLQuery:, Markdown code fences, or SQLResult.

Database schema:
{table_info}

User question:
{input}

Return at most {top_k} rows when the question requests rows.
"""
)
text_to_sql = create_sql_query_chain(llm, db, prompt=template)

# ============ 创建 SQL 执行工具 ============
# QuerySQLDatabaseTool 用于执行 SQL 并返回结果
execute_query = QuerySQLDatabaseTool(db=db)

# ============ 组合成完整链 ============
# 1. write_query: 把问题转换成 SQL
# 2. clean_sql: 去掉 SQLQuery: 这类说明文字，保留纯 SQL
# 3. execute_query: 执行 SQL 并返回结果
combined_chain = text_to_sql | execute_query


# ============ 测试 ============
invoke = combined_chain.invoke({"question": "How many employees are there?"})
print("查询结果:", invoke)
# 输出: "8" 或类似的数字

# 你还可以单独看生成的 SQL:
# sql = write_query.invoke({"question": "How many employees are there?"})
# print("生成的 SQL:", sql)
