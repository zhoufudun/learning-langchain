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
from langchain.chains import create_sql_query_chain        # 创建 SQL 生成链
from langchain_openai import ChatOpenAI

# ============ 连接数据库 ============
# from_uri() 从连接字符串创建数据库对象
# sqlite:///Chinook.db 表示当前目录下的 Chinook.db 文件
db = SQLDatabase.from_uri("sqlite:///Chinook.db")

# 查看数据库有哪些表
print("数据库表:", db.get_usable_table_names())
# 输出: ['Album', 'Artist', 'Customer', 'Employee', 'Genre', 'Invoice', ...]

# ============ 创建 SQL 生成链 ============
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# create_sql_query_chain() 创建一个链:
# 输入: 自然语言问题
# 输出: SQL 查询语句
write_query = create_sql_query_chain(llm, db)

# ============ 创建 SQL 执行工具 ============
# QuerySQLDatabaseTool 用于执行 SQL 并返回结果
execute_query = QuerySQLDatabaseTool(db=db)

# ============ 组合成完整链 ============
# 1. write_query: 把问题转换成 SQL
# 2. execute_query: 执行 SQL 并返回结果
combined_chain = write_query | execute_query

# ============ 测试 ============
result = combined_chain.invoke({"question": "How many employees are there?"})
print("查询结果:", result)
# 输出: "8" 或类似的数字

# 你还可以单独看生成的 SQL:
# sql = write_query.invoke({"question": "How many employees are there?"})
# print("生成的 SQL:", sql)
