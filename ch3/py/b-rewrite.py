from dotenv import load_dotenv
load_dotenv()
"""
Ch3-b: 查询重写（Query Rewriting）

【学习目标】
- 理解为什么需要查询重写
- 学会用 LLM 优化用户查询

【问题场景】
用户的查询可能包含无关信息，影响检索效果。
例如: "今天早上我刷了牙，然后看新闻，对了，古希腊哲学家有哪些？"
这种查询直接检索效果不好，需要提取核心问题。

【解决方案】
用 LLM 重写查询，提取核心问题，提高检索精度。

【Python 语法】
- strip(): 去除字符串两端的空白字符
- strip("**"): 去除字符串两端的 ** 符号
"""

# ============ 导入 ============
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain

# ============ 准备知识库和检索器 ============
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"

raw_documents = TextLoader('./test.txt', encoding='utf-8').load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(raw_documents)

embeddings_model = OpenAIEmbeddings()
db = PGVector.from_documents(
    documents, embeddings_model, connection=connection)

retriever = db.as_retriever(search_kwargs={"k": 2})

# ============ 测试有噪音的查询 ============
# 这个查询包含很多无关信息
query = 'Today I woke up and brushed my teeth, then I sat down to read the news. But then I forgot the food on the cooker. Who are some key figures in the ancient greek history of philosophy?'

# 直接检索（可能效果不好）
docs = retriever.invoke(query)
print("直接检索结果:")
print(docs[0].page_content[:200] + "...")

# ============ 基础 QA（不重写）============
prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context: {context} Question: {question} """
)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


@chain
def qa(input):
    docs = retriever.invoke(input)
    formatted = prompt.invoke({"context": docs, "question": input})
    answer = llm.invoke(formatted)
    return answer


result = qa.invoke(query)
print("
不重写的回答:", result.content)

print("
" + "="*50)
print("使用查询重写")
print("="*50 + "
")

# ============ 查询重写 ============
# 重写提示词：让 LLM 提取核心问题
rewrite_prompt = ChatPromptTemplate.from_template(
    """Provide a better search query for web search engine to answer the given question, end the queries with '**'. Question: {x} Answer:""")


def parse_rewriter_output(message):
    """
    解析重写结果，去掉引号和 ** 标记

    【Python 语法】
    - message.content: 获取 AIMessage 的文本内容
    - strip('"'): 去掉两端的引号
    - strip('**'): 去掉两端的 **
    """
    return message.content.strip('"').strip("**")


# 重写链: 提示词 → LLM → 解析
rewriter = rewrite_prompt | llm | parse_rewriter_output


# ============ 带重写的 QA ============
@chain
def qa_rrr(input):
    """
    RRR = Rewrite-Retrieve-Read
    先重写查询，再检索，最后生成回答
    """
    # 1. 重写查询
    new_query = rewriter.invoke(input)
    print("重写后的查询:", new_query)

    # 2. 用重写后的查询检索
    docs = retriever.invoke(new_query)

    # 3. 生成回答（注意：回答的问题还是原问题）
    formatted = prompt.invoke({"context": docs, "question": input})
    answer = llm.invoke(formatted)
    return answer


# 测试重写效果
print("使用重写后的回答:")
result = qa_rrr.invoke(query)
print(result.content)
