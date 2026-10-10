from dotenv import load_dotenv
load_dotenv()
"""
Ch3-a: 基础 RAG（检索增强生成）

【学习目标】
- 理解 RAG 的核心概念和流程
- 学会构建简单的问答系统

【什么是 RAG？】
- Retrieval-Augmented Generation（检索增强生成）
- 先从知识库检索相关文档，再让 LLM 基于文档回答问题
- 解决 LLM 知识过时、幻觉等问题

【RAG 流程图】
用户问题 → 检索器 → 相关文档 → 组装提示词 → LLM → 回答
                ↓
            向量数据库

【前置准备】
1. 启动 PostgreSQL: docker run --name pgvector-container -e POSTGRES_USER=langchain -e POSTGRES_PASSWORD=langchain -e POSTGRES_DB=langchain -p 6024:5432 -d pgvector/pgvector:pg16
2. pip install langchain_postgres
"""

# ============ 导入 ============
from langchain_community.document_loaders import TextLoader
# DeepSeek 没有 Embedding API，改用智谱 AI
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain


# ============ 分批嵌入的包装类 ============
class BatchingZhipuEmbeddings(ZhipuAIEmbeddings):
    """智谱嵌入，自动分批（单批最多 64 条）"""
    batch_size: int = 64

    def embed_documents(self, texts):
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            all_embeddings.extend(super().embed_documents(batch))
        return all_embeddings

# ============ 配置 ============
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"

# ============ 第一步：准备知识库 ============
# 加载文档
# 注意：用 load() 而不是 aload()，aload() 是异步方法
raw_documents = TextLoader('../../test2.txt', encoding='utf-8').load()

# 分割文档
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
split_documents = text_splitter.split_documents(raw_documents)

# 创建向量库
zhipu_embeddings = BatchingZhipuEmbeddings(model="embedding-3")

db_vector = PGVector.from_documents(documents=split_documents, embedding=zhipu_embeddings, connection=connection)

# ============ 第二步：创建检索器 ============
# as_retriever() 把向量库转换成检索器
# search_kwargs={"k": 2} 表示返回最相关的 2 个文档
as_retriever = db_vector.as_retriever(search_kwargs={"k": 2})

# ============ 第三步：测试检索 ============
query = 'Who are the key figures in the ancient greek history of philosophy?'

# invoke() 根据查询检索相关文档
docs = as_retriever.invoke(query)
print("检索到的文档内容:")
print(docs[0].page_content[:200] + "...")

# ============ 第四步：创建问答链 ============
# 提示词模板：要求 LLM 只根据上下文回答
prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context: {context} Question: {question} """
)

llm = ChatOpenAI(model_name="claude-sonnet-4-6", temperature=0)

# 用 | 连接提示词和模型
llm_chain = prompt | llm

# ============ 第五步：生成回答 ============
# 把检索到的文档和问题传给 LLM
result = llm_chain.invoke({"question": query, "context": docs})

print("\n回答:", result.content)

print("\n" + "="*50)
print("使用 @chain 封装完整流程")
print("="*50 + "\n")


# ============ 封装成完整的 QA 链 ============
@chain
def chainQuery(queryQuestion: str):
    """
        完整的 RAG 流程封装

        @chain 装饰器让这个函数变成 LangChain 的 Runnable
        可以使用 invoke()、batch()、stream() 等方法
        """
    # 1. 检索相关文档
    retriever_result = as_retriever.invoke(queryQuestion)
    # 2. 组装提示词
    formatted = prompt.invoke({"question": queryQuestion, "context": retriever_result})
    # 3. 调用 LLM 生成回答
    answer = llm.invoke(formatted)
    return answer

# 使用封装后的 QA 链
result = chainQuery.invoke(query)
print("封装后的回答:", result.content)
