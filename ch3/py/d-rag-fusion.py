from dotenv import load_dotenv
load_dotenv()
"""
Ch3-d: RAG Fusion（融合排序）

【学习目标】
- 理解 RAG Fusion 的原理
- 学会使用 RRF（倒数排名融合）算法

【什么是 RAG Fusion？】
多查询检索 + 融合排序
- 生成多个查询
- 每个查询独立检索
- 用 RRF 算法合并排序结果

【RRF 算法（Reciprocal Rank Fusion）】
- 每个文档在每次检索中有一个排名
- RRF 分数 = Σ(1 / (k + rank))
- k 是平滑参数（默认 60）
- 多次出现在前排的文档得分更高

【Python 语法】
- enumerate(list): 遍历时同时获取索引和元素
- sorted(dict, key=..., reverse=True): 按值降序排序
- lambda d: scores[d]: 匿名函数，用于排序的 key
"""

# ============ 导入 ============
from langchain_community.document_loaders import TextLoader
# DeepSeek 没有 Embedding API，改用智谱 AI
from langchain_community.embeddings import ZhipuAIEmbeddings


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

embeddings_model = BatchingZhipuEmbeddings(model="embedding-3")
db = PGVector.from_documents(
    documents, embeddings_model, connection=connection)

retriever = db.as_retriever(search_kwargs={"k": 5})

# ============ 查询生成 ============
prompt_rag_fusion = ChatPromptTemplate.from_template(
    """You are a helpful assistant that generates multiple search queries based on a single input query. 
 Generate multiple search queries related to: {question} 
 Output (4 queries):""")


def parse_queries_output(message):
    return message.content.split('
')


llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
query_gen = prompt_rag_fusion | llm | parse_queries_output

# 测试查询生成
query = "Who are the key figures in the ancient greek history of philosophy?"
generated_queries = query_gen.invoke(query)
print("生成的查询:", generated_queries)


# ============ RRF 融合排序算法 ============
def reciprocal_rank_fusion(results: list[list], k=60):
    """
    倒数排名融合算法

    参数:
        results: 多次检索的结果列表 [[docs1], [docs2], ...]
        k: 平滑参数，默认 60

    返回:
        按 RRF 分数排序的文档列表

    【算法原理】
    - 文档在排名 1 → 得分 1/(60+1) = 0.0164
    - 文档在排名 2 → 得分 1/(60+2) = 0.0161
    - 多次出现 → 分数累加
    - 越靠前、出现越多，分数越高
    """
    # 存储每个文档的 RRF 分数
    fused_scores = {}
    # 存储文档对象（用于最后返回）
    documents = {}

    for docs in results:
        # enumerate 同时获取排名（从 0 开始）和文档
        for rank, doc in enumerate(docs):
            doc_str = doc.page_content  # 用内容作为唯一标识

            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
                documents[doc_str] = doc

            # RRF 公式: 1 / (k + rank)
            fused_scores[doc_str] += 1 / (rank + k)

    # 按分数降序排序
    # sorted() 的 key 参数指定排序依据
    # reverse=True 表示降序
    reranked_doc_strs = sorted(
        fused_scores,                      # 要排序的字典键
        key=lambda d: fused_scores[d],     # 按分数排序
        reverse=True                       # 降序
    )

    # 返回排序后的文档对象列表
    return [documents[doc_str] for doc_str in reranked_doc_strs]


# ============ 完整检索链 ============
retrieval_chain = query_gen | retriever.batch | reciprocal_rank_fusion

result = retrieval_chain.invoke(query)
print("
融合排序后的第一个文档:")
print(result[0].page_content[:200] + "...")

# ============ RAG Fusion QA ============
prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context: {context} Question: {question} """
)

query = "Who are the some important yet not well known philosophers in the ancient greek history of philosophy?"


@chain
def rag_fusion(input):
    """RAG Fusion: 多查询 + RRF 融合排序"""
    docs = retrieval_chain.invoke(input)
    formatted = prompt.invoke({"context": docs, "question": input})
    answer = llm.invoke(formatted)
    return answer


print("
运行 RAG Fusion")
result = rag_fusion.invoke(query)
print(result.content)
