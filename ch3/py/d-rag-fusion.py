from asyncpg import TransactionResolutionUnknownError
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


# ============ 准备知识库和检索器 ============
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"

raw_documents = TextLoader('../../test2.txt', encoding='utf-8').load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
documents = text_splitter.split_documents(raw_documents)
# 用zhipu计算向量值
embeddings_model = BatchingZhipuEmbeddings(model="embedding-3")

db = PGVector.from_documents(
    documents,
    embeddings_model,
    connection=connection
)

retriever = db.as_retriever(search_kwargs={"k": 5})

# ============ 查询生成 ============
prompt_rag_fusion = ChatPromptTemplate.from_template(
    """You are a helpful assistant that generates multiple search queries based on a single input query. 
 Generate multiple search queries related to: {question} 
 Output (4 queries):""")


def parse_queries_output(message):
    """
        解析 LLM 输出，按换行符分割成多个有效查询。

        【Python 语法】
        - split('\n'): 按换行符分割
        - strip(): 去除字符串开头和结尾的空白字符
        - 返回字符串列表: ['查询1', '查询2', ...]
        """
    queries = []
    for line in message.content.split('\n'):
        # LLM 的输出中可能有空行；智谱 Embedding 不接受空字符串。
        cleaned_query = line.strip()
        # 只有非空查询才进入检索流程。
        if cleaned_query:
            queries.append(cleaned_query)
        return queries


llm = ChatOpenAI(model="deepseek-v4-flash-0731", temperature=0)
query_gen = prompt_rag_fusion | llm | parse_queries_output

# 测试查询生成
query = "Who are the key figures in the ancient greek history of philosophy?"
# invoke() 是 Chain 的标准调用方法，传入查询字符串
multi_query_result = query_gen.invoke(query)
print("生成的查询:", multi_query_result)

# ============ RRF 融合排序算法 ============
def reciprocal_rank_fusion(results: list[list], k=60):
    """
    倒数排名融合算法 (Reciprocal Rank Fusion)

    【什么是 RRF？】
    把多次检索的结果合并成一个排序列表的算法。
    核心思想：排名越靠前的文档，贡献的分数越高；
              被多个查询检索到的文档，分数会累加。

    【核心公式】
    RRF分数 = Σ (1 / (k + rank))
    - rank: 文档在某次检索中的排名（从 0 开始，0 是第一名）
    - k: 平滑参数（默认 60），防止排名第一的文档权重过大

    【举例】假设有 2 次检索：
    文档A: 检索1排名=0, 检索2排名=2
           分数 = 1/(60+0) + 1/(60+2) = 0.0167 + 0.0161 = 0.0328
    文档B: 检索1排名=1, 检索2排名=0
           分数 = 1/(60+1) + 1/(60+0) = 0.0164 + 0.0167 = 0.0331
    结果: B > A（B 在检索2中排第一，总分更高）

    【参数说明】
    results: list[list]
        多次检索的结果列表，结构: [[查询1的文档列表], [查询2的文档列表], ...]
        例如: [[DocA, DocB, DocC], [DocB, DocD, DocA]]

    k: int
        平滑参数，默认 60
        - k 越大，排名差异的影响越小
        - k=60 是经验值，在大多数场景下效果较好

    【返回值】
    list: 按 RRF 分数从高到低排序的文档列表

    【Python 语法】
    - list[list]: 类型注解，表示"列表的列表"（Python 3.9+）
    - k=60: 默认参数值
    """
    # -------- 初始化两个字典 --------
    # fused_scores: 存储每个文档的 RRF 累计分数
    # key 是文档内容（字符串），value 是分数（浮点数）
    fused_scores = {}

    # documents: 存储文档对象，用于最后返回
    # key 是文档内容（字符串），value 是 Document 对象
    documents = {}

    # -------- 遍历每次检索的结果 --------
    # results 结构: [[DocA, DocB], [DocB, DocC], ...]
    # docs 是其中一次检索返回的文档列表
    for docs in results:
        # -------- enumerate() 语法详解 --------
        # enumerate(docs) 返回 (索引, 元素) 的迭代器
        # 例如: enumerate(['A', 'B', 'C']) → (0,'A'), (1,'B'), (2,'C')
        # 这里 rank 就是排名（0=第一名，1=第二名...）
        for rank, doc in enumerate(docs):
            # 用文档内容作为唯一标识（去重用）
            page_content = doc.page_content
            # 如果是新文档，初始化分数为 0
            if page_content not in fused_scores:
                fused_scores[page_content] = 0
                documents[page_content] = doc # 保存文档对象

            # -------- RRF 核心公式 --------
            # 1 / (k + rank)
            # - rank=0（第一名）: 1/(60+0) = 0.0167
            # - rank=1（第二名）: 1/(60+1) = 0.0164
            # - rank=10: 1/(60+10) = 0.0143
            # 排名越靠前，加的分越多
            fused_scores[page_content] += 1/ (rank+k)

    # -------- 按分数排序 --------
    # sorted() 函数详解:
    #   sorted(iterable, key=函数, reverse=True/False)
    #   - iterable: 要排序的对象（这里是字典的 keys）
    #   - key: 指定排序依据的函数
    #   - reverse=True: 降序（从大到小）

    # lambda 匿名函数详解:
    #   lambda d: fused_scores[d]
    #   等价于:
    #   def get_score(d):
    #       return fused_scores[d]
    #   意思是：对于每个文档内容 d，取它的分数作为排序依据
    reranked_doc_strs = sorted(
        fused_scores,                    # 字典的 keys（文档内容字符串）
        key= lambda d:fused_scores[d],  # 按对应的分数排序
        reverse=True                     # 降序（分数高的排前面）
    )

    # -------- 列表推导式 --------
    # [documents[doc_str] for doc_str in reranked_doc_strs]
    # 把排序后的文档内容字符串转换回 Document 对象
    # 等价于:
    # result = []
    # for doc_str in reranked_doc_strs:
    #     result.append(documents[doc_str])
    # return result
    # return [documents[doc_str] for doc_str in reranked_doc_strs]
    return [documents[after_rerank_doc] for after_rerank_doc in reranked_doc_strs]


# ============ 完整检索链 ============
retrieval_chain = query_gen | retriever.batch | reciprocal_rank_fusion

# invoke() 是 Chain 的标准调用方法
result = retrieval_chain.invoke(query)
print("\n融合排序后的第一个文档:")
print(result[0].page_content[:200] + "...")

# ============ RAG Fusion QA ============
prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context: {context} Question: {question} """
)

query = "Who are the some important yet not well known philosophers in the ancient greek history of philosophy?"


@chain
def rag_fusion(input):
    """RAG Fusion: 多查询 + RRF 融合排序"""
    # invoke() 是 Chain 的标准调用方法
    docs = retrieval_chain.invoke(input)
    formatted = prompt.invoke({"context": docs, "question": input})
    answer = llm.invoke(formatted)
    return answer


print("\n运行 RAG Fusion")
result = rag_fusion.invoke(query)
print(result.content)
