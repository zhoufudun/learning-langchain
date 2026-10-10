from dotenv import load_dotenv
load_dotenv()
"""
Ch3-c: 多查询检索（Multi-Query Retrieval）

【学习目标】
- 理解多查询检索的原理
- 学会生成多个查询视角

【问题场景】
单一查询可能错过相关文档。
例如: "LLM 是什么？" 可能错过讨论 "大语言模型" 的文档。

【解决方案】
生成多个不同视角的查询，合并检索结果。
- 原问题: "LLM 是什么？"
- 变体1: "什么是大语言模型？"
- 变体2: "GPT 和 LLM 的关系？"
- 变体3: "语言模型的发展历史？"

【Python 语法】
- split('\n'): 按换行符分割字符串
- {doc.page_content: doc for ...}: 字典推导式，用于去重
- for sublist in document_lists for doc in sublist: 嵌套列表展开
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
    chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(raw_documents)
# 用zhipu 生成查询的向量
embeddings_model = BatchingZhipuEmbeddings(model="embedding-3")
db = PGVector.from_documents(documents, embeddings_model, connection=connection)

# k=5 因为多查询会检索更多文档
retriever = db.as_retriever(search_kwargs={"k": 5})

# ============ 多查询生成 ============
# 提示词：让 LLM 生成 5 个不同视角的查询
perspectives_prompt = ChatPromptTemplate.from_template(
    """You are an AI language model assistant. Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector database.
    By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based  similarity search.
    Provide these alternative questions separated by newlines.
    Original question: {question}""")

llm = ChatOpenAI(model="deepseek-v4-flash-0731")


def parse_queries_output(message):
    """
    解析 LLM 输出，按换行符分割成多个有效查询。

    【Python 语法】
    - split('\n'): 按换行符分割
    - strip(): 去除字符串开头和结尾的空白字符
    - 返回字符串列表: ['查询1', '查询2', ...]
    """
    queries = []

    # LLM 的输出中可能有空行；智谱 Embedding 不接受空字符串。
    for line in message.content.split('\n'):
        cleaned_query = line.strip()

        # 只有非空查询才进入检索流程。
        if cleaned_query:
            queries.append(cleaned_query)

    # 学习阶段打印出来，便于检查实际传给向量检索器的内容。
    print("LLM 生成的查询:", queries)
    return queries


# 多查询生成链
multi_query_gen = perspectives_prompt | llm | parse_queries_output


# ============ 去重函数 ============
def get_unique_union(document_lists):
    """
    合并多个文档列表并去重

    【Python 语法】
    - 嵌套循环推导式:
      for sublist in document_lists  # 遍历每个子列表
          for doc in sublist         # 遍历子列表中的每个文档
    - 字典推导式用于去重（相同内容只保留一个）
    """

    # 用字典去重：key 是文档内容，value 是文档对象
    # deduped_docs = {
    #     doc.page_content: doc
    #     for sublist in document_lists
    #     for doc in sublist
    # }

    # Python 的 dict 类似 Java 的 Map：
    # - key：doc.page_content，即文档正文
    # - value：doc，即完整的 Document 对象
    # 相同 key 再次赋值时会覆盖旧 value，因此能自动去重。
    deduped_docs = {}

    # document_lists 类似：[[文档A, 文档B], [文档A, 文档C]]。
    # 外层循环依次取每个查询对应的一组检索结果。
    for doc_list in document_lists:
        # 内层循环依次取这一组结果中的每篇文档。
        for doc in doc_list:
            # 类似 Java：dedupedDocs.put(doc.getPageContent(), doc)
            # 第二次遇到文档A时，key 相同，只会覆盖原来的文档A。
            deduped_docs[doc.page_content] = doc

    # dict.values() 获取 Map 中全部 Document；list() 转为普通列表。
    # 最终结果类似：[文档A, 文档B, 文档C]。
    return list(deduped_docs.values())


# ============ 完整检索链 ============
# 1. query_gen: 生成多个查询
# 2. retriever.batch: 批量检索（每个查询都检索）
# 3. get_unique_union: 合并去重
retrieval_chain = multi_query_gen | retriever.batch | get_unique_union

# ============ QA 链 ============
prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context: {context} Question: {question} """
)

query = "Who are the key figures in the ancient greek history of philosophy?"

@chain
def multi_query_qb(input: str):
    # 使用多查询检索
    docsList = retrieval_chain.invoke(input)
    print(f"检索到 {len(docsList)} 个文档（去重后）")

    chain = prompt | llm
    # 生成回答
    return chain.invoke({"context": docsList, "question": input})

# 运行
print("运行多查询 QA\n")
result = multi_query_qb.invoke(query)
print(result.content)
