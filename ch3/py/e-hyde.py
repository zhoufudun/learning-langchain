"""
Ch3-e: HyDE（假设文档嵌入）

【学习目标】
- 理解 HyDE 的原理
- 学会用假设答案提高检索效果

【什么是 HyDE？】
HyDE = Hypothetical Document Embeddings
- 先让 LLM 生成一个"假设的答案文档"
- 用这个假设文档去检索
- 因为假设文档比问题更像"答案"，检索效果更好

【为什么有效？】
- 问题: "谁是古希腊哲学家？"
- 假设答案: "苏格拉底、柏拉图、亚里士多德是古希腊著名哲学家..."
- 假设答案和真实文档的语义更接近

【流程图】
用户问题 → LLM 生成假设答案 → 用假设答案检索 → 返回真实文档 → 生成最终回答
"""

# ============ 导入 ============
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain
from langchain_core.output_parsers import StrOutputParser

# ============ 准备知识库和检索器 ============
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"

raw_documents = TextLoader('./test.txt', encoding='utf-8').load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(raw_documents)

embeddings_model = OpenAIEmbeddings()
db = PGVector.from_documents(
    documents, embeddings_model, connection=connection)

retriever = db.as_retriever(search_kwargs={"k": 5})

# ============ HyDE: 生成假设文档 ============
# 提示词：让 LLM 写一段回答问题的文章
prompt_hyde = ChatPromptTemplate.from_template(
    """Please write a passage to answer the question.\n Question: {question} \n Passage:""")

# 假设文档生成链
# StrOutputParser() 把 AIMessage 转换成纯字符串
generate_doc = (
    prompt_hyde
    | ChatOpenAI(temperature=0)
    | StrOutputParser()
)

# ============ HyDE 检索链 ============
# 1. 生成假设文档
# 2. 用假设文档去检索（不是用原问题）
retrieval_chain = generate_doc | retriever

# ============ 完整 QA 链 ============
query = "Who are some lesser known philosophers in the ancient greek history of philosophy?"

prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context: {context} Question: {question} """
)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


@chain
def qa(input):
    """
    HyDE QA 流程:
    1. 用 retrieval_chain 先生成假设文档，再用假设文档检索
    2. 用检索到的真实文档回答问题
    """
    # 获取检索结果（内部会先生成假设文档）
    docs = retrieval_chain.invoke(input)
    print(f"检索到 {len(docs)} 个文档")

    # 用真实文档生成回答
    formatted = prompt.invoke({"context": docs, "question": input})
    answer = llm.invoke(formatted)
    return answer


print("运行 HyDE")
result = qa.invoke(query)
print("\n回答:")
print(result.content)
