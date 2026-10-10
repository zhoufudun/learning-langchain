from dotenv import load_dotenv
load_dotenv()
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import InMemoryVectorStore
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
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


# --- Create an index of documents ---

urls = [
    "https://blog.langchain.dev/top-5-langgraph-agents-in-production-2024/",
    "https://blog.langchain.dev/langchain-state-of-ai-2024/",
    "https://blog.langchain.dev/introducing-ambient-agents/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

# Add to vectorDB
vectorstore = InMemoryVectorStore.from_documents(
    documents=doc_splits,
    embedding=BatchingZhipuEmbeddings(model="embedding-3"),
)
retriever = vectorstore.as_retriever()

# Retrieve the relevant documents
results = retriever.invoke(
    "What are 2 LangGraph agents used in production in 2024?")

print("Results: 
", results)


# --- Create a grader for retrieved documents ---

# Data model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


# LLM with structured output
llm = ChatOpenAI(temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# Prompt
system = """You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human",
         "Retrieved document: 

 {document} 

 User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader

# --- Grade retrieved documents ---

question = "What are 2 LangGraph agents used in production in 2024?"

# as an example retrieval_grader.invoke({"question": question, "document": doc_txt})
docs = retriever.invoke(question)

doc_txt = docs[0].page_content

result = retrieval_grader.invoke({"question": question, "document": doc_txt})

print("

Grade Result: 
", result)
